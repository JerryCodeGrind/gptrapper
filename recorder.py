"""
=============================================================================
  AUDIO FEATURE EXTRACTOR — instrument identification focused
=============================================================================

  Features extracted (instrument-relevant only):
    - MFCCs (13 coefficients, mean + std)
    - Spectral shape (centroid, bandwidth, rolloff, flatness, flux)
    - Band energy ratios (7 bands, sub-bass to air)
    - ADSR envelope (attack, decay, sustain, release)
    - HPSS harmonic/percussive ratio
    - Chroma (12 pitch classes)
    - ZCR (percussive vs tonal separator)

  Silence handling:
    - Trailing/leading silence trimmed automatically
    - Mid-recording pauses stripped via energy threshold
    - All features computed only on active (non-silent) audio

  DEPENDENCIES:
      pip install librosa numpy soundfile pydub rich

  USAGE:
      python audio_feature_extractor.py recording.m4a
      python audio_feature_extractor.py recording.m4a --top-db 25
=============================================================================
"""

import json
import warnings
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
        def rule(self, *a, **kw): print("─" * 60)
    console = Console()


# =============================================================================
#  DATACLASS — only instrument-relevant features
# =============================================================================

@dataclass
class AudioFeatures:
    """Instrument-focused acoustic feature profile."""

    # ── Metadata ──────────────────────────────────────────────
    file_path:               str   = ""
    duration_s:              float = 0.0   # duration of active audio (silence stripped)
    duration_original_s:     float = 0.0   # original file duration
    sample_rate:             int   = 0
    active_ratio:            float = 0.0   # fraction of original that was active audio

    # ── Zero Crossing Rate ────────────────────────────────────
    # High = percussive/noisy, Low = tonal. Fast and reliable separator.
    zcr_mean:                float = 0.0
    zcr_std:                 float = 0.0

    # ── Spectral Shape ────────────────────────────────────────
    spectral_centroid_mean:  float = 0.0
    spectral_centroid_std:   float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean:   float = 0.0
    spectral_flatness_mean:  float = 0.0
    spectral_flux_mean:      float = 0.0

    # ── Frequency Band Energy Ratios ──────────────────────────
    energy_sub_bass:         float = 0.0   # 20–80 Hz
    energy_bass:             float = 0.0   # 80–250 Hz
    energy_low_mid:          float = 0.0   # 250–500 Hz
    energy_mid:              float = 0.0   # 500–2k Hz
    energy_high_mid:         float = 0.0   # 2k–4k Hz
    energy_high:             float = 0.0   # 4k–8k Hz
    energy_air:              float = 0.0   # 8k–20k Hz

    # ── MFCCs — Timbre Fingerprint ────────────────────────────
    mfcc_means:              list  = field(default_factory=list)
    mfcc_stds:               list  = field(default_factory=list)

    # ── Chroma Features ───────────────────────────────────────
    chroma_means:            list  = field(default_factory=list)
    chroma_energy:           float = 0.0

    # ── ADSR Envelope ─────────────────────────────────────────
    attack_time_s:           float = 0.0
    attack_slope:            float = 0.0
    decay_time_s:            float = 0.0
    sustain_ratio:           float = 0.0
    release_time_s:          float = 0.0

    # ── Harmonic / Percussive Ratio ───────────────────────────
    harmonic_ratio:          float = 0.0
    percussive_ratio:        float = 0.0


# =============================================================================
#  LOADER
# =============================================================================

def load_audio(file_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()

    if ext in (".wav", ".flac", ".ogg", ".aiff"):
        return librosa.load(file_path, sr=target_sr, mono=True)

    try:
        from pydub import AudioSegment
        import tempfile, os
        audio_seg = AudioSegment.from_file(file_path)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(target_sr)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        audio_seg.export(tmp_path, format="wav")
        y, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
        os.unlink(tmp_path)
        return y, sr
    except ImportError:
        pass

    try:
        return librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load {file_path}.\n"
            "Install pydub and ffmpeg:\n"
            "  pip install pydub\n"
            "  brew install ffmpeg\n"
            f"Original error: {e}"
        )


# =============================================================================
#  SILENCE STRIPPER
# =============================================================================

def strip_silence(y: np.ndarray, sr: int, top_db: float = 20,
                  frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Remove silence from audio — both edges AND mid-recording pauses.

    top_db: frames quieter than (peak - top_db) dB are considered silent.
            20 dB works well for most instrument recordings.
            Increase to 30 if too much is being stripped.
    """
    rms    = librosa.feature.rms(y=y, frame_length=frame_length,
                                  hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    active_mask = rms_db > -top_db

    if not np.any(active_mask):
        return y  # entire signal silent — return as-is

    # Map frame mask back to samples
    active_samples = np.zeros(len(y), dtype=bool)
    for i, is_active in enumerate(active_mask):
        start = i * hop_length
        end   = min(start + hop_length, len(y))
        active_samples[start:end] = is_active

    y_active = y[active_samples]

    # Need at least 1 second for reliable features
    if len(y_active) < sr:
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return y_trimmed

    return y_active


# =============================================================================
#  HELPERS
# =============================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def band_energy_ratio(D_power: np.ndarray, sr: int,
                      f_low: float, f_high: float) -> float:
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(D_power.shape[0] - 1) * 2)
    mask  = (freqs >= f_low) & (freqs < f_high)
    total = D_power.sum()
    return 0.0 if total == 0 else float(D_power[mask].sum() / total)


def estimate_attack(rms: np.ndarray, sr: int, hop: int) -> tuple[float, float]:
    peak_idx  = int(np.argmax(rms))
    peak_val  = rms[peak_idx]
    start_idx = next(
        (i for i, v in enumerate(rms[:peak_idx]) if v >= 0.1 * peak_val), 0
    )
    attack_time = max(peak_idx - start_idx, 1) * hop / sr
    peak_db  = float(librosa.amplitude_to_db(np.array([peak_val]))[0])
    start_db = float(librosa.amplitude_to_db(np.array([rms[start_idx] + 1e-9]))[0])
    return attack_time, (peak_db - start_db) / attack_time


def estimate_decay(rms: np.ndarray, sr: int, hop: int) -> float:
    peak_idx = int(np.argmax(rms))
    target   = rms[peak_idx] / np.e
    for i in range(peak_idx, len(rms)):
        if rms[i] <= target:
            return (i - peak_idx) * hop / sr
    return (len(rms) - peak_idx) * hop / sr


def estimate_release(rms: np.ndarray, sr: int, hop: int) -> float:
    threshold = 0.05 * rms.max()
    start_idx = int(0.9 * len(rms))
    for i in range(len(rms) - 1, start_idx, -1):
        if rms[i] >= threshold:
            return (len(rms) - i) * hop / sr
    return 0.0


# =============================================================================
#  MAIN EXTRACTOR
# =============================================================================

def extract_features(file_path: str,
                     sr_target:  int   = 22050,
                     n_fft:      int   = 2048,
                     hop_length: int   = 512,
                     n_mfcc:     int   = 13,
                     n_mels:     int   = 128,
                     top_db:     float = 20.0) -> AudioFeatures:

    feat = AudioFeatures(file_path=str(file_path))

    # Load
    y_raw, sr            = load_audio(file_path, target_sr=sr_target)
    feat.sample_rate     = sr
    feat.duration_original_s = len(y_raw) / sr

    # Strip silence
    y                = strip_silence(y_raw, sr, top_db=top_db,
                                     frame_length=n_fft, hop_length=hop_length)
    feat.duration_s  = len(y) / sr
    feat.active_ratio = len(y) / len(y_raw)

    if RICH:
        console.print(
            f"[dim]Active audio: {feat.duration_s:.2f}s "
            f"({feat.active_ratio*100:.0f}% of "
            f"{feat.duration_original_s:.2f}s original)[/dim]"
        )

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(
              y, frame_length=n_fft, hop_length=hop_length)[0]
    feat.zcr_mean = float(np.mean(zcr))
    feat.zcr_std  = float(np.std(zcr))

    # STFT
    D       = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    D_mag   = np.abs(D)
    D_power = D_mag ** 2

    # Spectral shape
    sc = librosa.feature.spectral_centroid(S=D_mag, sr=sr, n_fft=n_fft)[0]
    feat.spectral_centroid_mean = float(np.mean(sc))
    feat.spectral_centroid_std  = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(S=D_mag, sr=sr, n_fft=n_fft)[0]
    feat.spectral_bandwidth_mean = float(np.mean(sb))

    sr_feat = librosa.feature.spectral_rolloff(
                  S=D_mag, sr=sr, n_fft=n_fft, roll_percent=0.85)[0]
    feat.spectral_rolloff_mean = float(np.mean(sr_feat))

    sf_feat = librosa.feature.spectral_flatness(S=D_mag, n_fft=n_fft)[0]
    feat.spectral_flatness_mean = float(np.mean(sf_feat))

    flux = np.sqrt(np.sum(np.diff(D_mag, axis=1) ** 2, axis=0))
    feat.spectral_flux_mean = float(np.mean(flux))

    # Band energy ratios
    feat.energy_sub_bass = band_energy_ratio(D_power, sr,    20,    80)
    feat.energy_bass     = band_energy_ratio(D_power, sr,    80,   250)
    feat.energy_low_mid  = band_energy_ratio(D_power, sr,   250,   500)
    feat.energy_mid      = band_energy_ratio(D_power, sr,   500,  2000)
    feat.energy_high_mid = band_energy_ratio(D_power, sr,  2000,  4000)
    feat.energy_high     = band_energy_ratio(D_power, sr,  4000,  8000)
    feat.energy_air      = band_energy_ratio(D_power, sr,  8000, 20000)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  n_fft=n_fft, hop_length=hop_length,
                                  n_mels=n_mels)
    feat.mfcc_means = [float(v) for v in np.mean(mfccs, axis=1)]
    feat.mfcc_stds  = [float(v) for v in np.std(mfccs,  axis=1)]

    # Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_mean       = np.mean(chroma, axis=1)
    feat.chroma_means  = [float(v) for v in chroma_mean]
    feat.chroma_energy = float(np.mean(chroma_mean))

    # ADSR
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    feat.attack_time_s, feat.attack_slope = estimate_attack(rms, sr, hop_length)
    feat.decay_time_s   = estimate_decay(rms, sr, hop_length)
    feat.release_time_s = estimate_release(rms, sr, hop_length)
    mid_s = len(rms) // 4
    mid_e = 3 * len(rms) // 4
    feat.sustain_ratio = float(
        np.mean(rms[mid_s:mid_e]) / (float(np.max(rms)) + 1e-9)
    )

    # HPSS
    y_harm, y_perc = librosa.effects.hpss(y)
    h_e   = float(np.mean(y_harm ** 2))
    p_e   = float(np.mean(y_perc ** 2))
    total = h_e + p_e + 1e-12
    feat.harmonic_ratio   = h_e / total
    feat.percussive_ratio = p_e / total

    return feat


# =============================================================================
#  REPORT PRINTER
# =============================================================================

def print_report(feat: AudioFeatures) -> None:
    if not RICH:
        print(json.dumps(asdict(feat), indent=2))
        return

    console.rule("[bold]Audio Feature Extraction Report[/bold]")

    console.print(Panel(
        f"[b]File:[/b] {feat.file_path}\n"
        f"[b]Active duration:[/b] {feat.duration_s:.3f}s   "
        f"[b]Original:[/b] {feat.duration_original_s:.3f}s   "
        f"[b]Active fraction:[/b] {feat.active_ratio*100:.0f}%   "
        f"[b]Sample rate:[/b] {feat.sample_rate} Hz",
        title="Metadata", border_style="dim"
    ))

    hp = ("mostly harmonic"   if feat.harmonic_ratio > 0.65 else
          "mostly percussive" if feat.harmonic_ratio < 0.35 else
          "balanced")
    console.print(Panel(
        f"[b]H/P balance:[/b] {hp}   "
        f"[b]Harmonic ratio:[/b] {feat.harmonic_ratio:.2f}   "
        f"[b]Tonal/noise:[/b] {1-feat.spectral_flatness_mean:.2f}   "
        f"[b]ZCR:[/b] {feat.zcr_mean:.4f}",
        title="Quick summary", border_style="blue"
    ))

    def tbl(title, rows):
        t = Table(title=title, box=box.SIMPLE, show_header=True,
                  header_style="bold", min_width=50)
        t.add_column("Feature", style="dim", min_width=28)
        t.add_column("Value", justify="right")
        for name, val in rows: t.add_row(name, val)
        return t

    console.print(tbl("Spectral Shape", [
        ("Centroid mean",  f"{feat.spectral_centroid_mean:.1f} Hz"),
        ("Centroid std",   f"{feat.spectral_centroid_std:.1f} Hz"),
        ("Bandwidth mean", f"{feat.spectral_bandwidth_mean:.1f} Hz"),
        ("Rolloff (85%)",  f"{feat.spectral_rolloff_mean:.1f} Hz"),
        ("Flatness",       f"{feat.spectral_flatness_mean:.4f}  (0=tonal 1=noise)"),
        ("Flux mean",      f"{feat.spectral_flux_mean:.3f}"),
    ]))

    console.print(tbl("Frequency Band Energy", [
        ("Sub-bass  20–80 Hz",  f"{feat.energy_sub_bass*100:.1f}%"),
        ("Bass      80–250 Hz", f"{feat.energy_bass*100:.1f}%"),
        ("Low-mid  250–500 Hz", f"{feat.energy_low_mid*100:.1f}%"),
        ("Mid      500–2k Hz",  f"{feat.energy_mid*100:.1f}%"),
        ("High-mid   2–4k Hz",  f"{feat.energy_high_mid*100:.1f}%"),
        ("High       4–8k Hz",  f"{feat.energy_high*100:.1f}%"),
        ("Air        8–20k Hz", f"{feat.energy_air*100:.1f}%"),
    ]))

    console.print(tbl("MFCCs — Timbre", [
        (f"MFCC {i+1:02d}  mean / std",
         f"{feat.mfcc_means[i]:+.2f}  /  {feat.mfcc_stds[i]:.2f}")
        for i in range(len(feat.mfcc_means))
    ]))

    console.print(tbl("Chroma", [
        (f"Chroma {NOTE_NAMES[i]}", f"{feat.chroma_means[i]:.3f}")
        for i in range(len(feat.chroma_means))
    ] + [("Overall chroma energy", f"{feat.chroma_energy:.3f}")]))

    console.print(tbl("ADSR Envelope", [
        ("Attack time",   f"{feat.attack_time_s*1000:.1f} ms"),
        ("Attack slope",  f"{feat.attack_slope:.1f} dB/s"),
        ("Decay time",    f"{feat.decay_time_s*1000:.1f} ms"),
        ("Sustain ratio", f"{feat.sustain_ratio:.3f}"),
        ("Release time",  f"{feat.release_time_s*1000:.1f} ms"),
    ]))

    console.print(tbl("Harmonic / Percussive", [
        ("Harmonic ratio",   f"{feat.harmonic_ratio:.3f}"),
        ("Percussive ratio", f"{feat.percussive_ratio:.3f}"),
        ("Balance",          hp),
    ]))

    console.rule()


# =============================================================================
#  JSON — numpy-safe serialiser
# =============================================================================

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def save_json(feat: AudioFeatures, out_path: Optional[str] = None) -> str:
    if out_path is None:
        out_path = str(
            Path(feat.file_path).parent /
            f"{Path(feat.file_path).stem}_features.json"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(feat), f, indent=2, cls=_NumpyEncoder)
    return out_path


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract instrument-identification audio features."
    )
    parser.add_argument("file",
                        help="Path to audio file (.m4a, .mp3, .wav, …)")
    parser.add_argument("--sr",       type=int,   default=22050)
    parser.add_argument("--n-fft",    type=int,   default=2048)
    parser.add_argument("--hop",      type=int,   default=512)
    parser.add_argument("--n-mfcc",   type=int,   default=13)
    parser.add_argument("--top-db",   type=float, default=20.0,
                        help="Silence threshold in dB (default: 20). "
                             "Raise to 30 if too much is being stripped.")
    parser.add_argument("--out",      type=str,   default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    if RICH:
        with Progress(SpinnerColumn(),
                      TextColumn("[progress.description]{task.description}"),
                      BarColumn(), transient=True) as progress:
            task = progress.add_task("Extracting features…", total=None)
            feat = extract_features(
                args.file, sr_target=args.sr, n_fft=args.n_fft,
                hop_length=args.hop, n_mfcc=args.n_mfcc, top_db=args.top_db
            )
            progress.update(task, completed=1, total=1)
    else:
        print("Extracting features…")
        feat = extract_features(
            args.file, sr_target=args.sr, n_fft=args.n_fft,
            hop_length=args.hop, n_mfcc=args.n_mfcc, top_db=args.top_db
        )

    if not args.json_only:
        print_report(feat)

    json_path = save_json(feat, args.out)

    if RICH:
        console.print(
            f"\n[green]✓[/green] Done in [b]{time.time()-t0:.2f}s[/b]"
            f"  →  [b]{json_path}[/b]"
        )
    else:
        print(f"\nDone in {time.time()-t0:.2f}s → {json_path}")


# =============================================================================
#  IMPORTABLE API
# =============================================================================

def extract_from_file(file_path: str, **kwargs) -> dict:
    """
    Use as a library:

        from audio_feature_extractor import extract_from_file
        features = extract_from_file("recording.m4a")
        print(features["harmonic_ratio"])
        print(features["mfcc_means"])
    """
    return asdict(extract_features(file_path, **kwargs))


if __name__ == "__main__":
    main()