import base64
import csv
import io
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

RECORDINGS      = sorted(glob("test*.m4a"))
INSTRUMENTS_DIR = "instrument_recordings"
SR              = 22050
TOP_DB          = 20
N_FFT           = 2048
HOP             = 512
N_MFCC          = 13
N_CHROMA        = 12

BANDS = [
    ("energy_sub_bass", 20,    80),
    ("energy_bass",     80,   250),
    ("energy_low_mid",  250,  500),
    ("energy_mid",      500,  2000),
    ("energy_high_mid", 2000, 4000),
    ("energy_high",     4000, 8000),
    ("energy_air",      8000, 20000),
]

SCALAR_FEATS = (
    ["spectral_centroid_mean", "spectral_bandwidth_mean",
     "spectral_rolloff_mean", "spectral_flatness_mean", "spectral_flux_mean"]
    + [b[0] for b in BANDS]
    + ["harmonic_ratio", "attack_time", "decay_time", "sustain_ratio", "release_time"]
)
MFCC_COLS   = [f"mfcc_{i}"   for i in range(N_MFCC)]
CHROMA_COLS = [f"chroma_{i}" for i in range(N_CHROMA)]
ALL_COLS    = SCALAR_FEATS + MFCC_COLS + CHROMA_COLS

_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def hz_to_note(hz: float) -> str:
    if hz <= 0:
        return "?"
    midi = round(12 * np.log2(hz / 440.0) + 69)
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

def nearest_c_hz(hz: float) -> float:
    """Return the frequency of the C note closest to hz."""
    if hz <= 0:
        return hz
    midi        = 12 * np.log2(hz / 440.0) + 69
    nearest_c   = round(midi / 12) * 12   # C notes fall on midi multiples of 12
    return float(440.0 * 2 ** ((nearest_c - 69) / 12))


# ── audio loading ─────────────────────────────────────────────

def load_audio(path):
    p = Path(path)
    if p.suffix.lower() in (".wav", ".flac", ".ogg", ".aiff"):
        return librosa.load(str(p), sr=SR, mono=True)
    seg = AudioSegment.from_file(str(p)).set_channels(1).set_frame_rate(SR)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        seg.export(tmp.name, format="wav")
        y, sr = librosa.load(tmp.name, sr=SR, mono=True)
        os.unlink(tmp.name)
    return y, sr


# ── silence stripping ─────────────────────────────────────────

def strip_silence(y):
    peak_sample = int(np.argmax(np.abs(y)))
    start  = max(0, peak_sample - int(0.002 * SR))
    rms    = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    end    = len(y)
    for i in range(peak_sample // HOP, len(rms)):
        if rms_db[i] < -TOP_DB:
            end = min(len(y), i * HOP)
            break
    return y[start:end]


# ── feature extraction ────────────────────────────────────────

@dataclass
class Features:
    spectral_centroid_mean:  float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_rolloff_mean:   float = 0.0
    spectral_flatness_mean:  float = 0.0
    spectral_flux_mean:      float = 0.0
    energy_sub_bass:         float = 0.0
    energy_bass:             float = 0.0
    energy_low_mid:          float = 0.0
    energy_mid:              float = 0.0
    energy_high_mid:         float = 0.0
    energy_high:             float = 0.0
    energy_air:              float = 0.0
    harmonic_ratio:          float = 0.0
    attack_time:             float = 0.0
    decay_time:              float = 0.0
    sustain_ratio:           float = 0.0
    release_time:            float = 0.0
    mfcc_means:              list  = field(default_factory=list)
    chroma_means:            list  = field(default_factory=list)


def _band_energy(D_power, f_low, f_high):
    freqs = librosa.fft_frequencies(sr=SR, n_fft=(D_power.shape[0] - 1) * 2)
    mask  = (freqs >= f_low) & (freqs < f_high)
    total = D_power.sum()
    return float(D_power[mask].sum() / total) if total > 0 else 0.0


def extract(y) -> Features:
    f = Features()

    D     = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    D_mag = np.abs(D)
    D_pow = D_mag ** 2

    f.spectral_centroid_mean  = float(np.mean(librosa.feature.spectral_centroid(S=D_mag, sr=SR, n_fft=N_FFT)[0]))
    f.spectral_bandwidth_mean = float(np.mean(librosa.feature.spectral_bandwidth(S=D_mag, sr=SR, n_fft=N_FFT)[0]))
    f.spectral_rolloff_mean   = float(np.mean(librosa.feature.spectral_rolloff(S=D_mag, sr=SR, n_fft=N_FFT, roll_percent=0.85)[0]))
    f.spectral_flatness_mean  = float(np.mean(librosa.feature.spectral_flatness(S=D_mag, n_fft=N_FFT)[0]))
    f.spectral_flux_mean      = float(np.mean(np.sqrt(np.sum(np.diff(D_mag, axis=1) ** 2, axis=0))))

    for key, f_low, f_high in BANDS:
        setattr(f, key, _band_energy(D_pow, f_low, f_high))

    mfccs        = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    f.mfcc_means = [float(v) for v in np.mean(mfccs, axis=1)]

    chroma         = librosa.feature.chroma_stft(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)
    f.chroma_means = [float(v) for v in np.mean(chroma, axis=1)]

    y_harm, y_perc = librosa.effects.hpss(y)
    h_e = float(np.mean(y_harm ** 2))
    p_e = float(np.mean(y_perc ** 2))
    f.harmonic_ratio = h_e / (h_e + p_e + 1e-12)

    rms      = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    peak_idx = int(np.argmax(rms))

    start_idx = next((i for i, v in enumerate(rms[:peak_idx + 1]) if v >= 0.1 * rms[peak_idx]), 0)
    f.attack_time = max(peak_idx - start_idx, 1) * HOP / SR

    decay_idx = len(rms)
    for i in range(peak_idx, len(rms)):
        if rms[i] < 0.5 * rms[peak_idx]:
            decay_idx = i
            break
    f.decay_time = max(decay_idx - peak_idx, 1) * HOP / SR

    mid_s, mid_e  = len(rms) // 4, 3 * len(rms) // 4
    f.sustain_ratio = float(np.mean(rms[mid_s:mid_e]) / (float(np.max(rms)) + 1e-9))

    f.release_time = max(len(rms) - decay_idx, 1) * HOP / SR

    return f


def feat_vec(f: Features) -> np.ndarray:
    return np.array(
        [getattr(f, k) for k in SCALAR_FEATS]
        + f.mfcc_means[:N_MFCC]
        + f.chroma_means[:N_CHROMA],
        dtype=float
    )


# ── database ──────────────────────────────────────────────────

def _phone_hpf(y: np.ndarray) -> np.ndarray:
    sos = butter(4, 150 / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def _spectral_centroid_hz(y) -> float:
    D_mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    return float(np.mean(librosa.feature.spectral_centroid(S=D_mag, sr=SR, n_fft=N_FFT)[0]))


def _detect_pitch(y) -> float:
    # 1. pyin — tonal instruments (guitar, cello, etc.)
    f0, voiced, _ = librosa.pyin(y, fmin=40, fmax=4000, sr=SR, hop_length=HOP, fill_na=np.nan)
    voiced_f0 = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
    if len(voiced_f0) > 0:
        return float(np.median(voiced_f0))
    # Use spectral centroid to decide what kind of percussive sound this is
    D_mag    = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=D_mag, sr=SR, n_fft=N_FFT)[0]))
    # 2. high centroid = cymbal/hihat (noise-like, energy above 2 kHz) — return centroid
    if centroid > 2000:
        return centroid
    # 3. low centroid = kickdrum/snare — return dominant low-freq FFT peak
    D_mean = D_mag.mean(axis=1)
    freqs  = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mask   = (freqs >= 30) & (freqs <= 500)
    return float(freqs[mask][np.argmax(D_mean[mask])]) if mask.any() else centroid


def build_db() -> list[dict]:
    rows = []
    for path in sorted(Path(INSTRUMENTS_DIR).iterdir()):
        if path.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            continue
        print(f"  {path.name}")
        y, _ = load_audio(str(path))
        y    = strip_silence(y)
        base_pitch = _detect_pitch(y)   # pitch from raw signal, same as detect_pitch.py
        y    = _phone_hpf(y)
        f    = extract(y)
        print(f"    base_pitch={base_pitch:.1f} Hz  ({hz_to_note(base_pitch)})")

        row = {"instrument": path.stem, "base_pitch_hz": base_pitch}
        for k in SCALAR_FEATS:
            row[k] = getattr(f, k)
        for i, v in enumerate(f.mfcc_means[:N_MFCC]):
            row[f"mfcc_{i}"] = v
        for i, v in enumerate(f.chroma_means[:N_CHROMA]):
            row[f"chroma_{i}"] = v
        rows.append(row)

    with open("instruments_db.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["instrument", "base_pitch_hz"] + ALL_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_db() -> list[dict]:
    with open("instruments_db.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        rows   = list(reader)
    if not rows or not all(c in rows[0] for c in ALL_COLS):
        print("instruments_db.csv schema mismatch — rebuilding…")
        return build_db()
    return [{k: (v if k == "instrument" else float(v)) for k, v in row.items()}
            for row in rows]


# ── morphing ──────────────────────────────────────────────────

def _apply_adsr_envelope(y, attack_s, decay_s, sustain_ratio, release_s):
    n       = len(y)
    a_n     = max(int(attack_s * SR), 1)
    d_n     = max(int(decay_s  * SR), 1)
    r_n     = max(int(release_s * SR), 1)
    sus_lvl = float(np.clip(sustain_ratio, 0.01, 1.0))

    env   = np.ones(n)
    a_end = min(a_n, n)
    env[:a_end] = np.linspace(0, 1, a_end)
    d_end = min(a_end + d_n, n)
    if d_end > a_end:
        env[a_end:d_end] = np.linspace(1, sus_lvl, d_end - a_end)
    s_end = max(n - r_n, d_end)
    env[d_end:s_end] = sus_lvl
    if s_end < n:
        env[s_end:] = np.linspace(sus_lvl, 0, n - s_end)

    return (y * env).astype(np.float32)


def morph(y, src: Features, target: dict) -> np.ndarray:
    rms_orig = np.sqrt(np.mean(y ** 2))

    # HPSS blend: scale harmonic or percussive component toward target ratio
    y_harm, y_perc = librosa.effects.hpss(y)
    tgt_h = target["harmonic_ratio"]
    src_h = src.harmonic_ratio
    if tgt_h < src_h:
        # target more percussive: reduce harmonic content proportionally
        w_harm    = float(np.clip(tgt_h / (src_h + 1e-12), 0, 1))
        y_blended = y_harm * w_harm + y_perc
    else:
        # target more tonal: reduce percussive content
        h_e   = float(np.mean(y_harm ** 2))
        p_e   = float(np.mean(y_perc ** 2))
        w_perc = float(np.clip(h_e * (1 - tgt_h) / (p_e * tgt_h + 1e-12), 0, 1)) if p_e > 1e-12 else 1.0
        y_blended = y_harm + y_perc * w_perc

    # Spectral flatness nudge on top of HPSS blend
    src_flat = src.spectral_flatness_mean
    tgt_flat = target["spectral_flatness_mean"]
    if tgt_flat > src_flat * 1.5:
        y_blended = 0.7 * y_blended + 0.3 * y_perc
    elif tgt_flat < src_flat * 0.5:
        y_blended = 0.7 * y_blended + 0.3 * y_harm

    # Spectral EQ — wider gain range (0.25–4.0) for aggressive transforms
    D     = librosa.stft(y_blended, n_fft=N_FFT, hop_length=HOP)
    D_mag = np.abs(D)
    D_pow = D_mag ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    gain  = np.ones(len(freqs))
    for key, f_low, f_high in BANDS:
        src_e = _band_energy(D_pow, f_low, f_high)
        tgt_e = target[key]
        g     = float(np.clip(np.sqrt(tgt_e / src_e) if src_e > 1e-9 else 1.0, 0.25, 4.0))
        gain[(freqs >= f_low) & (freqs < f_high)] = g
    y_eq = librosa.istft(D * gain[:, None], hop_length=HOP, length=len(y_blended))

    # ADSR envelope shaping (70% morphed, 30% original)
    y_adsr = _apply_adsr_envelope(
        y_eq,
        target["attack_time"], target["decay_time"],
        target["sustain_ratio"], target["release_time"]
    )
    y_eq = 0.7 * y_adsr + 0.3 * y_eq[:len(y_adsr)]

    # Wet/dry mix — 75% morphed so transforms actually take effect
    y_out = 0.75 * y_eq + 0.25 * y[:len(y_eq)]

    rms_out = np.sqrt(np.mean(y_out ** 2))
    if rms_out > 1e-9:
        y_out *= rms_orig / rms_out
    return y_out


# ── pitch adjustment ──────────────────────────────────────────

def pitch_adjust(y, target_pitch_hz: float, src_y=None, label: str = "") -> np.ndarray:
    if target_pitch_hz <= 0:
        return y
    detect_from = src_y if src_y is not None else y
    src_hz      = _detect_pitch(detect_from)
    n_steps = float(np.clip(12 * np.log2(target_pitch_hz / src_hz), -12, 12)) if src_hz > 0 else 0.0
    print(f"    pitch {label:<12s} detected={src_hz:.1f}Hz ({hz_to_note(src_hz)})  target={target_pitch_hz:.1f}Hz ({hz_to_note(target_pitch_hz)})  shift={n_steps:+.1f} st")
    if src_hz <= 0 or abs(n_steps) < 0.5:
        return y
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)


# ── server-layer constants and functions (used by server.py) ──

INSTRUMENT_PITCH: dict[str, "float | None"] = {
    "piano":     440.0,
    "violin":    440.0,
    "cello":     220.0,
    "aguitar":   196.0,
    "eguitar":   196.0,
    "bguitar":    98.0,
    "flute":     880.0,
    "trumpet":   440.0,
    "kickdrum":   None,
    "snaredrum":  None,
    "hihat":      None,
}

# General MIDI drum note → slot mapping
GM_DRUM_NOTES = {
    "kickdrum":  {35, 36},
    "snaredrum": {38, 40},
    "hihat":     {42, 44, 46},
}

# Per-slot mix gain. Bass dominates if equal-volume because its peaks coincide
# with everything else. Hi-hat trimmed because high transients are sharp.
SLOT_GAIN: dict[str, float] = {
    "piano":     0.7,
    "violin":    0.7,
    "cello":     0.6,
    "aguitar":   0.7,
    "eguitar":   0.7,
    "bguitar":   0.4,
    "flute":     0.7,
    "trumpet":   0.7,
    "kickdrum":  0.6,
    "snaredrum": 0.6,
    "hihat":     0.5,
}

AudioInput = Union[str, Path, bytes]


def _load_y_from_input(audio_input: AudioInput) -> np.ndarray:
    """Load audio (path or raw bytes) → mono float32 at SR with phone HPF."""
    if isinstance(audio_input, (str, Path)):
        y_raw, _ = load_audio(audio_input)
    else:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            tmp.write(audio_input)
            tmp_path = tmp.name
        try:
            y_raw, _ = load_audio(tmp_path)
        finally:
            os.unlink(tmp_path)
    return _phone_hpf(y_raw)


def classify_sound(audio_input: AudioInput) -> dict:
    """Classify against the 11 trained instruments. Returns scores + best_match."""
    y = strip_silence(_load_y_from_input(audio_input))
    db = load_db()

    mat = np.array([[row[c] for c in ALL_COLS] for row in db])
    mean = mat.mean(0)
    std = mat.std(0) + 1e-9
    weights = np.where([c.startswith("mfcc_") for c in ALL_COLS], 2.0, 1.0)
    db_n = (mat - mean) / std

    q_n = (feat_vec(extract(y)) - mean) / std
    dists = np.sqrt(np.sum(weights * (db_n - q_n) ** 2, axis=1))

    sims = 1.0 / (1.0 + dists)
    sims = sims / sims.sum()

    scores = {row["instrument"]: float(s) for row, s in zip(db, sims)}
    best = max(scores, key=scores.get)
    return {"scores": scores, "best_match": best}


def segment_sounds(
    audio_input: AudioInput,
    *,
    top_db: float = 30.0,
    min_duration_s: float = 0.06,
    min_peak_ratio_db: float = -25.0,
    max_segments: int = 32,
) -> list[dict]:
    """Split a long recording into discrete sounds and classify each.

    Quiet/short segments (likely room noise) are filtered out via
    `min_duration_s` and `min_peak_ratio_db` (vs the loudest peak in the take).
    Returns a list ordered by occurrence in the recording.
    """
    y = _load_y_from_input(audio_input)
    if len(y) == 0:
        return []

    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=N_FFT, hop_length=HOP
    )
    if len(intervals) == 0:
        return []

    overall_peak = float(np.max(np.abs(y))) or 1e-9

    db = load_db()
    mat = np.array([[row[c] for c in ALL_COLS] for row in db])
    mean = mat.mean(0)
    std = mat.std(0) + 1e-9
    weights = np.where([c.startswith("mfcc_") for c in ALL_COLS], 2.0, 1.0)
    db_n = (mat - mean) / std

    out: list[dict] = []
    for start, end in intervals:
        seg = y[int(start):int(end)]
        dur = (int(end) - int(start)) / SR
        if dur < min_duration_s:
            continue
        seg_peak = float(np.max(np.abs(seg)))
        peak_db = 20.0 * float(np.log10(max(seg_peak / overall_peak, 1e-9)))
        if peak_db < min_peak_ratio_db:
            continue

        try:
            q_n = (feat_vec(extract(seg)) - mean) / std
            dists = np.sqrt(np.sum(weights * (db_n - q_n) ** 2, axis=1))
            sims = 1.0 / (1.0 + dists)
            sims = sims / sims.sum()
            scores = {row["instrument"]: float(s) for row, s in zip(db, sims)}
            best = max(scores, key=scores.get)
        except Exception:
            scores = {}
            best = None

        buf = io.BytesIO()
        sf.write(buf, seg.astype(np.float32), SR, subtype="PCM_16", format="WAV")
        wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        out.append({
            "index": len(out),
            "start_s": float(int(start)) / SR,
            "end_s": float(int(end)) / SR,
            "duration_s": float(dur),
            "peak_db": float(peak_db),
            "scores": scores,
            "best_match": best,
            "audio_wav_b64": wav_b64,
        })
        if len(out) >= max_segments:
            break

    return out


def process_sound(audio_input: AudioInput, target_instrument: str) -> bytes:
    """Morph + pitch-adjust a sound toward target. Returns 16-bit PCM mono WAV bytes."""
    if target_instrument not in INSTRUMENT_PITCH:
        raise ValueError(f"unknown instrument: {target_instrument}")

    y = strip_silence(_load_y_from_input(audio_input))

    db = load_db()
    target_row = next((r for r in db if r["instrument"] == target_instrument), None)
    if target_row is None:
        raise ValueError(f"no DB row for {target_instrument} — rebuild instruments_db.csv?")

    src_features = extract(y)
    y_out = morph(y, src_features, target_row)

    target_hz = INSTRUMENT_PITCH.get(target_instrument)
    if target_hz is not None:
        y_out = pitch_adjust(y_out, target_hz, src_y=y, label=target_instrument)

    peak = float(np.max(np.abs(y_out)))
    if peak > 0:
        y_out = y_out * (0.891 / peak)

    buf = io.BytesIO()
    sf.write(buf, y_out.astype(np.float32), SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def identify_tracks(midi) -> dict:
    """Heuristically map MIDI tracks to instrument slots."""
    track_map: dict = {}

    drum_inst = None
    melodic = []
    for inst in midi.instruments:
        if inst.is_drum and len(inst.notes) > 0:
            if drum_inst is None or len(inst.notes) > len(drum_inst.notes):
                drum_inst = inst
        elif len(inst.notes) > 0:
            avg_pitch = float(np.mean([n.pitch for n in inst.notes]))
            melodic.append((avg_pitch, len(inst.notes), inst))

    if drum_inst is not None:
        track_map["__drums__"] = drum_inst

    if melodic:
        melodic_by_pitch = sorted(melodic, key=lambda x: x[0])
        bass_inst = melodic_by_pitch[0][2]
        track_map["bguitar"] = bass_inst

        rest = [m for m in melodic if m[2] is not bass_inst]
        if rest:
            rest.sort(key=lambda x: (x[1], x[0]), reverse=True)
            piano_inst = rest[0][2]
            track_map["piano"] = piano_inst

            extras = [m for m in rest if m[2] is not piano_inst]
            for avg_p, _n_notes, inst in extras:
                if avg_p >= 75:
                    candidates = ("violin", "flute", "trumpet")
                elif avg_p >= 60:
                    candidates = ("violin", "trumpet", "flute")
                else:
                    candidates = ("cello", "aguitar", "eguitar")
                for c in candidates:
                    if c not in track_map:
                        track_map[c] = inst
                        break

    return track_map


def pitch_shift_to_hz(
    sample: np.ndarray,
    target_hz: float,
    slot_key: str,
    src_hz_override: "float | None" = None,
    pitch_strength: float = 1.0,
) -> np.ndarray:
    """Pitch-shift a sample so its center pitch becomes target_hz.
    Octave-folds shifts beyond ±12 semitones to avoid librosa's metallic artifacts.
    Pass src_hz_override to use a detected pitch instead of the slot's nominal pitch.
    pitch_strength scales the shift: 1.0 = full shift to target_hz, 0.0 = no shift
    (sample plays at its natural pitch regardless of MIDI note).
    """
    src_hz = src_hz_override if src_hz_override is not None else INSTRUMENT_PITCH.get(slot_key)
    if src_hz is None or src_hz <= 0:
        return sample  # unpitched (drums) — never shift
    n_steps = 12.0 * float(np.log2(target_hz / src_hz))
    while n_steps > 12.0:
        n_steps -= 12.0
    while n_steps < -12.0:
        n_steps += 12.0
    n_steps *= float(pitch_strength)
    if abs(n_steps) < 0.5:
        return sample
    return librosa.effects.pitch_shift(sample, sr=SR, n_steps=n_steps).astype(np.float32)


def _trim_with_fade(sample: np.ndarray, n_samples: int, fade_samples: int = 661) -> np.ndarray:
    """Play `sample` for the MIDI note duration plus a short natural-decay tail.
    The tail (~120 ms) gives smooth note transitions without letting a long
    user recording (e.g. a 5-second boop) drone across the whole song and
    create a fake-reverb / fake-instrument-sustain effect.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    TAIL_S = 0.12
    tail = int(TAIL_S * SR)
    play_len = min(n_samples + tail, len(sample))
    if play_len <= 0:
        return np.zeros(0, dtype=np.float32)
    out = sample[:play_len].astype(np.float32, copy=True)
    fade = min(fade_samples, play_len)
    if fade > 0:
        ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out[-fade:] *= ramp
    return out


def _midi_pitch_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def render_song(
    midi_path: Union[str, Path],
    slot_samples: dict,
    tempo_percent: float = 100.0,
    raw_mode: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    """Render a MIDI file using the user's slot samples.

    raw_mode=False (default): assume `slot_samples` are already morphed (e.g. via
      /morph endpoint). Pitch-shift to MIDI notes using the slot's nominal pitch.
    raw_mode=True: `slot_samples` are user's raw audio. Pitch always shifts fully
      to the MIDI notes (so the melody is correct at any temperature).
      `temperature` ∈ [0, 1] controls TIMBRE only:
        0.0 = raw user timbre (a "boop" stays a boop, just pitched to each note).
        1.0 = morphed to the target instrument's timbre (sounds like a piano note).
        in between: linear timbre blend (raw vs morphed).
    """
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    tempo_factor = tempo_percent / 100.0
    duration = midi.get_end_time() / tempo_factor
    out_len = int(duration * SR) + SR
    out = np.zeros(out_len, dtype=np.float32)

    track_map = identify_tracks(midi)
    t = float(np.clip(temperature, 0.0, 1.0))

    # In raw mode, detect each user sample's natural pitch + precompute the morphed
    # version (if temperature > 0) so we can blend timbre per-slot.
    detected_src_hz: dict[str, float] = {}
    blended_samples: dict[str, np.ndarray] = {}
    if raw_mode:
        db = load_db() if t > 0.0 else []
        for k, s in slot_samples.items():
            arr = np.asarray(s, dtype=np.float32)
            if len(arr) == 0:
                continue
            if INSTRUMENT_PITCH.get(k) is not None:
                try:
                    detected_src_hz[k] = float(_detect_pitch(strip_silence(arr)))
                except Exception:
                    detected_src_hz[k] = INSTRUMENT_PITCH[k] or 220.0
            if t > 0.0:
                target_row = next((r for r in db if r["instrument"] == k), None)
                if target_row is not None:
                    try:
                        morphed = morph(arr, extract(arr), target_row).astype(np.float32)
                        n = min(len(arr), len(morphed))
                        blended_samples[k] = (1.0 - t) * arr[:n] + t * morphed[:n]
                    except Exception:
                        blended_samples[k] = arr

    # Build pool of user-filled melodic slots, ordered by INSTRUMENT_PITCH ascending
    # so we can pick a sensible fallback (lowest filled pitched slot for bass etc.)
    melodic_filled = [
        k for k in slot_samples.keys()
        if INSTRUMENT_PITCH.get(k) is not None and len(np.asarray(slot_samples[k])) > 0
    ]
    melodic_filled.sort(key=lambda k: INSTRUMENT_PITCH[k] or 0.0)

    shift_cache: dict = {}
    melodic_slot_keys = [k for k in track_map.keys() if k != "__drums__"]
    for slot_key in melodic_slot_keys:
        # Fallback: if the assigned slot isn't filled, pick a sensible filled slot.
        render_slot_key = slot_key
        if slot_key not in slot_samples or len(np.asarray(slot_samples[slot_key])) == 0:
            if not melodic_filled:
                continue
            # Bass track ("bguitar") → lowest filled pitched slot. Others → highest.
            render_slot_key = melodic_filled[0] if slot_key == "bguitar" else melodic_filled[-1]
        sample = slot_samples[render_slot_key].astype(np.float32)
        if raw_mode and render_slot_key in blended_samples:
            sample = blended_samples[render_slot_key].astype(np.float32)
        if len(sample) == 0:
            continue
        inst = track_map[slot_key]
        sorted_notes = sorted(inst.notes, key=lambda n: n.start)
        prev_chord_start_midi = -1.0
        chord_position = 0
        CHORD_WINDOW_S = 0.012
        CHORD_STAGGER_S = 0.005
        src_override = detected_src_hz.get(render_slot_key) if raw_mode else None
        # Pitch always tracks MIDI fully — temperature only blends timbre.
        slot_pitch_strength = 1.0

        for note in sorted_notes:
            if abs(note.start - prev_chord_start_midi) < CHORD_WINDOW_S:
                chord_position += 1
            else:
                chord_position = 0
                prev_chord_start_midi = note.start
            note_start = note.start / tempo_factor + chord_position * CHORD_STAGGER_S
            note_dur_sec = (note.end - note.start) / tempo_factor
            note_n_samples = int(note_dur_sec * SR)
            if note_n_samples <= 0:
                continue
            cache_key = (render_slot_key, int(note.pitch))
            shifted = shift_cache.get(cache_key)
            if shifted is None:
                target_hz = _midi_pitch_to_hz(int(note.pitch))
                shifted = pitch_shift_to_hz(
                    sample, target_hz, render_slot_key,
                    src_hz_override=src_override,
                    pitch_strength=slot_pitch_strength,
                )
                shift_cache[cache_key] = shifted
            trimmed = _trim_with_fade(shifted, note_n_samples)
            start_idx = int(note_start * SR)
            end_idx = min(start_idx + len(trimmed), out_len)
            if end_idx > start_idx:
                gain = (note.velocity / 127.0) * SLOT_GAIN.get(render_slot_key, 0.6)
                out[start_idx:end_idx] += trimmed[: end_idx - start_idx] * gain

    drum_inst = track_map.get("__drums__")
    if drum_inst is not None:
        # Fallback chain: if a specific drum slot isn't filled, pick any filled drum
        # slot. If no drum slots are filled at all, fall back to any filled slot.
        drum_filled = [k for k in GM_DRUM_NOTES.keys()
                       if k in slot_samples and len(np.asarray(slot_samples[k])) > 0]
        any_filled = [k for k in slot_samples.keys()
                      if len(np.asarray(slot_samples[k])) > 0]
        for note in drum_inst.notes:
            target_slot = None
            for slot_key, drum_pitches in GM_DRUM_NOTES.items():
                if int(note.pitch) not in drum_pitches:
                    continue
                if slot_key in slot_samples and len(np.asarray(slot_samples[slot_key])) > 0:
                    target_slot = slot_key
                elif drum_filled:
                    target_slot = drum_filled[0]
                elif any_filled:
                    target_slot = any_filled[0]
                break
            if target_slot is None:
                continue
            sample = slot_samples[target_slot].astype(np.float32)
            if len(sample) == 0:
                continue
            note_start = note.start / tempo_factor
            start_idx = int(note_start * SR)
            end_idx = min(start_idx + len(sample), out_len)
            if end_idx > start_idx:
                gain = (note.velocity / 127.0) * SLOT_GAIN.get(target_slot, 0.6)
                out[start_idx:end_idx] += sample[: end_idx - start_idx] * gain

    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out *= 0.891 / peak
    return out


# ── main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    db = load_db() if Path("instruments_db.csv").exists() else build_db()

    mat     = np.array([[row[c] for c in ALL_COLS] for row in db])
    mean    = mat.mean(0)
    std     = mat.std(0) + 1e-9
    weights = np.where([c.startswith("mfcc_") for c in ALL_COLS], 2.0, 1.0)
    db_n    = (mat - mean) / std

    print("Extracting features from recordings…")
    recs = []
    for path in RECORDINGS:
        y_raw, _ = load_audio(path)
        y        = strip_silence(y_raw)
        f        = extract(y)
        q_n      = (feat_vec(f) - mean) / std
        dists    = np.sqrt(np.sum(weights * (db_n - q_n) ** 2, axis=1))
        recs.append({"path": path, "y": y, "feats": f, "dists": dists})
        print(f"  {path}  ({len(y)/SR:.3f}s active)")

    cost             = np.stack([r["dists"] for r in recs])
    row_idx, col_idx = linear_sum_assignment(cost)

    # ── forced assignments (delete to revert to automatic) ───────
    FORCED = {
        "test6": "piano",
    }
    # ─────────────────────────────────────────────────────────────

    print("\nOptimal assignment:")
    total    = 0.0
    segments = []
    for ri, ci in zip(row_idx, col_idx):
        rec    = recs[ri]
        stem   = Path(rec["path"]).stem
        if stem in FORCED:
            forced_name = FORCED[stem]
            target = next((r for r in db if r["instrument"] == forced_name), db[ci])
        else:
            target = db[ci]
        name   = target["instrument"]
        dist   = cost[ri, ci]
        total += dist

        y_out     = morph(rec["y"], rec["feats"], target)
        raw_pitch = target["base_pitch_hz"]
        # noise instruments (hihat/cymbal, centroid > 2000 Hz): morph EQ handles it, skip pitch shift
        target_hz = nearest_c_hz(raw_pitch) if 0 < raw_pitch <= 2000 else 0.0
        y_out     = pitch_adjust(y_out, target_hz, src_y=rec["y"], label=name)
        peak  = np.max(np.abs(y_out))
        if peak > 0:
            y_out *= 0.891 / peak

        sf.write(f"{stem}_{name}.wav", y_out, SR)
        segments.append((name, target["base_pitch_hz"], y_out))
        print(f"  {rec['path']}  →  {name:<12s}  dist={dist:.3f}  → {stem}_{name}.wav")

    # ── jazz arrangement ──────────────────────────────────────
    BPM           = 110
    BEAT          = 60.0 / BPM
    TRIPLET       = BEAT / 3          # one triplet subdivision
    SWING         = TRIPLET * 2       # swing 8th = 2 triplets (long)
    BAR           = BEAT * 4
    N_BARS        = 8
    total_samples = int(BAR * N_BARS * SR)

    # Swing 8th note positions within a bar (long-short triplet feel)
    SWING_8THS = [b * BEAT + o for b in range(4) for o in [0, SWING]]

    # Jazz patterns (in seconds from bar start)
    JAZZ_PATTERNS = {
        "hihat":     SWING_8THS,                                          # ride-cymbal swing 8ths
        "kickdrum":  [0, 2 * BEAT + SWING],                              # beat 1 + and-of-3
        "snaredrum": [BEAT, 3 * BEAT],                                   # beats 2 and 4
        "bguitar":   [b * BEAT for b in range(4)],                       # walking bass every beat
        "aguitar":   [SWING, BEAT + SWING, 2 * BEAT + SWING, 3 * BEAT + SWING],  # offbeat comp
        "piano":     [BEAT, 2 * BEAT + SWING, 3 * BEAT],                # jazz comp stabs
        "cello":     [0, 2 * BEAT + SWING],
        "violin":    [0, SWING, 2 * BEAT],
        "flute":     [0, BEAT + SWING, 2 * BEAT, 3 * BEAT + SWING],
        "trumpet":   [0, 2 * BEAT],
    }

    # Walking bass offsets (per beat within a bar, stacked on top of chord root)
    WALKING = [0, 2, 5, 7]  # root → 2nd → 4th → 5th

    # Detect key from the lowest-pitched tonal instrument assigned
    # only instruments with a real musical pitch (< 2000 Hz) are tonal for arrangement purposes
    tonal_segs = [(name, bpitch) for name, bpitch, _ in segments if 0 < bpitch <= 2000]
    if tonal_segs:
        key_name, key_hz = min(tonal_segs, key=lambda x: x[1])
        key_note = hz_to_note(key_hz)
        # ii–V–I in detected key: ii=+2st, V=+7st, I=0st
        PROGRESSION = [2, 2, 7, 7, 0, 0, 0, 2]
        print(f"\nKey: {key_note}  (from {key_name} @ {key_hz:.1f} Hz)")
        chord_names = [hz_to_note(key_hz * 2 ** (s / 12)) for s in PROGRESSION]
        print(f"Progression: {' – '.join(chord_names)}")
    else:
        PROGRESSION = [0] * N_BARS
        print("\nNo tonal instruments detected — percussion only")

    print("\nBuilding jazz arrangement…")
    out = np.zeros(total_samples, dtype=np.float32)
    for name, base_pitch_hz, y_out in segments:
        pattern  = JAZZ_PATTERNS.get(name, [0])
        is_tonal = 0 < base_pitch_hz <= 2000

        # Pre-compute all needed pitch-shifted clips
        needed = set()
        for bar in range(N_BARS):
            root = PROGRESSION[bar] if is_tonal else 0
            if name == "bguitar" and is_tonal:
                for w in WALKING:
                    needed.add(root + w)
            else:
                needed.add(root)

        clips = {}
        for semitones in needed:
            clips[semitones] = (y_out if semitones == 0
                                else librosa.effects.pitch_shift(y_out, sr=SR, n_steps=float(semitones)))

        track = np.zeros(total_samples, dtype=np.float32)
        for bar in range(N_BARS):
            root = PROGRESSION[bar] if is_tonal else 0
            for beat_i, t_offset in enumerate(pattern):
                if name == "bguitar" and is_tonal:
                    semitones = root + WALKING[beat_i % 4]
                else:
                    semitones = root
                clip  = clips[semitones]
                start = int((bar * BAR + t_offset) * SR)
                end   = min(start + len(clip), total_samples)
                track[start:end] += clip[:end - start]

        peak_t = np.max(np.abs(track))
        print(f"  track {name:<12s}  peak={peak_t:.4f}  len={len(y_out)/SR:.3f}s  tonal={is_tonal}")
        if peak_t > 0:
            track *= 0.3 / peak_t
        out += track

    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.891 / peak
    sf.write("combined.wav", out, SR)

    print(f"\nTotal distance: {total:.3f}")
    print(f"Output: combined.wav  ({N_BARS} bars @ {BPM} BPM)")
