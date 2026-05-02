import csv
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────
RECORDINGS      = sorted(glob("test*.m4a"))
INSTRUMENTS_DIR = "instrument_recordings"
SR              = 22050
TOP_DB          = 20
N_FFT           = 2048
HOP             = 512
N_MFCC          = 13

# typical center pitch (Hz) per instrument; None = unpitched, skip pitch shift
INSTRUMENT_PITCH = {
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
# ─────────────────────────────────────────────────────────────

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
    ["zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean",
     "spectral_rolloff_mean", "spectral_flatness_mean", "spectral_flux_mean"]
    + [b[0] for b in BANDS]
    + ["harmonic_ratio", "attack_slope", "sustain_ratio"]
)
MFCC_COLS = [f"mfcc_{i}" for i in range(N_MFCC)]
ALL_COLS  = SCALAR_FEATS + MFCC_COLS


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
    zcr_mean:               float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean:float = 0.0
    spectral_rolloff_mean:  float = 0.0
    spectral_flatness_mean: float = 0.0
    spectral_flux_mean:     float = 0.0
    energy_sub_bass:        float = 0.0
    energy_bass:            float = 0.0
    energy_low_mid:         float = 0.0
    energy_mid:             float = 0.0
    energy_high_mid:        float = 0.0
    energy_high:            float = 0.0
    energy_air:             float = 0.0
    harmonic_ratio:         float = 0.0
    attack_slope:           float = 0.0
    sustain_ratio:          float = 0.0
    mfcc_means:             list  = field(default_factory=list)


def _band_energy(D_power, f_low, f_high):
    freqs = librosa.fft_frequencies(sr=SR, n_fft=(D_power.shape[0] - 1) * 2)
    mask  = (freqs >= f_low) & (freqs < f_high)
    total = D_power.sum()
    return float(D_power[mask].sum() / total) if total > 0 else 0.0


def extract(y) -> Features:
    f    = Features()
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0]
    f.zcr_mean = float(np.mean(zcr))

    D       = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    D_mag   = np.abs(D)
    D_power = D_mag ** 2

    f.spectral_centroid_mean  = float(np.mean(librosa.feature.spectral_centroid(S=D_mag, sr=SR, n_fft=N_FFT)[0]))
    f.spectral_bandwidth_mean = float(np.mean(librosa.feature.spectral_bandwidth(S=D_mag, sr=SR, n_fft=N_FFT)[0]))
    f.spectral_rolloff_mean   = float(np.mean(librosa.feature.spectral_rolloff(S=D_mag, sr=SR, n_fft=N_FFT, roll_percent=0.85)[0]))
    f.spectral_flatness_mean  = float(np.mean(librosa.feature.spectral_flatness(S=D_mag, n_fft=N_FFT)[0]))
    f.spectral_flux_mean      = float(np.mean(np.sqrt(np.sum(np.diff(D_mag, axis=1) ** 2, axis=0))))

    for key, f_low, f_high in BANDS:
        setattr(f, key, _band_energy(D_power, f_low, f_high))

    mfccs        = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    f.mfcc_means = [float(v) for v in np.mean(mfccs, axis=1)]

    rms       = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    peak_idx  = int(np.argmax(rms))
    start_idx = next((i for i, v in enumerate(rms[:peak_idx]) if v >= 0.1 * rms[peak_idx]), 0)
    attack_t  = max(peak_idx - start_idx, 1) * HOP / SR
    peak_db   = float(librosa.amplitude_to_db(np.array([rms[peak_idx]]))[0])
    start_db  = float(librosa.amplitude_to_db(np.array([rms[start_idx] + 1e-9]))[0])
    f.attack_slope  = (peak_db - start_db) / attack_t
    mid_s, mid_e    = len(rms) // 4, 3 * len(rms) // 4
    f.sustain_ratio = float(np.mean(rms[mid_s:mid_e]) / (float(np.max(rms)) + 1e-9))

    y_harm, y_perc = librosa.effects.hpss(y)
    h_e = float(np.mean(y_harm ** 2))
    p_e = float(np.mean(y_perc ** 2))
    f.harmonic_ratio = h_e / (h_e + p_e + 1e-12)

    return f


def feat_vec(f: Features) -> np.ndarray:
    return np.array([getattr(f, k) for k in SCALAR_FEATS] + f.mfcc_means[:N_MFCC], dtype=float)


# ── database ──────────────────────────────────────────────────

def _phone_hpf(y: np.ndarray) -> np.ndarray:
    sos = butter(4, 150 / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, y).astype(np.float32)


def build_db() -> list[dict]:
    rows = []
    for path in sorted(Path(INSTRUMENTS_DIR).iterdir()):
        if path.suffix.lower() not in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            continue
        print(f"  {path.name}")
        y, _ = load_audio(str(path))
        y    = _phone_hpf(y)
        y    = strip_silence(y)
        f    = extract(y)
        row  = {"instrument": path.stem}
        for k in SCALAR_FEATS:
            row[k] = getattr(f, k)
        for i, v in enumerate(f.mfcc_means[:N_MFCC]):
            row[f"mfcc_{i}"] = v
        rows.append(row)
    with open("instruments_db.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["instrument"] + ALL_COLS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_db() -> list[dict]:
    with open("instruments_db.csv", newline="") as fh:
        return [{k: (v if k == "instrument" else float(v)) for k, v in row.items()}
                for row in csv.DictReader(fh)]


# ── morphing ─────────────────────────────────────────────────

def morph(y, target: dict) -> np.ndarray:
    rms_orig = np.sqrt(np.mean(y ** 2))

    y_harm, y_perc = librosa.effects.hpss(y)
    h_e   = float(np.mean(y_harm ** 2))
    p_e   = float(np.mean(y_perc ** 2))
    tgt_h = target["harmonic_ratio"]
    src_h = h_e / (h_e + p_e + 1e-12)
    w     = float(np.clip(h_e * (1 - tgt_h) / (p_e * tgt_h), 0, 1)) if src_h < tgt_h and p_e > 1e-12 else 1.0
    y_blended = y_harm + y_perc * w

    D       = librosa.stft(y_blended, n_fft=N_FFT, hop_length=HOP)
    D_mag   = np.abs(D)
    D_power = D_mag ** 2
    freqs   = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    gain    = np.ones(len(freqs))
    for key, f_low, f_high in BANDS:
        src = _band_energy(D_power, f_low, f_high)
        tgt = target[key]
        g   = float(np.clip(np.sqrt(tgt / src) if src > 1e-9 else 1.0, 0.5, 2.0))
        gain[(freqs >= f_low) & (freqs < f_high)] = g
    y_eq  = librosa.istft(D * gain[:, None], hop_length=HOP, length=len(y_blended))
    y_out = 0.6 * y_eq + 0.4 * y[:len(y_eq)]

    rms_out = np.sqrt(np.mean(y_out ** 2))
    if rms_out > 1e-9:
        y_out *= rms_orig / rms_out
    return y_out


# ── pitch adjustment ─────────────────────────────────────────

def pitch_adjust(y, instrument_name: str) -> np.ndarray:
    target_hz = INSTRUMENT_PITCH.get(instrument_name)
    if target_hz is None:
        return y

    # try pyin for tonal pitch detection; fall back to spectral centroid
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=4000, sr=SR,
                                  hop_length=HOP, fill_na=np.nan)
    voiced_f0 = f0[voiced & ~np.isnan(f0)] if voiced is not None else np.array([])
    if len(voiced_f0) == 0:
        return y
    src_hz = float(np.median(voiced_f0))

    if src_hz <= 0:
        return y

    n_steps = float(np.clip(12 * np.log2(target_hz / src_hz), -12, 12))
    if abs(n_steps) < 0.5:
        return y

    return librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)


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
        q_n      = (feat_vec(extract(y)) - mean) / std
        dists    = np.sqrt(np.sum(weights * (db_n - q_n) ** 2, axis=1))
        recs.append({"path": path, "y": y, "dists": dists})
        print(f"  {path}  ({len(y)/SR:.3f}s active)")

    # Hungarian algorithm — find assignment minimising total distance
    cost             = np.stack([r["dists"] for r in recs])
    row_idx, col_idx = linear_sum_assignment(cost)

    print("\nOptimal assignment:")
    total    = 0.0
    segments = []   # list of (name, stem, y_out)
    for ri, ci in zip(row_idx, col_idx):
        rec    = recs[ri]
        target = db[ci]
        name   = target["instrument"]
        dist   = cost[ri, ci]
        total += dist

        y_out = morph(rec["y"], target)
        y_out = pitch_adjust(y_out, name)
        peak  = np.max(np.abs(y_out))
        if peak > 0:
            y_out *= 0.891 / peak

        stem = Path(rec["path"]).stem
        sf.write(f"{stem}_{name}.wav", y_out, SR)
        segments.append((name, y_out))
        print(f"  {rec['path']}  →  {name:<12s}  dist={dist:.3f}  → {stem}_{name}.wav")

    # ── music arrangement ────────────────────────────────────────
    BPM    = 120
    BEAT   = 60.0 / BPM
    BAR    = BEAT * 4
    N_BARS = 8
    total_samples = int(BAR * N_BARS * SR)

    PATTERNS = {
        "hihat":     [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
        "kickdrum":  [0, 2],
        "snaredrum": [1, 3],
        "piano":     [0, 2],
        "violin":    [0, 2],
        "cello":     [0, 2],
        "aguitar":   [0, 1, 2, 3],
        "eguitar":   [0, 2],
        "bguitar":   [0, 2],
        "flute":     [0, 2],
        "trumpet":   [0, 2],
    }

    out = np.zeros(total_samples, dtype=np.float32)
    for name, y_out in segments:
        pattern = PATTERNS.get(name, [0])
        track   = np.zeros(total_samples, dtype=np.float32)
        for bar in range(N_BARS):
            for beat_offset in pattern:
                start = int((bar * BAR + beat_offset * BEAT) * SR)
                end   = min(start + len(y_out), total_samples)
                track[start:end] += y_out[:end - start]
        peak_t = np.max(np.abs(track))
        print(f"  track {name:<12s}  peak={peak_t:.4f}  len={len(y_out)/SR:.3f}s")
        if peak_t > 0:
            track *= 0.3 / peak_t
        out += track

    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.891 / peak
    sf.write("combined.wav", out, SR)

    print(f"\nTotal distance: {total:.3f}")
    print(f"Output: combined.wav  ({N_BARS} bars @ {BPM} BPM)")
