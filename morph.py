"""
Morphs an audio file toward a target instrument's feature profile.

Usage:
    python morph.py test1.m4a piano_features.json
    python morph.py test1.m4a piano_features.json --out result.wav
"""

import json
import warnings
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

warnings.filterwarnings("ignore")

N_FFT      = 2048
HOP_LENGTH = 512

BANDS = [
    ("energy_sub_bass", 20,    80),
    ("energy_bass",     80,   250),
    ("energy_low_mid", 250,   500),
    ("energy_mid",     500,  2000),
    ("energy_high_mid",2000,  4000),
    ("energy_high",    4000,  8000),
    ("energy_air",     8000, 20000),
]


def band_energy_ratio(D_power, sr, f_low, f_high):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(D_power.shape[0] - 1) * 2)
    mask  = (freqs >= f_low) & (freqs < f_high)
    total = D_power.sum()
    return float(D_power[mask].sum() / total) if total > 0 else 0.0


def apply_spectral_eq(y, sr, target):
    """Reshape per-band energy to match target ratios."""
    D       = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_mag   = np.abs(D)
    D_power = D_mag ** 2
    freqs   = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    gain = np.ones(len(freqs))
    for key, f_low, f_high in BANDS:
        src = band_energy_ratio(D_power, sr, f_low, f_high)
        tgt = target[key]
        if src > 1e-9:
            # amplitude gain = sqrt(target_power_ratio / source_power_ratio)
            g = np.sqrt(tgt / src)
            g = np.clip(g, 0.0, 8.0)
        else:
            g = 0.0
        freqs_mask = (freqs >= f_low) & (freqs < f_high)
        gain[freqs_mask] = g

    D_eq = D * gain[:, np.newaxis]
    y_eq = librosa.istft(D_eq, hop_length=HOP_LENGTH, length=len(y))

    # preserve original RMS
    rms_in  = np.sqrt(np.mean(y ** 2))
    rms_out = np.sqrt(np.mean(y_eq ** 2))
    if rms_out > 1e-9:
        y_eq *= rms_in / rms_out
    return y_eq


def apply_harmonic_blend(y, target_harmonic_ratio):
    """Rebalance harmonic/percussive components to match target ratio."""
    y_harm, y_perc = librosa.effects.hpss(y)

    h_e = float(np.mean(y_harm ** 2))
    p_e = float(np.mean(y_perc ** 2))
    total = h_e + p_e + 1e-12
    src_h = h_e / total

    # If source is more percussive than target, attenuate percussive component.
    # perc_weight chosen so that resulting ratio hits target.
    if src_h < target_harmonic_ratio and p_e > 1e-12:
        # want: h_e / (h_e + w*p_e) = target_h  →  w = h_e*(1-target_h)/(p_e*target_h)
        w = h_e * (1.0 - target_harmonic_ratio) / (p_e * target_harmonic_ratio)
        w = np.clip(w, 0.0, 1.0)
    else:
        w = 1.0

    y_out = y_harm + y_perc * w

    # preserve original RMS
    rms_in  = np.sqrt(np.mean(y ** 2))
    rms_out = np.sqrt(np.mean(y_out ** 2))
    if rms_out > 1e-9:
        y_out *= rms_in / rms_out
    return y_out


def apply_envelope_shaping(y, sr, target):
    """Apply ADSR envelope that matches the target instrument's profile."""
    rms    = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    n      = len(rms)

    attack_s  = target["attack_time_s"]
    decay_s   = target["decay_time_s"]
    sustain_r = target["sustain_ratio"]
    release_s = target["release_time_s"]

    n_attack  = max(1, int(attack_s  * sr / HOP_LENGTH))
    n_decay   = max(1, int(decay_s   * sr / HOP_LENGTH))
    n_release = max(1, int(release_s * sr / HOP_LENGTH))
    n_sustain = max(0, n - n_attack - n_decay - n_release)

    # Build target envelope shape (normalised 0-1)
    target_env = np.concatenate([
        np.linspace(0.0, 1.0, n_attack),
        np.linspace(1.0, sustain_r, n_decay),
        np.full(n_sustain, sustain_r),
        np.linspace(sustain_r, 0.0, n_release),
    ])[:n]
    if len(target_env) < n:
        target_env = np.pad(target_env, (0, n - len(target_env)), constant_values=0.0)

    # Gain = target_shape / current_shape (clamped to avoid blowup in silent regions)
    gain_frames = np.where(rms > 1e-4, target_env / (rms + 1e-9), target_env)
    gain_frames = np.clip(gain_frames, 0.0, 6.0)

    # Upsample gain from frame rate to sample rate
    gain_samples = np.interp(
        np.arange(len(y)),
        np.arange(n) * HOP_LENGTH + HOP_LENGTH // 2,
        gain_frames,
    )

    y_shaped = y * gain_samples

    # preserve original RMS
    rms_in  = np.sqrt(np.mean(y ** 2))
    rms_out = np.sqrt(np.mean(y_shaped ** 2))
    if rms_out > 1e-9:
        y_shaped *= rms_in / rms_out
    return y_shaped


def morph(input_path: str, target: dict, output_path: str, sr: int = 22050):
    print(f"Loading {input_path}…")
    y, sr = librosa.load(input_path, sr=sr, mono=True)

    print("Enhancing harmonic content…")
    y = apply_harmonic_blend(y, target["harmonic_ratio"])

    print("Applying spectral EQ…")
    y = apply_spectral_eq(y, sr, target)

    print("Shaping envelope…")
    y = apply_envelope_shaping(y, sr, target)

    # Final peak-normalise to -1 dBFS
    peak = np.max(np.abs(y))
    if peak > 0:
        y *= 0.891 / peak

    sf.write(output_path, y, sr)
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",    help="Input audio file")
    parser.add_argument("features", help="Target features JSON")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--sr",     type=int, default=22050)
    args = parser.parse_args()

    with open(args.features) as f:
        target = json.load(f)

    out_path = args.out or f"{Path(args.input).stem}_piano.wav"
    morph(args.input, target, out_path, args.sr)


if __name__ == "__main__":
    main()
