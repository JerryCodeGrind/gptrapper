"""
Diagnostic: shows why each test recording matched its instrument,
and the per-feature distance breakdown vs every candidate.
"""
import csv, sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
from pipeline import (load_audio, strip_silence, extract, feat_vec,
                      ALL_COLS, SCALAR_FEATS, MFCC_COLS, RECORDINGS)

with open("instruments_db.csv") as fh:
    db = [{k: (v if k == "instrument" else float(v)) for k, v in r.items()}
          for r in csv.DictReader(fh)]

mat     = np.array([[row[c] for c in ALL_COLS] for row in db])
mean    = mat.mean(0)
std     = mat.std(0) + 1e-9
weights = np.where([c.startswith("mfcc_") for c in ALL_COLS], 2.0, 1.0)
db_n    = (mat - mean) / std

for path in RECORDINGS:
    print(f"\n{'='*70}")
    print(f"  {path}")
    print(f"{'='*70}")

    y_raw, _ = load_audio(path)
    y = strip_silence(y_raw)
    f = extract(y)

    print(f"\n  Key features of recording:")
    for k in ["spectral_centroid_mean","energy_sub_bass","energy_bass",
              "energy_low_mid","energy_mid","harmonic_ratio","zcr_mean"]:
        print(f"    {k:<30} {getattr(f, k):.4f}")

    q   = feat_vec(f)
    q_n = (q - mean) / std
    dists = np.sqrt(np.sum(weights * (db_n - q_n) ** 2, axis=1))
    ranked = sorted(zip(dists, [r["instrument"] for r in db]))

    print(f"\n  Distance ranking:")
    for dist, name in ranked:
        bar = "█" * int(dist * 2)
        print(f"    {name:<14} {dist:6.3f}  {bar}")

    best_name = ranked[0][1]
    worst2_name = ranked[1][1]

    best_idx = next(i for i, r in enumerate(db) if r["instrument"] == best_name)
    alt_idx  = next(i for i, r in enumerate(db) if r["instrument"] == worst2_name)

    print(f"\n  Why {best_name} beat {worst2_name} — biggest contributing features:")
    contribs = []
    for i, c in enumerate(ALL_COLS):
        bd = weights[i] * (db_n[best_idx, i] - q_n[i]) ** 2
        ad = weights[i] * (db_n[alt_idx,  i] - q_n[i]) ** 2
        contribs.append((ad - bd, c, bd, ad))
    contribs.sort(reverse=True)
    print(f"    {'feature':<28} {best_name:>12} {worst2_name:>12}  advantage")
    for diff, c, bd, ad in contribs[:10]:
        direction = f"  ← {best_name} wins by {diff:.3f}" if diff > 0 else f"  ← {worst2_name} wins by {-diff:.3f}"
        print(f"    {c:<28} {bd:>12.4f} {ad:>12.4f}{direction}")
