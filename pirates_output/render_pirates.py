"""Pirates of the Caribbean cover from random people_results/ samples.

Selection (option b — respect classifier labels where possible):
  - piano  : random pick among files classified as piano       (forced if only 1)
  - violin : random pick among files classified as violin
  - bguitar: NO file is classified as bguitar, so pick the file whose classifier
             score for bguitar is the highest ("closest to bass guitar").

Render: raw_mode=True, temperature=0.3 (mostly the person's voice timbre,
        nudged 30% toward the real instrument).
"""
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import SR, classify_sound, load_audio, render_song

PEOPLE_DIR = ROOT / "people_results"
MIDI_PATH  = ROOT / "web" / "public" / "midi" / "pirates-of-the-caribbean.mid"
OUT_DIR    = ROOT / "pirates_output"
TEMPERATURE = 0.3
SLOTS = ("piano", "violin", "bguitar")


def slot_of(filename: str) -> str:
    """Filename format is <Name>_<instrument>.wav"""
    return filename.rsplit("_", 1)[1].split(".")[0]


def pick_files() -> dict[str, Path]:
    files = sorted(PEOPLE_DIR.glob("*.wav"))
    by_slot: dict[str, list[Path]] = {}
    for f in files:
        by_slot.setdefault(slot_of(f.name), []).append(f)

    picks: dict[str, Path] = {}
    used: set[Path] = set()

    # piano + violin: random from their labelled bucket
    for slot in ("piano", "violin"):
        candidates = [f for f in by_slot.get(slot, []) if f not in used]
        if not candidates:
            raise SystemExit(f"no {slot}-classified files in {PEOPLE_DIR}")
        chosen = random.choice(candidates)
        picks[slot] = chosen
        used.add(chosen)

    # bguitar: highest bguitar-score among remaining files
    remaining = [f for f in files if f not in used]
    if not remaining:
        raise SystemExit("no files left to choose bguitar from")
    print("\nScoring remaining files for bguitar match…")
    best_file, best_score = None, -1.0
    for f in remaining:
        try:
            res = classify_sound(str(f))
        except Exception as e:
            print(f"  {f.name:<35s} classify failed: {e}")
            continue
        score = res["scores"].get("bguitar", 0.0)
        print(f"  {f.name:<35s} bguitar_score={score:.4f}")
        if score > best_score:
            best_score, best_file = score, f
    if best_file is None:
        raise SystemExit("could not score any file for bguitar")
    picks["bguitar"] = best_file
    return picks


def load_sample(path: Path) -> np.ndarray:
    y, _ = load_audio(str(path))
    return y.astype(np.float32)


def main() -> None:
    if not MIDI_PATH.exists():
        raise SystemExit(f"missing MIDI: {MIDI_PATH}")
    OUT_DIR.mkdir(exist_ok=True)
    random.seed()

    picks = pick_files()
    print("\nPicked:")
    for slot, path in picks.items():
        print(f"  {slot:<8s} → {path.name}")

    slot_samples = {slot: load_sample(path) for slot, path in picks.items()}

    print(f"\nRendering {MIDI_PATH.name}  (raw_mode=True, temperature={TEMPERATURE})…")
    out = render_song(
        str(MIDI_PATH),
        slot_samples,
        tempo_percent=100.0,
        raw_mode=True,
        temperature=TEMPERATURE,
    )

    tag = "_".join(picks[s].stem.split("_")[0] for s in SLOTS)
    out_wav = OUT_DIR / f"pirates_{tag}.wav"
    sf.write(str(out_wav), out, SR)

    manifest = {
        "midi": str(MIDI_PATH.relative_to(ROOT)),
        "raw_mode": True,
        "temperature": TEMPERATURE,
        "tempo_percent": 100.0,
        "selection_rule": "piano/violin: random labelled; bguitar: highest bguitar-score among remaining",
        "picks": {slot: picks[slot].name for slot in SLOTS},
        "output": out_wav.name,
        "duration_s": float(len(out) / SR),
    }
    (OUT_DIR / f"pirates_{tag}.manifest.json").write_text(json.dumps(manifest, indent=2))

    # drop a copy of this script next to the output so the algorithm lives with the result
    shutil.copy2(Path(__file__), OUT_DIR / Path(__file__).name)

    print(f"\nWrote {out_wav}  ({len(out)/SR:.1f}s)")
    print(f"     {out_wav.with_suffix('').name}.manifest.json")
    print(f"     {Path(__file__).name}  (algorithm copy)")


if __name__ == "__main__":
    main()
