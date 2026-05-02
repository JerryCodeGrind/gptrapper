"""Convert /people/*.m4a → wav, classify each via pipeline, rename with instrument."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydub import AudioSegment

from pipeline import SR, classify_sound

PEOPLE_DIR = ROOT / "people"
OUT_DIR    = ROOT / "people_results"


def to_wav(src: Path, dst: Path) -> None:
    seg = AudioSegment.from_file(str(src)).set_channels(1).set_frame_rate(SR)
    seg.export(str(dst), format="wav")


def safe_stem(name: str) -> str:
    return name.replace(" ", "_")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    m4a_files = sorted(PEOPLE_DIR.glob("*.m4a"))
    if not m4a_files:
        print(f"no .m4a files in {PEOPLE_DIR}")
        return

    print(f"Found {len(m4a_files)} files in {PEOPLE_DIR}\n")
    results = []
    for src in m4a_files:
        stem = safe_stem(src.stem)
        tmp_wav = OUT_DIR / f"{stem}.wav"
        try:
            to_wav(src, tmp_wav)
        except Exception as exc:
            print(f"  {src.name:<35s}  CONVERT FAILED: {exc}")
            continue

        try:
            result = classify_sound(str(tmp_wav))
        except Exception as exc:
            print(f"  {src.name:<35s}  CLASSIFY FAILED: {exc}")
            continue

        best = result["best_match"]
        score = result["scores"][best]
        final = OUT_DIR / f"{stem}_{best}.wav"
        if final.exists():
            final.unlink()
        tmp_wav.rename(final)
        results.append((src.name, best, score, final.name))
        print(f"  {src.name:<35s} → {best:<10s}  score={score:.3f}  ({final.name})")

    print(f"\nSaved {len(results)} classified WAVs to {OUT_DIR}")
    print("\nSummary by instrument:")
    counts: dict[str, int] = {}
    for _, best, *_ in results:
        counts[best] = counts.get(best, 0) + 1
    for inst, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {inst:<12s} {n}")


if __name__ == "__main__":
    main()
