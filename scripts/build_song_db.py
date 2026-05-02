"""Build data/song_db.json — 100 famous songs per genre with real getsongbpm metadata.

For each genre:
  1. Ask Claude for 100 famous song titles + artists in that genre.
  2. Look up each song on getsongbpm (https://api.getsong.co/) for real
     tempo / key / mode / time-sig / danceability / acousticness.
  3. Drop entries getsongbpm doesn't have.
  4. Save the surviving entries grouped by genre.

Idempotent: re-runs reuse data/song_lookup_cache.json so we don't re-hit either
API for songs already fetched. Safe to Ctrl-C and resume.

Usage:
  python scripts/build_song_db.py              # all 8 default genres
  python scripts/build_song_db.py rock,jazz    # subset
  python scripts/build_song_db.py --refresh    # ignore cached song lists
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the repo root importable so we can use claude_client + getsongbpm_client
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from anthropic import Anthropic  # noqa: E402

from getsongbpm_client import GetSongBPMError, lookup_song  # noqa: E402

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "song_db.json"
SONG_LIST_CACHE = DATA_DIR / "claude_song_lists.json"

DEFAULT_GENRES = ["pop", "indie pop", "hip-hop", "trap", "rock", "edm", "lo-fi", "jazz"]
SONGS_PER_GENRE = 100
PER_REQUEST_DELAY_S = 0.6  # ≤3000 req/hr, with retry-on-503 backoff in client

CLAUDE_MODEL = "claude-sonnet-4-6"

# Tool schema forces Claude to return a clean array.
SONG_LIST_TOOL = {
    "name": "song_list",
    "description": "Return famous songs in the requested genre.",
    "input_schema": {
        "type": "object",
        "properties": {
            "songs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "artist": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["artist", "title"],
                },
            },
        },
        "required": ["songs"],
        "additionalProperties": False,
    },
}


def _claude() -> Anthropic:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    return Anthropic(api_key=key)


def _load_song_list_cache() -> dict[str, list[dict]]:
    if SONG_LIST_CACHE.exists():
        try:
            return json.loads(SONG_LIST_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_song_list_cache(cache: dict) -> None:
    SONG_LIST_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def get_song_list(genre: str, refresh: bool = False) -> list[dict]:
    """Ask Claude for 100 famous songs in `genre`, cached on disk.
    Tries up to 3 prompts; some genres occasionally trip Claude's content filter."""
    cache = _load_song_list_cache()
    if not refresh and genre in cache and len(cache[genre]) >= SONGS_PER_GENRE * 0.8:
        return cache[genre]

    print(f"  Asking Claude for {SONGS_PER_GENRE} {genre} songs...", flush=True)
    client = _claude()

    prompt_variants = [
        # Original
        f"List {SONGS_PER_GENRE} famous {genre} songs spanning multiple decades and sub-styles. "
        f"Mix iconic classics with modern hits. Return ONLY via the `song_list` tool — no prose. "
        f"Use canonical artist/title spellings.",
        # Short, plain
        f"Give me {SONGS_PER_GENRE} well-known {genre} tracks (artist + title) via the song_list tool.",
        # Smaller batch — content filter sometimes trips on long outputs
        f"List 60 well-known {genre} songs (artist + title) via the song_list tool.",
    ]
    last_err: Exception | None = None
    for i, prompt in enumerate(prompt_variants):
        try:
            msg = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8000,
                tools=[SONG_LIST_TOOL],
                tool_choice={"type": "tool", "name": "song_list"},
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            print(f"    Claude attempt {i+1} failed: {e}", flush=True)
            last_err = e
            continue
        songs: list[dict] = []
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "song_list":
                songs = list(block.input.get("songs", []))
                break
        if songs:
            cache[genre] = songs
            _save_song_list_cache(cache)
            print(f"    Got {len(songs)} candidates from Claude", flush=True)
            return songs
    raise RuntimeError(f"Claude returned no songs for {genre} after {len(prompt_variants)} attempts ({last_err})")


def lookup_one(song: dict) -> dict | None:
    """Return enriched record or None if getsongbpm doesn't have it."""
    try:
        data = lookup_song(song["artist"], song["title"])
    except GetSongBPMError as e:
        print(f"    !! API error on '{song['artist']} - {song['title']}': {e}", flush=True)
        return None
    if not data or not data.get("features"):
        return None
    f = data["features"]
    if not f.get("tempo") or not f.get("key"):
        return None
    return {
        "artist": data["track"].get("artist") or song["artist"],
        "title": data["track"].get("title") or song["title"],
        "tempo": float(f["tempo"]),
        "key": f["key"],
        "mode": f["mode"],
        "time_signature": int(f.get("time_signature") or 4),
    }


def build_genre(genre: str, refresh: bool = False) -> list[dict]:
    print(f"\n=== {genre} ===", flush=True)
    candidates = get_song_list(genre, refresh=refresh)
    enriched: list[dict] = []
    seen_keys: set[str] = set()
    for i, song in enumerate(candidates, 1):
        key = f"{song['artist'].lower().strip()}::{song['title'].lower().strip()}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rec = lookup_one(song)
        time.sleep(PER_REQUEST_DELAY_S)
        if rec:
            enriched.append(rec)
            tag = "✓"
        else:
            tag = "·"
        print(f"  [{i:3d}/{len(candidates)}] {tag} {song['artist']} — {song['title']}", flush=True)
    print(f"\n  → {len(enriched)} of {len(candidates)} found in getsongbpm", flush=True)
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("genres", nargs="?", default=",".join(DEFAULT_GENRES))
    parser.add_argument("--refresh", action="store_true", help="Re-ask Claude even if cached")
    args = parser.parse_args()
    target_genres = [g.strip() for g in args.genres.split(",") if g.strip()]

    db: dict[str, list[dict]] = {}
    if DB_PATH.exists():
        try:
            db = json.loads(DB_PATH.read_text())
        except Exception:
            db = {}

    for g in target_genres:
        try:
            db[g] = build_genre(g, refresh=args.refresh)
        except Exception as e:
            print(f"  !! genre {g!r} failed: {e}", flush=True)
            continue
        DB_PATH.write_text(json.dumps(db, indent=2, sort_keys=True))
        print(f"  saved → {DB_PATH} ({sum(len(v) for v in db.values())} total songs)", flush=True)


if __name__ == "__main__":
    main()
