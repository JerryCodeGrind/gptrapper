"""Local song database — 566 real famous songs across 8 genres with
tempo/key/mode/time-sig from getsongbpm.

Loaded once at import time. Used by /text-to-music-params to seed genre-only
prompts with real song metadata instead of Claude's guesses.
"""
import json
import random
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.resolve()
DB_PATH = REPO_ROOT / "data" / "song_db.json"

# Genres present in the DB (built from data/song_db.json keys)
_DB: dict[str, list[dict]] = {}


def _load() -> dict[str, list[dict]]:
    global _DB
    if _DB:
        return _DB
    if not DB_PATH.exists():
        return {}
    try:
        _DB = json.loads(DB_PATH.read_text())
    except Exception:
        _DB = {}
    return _DB


def available_genres() -> list[str]:
    return sorted(_load().keys())


def total_songs() -> int:
    return sum(len(v) for v in _load().values())


def _normalize(s: str) -> str:
    return s.lower().replace("-", "").replace(" ", "").replace("_", "")


# Hand-curated synonym map: Claude's genre/subgenre output → DB key.
# Claude often returns "synth pop" / "alternative rock" / "rap" / "house" etc.
# We collapse those to one of our 8 buckets.
_GENRE_ALIASES: dict[str, str] = {
    # pop bucket
    "pop": "pop",
    "synthpop": "pop",
    "dancepop": "pop",
    "bubblegumpop": "pop",
    "popular": "pop",
    "kpop": "pop",
    "electropop": "pop",
    # indie pop
    "indiepop": "indie pop",
    "indie": "indie pop",
    "bedroompop": "indie pop",
    "dreampop": "indie pop",
    "indierock": "indie pop",
    "indiefolk": "indie pop",
    "altpop": "indie pop",
    "alternativepop": "indie pop",
    # hip-hop
    "hiphop": "hip-hop",
    "rap": "hip-hop",
    "boombap": "hip-hop",
    "consciousrap": "hip-hop",
    "oldschoolhiphop": "hip-hop",
    # trap
    "trap": "trap",
    "drill": "trap",
    "mumbleRap": "trap",
    "soundcloudrap": "trap",
    "phonk": "trap",
    # rock
    "rock": "rock",
    "alternativerock": "rock",
    "altrock": "rock",
    "punk": "rock",
    "punkrock": "rock",
    "metal": "rock",
    "thrashmetal": "rock",
    "heavymetal": "rock",
    "hardrock": "rock",
    "classicrock": "rock",
    "grunge": "rock",
    "garage": "rock",
    # edm
    "edm": "edm",
    "house": "edm",
    "techno": "edm",
    "trance": "edm",
    "dance": "edm",
    "electronic": "edm",
    "dubstep": "edm",
    "drumandbass": "edm",
    "dnb": "edm",
    # lo-fi
    "lofi": "lo-fi",
    "lofihiphop": "lo-fi",
    "chillhop": "lo-fi",
    "chillout": "lo-fi",
    "ambient": "lo-fi",
    "studybeats": "lo-fi",
    "chill": "lo-fi",
    # jazz
    "jazz": "jazz",
    "smoothjazz": "jazz",
    "jazzfusion": "jazz",
    "bebop": "jazz",
    "swing": "jazz",
    "bigband": "jazz",
    "bossanova": "jazz",
}


def resolve_genre(query: str) -> Optional[str]:
    """Map a free-form genre string to one of the DB's bucket keys, or None."""
    if not query:
        return None
    db = _load()
    if not db:
        return None
    norm = _normalize(query)
    # Direct alias hit
    if norm in _GENRE_ALIASES and _GENRE_ALIASES[norm] in db:
        return _GENRE_ALIASES[norm]
    # Substring match — sort by alias length DESC so longer aliases win
    # (otherwise "indiepop" matches "pop" before "indiepop", "trap" matches "rap", etc.)
    aliases_by_length = sorted(_GENRE_ALIASES.items(), key=lambda kv: -len(kv[0]))
    for alias, bucket in aliases_by_length:
        if alias in norm and bucket in db:
            return bucket
    # Substring on bucket names directly (also longest-first)
    buckets_by_length = sorted(db.keys(), key=lambda k: -len(k))
    for bucket in buckets_by_length:
        if _normalize(bucket) in norm:
            return bucket
    return None


def random_song(genre: str, *, rng: Optional[random.Random] = None) -> Optional[dict]:
    """Pick a random song record from the matching genre bucket. None if no match."""
    bucket = resolve_genre(genre)
    if not bucket:
        return None
    songs = _load().get(bucket, [])
    if not songs:
        return None
    r = rng or random
    pick = r.choice(songs)
    return {
        "artist": pick["artist"],
        "title": pick["title"],
        "tempo": float(pick["tempo"]),
        "key": pick["key"],
        "mode": pick["mode"],
        "time_signature": int(pick.get("time_signature") or 4),
        "genre_bucket": bucket,
    }
