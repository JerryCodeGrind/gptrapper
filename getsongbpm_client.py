"""GetSongBPM API client — free song-metadata lookup.

Used in place of Spotify's deprecated /audio-features endpoint to fetch
tempo, key, mode, and time signature for real songs the user names.
Docs: https://getsongbpm.com/api
"""
import json
import os
import threading
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).parent.resolve()
CACHE_PATH = REPO_ROOT / "data" / "songbpm_cache.json"

_API_BASE = "https://api.getsong.co"

_cache_lock = threading.Lock()


class GetSongBPMError(RuntimeError):
    pass


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _api_get(path: str, params: dict, max_retries: int = 5) -> dict:
    """Retry 5xx, 429, network errors, and non-JSON 200s (their server sometimes
    returns an HTML error page with status 200 under load)."""
    import json as _json
    import time as _time

    api_key = os.getenv("GETSONGBPM_API_KEY")
    if not api_key:
        raise GetSongBPMError("GETSONGBPM_API_KEY not configured in .env")
    full = {**params, "api_key": api_key}
    last_err = "unknown"
    for attempt in range(max_retries):
        backoff = min(2 ** attempt, 10)
        try:
            r = requests.get(f"{_API_BASE}{path}", params=full, timeout=15)
        except requests.RequestException as e:
            last_err = f"net:{e.__class__.__name__}"
            _time.sleep(backoff)
            continue
        if r.status_code in (500, 502, 503, 504, 429):
            last_err = f"{r.status_code}"
            _time.sleep(backoff)
            continue
        if r.status_code != 200:
            raise GetSongBPMError(f"GET {path} → {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except (_json.JSONDecodeError, ValueError):
            last_err = f"200-non-json (body: {r.text[:80]!r})"
            _time.sleep(backoff)
            continue
    raise GetSongBPMError(f"GET {path} → {last_err} after {max_retries} retries")


# GetSongBPM returns key as a string like "1A", "11B" (Camelot wheel) — convert to letter+mode.
# Mapping of Camelot codes → (key letter, mode). 'A' suffix = minor, 'B' suffix = major.
_CAMELOT = {
    "1A": ("A", "minor"),   "1B": ("B", "major"),
    "2A": ("E", "minor"),   "2B": ("F#", "major"),
    "3A": ("B", "minor"),   "3B": ("C#", "major"),
    "4A": ("F#", "minor"),  "4B": ("G#", "major"),
    "5A": ("C#", "minor"),  "5B": ("D#", "major"),
    "6A": ("G#", "minor"),  "6B": ("A#", "major"),
    "7A": ("D#", "minor"),  "7B": ("F", "major"),
    "8A": ("A#", "minor"),  "8B": ("C", "major"),
    "9A": ("F", "minor"),   "9B": ("G", "major"),
    "10A": ("C", "minor"),  "10B": ("D", "major"),
    "11A": ("G", "minor"),  "11B": ("A", "major"),
    "12A": ("D", "minor"),  "12B": ("E", "major"),
}


_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def _normalize_accidentals(s: str) -> str:
    """Convert unicode ♯/♭ to ASCII #/b so keys round-trip cleanly."""
    return s.replace("♯", "#").replace("♭", "b")


def _parse_key(key_str: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Parse various GetSongBPM key formats:
       'A'      → ('A',  'major')   (letter alone = major)
       'Em'     → ('E',  'minor')   (trailing m = minor)
       'F#m'    → ('F#', 'minor')
       'Bb'     → ('A#', 'major')   (flats normalized to sharps)
       '11B'    → ('A',  'major')   (Camelot notation)
       'C major'→ ('C',  'major')   (verbose form)
    """
    if not key_str:
        return (None, None)
    s = _normalize_accidentals(key_str.strip())
    if s in _CAMELOT:
        return _CAMELOT[s]
    parts = s.split()
    if len(parts) == 2:
        note, m = parts[0], parts[1].lower()
        if m in ("major", "minor"):
            return (_FLAT_TO_SHARP.get(note, note), m)
    # Compact form like 'A', 'Em', 'F#m', 'Bbm'
    is_minor = s.endswith("m") and not s.endswith("maj")
    note = s[:-1] if is_minor else s
    mode = "minor" if is_minor else "major"
    note = _FLAT_TO_SHARP.get(note, note)
    return (note, mode)


def lookup_song(artist: str, title: str) -> Optional[dict]:
    """Disk-cached. Returns {tempo, key, mode, time_sig, ...} or None if not found."""
    cache_key = f"{artist.strip().lower()}::{title.strip().lower()}"
    with _cache_lock:
        cache = _load_cache()
        if cache_key in cache:
            return cache[cache_key]

    # GetSongBPM search endpoint takes "lookup=song:TITLE artist:ARTIST"
    lookup = f"song:{title} artist:{artist}"
    data = _api_get("/search/", {"type": "both", "lookup": lookup})
    search = data.get("search")
    if not search or not isinstance(search, list) or len(search) == 0:
        return None
    hit = search[0]

    key_str = hit.get("key_of") or hit.get("song_key")
    key_letter, mode = _parse_key(key_str)

    result = {
        "track": {
            "id": hit.get("id"),
            "title": hit.get("song_title") or hit.get("title"),
            "artist": (hit.get("artist") or {}).get("name") if isinstance(hit.get("artist"), dict) else hit.get("artist"),
        },
        "features": {
            "tempo": float(hit["tempo"]) if hit.get("tempo") else None,
            "key": key_letter,
            "mode": mode,
            "time_signature": int(hit["time_sig"].split("/")[0]) if hit.get("time_sig") and "/" in str(hit["time_sig"]) else 4,
            "duration_ms": None,
        },
    }
    with _cache_lock:
        cache = _load_cache()
        cache[cache_key] = result
        _save_cache(cache)
    return result
