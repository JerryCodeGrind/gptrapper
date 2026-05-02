"""Spotify Web API — Client Credentials flow.

Used to look up real-song metadata (tempo, key, mode, time signature, mood
features) when the user names a specific track in the /studio prompt box.
No user-auth flow needed — Client Credentials is enough for public catalog data.
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import requests

REPO_ROOT = Path(__file__).parent.resolve()
CACHE_PATH = REPO_ROOT / "data" / "song_cache.json"

_TOKEN_URL = "https://accounts.spotify.com/api/token"
_API_BASE = "https://api.spotify.com/v1"

# Spotify maps key as integer 0–11 (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_token: Optional[str] = None
_token_expires_at: float = 0.0
_token_lock = threading.Lock()
_cache_lock = threading.Lock()


class SpotifyError(RuntimeError):
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


def _get_token() -> str:
    """Fetch a Client Credentials access token, caching until ~1 min before expiry."""
    global _token, _token_expires_at
    with _token_lock:
        if _token and time.time() < _token_expires_at - 60:
            return _token
        cid = os.getenv("SPOTIFY_CLIENT_ID")
        csec = os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not csec:
            raise SpotifyError("SPOTIFY_CLIENT_ID/SECRET not configured in .env")
        r = requests.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(cid, csec),
            timeout=10,
        )
        if r.status_code != 200:
            raise SpotifyError(f"token fetch failed ({r.status_code})")
        data = r.json()
        _token = data["access_token"]
        _token_expires_at = time.time() + int(data.get("expires_in", 3600))
        return _token


def _api_get(path: str, params: Optional[dict] = None) -> dict:
    headers = {"Authorization": f"Bearer {_get_token()}"}
    r = requests.get(f"{_API_BASE}{path}", headers=headers, params=params, timeout=10)
    if r.status_code == 401:
        # Token may have been revoked early; force a refresh and retry once.
        global _token_expires_at
        _token_expires_at = 0.0
        headers["Authorization"] = f"Bearer {_get_token()}"
        r = requests.get(f"{_API_BASE}{path}", headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        raise SpotifyError(f"GET {path} → {r.status_code}: {r.text[:200]}")
    return r.json()


def search_track(artist: str, title: str) -> Optional[dict]:
    """Return the top track match or None. Result has 'id', 'name', 'artists', 'popularity'."""
    q = f'track:"{title}" artist:"{artist}"'
    data = _api_get("/search", {"q": q, "type": "track", "limit": 1})
    items = data.get("tracks", {}).get("items", [])
    if not items:
        # fall back to looser query
        data = _api_get("/search", {"q": f"{title} {artist}", "type": "track", "limit": 1})
        items = data.get("tracks", {}).get("items", [])
    if not items:
        return None
    t = items[0]
    return {
        "id": t["id"],
        "name": t["name"],
        "artists": [a["name"] for a in t.get("artists", [])],
        "popularity": t.get("popularity"),
    }


def get_audio_features(track_id: str) -> dict:
    """Return tempo, key, mode, time_sig, energy, valence, danceability for a track."""
    raw = _api_get(f"/audio-features/{track_id}")
    return {
        "tempo": raw.get("tempo"),
        "key": _KEY_NAMES[raw["key"]] if isinstance(raw.get("key"), int) and 0 <= raw["key"] <= 11 else None,
        "mode": "major" if raw.get("mode") == 1 else "minor" if raw.get("mode") == 0 else None,
        "time_signature": raw.get("time_signature"),
        "energy": raw.get("energy"),
        "valence": raw.get("valence"),
        "danceability": raw.get("danceability"),
        "duration_ms": raw.get("duration_ms"),
    }


def lookup_song(artist: str, title: str) -> Optional[dict]:
    """Disk-cached: search + audio-features in one call."""
    cache_key = f"{artist.strip().lower()}::{title.strip().lower()}"
    with _cache_lock:
        cache = _load_cache()
        if cache_key in cache:
            return cache[cache_key]
    track = search_track(artist, title)
    if track is None:
        return None
    features = get_audio_features(track["id"])
    result = {
        "track": track,
        "features": features,
    }
    with _cache_lock:
        cache = _load_cache()
        cache[cache_key] = result
        _save_cache(cache)
    return result
