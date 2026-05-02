"""Found-Sound Song Maker — FastAPI backend.

Endpoints:
  GET  /health
  GET  /defaults-available
  GET  /default-sample/{slot}
  POST /classify              audio file       -> { scores, best_match }
  POST /morph                  audio + target   -> wav bytes
  POST /prep-raw              audio file       -> wav bytes (no morph)
  GET  /analyze-midi/{name}                     -> { needed_slots, ... }
  POST /render-song            midi + slots     -> wav bytes
  POST /text-to-music-params  { text }         -> { tempo, key, mode, instruments, ... }
  POST /apply-edit            { params, edit } -> updated params
"""
import base64
import io
import logging
import os
import re
from pathlib import Path

# Load .env BEFORE any modules that read os.getenv at import time.
from dotenv import load_dotenv

REPO_ROOT_FOR_ENV = Path(__file__).parent.resolve()
load_dotenv(REPO_ROOT_FOR_ENV / ".env")

logger = logging.getLogger("found-sound")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

for _key in ("ANTHROPIC_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET"):
    if not os.getenv(_key):
        logger.warning("%s missing from .env — /text-to-music-params and /apply-edit will fail until it's set", _key)

import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pipeline import (
    GM_DRUM_NOTES,
    INSTRUMENT_PITCH,
    SR,
    classify_sound,
    identify_tracks,
    process_sound,
    render_song,
    strip_silence,
    load_audio,
    _load_y_from_input,
)
from claude_client import ClaudeError, apply_edit, classify_prompt
from spotify_client import SpotifyError, lookup_song

REPO_ROOT = Path(__file__).parent.resolve()
MIDI_DIR = (REPO_ROOT / "web" / "public" / "midi").resolve()
SAMPLES_DIR = (REPO_ROOT / "samples").resolve()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Pre-morphed found-sound defaults (the "weird sounds" we recorded).
# Maps slot key → filename inside samples/. Slots not in this dict have NO default.
DEFAULT_SAMPLES: dict[str, str] = {
    "piano":     "test6_piano.wav",
    "aguitar":   "test6_aguitar.wav",
    "eguitar":   "test5_eguitar.wav",
    "bguitar":   "test2_bguitar.wav",
    "cello":     "test4_cello.wav",
    "kickdrum":  "test3_kickdrum.wav",
    "hihat":     "test1_hihat.wav",
}

app = FastAPI(title="Found-Sound Song Maker API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "track"


def _wav_bytes(y: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), SR, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@app.get("/health")
def health():
    return {"ok": True, "instruments_loaded": len(INSTRUMENT_PITCH)}


@app.get("/defaults-available")
def defaults_available():
    return {"slots": sorted(DEFAULT_SAMPLES.keys())}


@app.get("/default-sample/{instrument_key}")
def default_sample(instrument_key: str):
    if instrument_key not in INSTRUMENT_PITCH:
        raise HTTPException(status_code=400, detail=f"unknown instrument: {instrument_key}")
    if instrument_key not in DEFAULT_SAMPLES:
        raise HTTPException(
            status_code=404,
            detail=f"no found-sound default for {instrument_key} — please record or upload",
        )
    sample_path = SAMPLES_DIR / DEFAULT_SAMPLES[instrument_key]
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"missing sample file: {sample_path.name}")
    y, _ = load_audio(str(sample_path))
    y = strip_silence(y)
    return Response(content=_wav_bytes(y), media_type="audio/wav")


@app.post("/classify")
async def classify_endpoint(audio: UploadFile = File(...)):
    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio over 20 MB")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        return classify_sound(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"classify failed: {e}")


@app.post("/prep-raw")
async def prep_raw_endpoint(audio: UploadFile = File(...)):
    """Decode any audio (webm/ogg/wav/m4a/etc) → mono float32 SR WAV bytes.
    Used by the /raw page to convert recordings into a format /render-song can read.
    No morphing — preserves the user's natural timbre.
    """
    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio over 20 MB")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        y = strip_silence(_load_y_from_input(raw))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prep failed: {e}")
    if len(y) == 0:
        raise HTTPException(status_code=400, detail="audio is silent after trimming")
    return Response(content=_wav_bytes(y), media_type="audio/wav")


@app.post("/morph")
async def morph_endpoint(
    audio: UploadFile = File(...),
    target_instrument: str = Form(...),
):
    if target_instrument not in INSTRUMENT_PITCH:
        raise HTTPException(status_code=400, detail=f"unknown instrument: {target_instrument}")
    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio over 20 MB")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        wav_bytes = process_sound(raw, target_instrument)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"morph failed: {e}")
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/analyze-midi/{filename}")
def analyze_midi(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    midi_path = (MIDI_DIR / filename).resolve()
    try:
        midi_path.relative_to(MIDI_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="outside midi dir")
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail=f"midi not found: {filename}")

    import pretty_midi
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    track_map = identify_tracks(midi)

    needed: dict[str, dict] = {}
    for slot_key, inst in track_map.items():
        if slot_key == "__drums__":
            continue
        needed[slot_key] = {"notes": len(inst.notes), "track_name": inst.name or slot_key}

    drum_inst = track_map.get("__drums__")
    if drum_inst is not None:
        for slot_key, drum_pitches in GM_DRUM_NOTES.items():
            count = sum(1 for n in drum_inst.notes if int(n.pitch) in drum_pitches)
            if count > 0:
                needed[slot_key] = {"notes": count, "track_name": "Drums"}

    return {
        "filename": filename,
        "duration_sec": float(midi.get_end_time()),
        "needed_slots": needed,
    }


class RenderRequest(BaseModel):
    midi_filename: str
    slots: dict[str, str] = Field(default_factory=dict)
    tempo_percent: float = Field(default=100.0, ge=50.0, le=200.0)
    raw_mode: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


@app.post("/render-song")
def render_endpoint(req: RenderRequest):
    if "/" in req.midi_filename or "\\" in req.midi_filename or ".." in req.midi_filename:
        raise HTTPException(status_code=400, detail="invalid midi_filename")
    midi_path = (MIDI_DIR / req.midi_filename).resolve()
    try:
        midi_path.relative_to(MIDI_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="midi_filename outside midi dir")
    if not midi_path.exists():
        raise HTTPException(status_code=404, detail=f"midi not found: {req.midi_filename}")

    slot_samples: dict[str, np.ndarray] = {}
    for slot_key, b64 in req.slots.items():
        if slot_key not in INSTRUMENT_PITCH:
            raise HTTPException(status_code=400, detail=f"unknown slot: {slot_key}")
        try:
            raw = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail=f"slot {slot_key}: bad base64")
        if not raw:
            continue
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"slot {slot_key}: bad wav ({e})")
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        slot_samples[slot_key] = data.astype(np.float32)

    if not slot_samples:
        raise HTTPException(status_code=400, detail="no slots filled — fill at least 1")

    try:
        out = render_song(
            str(midi_path),
            slot_samples,
            req.tempo_percent,
            raw_mode=req.raw_mode,
            temperature=req.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"render failed: {e}")

    wav_bytes = _wav_bytes(out)
    title_slug = _slug(req.midi_filename.rsplit(".", 1)[0])
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{title_slug}-found-sound-cover.wav"'
        },
    )


# ─────────────────────────────────────────────────────────────
# /studio NLP endpoints
# ─────────────────────────────────────────────────────────────

class TextToParamsRequest(BaseModel):
    text: str
    bars: int = Field(default=16, description="Target length in bars (8/16/24/32)")


def _params_from_intent(intent: dict, bars: int) -> dict:
    """Merge Claude intent + Spotify lookup (if available) into final render params."""
    artist = intent.get("artist")
    title = intent.get("title")
    spotify_data: dict | None = None
    spotify_error: str | None = None

    if artist and title:
        try:
            spotify_data = lookup_song(artist, title)
        except SpotifyError as e:
            spotify_error = str(e)

    # Spotify wins for tempo/key/mode if it found the track; otherwise use Claude hints.
    if spotify_data and spotify_data.get("features"):
        f = spotify_data["features"]
        tempo = float(f.get("tempo") or intent.get("tempo_hint") or 110)
        key = f.get("key") or intent.get("key_hint") or "C"
        mode = f.get("mode") or intent.get("mode_hint") or "major"
        time_signature = int(f.get("time_signature") or 4)
        source = "spotify"
    else:
        tempo = float(intent.get("tempo_hint") or 110)
        key = intent.get("key_hint") or "C"
        mode = intent.get("mode_hint") or "major"
        time_signature = 4
        source = "claude"

    return {
        "tempo": tempo,
        "key": key,
        "mode": mode,
        "time_signature": time_signature,
        "instruments": intent.get("instruments") or ["piano", "kickdrum", "hihat"],
        "bars": bars,
        "mood": intent.get("mood"),
        "genre": intent.get("genre"),
        "subgenre": intent.get("subgenre"),
        "artist": artist,
        "title": title,
        "source": source,
        "spotify_error": spotify_error,
        "spotify_features": spotify_data.get("features") if spotify_data else None,
    }


@app.post("/text-to-music-params")
def text_to_music_params(req: TextToParamsRequest):
    if req.bars not in (8, 16, 24, 32):
        raise HTTPException(status_code=400, detail="bars must be 8/16/24/32")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    try:
        intent = classify_prompt(req.text)
    except ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"claude: {e}")
    return _params_from_intent(intent, req.bars)


class ApplyEditRequest(BaseModel):
    params: dict
    edit: str


@app.post("/apply-edit")
def apply_edit_endpoint(req: ApplyEditRequest):
    if not req.edit.strip():
        raise HTTPException(status_code=400, detail="empty edit")
    try:
        return apply_edit(req.params, req.edit)
    except ClaudeError as e:
        raise HTTPException(status_code=502, detail=f"claude: {e}")
