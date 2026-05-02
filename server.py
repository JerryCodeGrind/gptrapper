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
    segment_sounds,
    strip_silence,
    load_audio,
    _load_y_from_input,
)
from claude_client import ClaudeError, apply_edit, classify_prompt
from getsongbpm_client import GetSongBPMError, lookup_song
from song_db import random_song as random_song_from_db, total_songs as song_db_total

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
    return {
        "ok": True,
        "instruments_loaded": len(INSTRUMENT_PITCH),
        "song_db_count": song_db_total(),
    }


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
    """Decode any audio (webm/ogg/wav/m4a/etc) → mono float32 SR WAV bytes,
    AND return its detected fundamental pitch in the X-Detected-Pitch-Hz header
    (used by /studio to pick a key for Magenta generation).
    """
    from pipeline import _detect_pitch  # local import — avoids extra top-level coupling

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
    try:
        pitch_hz = float(_detect_pitch(y))
    except Exception:
        pitch_hz = 0.0
    return Response(
        content=_wav_bytes(y),
        media_type="audio/wav",
        headers={
            "X-Detected-Pitch-Hz": f"{pitch_hz:.2f}",
            # Without this CORS-Expose header, browser JS can't read X-Detected-Pitch-Hz.
            "Access-Control-Expose-Headers": "X-Detected-Pitch-Hz",
        },
    )


@app.post("/segment")
async def segment_endpoint(
    audio: UploadFile = File(...),
    top_db: float = Form(30.0),
    min_duration_s: float = Form(0.06),
    min_peak_ratio_db: float = Form(-25.0),
    max_segments: int = Form(32),
):
    """Split a long recording into discrete sounds, classify each.

    Body: multipart `audio` file. Optional form fields tune the splitter:
      - top_db: dB below the recording's peak that counts as silence (default 30)
      - min_duration_s: drop chunks shorter than this (default 0.06 s)
      - min_peak_ratio_db: drop chunks quieter than this vs overall peak (default -25 dB)
      - max_segments: hard cap on returned segments (default 32)
    Response: { "count": int, "segments": [ { index, start_s, end_s, duration_s,
                                               peak_db, scores, best_match,
                                               audio_wav_b64 } ] }
    `audio_wav_b64` is a 16-bit PCM mono WAV at SR — pass it straight to /morph
    or /render-song as a slot sample.
    """
    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio over 20 MB")
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="empty audio")
    try:
        segments = segment_sounds(
            raw,
            top_db=top_db,
            min_duration_s=min_duration_s,
            min_peak_ratio_db=min_peak_ratio_db,
            max_segments=max_segments,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"segment failed: {e}")
    return {"count": len(segments), "segments": segments}


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
    midi_filename: str | None = None
    midi_b64: str | None = None  # alternative to midi_filename — used by /studio for Magenta-generated MIDI
    slots: dict[str, str] = Field(default_factory=dict)
    tempo_percent: float = Field(default=100.0, ge=50.0, le=200.0)
    raw_mode: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


@app.post("/render-song")
def render_endpoint(req: RenderRequest):
    # Accept either midi_filename (server-side .mid in web/public/midi) or midi_b64
    # (raw MIDI bytes, e.g. from Magenta-generated NoteSequence).
    midi_path: Path
    midi_temp: Path | None = None
    if req.midi_b64:
        try:
            midi_bytes = base64.b64decode(req.midi_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="bad midi_b64")
        if not midi_bytes:
            raise HTTPException(status_code=400, detail="empty midi_b64")
        if len(midi_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="midi over 5 MB")
        midi_temp = MIDI_DIR / f".tmp-{os.getpid()}-{int.from_bytes(os.urandom(4), 'big'):08x}.mid"
        midi_temp.write_bytes(midi_bytes)
        midi_path = midi_temp
    elif req.midi_filename:
        if "/" in req.midi_filename or "\\" in req.midi_filename or ".." in req.midi_filename:
            raise HTTPException(status_code=400, detail="invalid midi_filename")
        midi_path = (MIDI_DIR / req.midi_filename).resolve()
        try:
            midi_path.relative_to(MIDI_DIR)
        except ValueError:
            raise HTTPException(status_code=400, detail="midi_filename outside midi dir")
        if not midi_path.exists():
            raise HTTPException(status_code=404, detail=f"midi not found: {req.midi_filename}")
    else:
        raise HTTPException(status_code=400, detail="provide midi_filename or midi_b64")

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

    # Diagnostic — log what the MIDI actually contains so we can debug silent output.
    try:
        import pretty_midi as _pm
        _m = _pm.PrettyMIDI(str(midi_path))
        logger.info("MIDI loaded: %d insts, end=%.2fs", len(_m.instruments), _m.get_end_time())
        for _i, _inst in enumerate(_m.instruments):
            if len(_inst.notes) == 0:
                logger.info("  inst[%d] is_drum=%s program=%d notes=0", _i, _inst.is_drum, _inst.program)
            else:
                pitches = [n.pitch for n in _inst.notes]
                logger.info("  inst[%d] is_drum=%s program=%d notes=%d pitches=%d–%d avg=%.1f",
                            _i, _inst.is_drum, _inst.program, len(_inst.notes),
                            min(pitches), max(pitches), sum(pitches) / len(pitches))
        logger.info("slots filled: %s", list(slot_samples.keys()))
        from pipeline import identify_tracks as _id_tr
        _tm = _id_tr(_m)
        logger.info("track_map: %s", {k: getattr(v, 'name', '?') for k, v in _tm.items()})
    except Exception as _e:
        logger.warning("MIDI diagnostic failed: %s", _e)

    try:
        out = render_song(
            str(midi_path),
            slot_samples,
            req.tempo_percent,
            raw_mode=req.raw_mode,
            temperature=req.temperature,
        )
        logger.info("render_song output: peak=%.4f rms=%.4f len=%.2fs",
                    float(np.max(np.abs(out))) if len(out) else 0.0,
                    float(np.sqrt(np.mean(out ** 2))) if len(out) else 0.0,
                    len(out) / SR if len(out) else 0.0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"render failed: {e}")
    finally:
        if midi_temp is not None:
            try:
                midi_temp.unlink()
            except OSError:
                pass

    wav_bytes = _wav_bytes(out)
    base_name = req.midi_filename or "generated"
    title_slug = _slug(base_name.rsplit(".", 1)[0])
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
    """Build render params from Claude's intent.

    Resolution order for tempo/key/mode/artist/title:
      1. Specific song named → live getsongbpm lookup (fresh data)
      2. Genre matches our 566-song local DB → random real song from that bucket
      3. Fallback → Claude's own tempo/key/mode estimates
    """
    artist = intent.get("artist")
    title = intent.get("title")
    lookup_data: dict | None = None
    lookup_error: str | None = None

    if artist and title:
        try:
            lookup_data = lookup_song(artist, title)
        except GetSongBPMError as e:
            lookup_error = str(e)

    # 1. Live API hit on a named song.
    if lookup_data and lookup_data.get("features"):
        f = lookup_data["features"]
        tempo = float(f.get("tempo") or intent.get("tempo") or 110)
        key = f.get("key") or intent.get("key") or "C"
        mode = f.get("mode") or intent.get("mode") or "major"
        time_signature = int(f.get("time_signature") or 4)
        source = "getsongbpm"
    else:
        # 2. Try the local DB if Claude gave us a genre/subgenre (no specific song).
        #    Search subgenre first (more specific), then top-level genre.
        db_pick: dict | None = None
        for q in (intent.get("subgenre"), intent.get("genre")):
            if q:
                db_pick = random_song_from_db(q)
                if db_pick:
                    break
        if db_pick:
            tempo = db_pick["tempo"]
            key = db_pick["key"]
            mode = db_pick["mode"]
            time_signature = db_pick["time_signature"]
            artist = db_pick["artist"]
            title = db_pick["title"]
            source = f"song_db:{db_pick['genre_bucket']}"
        else:
            # 3. Fallback to Claude's own values.
            tempo = float(intent.get("tempo") or 110)
            key = intent.get("key") or "C"
            mode = intent.get("mode") or "major"
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
        "lookup_error": lookup_error,
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
