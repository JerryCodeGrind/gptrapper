import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import librosa
import torch
from pydub import AudioSegment
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()

SR = 22050
HOP = 512
N_FFT = 2048

PANNS_SR = 32000

# AudioSet label index → our instrument label
# Indices from PANNs Cnn14 (527-class AudioSet model)
_LABEL_MAP = {
    142: "bass",     # Bass guitar
    153: "piano",    # Piano
    154: "piano",    # Electric piano
    162: "drum",     # Drum kit
    164: "drum",     # Drum
    165: "snare",    # Snare drum
    168: "drum",     # Bass drum
    171: "cymbal",   # Cymbal
    172: "hihat",    # Hi-hat
    196: "flute",    # Flute
}

_tagger = None


def _get_tagger():
    global _tagger
    if _tagger is None:
        from panns_inference import AudioTagging
        _tagger = AudioTagging(
            checkpoint_path=str(Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth"),
            device="cpu",
        )
    return _tagger

def _load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        seg = AudioSegment.from_file(tmp_path).set_channels(1).set_frame_rate(SR)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            tmp2_path = tmp2.name
        seg.export(tmp2_path, format="wav")
        try:
            y, _ = librosa.load(tmp2_path, sr=SR, mono=True)
        finally:
            os.unlink(tmp2_path)
    finally:
        os.unlink(tmp_path)
    return y.astype(np.float32)


def _strip_event_tail(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    peak_idx = int(np.argmax(rms))
    end = len(y)
    for i in range(peak_idx, len(rms)):
        if rms_db[i] < -top_db:
            end = min(len(y), i * HOP)
            break
    return y[:end]


def _classify_instrument(y: np.ndarray) -> Optional[str]:
    if len(y) < N_FFT:
        return None

    tagger = _get_tagger()
    y32 = librosa.resample(y, orig_sr=SR, target_sr=PANNS_SR)
    # CNN14 needs at least ~1s of audio (6 pooling layers → min 64 time frames at hop=320)
    min_len = PANNS_SR
    if len(y32) < min_len:
        y32 = np.pad(y32, (0, min_len - len(y32)))
    audio = y32[None, :]  # (1, samples)

    with torch.no_grad():
        (clipwise_output, _) = tagger.inference(audio)

    probs = clipwise_output[0]  # (527,)

    # Pick the highest-probability class that maps to one of our instruments
    scores: dict[str, float] = {}
    for idx, instrument in _LABEL_MAP.items():
        p = float(probs[idx])
        if p > scores.get(instrument, 0.0):
            scores[instrument] = p

    if not scores:
        return None

    best = max(scores, key=scores.get)
    print(f"[classify] scores: { {k: f'{v:.3f}' for k, v in sorted(scores.items(), key=lambda x: -x[1])} }")
    return best


def parse_clips(clips: List[bytes]) -> List[dict]:
    """
    Detect and isolate individual sound events across audio clips.

    Each clip may contain multiple events (claps, voice, etc.) mixed with
    background noise. Returns only the non-background events with their
    type, timing, and audio data.

    Args:
        clips: Raw audio bytes for each clip (any format pydub supports).
        min_snr_db: Minimum SNR above the clip's noise floor to keep an event.

    Returns:
        List of dicts:
          - clip_index (int): which input clip the event came from
          - type (str): "clap" | "voice" | "transient"
          - start_s (float): event start time within the original clip
          - duration_s (float): event duration after tail stripping
          - audio (np.ndarray): mono float32 at SR

    """

    results = []

    for clip_idx, audio_bytes in enumerate(clips):
        print(f"[parse] clip {clip_idx}: decoding {len(audio_bytes)} bytes")
        t0 = time.time()
        y = _load_audio_bytes(audio_bytes)
        print(f"[parse] clip {clip_idx}: decoded {len(y)/SR:.2f}s in {time.time()-t0:.2f}s")

        t0 = time.time()
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=SR, hop_length=HOP, backtrack=True, units="frames",
        )
        print(f"[parse] clip {clip_idx}: found {len(onset_frames)} onsets in {time.time()-t0:.2f}s")
        if len(onset_frames) == 0:
            continue

        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP)

        # Merge onsets that are within 0.5 s of each other (sustained sounds re-triggering)
        MIN_GAP = int(0.5 * SR)
        merged = [onset_samples[0]]
        for s in onset_samples[1:]:
            if s - merged[-1] >= MIN_GAP:
                merged.append(s)
        onset_samples = merged

        for i, start in enumerate(onset_samples):
            # Run to the next onset or 4 s — long enough not to cut off sustained sounds
            if i + 1 < len(onset_samples):
                end = min(int(onset_samples[i + 1]), start + int(4.0 * SR))
            else:
                end = min(start + int(4.0 * SR), len(y))

            event_y = y[start:end]

            # Absolute volume gate: skip events that are too quiet (noise floor bleed)
            event_rms = float(np.sqrt(np.mean(event_y ** 2)))
            if event_rms < 0.01:
                print(f"[parse]   onset {i} @ {start/SR:.2f}s skipped (too quiet: rms={event_rms:.4f})")
                continue

            # event_y = _strip_event_tail(event_y)
            # if len(event_y) < N_FFT:
            #     continue

            # t0 = time.time()
            # event_type = _classify_instrument(event_y)
            # print(f"[parse]   onset {i} @ {start/SR:.2f}s -> {event_type} ({time.time()-t0:.2f}s)")
            # if event_type is None:
            #     continue

            results.append({
                "clip_index": clip_idx,
                "type": None,
                "start_s": round(start / SR, 4),
                "duration_s": round(len(event_y) / SR, 4),
                "audio": event_y,
            })

    return results


@app.post("/generate")
async def generate(prompt: str = Form(...), clips: List[UploadFile] = File(...)):
    print("Prompt:", prompt)
    print("Clips received:", len(clips))

    clip_bytes = []
    for i, clip in enumerate(clips):
        content = await clip.read()
        print(f"  clip_{i}: {clip.filename}, {len(content)} bytes")
        clip_bytes.append(content)

    samples = parse_clips(clip_bytes)
    print(f"Parsed {len(samples)} samples:")
    for s in samples:
        print(f"  clip_{s['clip_index']}  type={s['type']}  "
              f"start={s['start_s']:.3f}s  dur={s['duration_s']:.3f}s")

    response = {
        "status": "ok",
        "clips_received": len(clips),
        "samples": [
            {
                "clip_index": s["clip_index"],
                "type": s["type"],
                "start_s": s["start_s"],
                "duration_s": s["duration_s"],
            }
            for s in samples
        ],
    };

    print(response);

    return response;

# type InstrumentType =
#   | "bass" | "piano" | "flute"
#   | "snare" | "hihat" | "drum" | "cymbal"
#
# type Sample = {
#   clip_index: number
#   type: InstrumentType
#   start_s: number
#   duration_s: number
# }
#
# type GenerateResponse = {
#   status: string
#   clips_received: number
#   samples: Sample[]
# }
