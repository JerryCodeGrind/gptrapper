# Found-Sound Song Maker

Make a song using your own found sounds. Record taps, scrapes, voice noises, or anything that makes a sound — the app maps each sound to an instrument slot, then plays a real song using your sounds in place of instruments.

## How it works
- **Frontend** — Next.js (App Router) at `web/`
- **Backend** — FastAPI in `server.py`, classifies and morphs sounds via `pipeline.py`
- **Routes**
  - `/` — morph mode: pick a song, fill slots, generate
  - `/raw` — raw mode: skip morphing, just pitch-shift to MIDI notes; temperature slider blends timbre
  - `/studio` — text-to-music mode (in progress)

## Running locally
```bash
# backend
uvicorn server:app --port 8000 --host 127.0.0.1 --reload

# frontend
cd web && npm run dev -- --port 3002
```

## Data sources

Song metadata (tempo, key, mode, time signature) provided by [GetSongBPM.com](https://getsongbpm.com/).
Natural language understanding via [Anthropic Claude](https://www.anthropic.com/).
Music generation via [Magenta.js](https://magenta.tensorflow.org/).
