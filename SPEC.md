# Found-Sound Song Maker — MVP Spec (v2, post-pipeline pivot)

## What we're building

A web app where users record/upload everyday sounds (tapping, hitting a bottle, humming) and assign each one to an **instrument slot** (Piano, Bass Guitar, Kick Drum, etc.). Then they pick a popular song from a bundled MIDI library, and the backend renders a "cover" of that song where each instrument's part is played using the user's morphed-and-pitch-shifted sound.

Think: "Blinding Lights, but every instrument is performed by something the user tapped on their desk."

The output should be **recognizable as the chosen song** but clearly transformed by the user's noises.

---

## Architecture

```
┌─ Next.js (web/) ─────────────────────┐    ┌─ FastAPI (server.py) ──────────────┐
│                                        │    │                                     │
│ Step 1 — Fill instrument slots         │    │ POST /classify                      │
│   11 cards, one per instrument.         │    │   in:  audio blob                   │
│   Click a card → record mic OR upload  │ ──→│   out: { piano: 0.78, cello: 0.22,..}│
│   → audio is sent to /morph,           │    │                                     │
│   morphed-to-instrument blob comes back│    │ POST /morph                         │
│   → stored client-side as that slot's   │ ──→│   in:  audio blob + target_instrument│
│   sample.                               │    │   out: morphed .wav (16-bit PCM)    │
│                                         │    │                                     │
│ Step 2 — Pick a song                   │    │ POST /render-song                   │
│   Grid of cards reading                 │ ──→│   in:  midi filename + slot map     │
│   /midi/index.json (15 bundled MIDI).  │    │         (instrument → morphed wav)  │
│                                         │    │   out: combined.wav (rendered cover)│
│ Step 3 — Generate                      │    │                                     │
│   POST /render-song with chosen MIDI   │    │  Uses pipeline.py morph + pitch +   │
│   + filled slots → server returns      │    │  pretty_midi to schedule notes per  │
│   combined.wav → play in browser.      │    │  track + numpy mixdown.             │
│                                         │    │                                     │
│ [Download .wav]                        │    │  Tracks whose instrument slot is    │
└────────────────────────────────────────┘    │  empty render as silence.           │
                                              └─────────────────────────────────────┘
```

**Crucial design decisions:**

- **Server-side rendering.** Python parses the MIDI, pitch-shifts the morphed sample for each note, mixes into a single .wav. No Tone.js, no client-side audio scheduling. To change tempo, the user re-renders.
- **User-driven slot assignment.** The classifier exists as a confidence hint ("this sounds 78% piano-like") but the user explicitly chooses which slot a sound goes into.
- **11 fixed instrument slots.** User fills any subset; unfilled slots → silent for that track in the rendered song.

---

## Tech stack (do not deviate)

### Backend (Python 3.11+)
- **FastAPI** for the HTTP API
- **uvicorn** as the ASGI server
- **librosa** + **soundfile** + **pydub** + **scipy** + **numpy** — already used by `pipeline.py`
- **pretty_midi** — for MIDI parsing in `render-song`
- **python-multipart** — for FastAPI file uploads

### Frontend (Next.js 14+, App Router)
- **Next.js** with the App Router and the **/web** subfolder layout
- **TypeScript**
- Built-in fetch, **MediaRecorder** for mic capture
- **No Tone.js, no @tonejs/midi, no Meyda** — backend handles all audio work
- Tailwind optional; vanilla CSS modules are fine

### No third-party state management — `useState` and `useReducer` are enough.

---

## Repo layout

```
/                        ← Python backend lives at root (Jerry's existing code)
├── pipeline.py          ← classifier + morph + pitch_adjust (Jerry, refactored in N1)
├── server.py            ← FastAPI app (NEW, mission N2)
├── debug.py             ← classifier debug tool (Jerry, keep as-is)
├── rebuild_db.py        ← rebuilds instruments_db.csv (Jerry, keep)
├── instrument_recordings/   ← training set, 11 wavs/mp3s (Jerry, KEEP)
├── instruments_db.csv   ← extracted features for the training set (KEEP)
├── samples/             ← Jerry's old test artifacts (test*.m4a, *_*.wav, combined.wav)
├── SPEC.md              ← this file
├── requirements.txt     ← Python deps (NEW, mission N1)
└── web/                 ← Next.js app
    ├── package.json
    ├── next.config.js
    ├── tsconfig.json
    ├── public/
    │   └── midi/
    │       ├── index.json           ← list of bundled songs
    │       ├── blinding-lights.mid
    │       └── ...
    ├── src/
    │   └── app/
    │       ├── layout.tsx
    │       ├── page.tsx              ← Step1+2+3 single page
    │       ├── components/
    │       │   ├── InstrumentSlot.tsx
    │       │   ├── SongPicker.tsx
    │       │   ├── GenerateButton.tsx
    │       │   └── Player.tsx
    │       └── lib/
    │           ├── api.ts            ← fetch wrappers for /morph, /classify, /render-song
    │           └── recorder.ts       ← MediaRecorder wrapper
    └── styles/
        └── globals.css
```

---

## The 11 instrument slots

Match `pipeline.py`'s `INSTRUMENT_PITCH` keys exactly:

| Display name | Slot key | Center pitch (Hz) |
|---|---|---|
| Piano | `piano` | 440 |
| Acoustic Guitar | `aguitar` | 196 |
| Electric Guitar | `eguitar` | 196 |
| Bass Guitar | `bguitar` | 98 |
| Violin | `violin` | 440 |
| Cello | `cello` | 220 |
| Trumpet | `trumpet` | 440 |
| Flute | `flute` | 880 |
| Kick Drum | `kickdrum` | unpitched |
| Snare Drum | `snaredrum` | unpitched |
| Hi-Hat | `hihat` | unpitched |

---

## API contracts

### `POST /classify`
**Request:** multipart `audio` field (any wav/mp3/m4a/ogg/webm under 20MB).
**Response:**
```json
{
  "scores": {
    "piano": 0.78,
    "cello": 0.22,
    "violin": 0.05,
    ...
  },
  "best_match": "piano"
}
```
- `scores` is normalized: 1.0 = perfect match, 0.0 = no resemblance. Computed as `1 / (1 + distance)` then re-normalized so all 11 sum to 1.
- Used by the frontend purely as a hint under each slot.

### `POST /morph`
**Request:** multipart `audio` + form field `target_instrument` (one of the 11 keys).
**Response:** `audio/wav` body, 16-bit PCM, 22050Hz mono. The user's sound morphed and pitch-adjusted to match `target_instrument`'s spectral profile.

### `POST /render-song`
**Request:** JSON
```json
{
  "midi_filename": "blinding-lights.mid",
  "slots": {
    "piano":     "<base64 wav from prior /morph>",
    "bguitar":   "<base64 wav from prior /morph>",
    "kickdrum":  "<base64 wav from prior /morph>",
    ...
  },
  "tempo_percent": 100
}
```
- Slots may include any subset of the 11. Unprovided slots are silent.
- `midi_filename` must match a file in `web/public/midi/`. Frontend tells the server the relative path; server resolves it against a configured MIDI directory.
**Response:** `audio/wav` of the rendered cover.

---

## Server-side MIDI rendering algorithm (the crucial new code)

```python
def render_song(midi_path: str, slot_samples: dict[str, np.ndarray], tempo_percent: float = 100) -> np.ndarray:
    """
    midi_path: absolute path to the .mid file
    slot_samples: { instrument_key: morphed_audio_array_at_22050hz_mono }
    Returns: combined audio array at 22050Hz mono.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Identify tracks by heuristic
    track_map = identify_tracks(midi)
    # → { 'piano': <Instrument>, 'bass': <Instrument>, 'drums': <Instrument>, ... }
    # Mapping rules per SPEC: lowest-avg-pitch percussion → drums; lowest-avg-pitch
    # non-drum → bass; most-notes non-drum-non-bass → melody (treated as piano);
    # any remaining tracks → match by GM program number to closest of the 11 slots.

    # Apply tempo modifier
    duration = midi.get_end_time() / (tempo_percent / 100.0)
    sr = 22050
    out = np.zeros(int(duration * sr) + sr, dtype=np.float32)

    for slot_key, instrument_obj in track_map.items():
        if slot_key not in slot_samples:
            continue  # user didn't fill this slot → silent
        sample = slot_samples[slot_key]
        for note in instrument_obj.notes:
            note_start = note.start / (tempo_percent / 100.0)
            note_pitch_hz = 440 * 2 ** ((note.pitch - 69) / 12)
            shifted = pitch_shift_to_hz(sample, note_pitch_hz, slot_key)
            # for unpitched (drums), skip the shift
            start_idx = int(note_start * sr)
            end_idx = min(start_idx + len(shifted), len(out))
            out[start_idx:end_idx] += shifted[:end_idx - start_idx] * note.velocity / 127

    # Master limiter
    peak = np.max(np.abs(out))
    if peak > 0:
        out *= 0.891 / peak
    return out
```

Notes:
- `pitch_shift_to_hz` is a wrapper around `librosa.effects.pitch_shift` that computes semitone delta from the slot's center pitch (per `INSTRUMENT_PITCH`) to the MIDI note's frequency, clamped to ±24 semitones.
- For drums (`kickdrum`, `snaredrum`, `hihat`), do NOT pitch-shift — just play the morphed sample at every note's start time.
- `tempo_percent` scales note start times. Default 100. Range 50–200 acceptable.

---

## UI flow (single page, top to bottom)

```
┌─────────────────────────────────────────────────────────┐
│  Found-Sound Song Maker                                  │
├─────────────────────────────────────────────────────────┤
│  Step 1: Fill instrument slots                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │ Piano   │ │ Bass    │ │ Kick    │ │ Hi-hat  │        │
│  │ [+ Add] │ │ [▶][✕]  │ │ [+ Add] │ │ [+ Add] │        │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │
│  ... (11 total — auto-flow grid) ...                     │
│                                                            │
│  Click slot → modal: [🎙 Record] [📁 Upload]              │
│  After morph: shows hint "your sound was 78% piano-like" │
├─────────────────────────────────────────────────────────┤
│  Step 2: Pick a song                                      │
│  ┌──────┐ ┌──────┐ ┌──────┐                              │
│  │Blind.│ │Mario │ │Africa│  (15 cards, click selects)   │
│  └──────┘ └──────┘ └──────┘                              │
├─────────────────────────────────────────────────────────┤
│  Step 3: Generate                                         │
│  Tempo: [---●---] 100%   [✨ Generate]                    │
├─────────────────────────────────────────────────────────┤
│  ▶ ⏸  [============●----]  0:42/1:30                      │
│  [⬇ Download .wav]                                       │
└─────────────────────────────────────────────────────────┘
```

Style: dark mode, rounded corners, accent color #fbbf24, big chunky buttons. Don't over-design.

---

## Build order (the 6 missions)

Do these one at a time. Each must self-test before the next runs.

### N1 — Refactor `pipeline.py` + add MIDI rendering
- Extract `load_audio`, `strip_silence`, `extract`, `morph`, `pitch_adjust`, `_phone_hpf` from pipeline.py as importable library functions. They must work without depending on the `__main__` block.
- Move the existing `if __name__ == "__main__"` fixed-loop demo to a separate `experiments/fixed_loop_demo.py` (preserve it for reference; it was Jerry's working test).
- Add `classify_sound(audio_bytes_or_path) -> dict[str, float]` that returns confidence scores per instrument (1/(1+distance), normalized so they sum to 1).
- Add `process_sound(audio_bytes, target_instrument) -> bytes` that runs strip_silence + morph + pitch_adjust against the named target and returns 16-bit PCM 22050Hz mono WAV bytes.
- Add `render_song(midi_path, slot_samples, tempo_percent=100) -> np.ndarray` per the algorithm above. Use `pretty_midi`. Implement `identify_tracks` heuristically.
- Add `requirements.txt` with: librosa, numpy, scipy, soundfile, pydub, fastapi, uvicorn, python-multipart, pretty_midi.
- Self-test: write `experiments/test_render.py` that loads two of Jerry's training samples (`instrument_recordings/piano.wav` and `instrument_recordings/kickdrum.wav`), renders a tiny test MIDI through them via `render_song`, writes the output, and confirms the output has non-silent samples.

### N2 — FastAPI server (`server.py`)
- `POST /classify` → calls `classify_sound`.
- `POST /morph` → calls `process_sound`.
- `POST /render-song` → decodes base64 slot samples, calls `render_song`, returns wav.
- CORS allow-list: `http://localhost:3000` and `http://localhost:5173`.
- `/health` → `{"ok": true, "instruments_loaded": 11}`.
- Boots via `uvicorn server:app --reload --port 8000`. Document the command in a README.
- Self-test: hit `/health` and `/classify` with `instrument_recordings/piano.wav` via curl. Confirm `best_match: "piano"`.

### N3 — MIDI library sourcing
- Research top-30 most iconic / instantly-recognizable songs with free MIDI online. Pick 15 with smallest, cleanest track structure (under 200KB each, parseable by `pretty_midi`, identifiable bass + drums + melody tracks).
- Place in `web/public/midi/`. Generate `web/public/midi/index.json`: `[{filename, title, artist}]`.
- Document sources in `web/public/midi/SOURCES.md`.
- Validation script `web/scripts/validate-midi.mjs` (or a Python script if simpler) — every .mid must parse without error.
- Self-test: run the validation script. All 15 PASS.

### N4 — Next.js scaffold + 11 instrument slot UI
- `npx create-next-app@latest web --typescript --app --no-tailwind --import-alias "@/*"` (or with tailwind — agent's call, but document it).
- Build the InstrumentSlot component: 11 cards, click opens a modal with Record + Upload buttons. On audio captured, POST to `http://localhost:8000/morph` with the slot's `target_instrument`. Receive morphed wav blob, store in slot state.
- Show classifier hint under each slot — POST in parallel to `/classify` for the user-feedback display ("78% piano").
- Slot state lives in a single `useReducer` in the page component for now.
- Self-test: with the FastAPI server running, run `npm run dev` from `web/`, manually fill 2 slots, confirm morphed previews play.

### N5 — Song picker + generate + playback
- SongPicker component reads `/midi/index.json` (Next.js serves from `public/`).
- GenerateButton: when at least 1 slot is filled and a song is selected, POST to `/render-song` with the MIDI filename + base64 of each slot's morphed wav blob.
- Player: receive the rendered wav, play it via `<audio>` element. Show progress bar.
- Download button: trigger browser download with filename `{slug-of-song}-found-sound-cover.wav`.
- Tempo slider: 50%–150%, default 100%, sent as `tempo_percent` to render.
- Self-test: with backend up, fill 3 slots with `instrument_recordings/{piano,bguitar,kickdrum}.wav` (treat them as user uploads), pick a song, hit Generate, confirm audible playback recognizable as that song. Download → verify .wav opens in QuickTime/VLC.

### N6 — Polish + acceptance
- Dark mode (#0d0e12 bg, #fbbf24 accent), rounded buttons, spacing.
- Loading states: "Morphing..." while waiting for /morph; "Rendering..." while waiting for /render-song.
- Error toasts: mic-permission-denied, file-too-large, render-failed, server-down.
- Empty/disabled states: Generate disabled until ≥1 slot filled and a song selected.
- Acceptance walkthrough: write `ACCEPTANCE.md` with PASS/FAIL against every criterion in the next section.

---

## Acceptance criteria

- [ ] Backend `/health` returns `instruments_loaded: 11`.
- [ ] Backend `/classify` correctly identifies `instrument_recordings/piano.wav` as piano.
- [ ] Backend `/morph` returns a non-silent wav for any of Jerry's training samples + any target instrument.
- [ ] Backend `/render-song` returns a non-silent wav given a MIDI from the bundled library and at least 1 slot filled.
- [ ] Frontend boots cleanly on `npm run dev` with no console errors on initial load.
- [ ] User can fill any subset of 11 slots via mic recording.
- [ ] User can fill any subset of 11 slots via file upload.
- [ ] Each slot displays the classifier confidence hint as a percentage.
- [ ] Song picker grid renders ≥10 songs from `/midi/index.json`.
- [ ] Clicking Generate posts to /render-song and plays the returned audio in the browser.
- [ ] Output sounds **recognizably like the chosen song** but clearly made of the user's noises.
- [ ] User can download the result as a .wav.
- [ ] App works in Chrome and Firefox with no console errors.
- [ ] No repo-root Python files (`pipeline.py`, `debug.py`, `rebuild_db.py`, `instruments_db.csv`, `instrument_recordings/`) are deleted or broken by the frontend work.

---

## Out of scope (do NOT build these)

- Real-time tempo control during playback (tempo only applies on re-render).
- AI-generated melodies.
- User accounts, project saving, sharing.
- Mobile-optimized UI.
- Hungarian-algorithm auto-assignment of sounds to slots (user picks slot manually — the user explicitly opted out of auto-assignment).
- Multiple concurrent users / production deployment / authentication.

---

## Notes on existing files

`samples/` contains Jerry's old test artifacts — `test*.m4a` (his test inputs) and `test*_<instrument>.wav` (their morphed outputs) and `combined.wav` (his fixed-loop 8-bar arrangement). These are pure dev fixtures from before this spec existed. **Do not delete them** (useful as reference) but they are not part of the runtime.

`pipeline.py`'s `if __name__ == "__main__"` block (the 8-bar fixed-loop demo) gets moved to `experiments/fixed_loop_demo.py` in mission N1. That demo was Jerry's end-to-end smoke test before the FastAPI layer existed.

---

## Hackathon priorities (if time runs out)

Cut in this order:
1. First, cut the **classifier hint** display under each slot (the value-add is small for the cost).
2. Then cut the **tempo slider** (always render at 100%).
3. Then cut **download** (judges can record demo audio).
4. Then cut **manual cleanup of unused songs** in the MIDI library — ship 10 instead of 15.

Do **NOT** cut: instrument slots, MIDI song picker, /render-song, audible playback. That's the demo.
