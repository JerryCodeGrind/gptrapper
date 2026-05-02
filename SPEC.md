# Found-Sound Song Maker — MVP Spec

## What we're building

A browser app where users upload everyday sounds (tapping a table, hitting a water bottle, stomping) and pick a popular song. The app then plays a "cover" of that song using the user's sounds as the instruments — pitch-shifted across the song's melody, bassline, and drum pattern.

Think: "Blinding Lights, but performed by someone tapping their desk."

The output should be **recognizable as the chosen song** but clearly transformed by the user's weird sounds — not a literal copy.

---

## Tech stack (do not deviate)

- **Vite + Vanilla JS** (no React — keep it lightweight, faster to ship)
- **Tone.js** — audio playback, sampling, pitch-shifting, scheduling, effects
- **@tonejs/midi** — parse bundled MIDI files into JS note data
- **Meyda.js** — extract audio features for auto-classifying user sounds
- **No backend.** Everything runs in the browser. Deploy to Vercel/Netlify when done.

Install commands:
```bash
npm create vite@latest found-sound-song-maker -- --template vanilla
cd found-sound-song-maker
npm install tone @tonejs/midi meyda
```

---

## File structure

```
/
├── index.html
├── src/
│   ├── main.js               # App entry, wires everything together
│   ├── ui.js                 # DOM rendering and event handlers
│   ├── recorder.js           # MediaRecorder wrapper for capturing user sounds
│   ├── classifier.js         # Meyda-based sound classification
│   ├── midi-loader.js        # Loads + parses bundled MIDI files
│   ├── song-engine.js        # Tone.js sampler + scheduling + transformations
│   ├── exporter.js           # Render song to .wav using Tone.Offline
│   └── styles.css
├── public/
│   └── midi/                 # Bundled MIDI files (10–20 popular songs)
│       ├── blinding-lights.mid
│       ├── seven-nation-army.mid
│       └── ...
└── package.json
```

---

## Core MVP features (must ship)

### 1. Sound input

- User can **record** a sound from their mic (MediaRecorder API, save as WebM/OGG blob).
- User can **upload** an audio file (.mp3, .wav, .ogg, .webm — accept all).
- User can add **4–8 sounds total**.
- Each sound shows in a list with a name (user-editable), a play button to preview, and a delete button.

### 2. Sound classification (auto + manual override)

When a sound is added, run it through Meyda to extract:
- **Duration** (seconds)
- **Spectral centroid** (average — measure of brightness)
- **RMS / loudness**
- **Zero-crossing rate** (proxy for noisiness vs. pitched)

Then apply a simple rule tree to assign a role:

```
if duration < 0.2s and centroid is low (< 800Hz)   → KICK
if duration < 0.2s and centroid is high (> 3000Hz) → HI_HAT
if duration < 0.4s and centroid is mid             → SNARE
if duration > 0.4s and zero-crossing rate is low   → MELODIC (pitched-ish)
if duration > 0.4s and zero-crossing rate is high  → BASS / SUB
```

**Always show the assigned role in the UI with a dropdown so the user can override.** Roles available: `KICK`, `SNARE`, `HI_HAT`, `BASS`, `MELODIC`, `PERC`.

### 3. Song picker

- Bundle **10–15 MIDI files** in `public/midi/` (suggestions: Blinding Lights, Seven Nation Army, Take On Me, Smoke on the Water, Megalovania, Africa, Stayin' Alive, Sweet Child O' Mine, Hey Jude, Pirates of the Caribbean theme, Mario Theme, Tetris Theme, Happy Birthday).
- Show as a grid of cards with song name + artist.
- User clicks one to select it.

### 4. Song generation engine

When user clicks **"Generate Song"**:

1. **Load and parse the selected MIDI file** with `@tonejs/midi`.
2. **Identify tracks** by heuristic:
   - Track with lowest average pitch and `isPercussion` true → **drums**
   - Track with lowest average pitch (non-drum) → **bass**
   - Track with most notes / highest avg pitch → **melody**
   - Other tracks → **chords / harmony**
3. **Map user sounds to tracks** based on classified roles:
   - `KICK` / `SNARE` / `HI_HAT` sounds → drums (split by hit type if available, otherwise just KICK on every drum hit)
   - `BASS` sound → bass track
   - `MELODIC` sound → melody track
   - If a role is missing (e.g. user only uploaded percussion sounds), **gracefully skip that track** rather than crashing.
4. **Build a Tone.js Sampler for each role** using the user's sound as the sample. Set the sample's base note to C4.
5. **Schedule all notes** using `Tone.Transport` and the timing from the MIDI.
6. **Apply transformations** (see next section).
7. **Press play.** Show a stop button and a progress bar.

### 5. Transformations (this is what makes it not sound like a literal copy)

Apply these to every generation:

- **Pitch envelope shift**: shift the entire melody track up or down by 1–2 octaves randomly, so the user's bottle ping doesn't try to play the song's actual original pitches (which would sound off because of timbre mismatch). Same for bass — shift it down 1 octave so it's properly thumpy.
- **Tempo**: read tempo from MIDI, but offer a UI slider (50%–150% of original BPM) defaulting to 100%.
- **Timing humanization**: jitter every note's start time by ±15ms (random) so it feels handmade, not robotic.
- **Effects chain on each role** (using Tone.js built-ins):
  - Drums: subtle compression + a touch of reverb
  - Bass: lowpass filter at ~400Hz + light distortion
  - Melody: medium reverb + slight delay
- **Master bus**: gentle compressor + limiter to avoid clipping.

### 6. Export to .wav

- Button: **"Download Song"**
- Use `Tone.Offline()` to render the entire song into an AudioBuffer, then encode to .wav and trigger a browser download.
- Filename: `{song-name}-found-sound-cover.wav`

### 7. UI flow (single page, top to bottom)

```
┌─────────────────────────────────────┐
│  Found-Sound Song Maker             │
├─────────────────────────────────────┤
│  Step 1: Add your sounds            │
│  [🎙 Record]  [📁 Upload]            │
│  ┌─────────────────────────────┐    │
│  │ table-tap     [▶] [KICK ▾] [✕] │    │
│  │ bottle-ping   [▶] [HI_HAT ▾][✕]│    │
│  │ stomp         [▶] [BASS ▾] [✕] │    │
│  └─────────────────────────────┘    │
├─────────────────────────────────────┤
│  Step 2: Pick a song                │
│  ┌──────┐ ┌──────┐ ┌──────┐         │
│  │Blind.│ │7-Nat.│ │Mario │  ...    │
│  └──────┘ └──────┘ └──────┘         │
├─────────────────────────────────────┤
│  Step 3: Generate                   │
│  Tempo: [-----●-----]  100%         │
│  [✨ Generate Song]                  │
├─────────────────────────────────────┤
│  ▶ ⏸  [============●----]  0:42/1:30│
│  [⬇ Download .wav]                  │
└─────────────────────────────────────┘
```

Style: dark mode, big chunky buttons, a little playful (slight rounded corners, maybe a subtle waveform animation when playing). Don't over-design it.

---

## Code patterns (the tricky bits, written out)

### Tone.js Sampler from a user-uploaded blob

```js
import * as Tone from 'tone';

async function makeSamplerFromBlob(blob, baseNote = 'C4') {
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await Tone.getContext().decodeAudioData(arrayBuffer);
  
  const sampler = new Tone.Sampler({
    urls: { [baseNote]: audioBuffer },
    release: 1,
  }).toDestination();
  
  await Tone.loaded();
  return sampler;
}
```

### Parsing MIDI and scheduling notes

```js
import { Midi } from '@tonejs/midi';

const midi = await Midi.fromUrl('/midi/blinding-lights.mid');

midi.tracks.forEach((track) => {
  const sampler = samplersByTrack[track.name]; // assigned earlier
  if (!sampler) return;
  
  track.notes.forEach((note) => {
    Tone.Transport.schedule((time) => {
      sampler.triggerAttackRelease(
        note.name,           // pitch (e.g. "C4")
        note.duration,
        time + (Math.random() - 0.5) * 0.03, // ±15ms humanization
        note.velocity
      );
    }, note.time);
  });
});

Tone.Transport.bpm.value = midi.header.tempos[0]?.bpm ?? 120;
Tone.Transport.start();
```

### Meyda feature extraction for classification

```js
import Meyda from 'meyda';

async function classify(audioBuffer) {
  const features = Meyda.extract(
    ['rms', 'spectralCentroid', 'zcr'],
    audioBuffer.getChannelData(0).slice(0, 2048)
  );
  const duration = audioBuffer.duration;
  
  // Apply rule tree from spec
  if (duration < 0.2 && features.spectralCentroid < 30) return 'KICK';
  if (duration < 0.2 && features.spectralCentroid > 100) return 'HI_HAT';
  if (duration < 0.4) return 'SNARE';
  if (features.zcr < 0.1) return 'MELODIC';
  return 'BASS';
}
```
*(Note: Meyda's spectral centroid is in bins, not Hz — tune the thresholds empirically once it's running.)*

### Offline render to .wav

```js
const buffer = await Tone.Offline(({ transport }) => {
  // Re-build the entire song graph inside this callback
  setupSamplersAndScheduleSong();
  transport.start();
}, songDurationSeconds);

// Convert AudioBuffer to .wav blob (use a small wav-encoder helper)
const wavBlob = audioBufferToWav(buffer);
downloadBlob(wavBlob, `${songName}-found-sound-cover.wav`);
```

For `audioBufferToWav`, use the `audiobuffer-to-wav` npm package (~30 lines, no dependencies).

---

## Build order (follow this exactly)

Do these one at a time and test each before moving on. Do not try to build everything at once.

1. **Vite scaffold + install deps.** Get a blank page rendering.
2. **Tone.js Sampler proof-of-concept.** Hardcode one audio file in `public/`, load it into a Sampler, play a C major scale on button click. **Until this works, nothing else matters.**
3. **MIDI playback proof-of-concept.** Load one bundled MIDI file with `@tonejs/midi`, play it through the hardcoded sampler. Confirm timing sounds right.
4. **Sound recording UI.** Record from mic → store blob in memory → preview playback. No classification yet.
5. **Sound upload UI.** Same as recording but via file input.
6. **Classification.** Wire up Meyda, assign roles, show role dropdowns.
7. **Song picker UI.** Grid of cards, clicking selects a song.
8. **Full song generation.** Map sounds to tracks, build samplers per role, schedule MIDI notes through them. **First milestone where it actually demos.**
9. **Transformations.** Pitch shifts, tempo control, humanization, effects chain.
10. **Export to .wav.** Tone.Offline render + download.
11. **Polish UI.** Dark mode, animations, loading states, error handling.

---

## Acceptance criteria (definition of "done")

- [ ] User can record at least 4 sounds from mic
- [ ] User can upload sounds as alternative
- [ ] Each sound auto-classifies into a role with manual override available
- [ ] User can pick from at least 10 bundled songs
- [ ] Clicking "Generate" produces audible playback of the song using user's sounds
- [ ] Playback has bass + melody + drums all clearly using different user sounds
- [ ] Tempo slider works
- [ ] Pitch transformations are applied (sounds aren't trying to hit unreasonable original-song pitches)
- [ ] Output sounds **recognizably like the chosen song** but clearly made of the user's noises
- [ ] User can download the result as a .wav file
- [ ] App works in Chrome and Firefox without errors
- [ ] No backend, no API keys, runs entirely in the browser

---

## Out of scope (do NOT build these)

- "Type any song name" search — just use the bundled library.
- AI-generated melodies (Magenta.js etc.) — using existing MIDI is faster and better.
- User accounts, saving projects, sharing.
- Mobile-optimized UI — desktop only is fine.
- Multiple genre/style presets — one good generation pipeline is enough.
- Real-time effects tweaking — apply on generation, that's it.
- Polyphonic samplers per drum hit — one sound per role is fine.

---

## Notes on MIDI files

- Source MIDI files from [BitMidi](https://bitmidi.com/) or similar **free** repositories.
- Test each bundled MIDI before committing — some have weird track structures (e.g. melody split across 4 tracks). Either pick clean MIDI files or write a track-merging step.
- Keep MIDI files **under 200KB each** to keep the bundle small.

---

## Hackathon priorities (if you run out of time)

Cut in this order:
1. First, cut the **export to .wav** (judges can record demo audio if needed).
2. Then cut **manual classification override** (auto-classify only).
3. Then cut **tempo slider** (use original BPM always).
4. Then cut **effects chain** (dry playback is fine).

Do **not** cut: sound input, song picker, MIDI playback through user sounds. That's the demo.

Good luck. Ship it.
