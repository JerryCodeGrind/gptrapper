/* eslint-disable @typescript-eslint/no-explicit-any */
// Magenta.js wrapper — generates a multitrack NoteSequence (lead + bass + drums)
// using MusicVAE.trio_16bar, then converts to MIDI bytes.
//
// Magenta.js is a heavy browser-only library (~30 MB checkpoint download on
// first use). This module guards against SSR and loads lazily.

// mel_2bar_small is only ~20 MB. trio_16bar (the multitrack model) is 860 MB
// of weights — too big for a browser in a hackathon. We sample a 2-bar melody
// and synthesize drums + bass programmatically below.
const MEL_CKPT =
  'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small';

type MagentaCore = any;
type MagentaVAE = any;
let _core: MagentaCore | null = null;
let _vae: MagentaVAE | null = null;
let _trio: any | null = null;
let _trioReady: Promise<any> | null = null;
let _player: any | null = null;
let _playerReady: Promise<any> | null = null;

async function getMagenta(): Promise<{ core: MagentaCore; vae: MagentaVAE }> {
  if (typeof window === 'undefined') {
    throw new Error('Magenta is browser-only');
  }
  if (!_core) {
    console.log('[magenta] importing @magenta/music/es6/core …');
    const t0 = performance.now();
    _core = await import('@magenta/music/es6/core');
    console.log(`[magenta] core ready (${Math.round(performance.now() - t0)} ms)`);
  }
  if (!_vae) {
    console.log('[magenta] importing @magenta/music/es6/music_vae …');
    const t0 = performance.now();
    _vae = await import('@magenta/music/es6/music_vae');
    console.log(`[magenta] music_vae ready (${Math.round(performance.now() - t0)} ms)`);
  }
  return { core: _core, vae: _vae };
}

const SF_URL = 'https://storage.googleapis.com/magentadata/js/soundfonts/sgm_plus';

async function getPlayer(): Promise<any> {
  if (_player) return _player;
  if (_playerReady) return _playerReady;
  _playerReady = (async () => {
    const { core } = await getMagenta();
    const p = new core.SoundFontPlayer(SF_URL);
    _player = p;
    return p;
  })();
  return _playerReady;
}

async function getMelodyModel(): Promise<any> {
  if (_trio) return _trio;
  if (_trioReady) return _trioReady;
  _trioReady = (async () => {
    const { vae } = await getMagenta();
    console.log('[magenta] loading mel_2bar_small (~20 MB) from', MEL_CKPT);
    const t0 = performance.now();
    const model = new vae.MusicVAE(MEL_CKPT);
    try {
      await model.initialize();
    } catch (e) {
      console.error('[magenta] model.initialize() failed:', e);
      throw e;
    }
    console.log(`[magenta] mel model ready in ${Math.round(performance.now() - t0)} ms`);
    _trio = model;
    return model;
  })();
  return _trioReady;
}

function hzToMidi(hz: number): number {
  return 69 + 12 * Math.log2(hz / 440);
}

/** Mutate a quantized NoteSequence so its melodic tonic ≈ targetHz.
 *  Drums (is_drum=true notes) are left alone. We compute the lowest pitched
 *  note in the lead/bass tracks as a stand-in for the tonic, then shift by
 *  the closest interval (mod 12) so the new tonic equals targetHz's pitch class.
 */
function transposeToHz(seq: any, targetHz: number): void {
  if (!targetHz || targetHz <= 0) return;
  const pitchedNotes = seq.notes.filter((n: any) => !n.isDrum && typeof n.pitch === 'number');
  if (pitchedNotes.length === 0) return;
  const seqTonic = Math.min(...pitchedNotes.map((n: any) => n.pitch));
  const targetMidi = hzToMidi(targetHz);
  // Shift only by pitch-class so the song stays in a sensible octave range
  let semitones = Math.round(targetMidi - seqTonic);
  // Reduce to the closest pitch-class shift (-6..+6 semitones), so we don't
  // teleport everything 4 octaves up if the user's noise is very high or low.
  while (semitones > 6) semitones -= 12;
  while (semitones < -6) semitones += 12;
  if (semitones === 0) return;
  for (const n of seq.notes) {
    if (!n.isDrum && typeof n.pitch === 'number') {
      n.pitch = Math.max(0, Math.min(127, n.pitch + semitones));
    }
  }
}

/** GM drum MIDI pitches we use programmatically. */
const KICK = 36;
const SNARE = 38;
const HIHAT_CLOSED = 42;

/** Add drums + bass to a melody NoteSequence, looping the melody to fill `bars`.
 *  The melody comes from MusicVAE.mel_2bar_small (2 bars). We:
 *   1. Loop it to N bars
 *   2. Add a programmatic kick/snare/hihat pattern on each bar
 *   3. Add a bass track = melody pitches shifted down 2 octaves, only on beat 1 and 3
 */
function buildBeatSequence(
  melodySeq: any,
  bars: number,
  stepsPerBar: number,
  tempoBpm: number,
): any {
  const totalSteps = bars * stepsPerBar;
  const melodySteps = melodySeq.totalQuantizedSteps ?? (2 * stepsPerBar);
  const notes: any[] = [];

  // 1. Looped melody (instrument 0)
  for (let loop = 0; loop * melodySteps < totalSteps; loop++) {
    for (const n of melodySeq.notes) {
      const start = (n.quantizedStartStep ?? 0) + loop * melodySteps;
      const end = (n.quantizedEndStep ?? start + 1) + loop * melodySteps;
      if (start >= totalSteps) continue;
      notes.push({
        pitch: n.pitch,
        quantizedStartStep: start,
        quantizedEndStep: Math.min(end, totalSteps),
        velocity: n.velocity ?? 90,
        instrument: 0,
        program: 0,
        isDrum: false,
      });
    }
  }

  // 2. Programmatic drums (instrument 1, isDrum=true)
  // Kick on beat 1 and 3 (steps 0, 8 of each 16-step bar)
  // Snare on beat 2 and 4 (steps 4, 12)
  // Closed hihat on every 8th (steps 0, 2, 4, ..., 14)
  for (let bar = 0; bar < bars; bar++) {
    const base = bar * stepsPerBar;
    // Kick
    for (const beat of [0, 8]) {
      notes.push({
        pitch: KICK,
        quantizedStartStep: base + beat,
        quantizedEndStep: base + beat + 1,
        velocity: 110,
        instrument: 1,
        isDrum: true,
      });
    }
    // Snare
    for (const beat of [4, 12]) {
      notes.push({
        pitch: SNARE,
        quantizedStartStep: base + beat,
        quantizedEndStep: base + beat + 1,
        velocity: 100,
        instrument: 1,
        isDrum: true,
      });
    }
    // Hihat 8ths
    for (let s = 0; s < stepsPerBar; s += 2) {
      notes.push({
        pitch: HIHAT_CLOSED,
        quantizedStartStep: base + s,
        quantizedEndStep: base + s + 1,
        velocity: 75,
        instrument: 1,
        isDrum: true,
      });
    }
  }

  // 3. Bass — melody shifted down 2 octaves, but only the first note of each beat
  //    so it doesn't muddy the mix. Instrument 2.
  const beatBoundaries = new Set<number>();
  for (let b = 0; b < bars; b++) {
    for (const beat of [0, 4, 8, 12]) beatBoundaries.add(b * stepsPerBar + beat);
  }
  for (const n of melodySeq.notes) {
    for (let loop = 0; loop * melodySteps < totalSteps; loop++) {
      const start = (n.quantizedStartStep ?? 0) + loop * melodySteps;
      if (start >= totalSteps) continue;
      if (!beatBoundaries.has(start)) continue;
      notes.push({
        pitch: Math.max(24, n.pitch - 24),
        quantizedStartStep: start,
        quantizedEndStep: Math.min(start + 4, totalSteps),
        velocity: 95,
        instrument: 2,
        program: 33, // GM Electric Bass (finger)
        isDrum: false,
      });
    }
  }

  return {
    quantizationInfo: { stepsPerQuarter: stepsPerBar / 4 },
    totalQuantizedSteps: totalSteps,
    tempos: [{ time: 0, qpm: tempoBpm }],
    notes,
  };
}

/** Generate a "trio-style" beat:
 *   - Melody from MusicVAE.mel_2bar_small (real Magenta-generated notes)
 *   - Drums + bass added programmatically
 *  The result feels like a multitrack beat without the 860 MB trio_16bar download.
 */
export async function generateTrio(opts: {
  tempo: number;
  bars: 8 | 16 | 24 | 32;
  temperature?: number;
  transposeToHz?: number;
}): Promise<{ midiBytes: Uint8Array }> {
  const model = await getMelodyModel();
  console.log('[magenta] sampling 2-bar melody…');
  const tSample = performance.now();
  const samples: any[] = await model.sample(1, opts.temperature ?? 1.0);
  console.log(`[magenta] sampled in ${Math.round(performance.now() - tSample)} ms`);
  const melodySeq = samples[0];
  const stepsPerBar = 16;

  let seq = buildBeatSequence(melodySeq, opts.bars, stepsPerBar, opts.tempo);
  if (opts.transposeToHz) transposeToHz(seq, opts.transposeToHz);

  const { core } = await getMagenta();
  const unq = core.sequences.unquantizeSequence(seq);
  const midiBytes: Uint8Array = core.sequenceProtoToMidi(unq);
  return { midiBytes };
}

/** Generate a trio NoteSequence and play it via Magenta's SoundFontPlayer.
 *  Returns the player and the unquantized sequence; caller can stop playback
 *  via `player.stop()`. No slot-filling, no backend — pure browser preview.
 */
export async function previewTrio(opts: {
  tempo: number;
  bars: 8 | 16 | 24 | 32;
  temperature?: number;
  transposeToHz?: number;
  onNote?: (note: any) => void;
}): Promise<{ stop: () => void; durationSec: number }> {
  const model = await getMelodyModel();
  console.log('[magenta:preview] sampling 2-bar melody…');
  const tSample = performance.now();
  const samples: any[] = await model.sample(1, opts.temperature ?? 1.0);
  console.log(`[magenta:preview] sampled in ${Math.round(performance.now() - tSample)} ms`);
  const melodySeq = samples[0];
  const stepsPerBar = 16;

  let seq = buildBeatSequence(melodySeq, opts.bars, stepsPerBar, opts.tempo);
  if (opts.transposeToHz) transposeToHz(seq, opts.transposeToHz);

  const { core } = await getMagenta();
  const unq = core.sequences.unquantizeSequence(seq);
  const player = await getPlayer();
  if (opts.onNote) {
    player.callbackObject = { run: opts.onNote, stop: () => {} };
  }
  // SoundFontPlayer needs to load instrument samples for the notes in this sequence
  await player.loadSamples(unq);
  player.start(unq);
  const durationSec = (opts.bars * stepsPerBar) * (60.0 / opts.tempo) / 4.0; // 16 steps/bar @ 4/4
  return {
    stop: () => { try { player.stop(); } catch { /* already stopped */ } },
    durationSec,
  };
}

export function uint8ToBase64(arr: Uint8Array): string {
  const CHUNK = 0x8000;
  let bin = '';
  for (let i = 0; i < arr.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(arr.subarray(i, i + CHUNK)));
  }
  return btoa(bin);
}
