'use client';

import { useEffect, useRef, useState } from 'react';
import {
  applyEdit,
  health,
  prepRaw,
  renderSong,
  textToMusicParams,
  type MusicParams,
} from '../lib/api';
import { recordAudio } from '../lib/recorder';
import { INSTRUMENTS, type InstrumentKey } from '../lib/instruments';

type FilledSlot = { wav: Blob; source: 'recorded' | 'uploaded'; pitchHz: number };
type SlotState = FilledSlot | { loading: true } | null;

// Priority for picking which slot's pitch sets the song's key.
// Lead/melodic instruments first, then bass, then anything else pitched.
const PITCH_PRIORITY: InstrumentKey[] = [
  'piano', 'aguitar', 'eguitar', 'violin', 'cello', 'flute', 'trumpet', 'bguitar',
];

function hzToNoteName(hz: number): string {
  if (!hz || hz <= 0) return '—';
  const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const midi = Math.round(69 + 12 * Math.log2(hz / 440));
  return `${NOTES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

const isFilled = (s: SlotState): s is FilledSlot => s != null && !('loading' in s);
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

type Status = { kind: 'idle' } | { kind: 'busy'; msg: string } | { kind: 'error'; msg: string };

export default function StudioPage() {
  const [slots, setSlots] = useState<Record<string, SlotState>>({});
  const [healthOk, setHealthOk] = useState(false);

  // NLP state
  const [prompt, setPrompt] = useState('a chill lo-fi study beat');
  const [bars, setBars] = useState<8 | 16 | 24 | 32>(16);
  const [temperature, setTemperature] = useState(80); // raw-mode timbre blend

  const [params, setParams] = useState<MusicParams | null>(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [lastExplanation, setLastExplanation] = useState<string | null>(null);

  const [rendered, setRendered] = useState<Blob | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });

  const renderedUrl = useBlobUrl(rendered);

  useEffect(() => {
    const ping = () =>
      health()
        .then(() => setHealthOk(true))
        .catch(() => setHealthOk(false));
    ping();
    const id = setInterval(ping, 4000);
    return () => clearInterval(id);
  }, []);

  function setSlot(key: InstrumentKey, s: SlotState) {
    setSlots((prev) => ({ ...prev, [key]: s }));
  }

  async function fillSlot(key: InstrumentKey, audio: Blob, source: 'recorded' | 'uploaded') {
    setSlot(key, { loading: true });
    try {
      const { wav, pitchHz } = await prepRaw(audio);
      setSlot(key, { wav, source, pitchHz });
    } catch (e) {
      alert(`${key}: ${e instanceof Error ? e.message : 'failed'}`);
      setSlot(key, null);
    }
  }

  /** Pick the user's "lead" pitched slot to drive the song's key.
   *  Skips drum slots (no fundamental). Uses PITCH_PRIORITY ordering. */
  function primaryPitchHz(): number | null {
    for (const key of PITCH_PRIORITY) {
      const s = slots[key];
      if (isFilled(s) && s.pitchHz > 0) return s.pitchHz;
    }
    return null;
  }

  async function record(key: InstrumentKey) {
    try {
      const blob = await recordAudio(5000);
      await fillSlot(key, blob, 'recorded');
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'failed';
      alert(msg === 'mic-permission-denied' ? 'Microphone access denied.' : msg);
    }
  }

  async function classify() {
    if (!prompt.trim()) return;
    setStatus({ kind: 'busy', msg: 'asking Claude…' });
    setParams(null);
    setLastExplanation(null);
    try {
      const p = await textToMusicParams(prompt.trim(), bars);
      setParams(p);
      setStatus({ kind: 'idle' });
    } catch (e) {
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'classify failed' });
    }
  }

  async function applyEditClick() {
    if (!params || !editPrompt.trim()) return;
    setStatus({ kind: 'busy', msg: 'applying edit…' });
    try {
      const updated = await applyEdit(params, editPrompt.trim());
      setParams({ ...params, ...updated });
      setLastExplanation(updated.explanation ?? null);
      setEditPrompt('');
      setStatus({ kind: 'idle' });
    } catch (e) {
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'edit failed' });
    }
  }

  async function generate() {
    if (!params) {
      setStatus({ kind: 'error', msg: 'classify a prompt first' });
      return;
    }
    const filled: Record<string, Blob> = {};
    for (const i of INSTRUMENTS) {
      const s = slots[i.key];
      if (isFilled(s)) filled[i.key] = s.wav;
    }
    if (Object.keys(filled).length === 0) {
      setStatus({ kind: 'error', msg: 'fill at least one slot first' });
      return;
    }
    setStatus({ kind: 'busy', msg: 'loading Magenta (~30 MB on first run)…' });
    setRendered(null);
    try {
      const { generateTrio, uint8ToBase64 } = await import('../lib/magenta');
      const userPitchHz = primaryPitchHz();
      setStatus({
        kind: 'busy',
        msg: userPitchHz
          ? `generating Magenta beat in your noise's key (${hzToNoteName(userPitchHz)})…`
          : 'generating Magenta beat…',
      });
      const { midiBytes } = await generateTrio({
        tempo: params.tempo,
        bars: params.bars as 8 | 16 | 24 | 32,
        temperature: 1.0,
        transposeToHz: userPitchHz ?? undefined,
      });
      const midiB64 = uint8ToBase64(midiBytes);

      setStatus({ kind: 'busy', msg: 'rendering with your sounds…' });
      const wav = await renderSong({
        midiB64,
        slots: filled,
        rawMode: true,
        temperature: temperature / 100,
      });
      setRendered(wav);
      setStatus({ kind: 'idle' });
    } catch (e) {
      console.error(e);
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'generation failed' });
    }
  }

  const filledCount = INSTRUMENTS.filter((i) => isFilled(slots[i.key])).length;
  const canGenerate = params != null && filledCount > 0 && status.kind !== 'busy';

  return (
    <main>
      <header>
        <h1>Studio <span>Text → Song</span></h1>
        <span className={`health ${healthOk ? 'ok' : ''}`}>
          {healthOk ? '● backend online' : '● backend offline'}
        </span>
      </header>

      <section style={{ background: 'rgba(251,191,36,0.06)', borderColor: '#fbbf24' }}>
        <h2>Studio mode</h2>
        <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.9rem' }}>
          Fill slots with your sounds, type what you want to hear, and Magenta generates a multitrack MIDI
          (lead + bass + drums) that we render with your sounds.{' '}
          <a href="/raw" style={{ color: '#fbbf24' }}>Switch to /raw →</a>
        </p>
      </section>

      <section>
        <h2>1. Fill instrument slots ({filledCount}/{INSTRUMENTS.length})</h2>
        <div style={{ marginBottom: 12 }}>
          <button className="secondary" onClick={() => setSlots({})} disabled={filledCount === 0}>Clear all</button>
        </div>
        <div className="slots">
          {INSTRUMENTS.map((inst) => {
            const s = slots[inst.key];
            const filled = isFilled(s);
            const loading = s != null && 'loading' in s;
            return (
              <div key={inst.key} className={`slot ${filled ? 'filled' : ''} ${loading ? 'loading' : ''}`}>
                <div className="emoji">{inst.emoji}</div>
                <div className="label">{inst.label}</div>
                <div className="status">
                  {loading ? 'working…' : filled
                    ? (
                      <>
                        raw ({s.source})
                        {s.pitchHz > 0 && (
                          <span style={{ color: '#fbbf24', marginLeft: 4 }}>
                            · {hzToNoteName(s.pitchHz)}
                          </span>
                        )}
                      </>
                    )
                    : 'empty'}
                </div>
                <div className="actions">
                  {filled ? (
                    <>
                      <Preview blob={s.wav} />
                      <button className="danger" onClick={() => setSlot(inst.key, null)}>✕</button>
                    </>
                  ) : (
                    <>
                      <button onClick={() => record(inst.key)} disabled={loading || !healthOk}>🎙 Rec</button>
                      <UploadBtn onFile={(f) => fillSlot(inst.key, f, 'uploaded')} disabled={loading || !healthOk} />
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h2>2. Describe the song</h2>
        <textarea
          className="prompt"
          rows={2}
          placeholder='e.g. "a chill lo-fi study beat" or "laufey from the start"'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <div className="bars-toggle">
            {([8, 16, 24, 32] as const).map((b) => (
              <button
                key={b}
                className={bars === b ? '' : 'secondary'}
                onClick={() => setBars(b)}
              >
                {b} bars
              </button>
            ))}
          </div>
          <button className="secondary" onClick={classify} disabled={status.kind === 'busy'}>
            🤖 Classify
          </button>
        </div>
      </section>

      {params && (
        <section>
          <h2>3. Generated parameters</h2>
          <div className="params">
            <Param k="Tempo" v={`${Math.round(params.tempo)} BPM`} />
            <Param k="Key" v={`${params.key} ${params.mode}`} />
            <Param k="Bars" v={String(params.bars)} />
            <Param k="Genre" v={params.subgenre || params.genre || '—'} />
            <Param k="Mood" v={params.mood || '—'} />
            {params.artist && params.title && (
              <Param k="Seed song" v={`${params.artist} – ${params.title}`} />
            )}
            <Param k="Instruments" v={params.instruments.join(', ')} />
          </div>
          <div className="row" style={{ marginTop: 12, gap: 8 }}>
            <input
              className="prompt"
              placeholder='Edit: e.g. "make it slower" or "switch to minor"'
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') applyEditClick(); }}
            />
            <button className="secondary" onClick={applyEditClick} disabled={status.kind === 'busy' || !editPrompt.trim()}>
              Apply edit
            </button>
          </div>
          {lastExplanation && (
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '8px 0 0' }}>
              <strong>Claude:</strong> {lastExplanation}
            </p>
          )}
        </section>
      )}

      <section>
        <h2>4. Temperature — timbre blend</h2>
        <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '0 0 12px' }}>
          0 = pure raw user sound · 100 = morphed to real instrument
        </p>
        <div className="temp">
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
          <div className="temp-readout">
            <span>0 your timbre</span>
            <span className="temp-value">{temperature}</span>
            <span>100 real instrument</span>
          </div>
        </div>
      </section>

      <section>
        <h2>5. Generate</h2>
        {(() => {
          const ph = primaryPitchHz();
          if (ph) {
            return (
              <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '0 0 12px' }}>
                Your noise rings at <strong style={{ color: '#fbbf24' }}>{hzToNoteName(ph)}</strong> ({Math.round(ph)} Hz) —
                the Magenta beat will be transposed to that key, regardless of the seed song.
              </p>
            );
          }
          return null;
        })()}
        <button className="big" disabled={!canGenerate} onClick={generate}>
          {status.kind === 'busy' ? status.msg : '✨ Generate Song'}
        </button>
        {status.kind === 'error' && (
          <p style={{ color: '#ef4444', marginTop: 8, fontSize: '0.85rem' }}>{status.msg}</p>
        )}
      </section>

      {renderedUrl && (
        <section>
          <h2>Your generated cover</h2>
          <div className="player">
            <audio controls src={renderedUrl} />
            <a className="btn-link" href={renderedUrl} download={`${slug(prompt).slice(0, 40) || 'studio'}-found-sound.wav`}>
              ⬇ Download .wav
            </a>
          </div>
        </section>
      )}
    </main>
  );
}

function Param({ k, v }: { k: string; v: string }) {
  return (
    <div className="param">
      <div className="param-k">{k}</div>
      <div className="param-v">{v}</div>
    </div>
  );
}

function useBlobUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob) { setUrl(null); return; }
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);
  return url;
}

function Preview({ blob }: { blob: Blob }) {
  const ref = useRef<HTMLAudioElement | null>(null);
  const url = useBlobUrl(blob);
  return (
    <>
      <button className="secondary" onClick={() => { ref.current?.play(); }}>▶</button>
      {url && <audio ref={ref} src={url} preload="auto" />}
    </>
  );
}

function UploadBtn({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  const ref = useRef<HTMLInputElement | null>(null);
  return (
    <>
      <button className="secondary" disabled={disabled} onClick={() => ref.current?.click()}>📁 Upload</button>
      <input
        ref={ref}
        type="file"
        accept="audio/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
    </>
  );
}
