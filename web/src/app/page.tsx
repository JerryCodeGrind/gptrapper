'use client';

import { useEffect, useRef, useState } from 'react';
import {
  analyzeMidi,
  classify,
  defaultSample,
  defaultsAvailable,
  health,
  morph,
  renderSong,
  type SongAnalysis,
  type SongMeta,
} from './lib/api';
import { recordAudio } from './lib/recorder';
import { INSTRUMENTS, type InstrumentKey } from './lib/instruments';

type FilledSlot = { wav: Blob; source: 'recorded' | 'uploaded' | 'default'; bestMatch: string | null };
type SlotState = FilledSlot | { loading: true } | null;

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const isFilled = (s: SlotState): s is FilledSlot => s != null && !('loading' in s);

export default function Page() {
  const [slots, setSlots] = useState<Record<string, SlotState>>({});
  const [songs, setSongs] = useState<SongMeta[]>([]);
  const [selected, setSelected] = useState<SongMeta | null>(null);
  const [analysis, setAnalysis] = useState<SongAnalysis | null>(null);
  const [generating, setGenerating] = useState(false);
  const [rendered, setRendered] = useState<Blob | null>(null);
  const [healthOk, setHealthOk] = useState(false);
  const [defaultSlots, setDefaultSlots] = useState<Set<string>>(new Set());

  useEffect(() => {
    const ping = () =>
      health()
        .then(() => setHealthOk(true))
        .catch(() => setHealthOk(false));
    ping();
    const id = setInterval(ping, 4000);
    fetch('/midi/index.json').then((r) => r.json()).then(setSongs).catch(console.error);
    defaultsAvailable().then((arr) => setDefaultSlots(new Set(arr))).catch(console.error);
    return () => clearInterval(id);
  }, []);

  // Re-fetch defaults list when health flips to ok
  useEffect(() => {
    if (healthOk && defaultSlots.size === 0) {
      defaultsAvailable().then((arr) => setDefaultSlots(new Set(arr))).catch(console.error);
    }
  }, [healthOk, defaultSlots.size]);

  // When the user picks a song, ask the backend which slots that MIDI actually uses
  useEffect(() => {
    if (!selected) {
      setAnalysis(null);
      return;
    }
    let cancelled = false;
    analyzeMidi(selected.filename)
      .then((a) => { if (!cancelled) setAnalysis(a); })
      .catch(() => { if (!cancelled) setAnalysis(null); });
    return () => { cancelled = true; };
  }, [selected]);

  const renderedUrl = useBlobUrl(rendered);

  function setSlot(key: InstrumentKey, s: SlotState) {
    setSlots((prev) => ({ ...prev, [key]: s }));
  }

  async function fillSlot(key: InstrumentKey, audio: Blob, source: 'recorded' | 'uploaded') {
    setSlot(key, { loading: true });
    try {
      const [wav, cls] = await Promise.all([
        morph(audio, key),
        classify(audio).catch(() => null),
      ]);
      setSlot(key, { wav, source, bestMatch: cls?.best_match ?? null });
    } catch (e) {
      alert(`${key}: ${e instanceof Error ? e.message : 'failed'}`);
      setSlot(key, null);
    }
  }

  async function useDefault(key: InstrumentKey) {
    setSlot(key, { loading: true });
    try {
      const wav = await defaultSample(key);
      setSlot(key, { wav, source: 'default', bestMatch: null });
    } catch (e) {
      alert(`${key} default: ${e instanceof Error ? e.message : 'failed'}`);
      setSlot(key, null);
    }
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

  async function fillAllDefaults() {
    await Promise.all(
      INSTRUMENTS.filter((i) => defaultSlots.has(i.key)).map((i) => useDefault(i.key)),
    );
  }

  async function generate() {
    if (!selected) return;
    const filled: Record<string, Blob> = {};
    for (const i of INSTRUMENTS) {
      const s = slots[i.key];
      if (isFilled(s)) filled[i.key] = s.wav;
    }
    if (Object.keys(filled).length === 0) {
      alert('Fill at least one slot first');
      return;
    }
    setGenerating(true);
    setRendered(null);
    try {
      const wav = await renderSong({ midiFilename: selected.filename, slots: filled });
      setRendered(wav);
    } catch (e) {
      alert(`Render failed: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      setGenerating(false);
    }
  }

  const filledCount = INSTRUMENTS.filter((i) => isFilled(slots[i.key])).length;
  const canGenerate = selected != null && filledCount > 0 && !generating;

  return (
    <main>
      <header>
        <h1>Found-Sound <span>Song Maker</span></h1>
        <span className={`health ${healthOk ? 'ok' : ''}`}>
          {healthOk ? '● backend online' : '● backend offline'}
        </span>
      </header>

      <section>
        <h2>1. Fill instrument slots ({filledCount}/{INSTRUMENTS.length})</h2>
        <div style={{ marginBottom: 12 }}>
          <button className="secondary" onClick={fillAllDefaults} disabled={!healthOk}>⚡ Fill all defaults</button>{' '}
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
                  {loading ? 'working…' : filled ? `filled (${s.source})` : 'empty'}
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
                      <button
                        className="secondary"
                        onClick={() => useDefault(inst.key)}
                        disabled={loading || !healthOk || !defaultSlots.has(inst.key)}
                        title={defaultSlots.has(inst.key) ? "Use the found-sound default" : 'No found-sound default — record or upload'}
                      >
                        {defaultSlots.has(inst.key) ? 'Default' : 'No default'}
                      </button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <h2>2. Pick a song</h2>
        {songs.length === 0 ? (
          <p style={{ color: '#9ca3af' }}>Loading songs…</p>
        ) : (
          <div className="songs">
            {songs.map((song) => (
              <div
                key={song.filename}
                className={`song ${selected?.filename === song.filename ? 'selected' : ''}`}
                onClick={() => setSelected(song)}
              >
                <div className="title">{song.title}</div>
                <div className="artist">{song.artist}</div>
                {song.tracks && <div className="tracks">uses: {song.tracks.join(', ')}</div>}
              </div>
            ))}
          </div>
        )}
      </section>

      {selected && analysis && (
        <section>
          <h2>This song needs:</h2>
          <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '0 0 12px' }}>
            Detected from MIDI ({analysis.duration_sec.toFixed(1)}s). Filling these slots gives the best result.
          </p>
          <div className="slots">
            {Object.entries(analysis.needed_slots).map(([key, info]) => {
              const inst = INSTRUMENTS.find((i) => i.key === key);
              if (!inst) return null;
              const filled = isFilled(slots[key]);
              return (
                <div key={key} className={`slot ${filled ? 'filled' : ''}`}>
                  <div className="emoji">{inst.emoji}</div>
                  <div className="label">{inst.label}</div>
                  <div className="status">{info.notes} notes · {filled ? '✓ filled' : 'empty'}</div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <section>
        <h2>3. Generate</h2>
        <button className="big" disabled={!canGenerate} onClick={generate}>
          {generating ? 'Rendering…' : '✨ Generate Song'}
        </button>
      </section>

      {renderedUrl && selected && (
        <section>
          <h2>Your cover</h2>
          <div className="player">
            <audio controls src={renderedUrl} />
            <a className="btn-link" href={renderedUrl} download={`${slug(selected.title)}-found-sound-cover.wav`}>
              ⬇ Download .wav
            </a>
          </div>
        </section>
      )}
    </main>
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
