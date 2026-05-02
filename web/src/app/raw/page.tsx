'use client';

import { useEffect, useRef, useState } from 'react';
import {
  analyzeMidi,
  health,
  prepRaw,
  renderSong,
  type SongAnalysis,
  type SongMeta,
} from '../lib/api';
import { recordAudio } from '../lib/recorder';
import { INSTRUMENTS, type InstrumentKey } from '../lib/instruments';

type FilledSlot = { wav: Blob; source: 'recorded' | 'uploaded' };
type SlotState = FilledSlot | { loading: true } | null;

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const isFilled = (s: SlotState): s is FilledSlot => s != null && !('loading' in s);

export default function RawPage() {
  const [slots, setSlots] = useState<Record<string, SlotState>>({});
  const [songs, setSongs] = useState<SongMeta[]>([]);
  const [selected, setSelected] = useState<SongMeta | null>(null);
  const [analysis, setAnalysis] = useState<SongAnalysis | null>(null);
  const [generating, setGenerating] = useState(false);
  const [rendered, setRendered] = useState<Blob | null>(null);
  const [healthOk, setHealthOk] = useState(false);
  const [temperature, setTemperature] = useState(100); // 0 = original, 100 = pitched to music

  useEffect(() => {
    const ping = () =>
      health()
        .then(() => setHealthOk(true))
        .catch(() => setHealthOk(false));
    ping();
    const id = setInterval(ping, 4000);
    fetch('/midi/index.json').then((r) => r.json()).then(setSongs).catch(console.error);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!selected) { setAnalysis(null); return; }
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
      const wav = await prepRaw(audio);
      setSlot(key, { wav, source });
    } catch (e) {
      alert(`${key}: ${e instanceof Error ? e.message : 'failed'}`);
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
      const wav = await renderSong({
        midiFilename: selected.filename,
        slots: filled,
        rawMode: true,
        temperature: temperature / 100,
      });
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
        <h1>Raw <span>Found-Sound</span></h1>
        <span className={`health ${healthOk ? 'ok' : ''}`}>
          {healthOk ? '● backend online' : '● backend offline'}
        </span>
      </header>

      <section style={{ background: 'rgba(251,191,36,0.06)', borderColor: '#fbbf24' }}>
        <h2>Raw mode</h2>
        <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.9rem' }}>
          Your sounds are pitch-shifted to match the MIDI but <strong>not morphed</strong> —
          you&apos;ll hear your own voice/noise distinctly, not a fake instrument.{' '}
          <a href="/" style={{ color: '#fbbf24' }}>Switch to morphed mode →</a>
        </p>
      </section>

      <section>
        <h2>1. Fill slots with raw sounds ({filledCount}/{INSTRUMENTS.length})</h2>
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
                  {loading ? 'working…' : filled ? `raw (${s.source})` : 'empty'}
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
            Detected from MIDI ({analysis.duration_sec.toFixed(1)}s).
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
        <h2>3. Temperature — timbre blend</h2>
        <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '0 0 12px' }}>
          Melody is always correct (pitch tracks the MIDI). Temperature only changes the timbre.{' '}
          <strong>0</strong> = a &quot;boop&quot; stays a boop, just pitched to each note &nbsp;·&nbsp;{' '}
          <strong>100</strong> = morphed to actually sound like the real instrument
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
        <h2>4. Generate (raw)</h2>
        <button className="big" disabled={!canGenerate} onClick={generate}>
          {generating ? 'Rendering…' : `✨ Generate (temp ${temperature})`}
        </button>
      </section>

      {renderedUrl && selected && (
        <section>
          <h2>Your raw cover</h2>
          <div className="player">
            <audio controls src={renderedUrl} />
            <a className="btn-link" href={renderedUrl} download={`${slug(selected.title)}-raw.wav`}>
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
