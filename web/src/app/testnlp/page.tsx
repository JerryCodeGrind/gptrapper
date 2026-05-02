'use client';

import { useEffect, useRef, useState } from 'react';
import { applyEdit, health, textToMusicParams, type MusicParams } from '../lib/api';

type Status = { kind: 'idle' } | { kind: 'busy'; msg?: string } | { kind: 'error'; msg: string };

export default function TestNlpPage() {
  const [healthOk, setHealthOk] = useState(false);
  const [prompt, setPrompt] = useState('a chill lo-fi study beat');
  const [bars, setBars] = useState<8 | 16 | 24 | 32>(16);
  const [params, setParams] = useState<MusicParams | null>(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [explanation, setExplanation] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const [history, setHistory] = useState<{ prompt: string; params: MusicParams }[]>([]);
  const [playing, setPlaying] = useState(false);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    const ping = () => health().then(() => setHealthOk(true)).catch(() => setHealthOk(false));
    ping();
    const id = setInterval(ping, 4000);
    return () => clearInterval(id);
  }, []);

  async function classify() {
    if (!prompt.trim()) return;
    setStatus({ kind: 'busy' });
    setExplanation(null);
    try {
      const p = await textToMusicParams(prompt.trim(), bars);
      setParams(p);
      setHistory((h) => [{ prompt: prompt.trim(), params: p }, ...h].slice(0, 20));
      setStatus({ kind: 'idle' });
    } catch (e) {
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'failed' });
    }
  }

  async function applyEditClick() {
    if (!params || !editPrompt.trim()) return;
    setStatus({ kind: 'busy' });
    try {
      const updated = await applyEdit(params, editPrompt.trim());
      setParams({ ...params, ...updated });
      setExplanation(updated.explanation ?? null);
      setEditPrompt('');
      setStatus({ kind: 'idle' });
    } catch (e) {
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'failed' });
    }
  }

  async function previewMagenta() {
    if (!params) return;
    if (playing && stopRef.current) {
      stopRef.current();
      stopRef.current = null;
      setPlaying(false);
      return;
    }
    setStatus({ kind: 'busy', msg: 'loading Magenta + soundfont…' });
    try {
      const { previewTrio } = await import('../lib/magenta');
      setStatus({ kind: 'busy', msg: 'sampling MusicVAE…' });
      const handle = await previewTrio({
        tempo: params.tempo,
        bars: params.bars as 8 | 16 | 24 | 32,
        temperature: 1.0,
      });
      stopRef.current = handle.stop;
      setPlaying(true);
      setStatus({ kind: 'idle' });
      // Auto-clear playing flag after the sequence ends
      window.setTimeout(() => {
        setPlaying(false);
        stopRef.current = null;
      }, Math.ceil(handle.durationSec * 1000) + 500);
    } catch (e) {
      console.error(e);
      setStatus({ kind: 'error', msg: e instanceof Error ? e.message : 'preview failed' });
    }
  }

  // Stop playback on unmount
  useEffect(() => () => { stopRef.current?.(); }, []);

  return (
    <main>
      <header>
        <h1>Test NLP <span>Claude → params</span></h1>
        <span className={`health ${healthOk ? 'ok' : ''}`}>
          {healthOk ? '● backend online' : '● backend offline'}
        </span>
      </header>

      <section style={{ background: 'rgba(251,191,36,0.06)', borderColor: '#fbbf24' }}>
        <h2>NLP + Magenta sandbox</h2>
        <p style={{ color: '#9ca3af', margin: 0, fontSize: '0.9rem' }}>
          Type any music prompt → Claude classifies it → we pull a real song&apos;s tempo/key from
          our 566-song database (or live API if you name a song) → hit{' '}
          <strong style={{ color: '#fbbf24' }}>Preview</strong> and Magenta.js generates a brand-new
          beat in your browser using those params as a style hint.{' '}
          <strong style={{ color: '#fbbf24' }}>The seed song is never copied or played</strong> — only its
          tempo and key are used.{' '}
          <a href="/studio" style={{ color: '#fbbf24' }}>Full studio →</a>
        </p>
      </section>

      <section>
        <h2>1. Prompt</h2>
        <textarea
          className="prompt"
          rows={2}
          placeholder='e.g. "laufey from the start", "sad lofi piano", "drake gods plan"'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) classify(); }}
        />
        <div className="row" style={{ marginTop: 12 }}>
          <div className="bars-toggle">
            {([8, 16, 24, 32] as const).map((b) => (
              <button key={b} className={bars === b ? '' : 'secondary'} onClick={() => setBars(b)}>
                {b} bars
              </button>
            ))}
          </div>
          <button onClick={classify} disabled={status.kind === 'busy' || !healthOk}>
            🤖 Classify
          </button>
          <span style={{ color: '#6b7280', fontSize: '0.75rem' }}>(⌘+Enter / Ctrl+Enter)</span>
        </div>
        {status.kind === 'error' && (
          <p style={{ color: '#ef4444', marginTop: 8, fontSize: '0.85rem' }}>{status.msg}</p>
        )}
      </section>

      {params && (
        <section>
          <h2>2. Result</h2>
          <div className="params">
            <Param k="Tempo" v={`${Math.round(params.tempo)} BPM`} />
            <Param k="Key" v={`${params.key} ${params.mode}`} />
            <Param k="Bars" v={String(params.bars)} />
            <Param k="Time sig" v={`${params.time_signature}/4`} />
            <Param k="Genre" v={params.genre || '—'} />
            <Param k="Subgenre" v={params.subgenre || '—'} />
            <Param k="Mood" v={params.mood || '—'} />
            <Param k="Source" v={params.source || '—'} />
            {params.artist && params.title && (
              <Param
                k="Seed song"
                v={`${params.artist} – ${params.title}`}
                hint="Where the tempo + key were sourced. Magenta does NOT copy this song — it generates a fresh beat using the same tempo/key."
              />
            )}
            <Param k="Instruments" v={params.instruments.join(', ')} />
          </div>

          <div className="row" style={{ marginTop: 12, gap: 8 }}>
            <button onClick={previewMagenta} disabled={status.kind === 'busy'}>
              {playing ? '⏹ Stop' : status.kind === 'busy' ? (status.msg ?? 'working…') : '🎵 Preview Magenta beat'}
            </button>
            <span style={{ color: '#6b7280', fontSize: '0.75rem' }}>
              first run downloads a ~30 MB model + soundfont
            </span>
          </div>

          <details style={{ marginTop: 12 }}>
            <summary style={{ cursor: 'pointer', color: '#9ca3af', fontSize: '0.85rem' }}>raw JSON</summary>
            <pre className="codeblock">{JSON.stringify(params, null, 2)}</pre>
          </details>

          <h3 style={{ marginTop: 20, fontSize: '0.95rem' }}>Apply edit</h3>
          <div className="row" style={{ gap: 8 }}>
            <input
              className="prompt"
              placeholder='e.g. "make it slower" or "switch to minor"'
              value={editPrompt}
              onChange={(e) => setEditPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') applyEditClick(); }}
            />
            <button className="secondary" onClick={applyEditClick} disabled={status.kind === 'busy' || !editPrompt.trim()}>
              Apply
            </button>
          </div>
          {explanation && (
            <p style={{ color: '#9ca3af', fontSize: '0.85rem', margin: '8px 0 0' }}>
              <strong>Claude:</strong> {explanation}
            </p>
          )}
        </section>
      )}

      {history.length > 1 && (
        <section>
          <h2>History</h2>
          <ul className="history">
            {history.map((h, i) => (
              <li key={i}>
                <span className="hist-prompt">&ldquo;{h.prompt}&rdquo;</span>{' '}
                <span className="hist-meta">
                  → {Math.round(h.params.tempo)} BPM · {h.params.key} {h.params.mode} ·{' '}
                  {h.params.subgenre || h.params.genre} ·{' '}
                  {h.params.artist ? `${h.params.artist} - ${h.params.title}` : h.params.mood}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

function Param({ k, v, hint }: { k: string; v: string; hint?: string }) {
  return (
    <div className="param" title={hint}>
      <div className="param-k">{k}{hint && <span style={{ marginLeft: 4, opacity: 0.6 }}>ⓘ</span>}</div>
      <div className="param-v">{v}</div>
    </div>
  );
}
