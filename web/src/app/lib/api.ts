const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000';

export type ClassifyResult = {
  scores: Record<string, number>;
  best_match: string;
};

export type SongMeta = {
  filename: string;
  title: string;
  artist: string;
  tracks?: string[];
};

export type SongAnalysis = {
  filename: string;
  duration_sec: number;
  needed_slots: Record<string, { notes: number; track_name: string }>;
};

export type MusicParams = {
  tempo: number;
  key: string;
  mode: 'major' | 'minor' | string;
  time_signature: number;
  instruments: string[];
  bars: number;
  mood?: string;
  genre?: string;
  subgenre?: string;
  artist?: string | null;
  title?: string | null;
  source?: string;
};

export async function textToMusicParams(text: string, bars: number): Promise<MusicParams> {
  const r = await fetch(`${API_BASE}/text-to-music-params`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, bars }),
  });
  if (!r.ok) throw new Error(`text-to-music-params ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function applyEdit(params: MusicParams, edit: string): Promise<MusicParams & { explanation?: string }> {
  const r = await fetch(`${API_BASE}/apply-edit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ params, edit }),
  });
  if (!r.ok) throw new Error(`apply-edit ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function health(): Promise<{ ok: boolean; instruments_loaded: number }> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`/health ${r.status}`);
  return r.json();
}

export async function defaultsAvailable(): Promise<string[]> {
  const r = await fetch(`${API_BASE}/defaults-available`);
  if (!r.ok) throw new Error(`defaults-available ${r.status}`);
  const j = await r.json();
  return j.slots as string[];
}

export async function classify(audio: Blob): Promise<ClassifyResult> {
  const fd = new FormData();
  fd.append('audio', audio, 'clip.webm');
  const r = await fetch(`${API_BASE}/classify`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`classify ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function prepRaw(audio: Blob): Promise<{ wav: Blob; pitchHz: number }> {
  const fd = new FormData();
  fd.append('audio', audio, 'clip.webm');
  const r = await fetch(`${API_BASE}/prep-raw`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`prep-raw ${r.status}: ${await r.text()}`);
  const wav = await r.blob();
  const pitchHz = Number(r.headers.get('X-Detected-Pitch-Hz') ?? '0') || 0;
  return { wav, pitchHz };
}

export async function morph(audio: Blob, targetInstrument: string): Promise<Blob> {
  const fd = new FormData();
  fd.append('audio', audio, 'clip.webm');
  fd.append('target_instrument', targetInstrument);
  const r = await fetch(`${API_BASE}/morph`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`morph ${r.status}: ${await r.text()}`);
  return r.blob();
}

export async function defaultSample(instrumentKey: string): Promise<Blob> {
  const r = await fetch(`${API_BASE}/default-sample/${instrumentKey}`);
  if (!r.ok) throw new Error(`default-sample ${r.status}: ${await r.text()}`);
  return r.blob();
}

export async function analyzeMidi(filename: string): Promise<SongAnalysis> {
  const r = await fetch(`${API_BASE}/analyze-midi/${encodeURIComponent(filename)}`);
  if (!r.ok) throw new Error(`analyze-midi ${r.status}`);
  return r.json();
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  let bin = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
  }
  return btoa(bin);
}

export async function renderSong(opts: {
  midiFilename?: string;
  midiB64?: string;
  slots: Record<string, Blob>;
  tempoPercent?: number;
  rawMode?: boolean;
  temperature?: number;
}): Promise<Blob> {
  if (!opts.midiFilename && !opts.midiB64) {
    throw new Error('renderSong: provide midiFilename or midiB64');
  }
  const slotsB64: Record<string, string> = {};
  for (const [k, blob] of Object.entries(opts.slots)) {
    slotsB64[k] = arrayBufferToBase64(await blob.arrayBuffer());
  }
  const r = await fetch(`${API_BASE}/render-song`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      midi_filename: opts.midiFilename,
      midi_b64: opts.midiB64,
      slots: slotsB64,
      tempo_percent: opts.tempoPercent ?? 100,
      raw_mode: opts.rawMode ?? false,
      temperature: opts.temperature ?? 1.0,
    }),
  });
  if (!r.ok) throw new Error(`render-song ${r.status}: ${await r.text()}`);
  return r.blob();
}
