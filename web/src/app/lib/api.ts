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

export async function prepRaw(audio: Blob): Promise<Blob> {
  const fd = new FormData();
  fd.append('audio', audio, 'clip.webm');
  const r = await fetch(`${API_BASE}/prep-raw`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`prep-raw ${r.status}: ${await r.text()}`);
  return r.blob();
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
  midiFilename: string;
  slots: Record<string, Blob>;
  tempoPercent?: number;
  rawMode?: boolean;
  temperature?: number;
}): Promise<Blob> {
  const slotsB64: Record<string, string> = {};
  for (const [k, blob] of Object.entries(opts.slots)) {
    slotsB64[k] = arrayBufferToBase64(await blob.arrayBuffer());
  }
  const r = await fetch(`${API_BASE}/render-song`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      midi_filename: opts.midiFilename,
      slots: slotsB64,
      tempo_percent: opts.tempoPercent ?? 100,
      raw_mode: opts.rawMode ?? false,
      temperature: opts.temperature ?? 1.0,
    }),
  });
  if (!r.ok) throw new Error(`render-song ${r.status}: ${await r.text()}`);
  return r.blob();
}
