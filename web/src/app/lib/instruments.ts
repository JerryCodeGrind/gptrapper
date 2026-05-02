export const INSTRUMENTS = [
  { key: 'piano',     label: 'Piano',           emoji: '🎹' },
  { key: 'aguitar',   label: 'Acoustic Guitar', emoji: '🎸' },
  { key: 'eguitar',   label: 'Electric Guitar', emoji: '🎸' },
  { key: 'bguitar',   label: 'Bass Guitar',     emoji: '🎸' },
  { key: 'violin',    label: 'Violin',          emoji: '🎻' },
  { key: 'cello',     label: 'Cello',           emoji: '🎻' },
  { key: 'trumpet',   label: 'Trumpet',         emoji: '🎺' },
  { key: 'flute',     label: 'Flute',           emoji: '🪈' },
  { key: 'kickdrum',  label: 'Kick Drum',       emoji: '🥁' },
  { key: 'snaredrum', label: 'Snare Drum',      emoji: '🥁' },
  { key: 'hihat',     label: 'Hi-Hat',          emoji: '🎩' },
] as const;

export type InstrumentKey = (typeof INSTRUMENTS)[number]['key'];
