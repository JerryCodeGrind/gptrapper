import os
import json
import re
import sys
from io import BytesIO

import anthropic
from midiutil import MIDIFile

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

NOTE_MAP = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

DURATION_MAP = {
    "0": 0.0,
    "1": 4.0, "2": 2.0, "4": 1.0, "8": 0.5, "16": 0.25,
    "d4": 1.5, "d2": 3.0, "d8": 0.75,
}


def pitch_to_midi(pitch: str) -> int:
    match = re.match(r"([A-Ga-g][#b]?)(\d)", pitch)
    if not match:
        raise ValueError(f"Invalid pitch: {pitch}")
    note, octave = match.group(1), int(match.group(2))
    return (octave + 1) * 12 + NOTE_MAP[note]


def duration_to_beats(duration: str) -> float:
    if duration in DURATION_MAP:
        return DURATION_MAP[duration]
    try:
        return float(duration)
    except (ValueError, TypeError):
        return 1.0


def compose(description: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16192,
        system="""You are a music composer. Respond ONLY with a valid JSON object, no markdown or explanation:

{
  "title": "string",
  "tempo": 120,
  "tracks": [
    {
      "name": "string",
      "instrument": 0,
      "notes": [
        { "pitch": "C4", "duration": "4", "velocity": 80, "wait": "0" }
      ]
    }
  ]
}

- pitch: scientific notation e.g. C4, D#3, Gb5
- duration: "1"=whole "2"=half "4"=quarter "8"=eighth "16"=sixteenth "d4"=dotted quarter
- velocity: 1-127
- wait: rest before this note ("0" = no rest)
- instrument: General MIDI number (0=piano, 25=guitar, 32=bass, 48=strings, 73=flute)
- 16-32 notes per track, 1-3 tracks, make it musical""",
        messages=[{"role": "user", "content": description}],
    )

    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response was truncated — increase max_tokens or request fewer notes")
    raw = response.content[0].text.strip()
    cleaned = re.sub(r"^```json\s*|```\s*$", "", raw, flags=re.IGNORECASE).strip()
    return json.loads(cleaned)


def build_midi(composition: dict) -> bytes:
    tracks = composition["tracks"]
    tempo = composition.get("tempo", 120)
    midi = MIDIFile(len(tracks))

    for i, track in enumerate(tracks):
        midi.addTempo(i, 0, tempo)
        midi.addProgramChange(i, 0, 0, track.get("instrument", 0))
        time = 0.0
        for note in track["notes"]:
            time += duration_to_beats(note.get("wait", "0"))
            midi.addNote(i, 0, pitch_to_midi(note["pitch"]), time,
                         duration_to_beats(note["duration"]), note.get("velocity", 80))
            time += duration_to_beats(note["duration"])

    buf = BytesIO()
    midi.writeFile(buf)
    return buf.getvalue()


if __name__ == "__main__":
    description = " ".join(sys.argv[1:]) or "a full rock song with electric guitar, piano, drums, and some fun adlib type noises"
    print(f"Composing: {description}")

    composition = compose(description)
    print(f"Title: {composition['title']}  |  Tempo: {composition['tempo']} BPM")
    for t in composition["tracks"]:
        print(f"  {t['name']}: {len(t['notes'])} notes")

    midi_bytes = build_midi(composition)

    filename = re.sub(r"[^a-z0-9-]", "", composition["title"].lower().replace(" ", "-")) + ".mid"
    with open(filename, "wb") as f:
        f.write(midi_bytes)

    print(f"Saved: {filename}")