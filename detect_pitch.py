import sys
sys.path.insert(0, ".")
from pipeline import load_audio, strip_silence, _detect_pitch, hz_to_note

path = sys.argv[1] if len(sys.argv) > 1 else "instrument_recordings/hihat.wav"
y, _ = load_audio(path)
y    = strip_silence(y)
hz   = _detect_pitch(y)
print(f"{path}: {hz:.1f} Hz  ({hz_to_note(hz)})")
