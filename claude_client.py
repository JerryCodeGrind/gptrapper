"""Claude / Anthropic — NLP for the /studio text prompt.

Two helpers:
  classify_prompt(text)  → {artist, title, genre, subgenre, mood, instruments, ...}
  apply_edit(params, edit_text) → updated params dict

Both use Claude tool-use to force structured JSON output (no fragile string parsing).
"""
import json
import os
from typing import Optional

from anthropic import Anthropic

# Slot keys must match web/src/app/lib/instruments.ts
_VALID_SLOTS = {
    "piano", "aguitar", "eguitar", "bguitar", "violin", "cello",
    "trumpet", "flute", "kickdrum", "snaredrum", "hihat",
}

_MODEL = "claude-sonnet-4-6"  # Sonnet 4.6 — best for structured output, cheap enough

_client: Optional[Anthropic] = None


class ClaudeError(RuntimeError):
    pass


def _get_client() -> Anthropic:
    global _client
    if _client is not None:
        return _client
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ClaudeError("ANTHROPIC_API_KEY not configured in .env")
    _client = Anthropic(api_key=key)
    return _client


# Tool schema Claude is forced to fill in for the initial classify call.
_CLASSIFY_TOOL = {
    "name": "music_intent",
    "description": (
        "Capture the user's intent for generating music. If the user named a "
        "specific song/artist, fill artist+title. Otherwise leave them null and "
        "rely on genre/subgenre/mood. Always pick instruments from the allowed list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "artist": {
                "type": ["string", "null"],
                "description": "Artist name if user explicitly named one (e.g. 'Laufey', 'Drake'). Null otherwise.",
            },
            "title": {
                "type": ["string", "null"],
                "description": "Song title if user named a specific track (e.g. 'From The Start'). Null otherwise.",
            },
            "genre": {
                "type": "string",
                "description": "Top-level genre — one of: pop, indie pop, rock, hip-hop, trap, edm, lo-fi, jazz, classical, country, r&b, folk, metal.",
            },
            "subgenre": {
                "type": "string",
                "description": "More specific style descriptor, e.g. 'slow indie pop', 'jazzy bedroom pop', 'aggressive trap'.",
            },
            "mood": {
                "type": "string",
                "description": "One short phrase: 'melancholic', 'upbeat', 'dreamy', 'energetic', etc.",
            },
            "tempo": {
                "type": "number",
                "description": "BPM. If a famous song is named, use your training-data knowledge of that track's actual tempo. Otherwise pick a value typical for the genre/mood. Range 50–200.",
            },
            "key": {
                "type": "string",
                "description": "Musical key letter. If a famous song is named, use that song's actual key. Otherwise pick a sensible default. One of: C, C#, D, D#, E, F, F#, G, G#, A, A#, B.",
                "enum": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
            },
            "mode": {
                "type": "string",
                "enum": ["major", "minor"],
                "description": "Major or minor. Use the named song's actual mode if known; otherwise pick what fits the mood (minor for melancholic/dark, major for upbeat/dreamy).",
            },
            "instruments": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": sorted(_VALID_SLOTS),
                },
                "description": "2–5 instrument slot keys that suit the song. Always include at least one rhythm element (kickdrum/hihat/snaredrum) and one melodic element (piano/aguitar/etc).",
            },
        },
        "required": ["genre", "subgenre", "mood", "tempo", "key", "mode", "instruments"],
        "additionalProperties": False,
    },
}


def classify_prompt(text: str) -> dict:
    """User text → structured intent. Raises ClaudeError on failure."""
    if not text or not text.strip():
        raise ClaudeError("empty prompt")
    client = _get_client()
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "music_intent"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this music-generation request. Respond ONLY by calling the "
                    "`music_intent` tool. Never reply in text.\n\n"
                    f"User request: {text.strip()}"
                ),
            }
        ],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "music_intent":
            data = dict(block.input)
            # Filter instruments to allowed set defensively.
            data["instruments"] = [i for i in data.get("instruments", []) if i in _VALID_SLOTS]
            return data
    raise ClaudeError("Claude returned no tool_use block")


# Tool schema for live edits: Claude diffs the current params and returns updated values.
_EDIT_TOOL = {
    "name": "edit_params",
    "description": (
        "Apply the user's edit instruction to the current music parameters. "
        "Return the FULL updated params object — keep fields that didn't change."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tempo": {"type": "number"},
            "key": {"type": "string"},
            "mode": {"type": "string", "enum": ["major", "minor"]},
            "instruments": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_VALID_SLOTS)},
            },
            "bars": {"type": "integer", "enum": [8, 16, 24, 32]},
            "mood": {"type": "string"},
            "explanation": {
                "type": "string",
                "description": "One-sentence explanation of what you changed and why.",
            },
        },
        "required": ["tempo", "key", "mode", "instruments", "bars", "mood", "explanation"],
        "additionalProperties": False,
    },
}


def apply_edit(current_params: dict, edit_text: str) -> dict:
    """Take existing params + a user instruction → updated params."""
    if not edit_text or not edit_text.strip():
        raise ClaudeError("empty edit instruction")
    client = _get_client()
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        tools=[_EDIT_TOOL],
        tool_choice={"type": "tool", "name": "edit_params"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Current music parameters (JSON):\n"
                    f"{json.dumps(current_params, indent=2)}\n\n"
                    f"User edit instruction: {edit_text.strip()}\n\n"
                    "Apply the edit and return the FULL updated parameters via the "
                    "`edit_params` tool. Keep unchanged fields the same."
                ),
            }
        ],
    )
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "edit_params":
            data = dict(block.input)
            data["instruments"] = [i for i in data.get("instruments", []) if i in _VALID_SLOTS]
            return data
    raise ClaudeError("Claude returned no tool_use block")
