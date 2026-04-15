"""Ask Gemini on Vertex AI to pick paragraph-break points inside speaker turns.

Design:
- One Gemini call per transcribe. We ship a chunk-per-line input with
  anonymous speaker letters (A/B/…) and ask for a JSON list of chunk IDs
  to break after. Structured output (`response_schema`) guarantees the
  shape, so there's no parsing drift to handle.
- Round-trip safe by construction: Gemini never re-emits text, it only
  picks break points. We slice chunks into paragraphs on our side.
- Two post-process passes clean up what the LLM is unreliable about:
  short trailing fragments get merged back into their setup (punchlines
  belong with the thought they resolve), and oversized paragraphs get
  split at the nearest discourse marker.

If Gemini isn't reachable (ADC missing, no project, API error) the caller
falls back to one-paragraph-per-turn — the legacy wall-of-text shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from .transcript import _group_into_turns, _Turn
from .types import ChunkLabel

# `global` has the widest model availability — the flash-lite preview lives
# there but not in regional endpoints. Override via $GOOGLE_CLOUD_LOCATION
# if you need in-region traffic.
PARAGRAPHIZE_MODEL = os.environ.get("POLYPHONY_PARAGRAPHIZE_MODEL", "gemini-3.1-flash-lite-preview")
PARAGRAPHIZE_LOCATION_DEFAULT = "global"

_STRANDED_TAIL_MAX_CHARS = 120
_OVERSIZED_PARAGRAPH_MIN_CHARS = 1000

# Discourse markers we'll split an oversized paragraph at, roughly in order
# of "hardness". The leading ". " means the marker only counts when it
# starts a new sentence, so we won't split inside a clause.
_DISCOURSE_MARKERS = (
    ". And then ",
    ". But then ",
    ". So then ",
    ". Now ",
    ". So, ",
    ". So ",
    ". But ",
    ". And so, ",
    ". And so ",
    ". And ",
    ". Then ",
    ". Meanwhile, ",
)


def paragraphize_via_gemini(
    labels: list[ChunkLabel],
    names: list[str] | None,
    context_hint: str | None = None,
    project: str | None = None,
    location: str | None = None,
    source_audio: Path | None = None,
) -> list[list[str]] | None:
    """Ask Gemini on Vertex AI to pick paragraph-break points.

    Returns a list aligned with `_group_into_turns`; each inner list is
    the paragraphs for that turn. Returns None on any failure (ADC
    missing, no project configured, API error, malformed response).
    Callers pass the result straight to `build_transcript`; if None,
    the markdown falls back to one-paragraph-per-turn.
    """
    turns = _group_into_turns(labels, review_threshold=100)
    if not turns:
        return []

    if source_audio is not None:
        from .cache import load_paragraphs

        cached = load_paragraphs(source_audio, len(turns))
        if cached is not None:
            return cached

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.warning(
            "No GCP project set (pass --project or export $GOOGLE_CLOUD_PROJECT); skipping paragraphize pass."
        )
        return None
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or PARAGRAPHIZE_LOCATION_DEFAULT

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        logger.warning(f"google-genai not installed ({e}); skipping paragraphize pass.")
        return None

    prompt = _build_prompt(labels, context_hint)
    valid_ids = {lbl.chunk.idx for lbl in labels}
    schema = {
        "type": "object",
        "properties": {
            "break_after_chunk": {
                "type": "array",
                "items": {"type": "integer"},
            }
        },
        "required": ["break_after_chunk"],
    }
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=schema,
        # Gemini 2.5+ Flash burns thinking tokens against max_output_tokens
        # before emitting the visible response, so we need significant
        # headroom even though the JSON we actually want is <1KB.
        max_output_tokens=32768,
    )

    logger.info(
        f"Paragraphizing {len(labels)} chunks / {len(turns)} turns in one Gemini call "
        f"({PARAGRAPHIZE_MODEL}, project={project}, location={location})…"
    )
    try:
        client = genai.Client(vertexai=True, project=project, location=location)
        response = client.models.generate_content(
            model=PARAGRAPHIZE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=config,
        )
    except Exception as e:
        logger.error(f"Paragraphize Gemini call failed: {e}")
        return None

    break_ids = _parse_break_ids(response.text or "", valid_ids)
    if break_ids is None:
        return None

    result = _apply_breaks_to_turns(turns, break_ids)
    logger.info(f"Paragraphize: {len(result)} turn(s) split into {sum(len(p) for p in result)} paragraph(s).")

    if source_audio is not None:
        from .cache import save_paragraphs

        save_paragraphs(source_audio, result)
    return result


# ---------- prompt ----------


def _speaker_letter(speaker_id: int) -> str:
    # A, B, C, ... for speakers 1, 2, 3, ...; past Z we bail to S<n>.
    n = speaker_id - 1
    if 0 <= n < 26:
        return chr(ord("A") + n)
    return f"S{speaker_id}"


def _build_prompt(labels: list[ChunkLabel], context_hint: str | None) -> str:
    # One chunk per line: "<chunk_id> <speaker_letter>: <text>". Speaker
    # letters (A, B, …) are anonymous so we don't leak PII and save tokens
    # over spelling real names hundreds of times.
    lines = [f"{lbl.chunk.idx} {_speaker_letter(lbl.final)}: {lbl.chunk.text.strip()}" for lbl in labels]
    transcript = "\n".join(lines)
    context_line = (
        f"Context: {context_hint.strip()}" if context_hint else "Context: infer the domain from the transcript itself."
    )
    return f"""You identify paragraph-break points in a speaker-labeled transcript.

{context_line}

Each line below is one transcript chunk, formatted as:
  <chunk_id> <speaker_letter>: <chunk_text>

A "turn" is a run of consecutive lines with the same speaker letter.
Most turns are already short (1-3 chunks); those stay as a single
paragraph and you do NOT emit a break for them.

Long turns — several thousand characters or 10+ chunks of the same
speaker in a row — should be broken at natural topic or breath
boundaries, roughly every 3-6 sentences or ~300-500 characters worth
of content.

A break is placed AFTER a given chunk; the next chunk of the SAME
speaker starts a new paragraph. Do not emit breaks at turn boundaries
(different speakers on the next line) — those are already separate
paragraphs.

IMPORTANT NUANCES:

- Do NOT strand a short closing sentence (under ~120 chars) on its own
  at the end of a turn. If the last chunk of a turn is a punchline,
  contrast, or one-line conclusion to what came before, keep it in the
  same paragraph as its setup. Rhetorical buttons belong with the
  thought they resolve.

- If a paragraph (the stretch between two breaks, or between a turn
  start and its first break) would exceed ~1000 characters of content,
  emit at least one additional break inside it at the strongest
  available discourse marker ("And then…", "So…", "But…", "Now…").
  No single paragraph should run past ~1000 chars if a reasonable
  break point exists.

Return a JSON object with a single key `break_after_chunk` whose value
is the sorted list of chunk_ids to break after. Empty list is valid.

TRANSCRIPT:
{transcript}
"""


# ---------- parsing ----------


def _parse_break_ids(raw: str, valid_ids: set[int]) -> list[int] | None:
    raw = (raw or "").strip()
    if not raw:
        logger.error("Paragraphize returned empty response.")
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Paragraphize JSON parse failed: {e}; raw: {raw[:500]}")
        return None

    if not isinstance(data, dict):
        logger.error(f"Paragraphize result was not a JSON object (got {type(data).__name__}).")
        return None

    raw_ids = data.get("break_after_chunk")
    if not isinstance(raw_ids, list):
        logger.error("Paragraphize result had no 'break_after_chunk' list.")
        return None

    clean: list[int] = []
    skipped = 0
    for v in raw_ids:
        if isinstance(v, int) and v in valid_ids:
            clean.append(v)
        else:
            skipped += 1
    if skipped:
        logger.info(f"Paragraphize ignored {skipped} out-of-range or malformed break id(s).")
    return sorted(set(clean))


# ---------- apply + polish ----------


def _apply_breaks_to_turns(turns: list[_Turn], break_ids: list[int]) -> list[list[str]]:
    breaks = set(break_ids)
    out: list[list[str]] = []
    for t in turns:
        paragraphs: list[str] = []
        buf: list[str] = []
        for ch in t.chunks:
            text = ch.text.strip()
            if text:
                buf.append(text)
            if ch.idx in breaks and buf:
                paragraphs.append(" ".join(buf))
                buf = []
        if buf:
            paragraphs.append(" ".join(buf))
        if not paragraphs:
            paragraphs = [""]
        out.append(_polish_paragraphs(paragraphs))
    return out


def _polish_paragraphs(paragraphs: list[str]) -> list[str]:
    """Enforce what the LLM isn't reliable at: merge a stranded trailing
    punchline into its setup, and split a paragraph that came back too
    long at the nearest discourse marker to its middle.
    """
    merged = _merge_stranded_tail(paragraphs)
    split: list[str] = []
    for p in merged:
        split.extend(_split_oversized(p))
    return split


def _merge_stranded_tail(paragraphs: list[str]) -> list[str]:
    if len(paragraphs) < 2:
        return paragraphs
    if len(paragraphs[-1]) >= _STRANDED_TAIL_MAX_CHARS:
        return paragraphs
    head = paragraphs[:-2]
    merged_tail = paragraphs[-2] + " " + paragraphs[-1]
    return [*head, merged_tail]


def _split_oversized(paragraph: str) -> list[str]:
    if len(paragraph) < _OVERSIZED_PARAGRAPH_MIN_CHARS:
        return [paragraph]
    # Find the discourse marker whose position is closest to the middle of
    # the paragraph; that gives the most balanced split. Ties break by
    # earliest-in-list (i.e. strongest marker).
    mid = len(paragraph) // 2
    best: tuple[int, int] | None = None  # (abs_distance_from_mid, split_position)
    for marker in _DISCOURSE_MARKERS:
        idx = paragraph.find(marker)
        while idx != -1:
            split_at = idx + len(". ")  # keep the period with the first half
            dist = abs(split_at - mid)
            if best is None or dist < best[0]:
                best = (dist, split_at)
            idx = paragraph.find(marker, idx + 1)
    if best is None:
        return [paragraph]
    split_pos = best[1]
    left = paragraph[:split_pos].rstrip()
    right = paragraph[split_pos:].lstrip()
    if not left or not right:
        return [paragraph]
    # Recurse once in case the paragraph was really huge; each level halves.
    return _split_oversized(left) + _split_oversized(right)
