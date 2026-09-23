"""Ask the LLM to pick paragraph-break points inside speaker turns.

Design:
- One LLM call per transcribe. We ship a chunk-per-line input with
  anonymous speaker letters (A/B/…) and ask for the chunk IDs to break
  after. Structured output guarantees the shape, so there's no parsing
  drift to handle.
- Round-trip safe by construction: the LLM never re-emits text, it only
  picks break points. We slice chunks into paragraphs on our side.
- Two post-process passes clean up what the LLM is unreliable about:
  short trailing fragments get merged back into their setup (punchlines
  belong with the thought they resolve), and oversized paragraphs get
  split at the nearest discourse marker.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured
from .transcript import _group_into_turns, _Turn
from .types import ChunkLabel

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


class _Breaks(BaseModel):
    break_after_chunk: list[int] = Field(description="Sorted chunk ids to break after. Empty list is valid.")


def paragraphize(
    labels: list[ChunkLabel],
    model: str,
    context_hint: str | None = None,
    source_audio: Path | None = None,
) -> list[int] | None:
    """Ask the LLM to pick paragraph-break points.

    Returns the sorted chunk ids to break after — stored in the sidecar so a
    review can re-render after speaker overrides without another LLM call.
    Turn them into paragraphs with `paragraphs_for`. Returns None on any
    failure; the markdown then falls back to one-paragraph-per-turn.
    """
    if not labels:
        return []

    prompt = _build_prompt(labels, context_hint)
    prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if source_audio is not None:
        from .cache import load_paragraph_breaks

        cached = load_paragraph_breaks(source_audio, model, prompt_digest)
        if cached is not None:
            return cached

    logger.info(f"Paragraphizing {len(labels)} chunks via {model}…")
    try:
        result = run_structured(prompt, _Breaks, model)
    except Exception as e:
        logger.error(f"Paragraphize LLM call failed ({model}): {e}")
        return None

    break_ids = _clean_break_ids(result.break_after_chunk, {lbl.chunk.idx for lbl in labels})
    logger.info(f"Paragraphize: {len(break_ids)} break(s) chosen.")
    if source_audio is not None:
        from .cache import save_paragraph_breaks

        save_paragraph_breaks(source_audio, model, prompt_digest, break_ids)
    return break_ids


def paragraphs_for(labels: list[ChunkLabel], break_ids: list[int]) -> list[list[str]]:
    """Per-turn paragraphs (aligned with `_group_into_turns`) for `build_transcript`."""
    return _apply_breaks_to_turns(_group_into_turns(labels, review_threshold=100), break_ids)


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

Return the sorted list of chunk_ids to break after. Empty list is valid.

TRANSCRIPT:
{transcript}
"""


# ---------- validation ----------


def _clean_break_ids(raw_ids: list[int], valid_ids: set[int]) -> list[int]:
    clean = sorted({i for i in raw_ids if i in valid_ids})
    if skipped := len(raw_ids) - sum(1 for i in raw_ids if i in valid_ids):
        logger.info(f"Paragraphize ignored {skipped} out-of-range break id(s).")
    return clean


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
