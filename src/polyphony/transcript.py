"""Render ChunkLabels into the final markdown transcript.

Assembly only: speaker-labeled chunks → `Name: …` markdown with ⚠️
markers on low-confidence turns. The Gemini-based paragraph-splitting
lives in `paragraphize.py`; call that first and pass the result in via
`per_turn_paragraphs` to get per-turn paragraph breaks. Without it each
turn renders as a single space-joined paragraph — the legacy
wall-of-text shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from .types import Chunk, ChunkLabel


def display_name(speaker_id: int, names: list[str] | None) -> str:
    if names and 1 <= speaker_id <= len(names):
        return names[speaker_id - 1]
    return f"Speaker {speaker_id}"


@dataclass
class _Turn:
    speaker: int
    chunks: list[Chunk]
    min_conf: int
    flagged: bool
    notes: list[str]


def _group_into_turns(labels: list[ChunkLabel], review_threshold: int) -> list[_Turn]:
    turns: list[_Turn] = []
    cur: _Turn | None = None
    for lbl in labels:
        spk = lbl.effective_speaker
        if cur is None or cur.speaker != spk:
            cur = _Turn(speaker=spk, chunks=[], min_conf=100, flagged=False, notes=[])
            turns.append(cur)
        cur.chunks.append(lbl.chunk)
        cur.min_conf = min(cur.min_conf, lbl.confidence)
        if lbl.confidence < review_threshold:
            cur.flagged = True
        if lbl.note:
            cur.notes.append(f"chunk {lbl.chunk.idx}: {lbl.note}")
    return turns


def _join_turn_text(chunks: list[Chunk]) -> str:
    return " ".join(ch.text.strip() for ch in chunks if ch.text.strip())


def build_transcript(
    labels: list[ChunkLabel],
    names: list[str] | None,
    review_threshold: int = 70,
    per_turn_paragraphs: list[list[str]] | None = None,
) -> str:
    """Fold labeled chunks into speaker turns and render as markdown.

    If `per_turn_paragraphs` is provided (same length as the turn list),
    we use it verbatim. Otherwise each turn collapses to a single
    space-joined paragraph — the legacy wall-of-text shape. Call
    `paragraphize.paragraphize_via_gemini` from the pipeline first for
    the good shape.
    """
    turns = _group_into_turns(labels, review_threshold=review_threshold)

    if per_turn_paragraphs is not None and len(per_turn_paragraphs) != len(turns):
        logger.warning(
            f"per_turn_paragraphs length mismatch ({len(per_turn_paragraphs)} vs {len(turns)}); "
            "using single-paragraph-per-turn fallback."
        )
        per_turn_paragraphs = None

    lines: list[str] = []
    for i, t in enumerate(turns):
        name = display_name(t.speaker, names)
        if per_turn_paragraphs is not None:
            paragraphs = per_turn_paragraphs[i]
        else:
            text = _join_turn_text(t.chunks)
            paragraphs = [text] if text else []
        prefix = "⚠️ " if t.flagged else ""
        if t.flagged:
            summary = "; ".join(n for n in t.notes if n)[:240] or "low-confidence turn"
            lines.append(f"<!-- review needed (min confidence {t.min_conf}): {summary} -->")
        if paragraphs:
            head, *tail = paragraphs
            lines.append(f"{prefix}{name}: {head}")
            lines.extend(tail)

    return "\n\n".join(lines) + "\n"
