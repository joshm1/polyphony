"""Apply review decisions from `polyphony serve` to produce the reviewed transcript.

Deterministic by design: speaker overrides and word corrections are exact
edits on the sidecar's chunks, so the reviewed markdown is re-rendered from
data rather than rewritten by an LLM. The raw transcript is never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paragraphize import paragraphs_for
from .transcript import build_transcript
from .types import Chunk, ChunkLabel

DEFAULT_REVIEW_THRESHOLD = 70


def reviewed_path(sidecar_path: Path) -> Path:
    """`x.transcript.polyphony.json` → `x.transcript.reviewed.md`, next to the raw transcript."""
    return sidecar_path.with_name(sidecar_path.name.removesuffix(".polyphony.json") + ".reviewed.md")


def resolve_corrections(
    flags: list[dict[str, Any]], word_decisions: dict[str, dict[str, str]]
) -> dict[int, list[tuple[str, str]]]:
    """Per-chunk (original, replacement) pairs. An undecided flag takes its suggestion, matching the UI."""
    out: dict[int, list[tuple[str, str]]] = {}
    for i, flag in enumerate(flags):
        decision = word_decisions.get(str(i)) or {"kind": "suggested", "value": flag["suggested"]}
        if decision["kind"] == "original":
            continue
        out.setdefault(flag["chunk_idx"], []).append((flag["original"], decision["value"]))
    return out


def apply_corrections(text: str, corrections: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Replace each `original` span once, earliest match first, never re-matching replaced text.

    Mirrors the review UI's `renderChunkTokens` so the file matches what the
    reviewer saw. Returns the new text plus originals that weren't found.
    """
    pending = list(corrections)
    parts: list[str] = []
    remaining = text
    while remaining and pending:
        found = [(remaining.find(orig), i) for i, (orig, _) in enumerate(pending) if orig and orig in remaining]
        if not found:
            break
        at, i = min(found)
        original, replacement = pending.pop(i)
        parts.append(remaining[:at] + replacement)
        remaining = remaining[at + len(original) :]
    parts.append(remaining)
    return "".join(parts), [orig for orig, _ in pending]


def apply_review(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Render the reviewed markdown from a sidecar payload carrying a `review` block.

    Returns (markdown, skipped) where `skipped` lists corrections whose
    original span no longer appears in its chunk.
    """
    review = data.get("review") or {}
    overrides = {int(k): int(v) for k, v in (review.get("overrides") or {}).items()}
    corrections = resolve_corrections(data.get("asr_flags") or [], review.get("word_decisions") or {})

    labels: list[ChunkLabel] = []
    skipped: list[dict[str, Any]] = []
    for c in data["chunks"]:
        idx = c["idx"]
        text, missed = apply_corrections(c["text"], corrections.get(idx, []))
        skipped.extend({"chunk_idx": idx, "original": m} for m in missed)
        override = overrides.get(idx)
        labels.append(
            ChunkLabel(
                chunk=Chunk(idx=idx, start=c["start"], end=c["end"], text=text),
                audio=c.get("audio"),
                llm=c.get("llm"),
                final=c["final"],
                # A reviewer-assigned speaker is confirmed, so it shouldn't keep the turn flagged.
                confidence=100 if override is not None else c["confidence"],
                note="speaker set in review" if override is not None else c.get("note", ""),
                override=override,
            )
        )

    breaks = data.get("paragraph_breaks")
    markdown = build_transcript(
        labels,
        data.get("names") or None,
        review_threshold=data.get("review_threshold", DEFAULT_REVIEW_THRESHOLD),
        per_turn_paragraphs=paragraphs_for(labels, breaks) if breaks is not None else None,
    )
    return markdown, skipped
