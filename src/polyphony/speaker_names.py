"""Map candidate names onto anonymous speaker ids using the transcript text."""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured
from .types import ChunkLabel


class _Assignment(BaseModel):
    speaker: int = Field(ge=1, description="Speaker id from the transcript (the number after S).")
    name: str = Field(description="One of the candidate names, exactly as given.")
    reason: str = Field(default="", description="≤15 words pointing at the evidence.")


class _Assignments(BaseModel):
    assignments: list[_Assignment] = Field(description="Only speakers you can identify; omit the rest.")


def identify_speakers(
    labels: list[ChunkLabel],
    candidates: list[str],
    fixed: dict[int, str],
    model: str,
    context_hint: str | None = None,
) -> dict[int, str]:
    """Assign `candidates` to speakers not already in `fixed`. Returns only new assignments; {} on failure."""
    speakers = sorted({lbl.effective_speaker for lbl in labels})
    unnamed = [s for s in speakers if s not in fixed]
    remaining = [c for c in candidates if c not in fixed.values()]
    if not unnamed or not remaining:
        return {}

    known = "\n".join(f"  S{s} = {name}" for s, name in sorted(fixed.items())) or "  (none)"
    transcript = "\n".join(f"S{lbl.effective_speaker}: {lbl.chunk.text.strip()}" for lbl in labels)
    context_line = f"Context: {context_hint.strip()}\n\n" if context_hint else ""
    prompt = f"""{context_line}Identify which speaker in this transcript is which person.

Speakers already identified:
{known}

Candidate names for the remaining speakers ({", ".join(f"S{s}" for s in unnamed)}):
{chr(10).join(f"  - {c}" for c in remaining)}

Use evidence in the text: people addressing each other by name, self-introductions,
roles implied by the conversation. Assign each candidate to at most one speaker and
omit speakers you can't identify — a wrong name is worse than none.

Transcript:
{transcript}
"""
    logger.info(f"Identifying {len(unnamed)} speaker(s) from {len(remaining)} candidate name(s) via {model}…")
    try:
        result = run_structured(prompt, _Assignments, model)
    except Exception as e:
        logger.error(f"Speaker identification failed ({model}): {e}")
        return {}

    out: dict[int, str] = {}
    for a in result.assignments:
        if a.speaker in unnamed and a.name in remaining and a.speaker not in out and a.name not in out.values():
            out[a.speaker] = a.name
            logger.info(f"  S{a.speaker} = {a.name} ({a.reason})")
    return out
