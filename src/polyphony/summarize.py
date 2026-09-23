"""LLM summary of a reviewed transcript, stored in the sidecar and rendered atop the reviewed note."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured


class ActionItem(BaseModel):
    owner: str | None = Field(default=None, description="Who committed to it, as named in the transcript.")
    task: str


class TranscriptSummary(BaseModel):
    tldr: str = Field(description="2-3 sentences: what the conversation was about and where it landed.")
    key_points: list[str] = Field(description="The substantive points, most important first. 3-8 items.")
    decisions: list[str] = Field(description="Things explicitly agreed or decided. Empty if none.")
    action_items: list[ActionItem] = Field(description="Concrete follow-ups someone committed to. Empty if none.")


def transcript_digest(transcript_markdown: str) -> str:
    return hashlib.sha256(transcript_markdown.encode()).hexdigest()[:16]


def summarize(transcript_markdown: str, context_hint: str | None, model: str) -> dict[str, Any]:
    """Summarize and return the sidecar `summary` record (summary fields + provenance)."""
    context_line = f"Context: {context_hint.strip()}\n\n" if context_hint else ""
    prompt = f"""{context_line}Summarize this conversation for the participants' own notes.

Refer to people by the speaker names used in the transcript. Only include what was
actually said — no speculation, no advice of your own. Keep items short and specific
(names, numbers, dates when stated). Leave a list empty rather than padding it.

Transcript:
{transcript_markdown}
"""
    logger.info(f"Summarizing transcript via {model}…")
    summary = run_structured(prompt, TranscriptSummary, model)
    return {
        **summary.model_dump(),
        "model": model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_digest": transcript_digest(transcript_markdown),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["## Summary", "", summary["tldr"], ""]
    sections = [
        ("Key points", summary.get("key_points") or []),
        ("Decisions", summary.get("decisions") or []),
        # Checkbox items so they show up as tasks in Obsidian.
        (
            "Action items",
            [
                f"[ ] **{a['owner']}**: {a['task']}" if a.get("owner") else f"[ ] {a['task']}"
                for a in summary.get("action_items") or []
            ],
        ),
    ]
    for heading, items in sections:
        if items:
            lines += [f"### {heading}", "", *(f"- {item}" for item in items), ""]
    return "\n".join(lines) + "\n## Transcript\n\n"
