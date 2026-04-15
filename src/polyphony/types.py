"""Shared dataclasses that flow through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A Whisper transcription chunk with its audio time range."""

    idx: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class PyannoteSegment:
    """Raw pyannote speaker turn (audio time range + opaque speaker tag)."""

    start: float
    end: float
    speaker: str  # e.g. "SPEAKER_00"


@dataclass
class ChunkLabel:
    """Final per-chunk speaker decision after ensemble reconciliation."""

    chunk: Chunk
    # Each candidate holds a 1-indexed speaker id or None when the backend
    # had nothing to say for this chunk (e.g. a silent window in pyannote).
    pyannote: int | None
    claude: int | None
    # Reconciled speaker id (always 1-indexed, never None).
    final: int
    # 0-100. Higher = more trustworthy. 100 = both backends agreed from
    # the start; lower values indicate reconciler uncertainty.
    confidence: int
    # Short human-readable rationale from the reconciler (or "agreed" / "only X available").
    note: str = ""
    # User override applied post-hoc via the review playground.
    override: int | None = None

    @property
    def effective_speaker(self) -> int:
        return self.override if self.override is not None else self.final
