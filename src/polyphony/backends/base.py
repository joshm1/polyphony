"""Backend adapter protocol — swap the transcription+diarization strategy.

A backend owns the *path from audio to labeled chunks*. Everything downstream
(transcript markdown, review playground, optional ASR correction pass) is
backend-agnostic: it consumes `list[ChunkLabel]`.

Backends behind this interface today:

  - local       — Whisper + pyannote + LLM-diarize + ensemble reconciler
  - gemini      — one Gemini 2.5 call on Vertex AI that does both at once
  - assemblyai  — hosted ASR + voice diarization, then LLM-diarize + reconciler

Adding another (e.g. Deepgram) is: subclass Backend, implement
`preflight` and `run`, register in `backends/__init__.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..llm import DEFAULT_LLM_MODEL
from ..types import ChunkLabel


@dataclass(frozen=True)
class BackendConfig:
    """Shared knobs every backend accepts (may ignore ones it doesn't use)."""

    names: list[str] | None = None
    context_hint: str | None = None
    device: str = "auto"  # torch device, local-backend only
    llm_model: str = DEFAULT_LLM_MODEL  # pydantic-ai model string for text-side diarization + reconciler
    # Hosted-backend knobs (ignored by local backend)
    project: str | None = None
    location: str | None = None
    model: str | None = None  # Gemini model id, or AssemblyAI speech model
    gcs_bucket: str | None = None


class BackendUnavailable(RuntimeError):
    """Raised by preflight when a backend cannot be used in the current env."""


class Backend(ABC):
    """Adapter interface — one end-to-end strategy for getting ChunkLabels."""

    name: str = "base"

    @abstractmethod
    def preflight(self, cfg: BackendConfig) -> None:
        """Fail fast if this backend isn't usable. Raises BackendUnavailable."""

    @abstractmethod
    def run(self, audio_path: Path, cfg: BackendConfig) -> list[ChunkLabel]:
        """Produce the full labeled chunk list for this audio."""
