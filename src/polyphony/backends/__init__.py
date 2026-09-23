"""Backend registry — import adapters and map by name."""

from __future__ import annotations

import os

from ..diarize import get_hf_token
from .assemblyai import API_KEY_ENV as ASSEMBLYAI_API_KEY_ENV
from .assemblyai import AssemblyAIBackend
from .base import Backend, BackendConfig, BackendUnavailable
from .gemini import GeminiBackend
from .local import LocalEnsembleBackend

BACKENDS: dict[str, type[Backend]] = {
    "local": LocalEnsembleBackend,
    "gemini": GeminiBackend,
    "assemblyai": AssemblyAIBackend,
}


def resolve_backend(choice: str) -> Backend:
    """Instantiate a backend by name. 'auto' picks the fastest usable one."""
    if choice == "auto":
        # AssemblyAI wins when its key is set: it's the only fast path that
        # keeps the audio + text ensemble. Then Gemini when google-genai imports;
        # else local (which has its own preflight to report license/token issues).
        if os.environ.get(ASSEMBLYAI_API_KEY_ENV):
            return AssemblyAIBackend()
        try:
            import google.genai  # noqa: F401

            return GeminiBackend()
        except ImportError:
            pass
        if get_hf_token():
            return LocalEnsembleBackend()
        # Neither fully ready — still return gemini (clearer error than a half-local path).
        return GeminiBackend()

    cls = BACKENDS.get(choice)
    if cls is None:
        raise ValueError(f"Unknown backend {choice!r}. Options: {sorted(BACKENDS)} or 'auto'.")
    return cls()


__all__ = [
    "AssemblyAIBackend",
    "Backend",
    "BackendConfig",
    "BackendUnavailable",
    "GeminiBackend",
    "LocalEnsembleBackend",
    "BACKENDS",
    "resolve_backend",
]
