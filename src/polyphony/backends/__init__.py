"""Backend registry — import adapters and map by name."""

from __future__ import annotations

import os

from ..gemini_client import gemini_api_key
from .assemblyai import API_KEY_ENV as ASSEMBLYAI_API_KEY_ENV
from .assemblyai import AssemblyAIBackend
from .base import Backend, BackendConfig, BackendUnavailable
from .gemini import GeminiBackend
from .local import LocalEnsembleBackend, missing_local_dependencies

BACKENDS: dict[str, type[Backend]] = {
    "local": LocalEnsembleBackend,
    "gemini": GeminiBackend,
    "assemblyai": AssemblyAIBackend,
}


def resolve_backend(choice: str) -> Backend:
    """Instantiate a backend by name. 'auto' picks the fastest one with credentials configured."""
    if choice == "auto":
        # AssemblyAI first: it's the only fast path that keeps the audio + text
        # ensemble. Then Gemini when it has credentials (a GCP project alone only
        # counts if local can't run). Otherwise local, whose preflight explains
        # what's missing and names the hosted alternatives.
        if os.environ.get(ASSEMBLYAI_API_KEY_ENV):
            return AssemblyAIBackend()
        if gemini_api_key() or (os.environ.get("GOOGLE_CLOUD_PROJECT") and missing_local_dependencies()):
            return GeminiBackend()
        return LocalEnsembleBackend()

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
