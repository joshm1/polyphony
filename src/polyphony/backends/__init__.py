"""Backend registry — import adapters and map by name."""

from __future__ import annotations

import shutil

from ..diarize import get_hf_token
from .base import Backend, BackendConfig, BackendUnavailable
from .gemini import GeminiBackend
from .local import LocalEnsembleBackend

BACKENDS: dict[str, type[Backend]] = {
    "local": LocalEnsembleBackend,
    "gemini": GeminiBackend,
}


def resolve_backend(choice: str) -> Backend:
    """Instantiate a backend by name. 'auto' picks the fastest usable one."""
    if choice == "auto":
        # Gemini wins when ADC + google-genai are ready; else fall back to local
        # (which has its own preflight to report license/token issues).
        try:
            import google.genai  # noqa: F401

            return GeminiBackend()
        except ImportError:
            pass
        if get_hf_token() and shutil.which("claude"):
            return LocalEnsembleBackend()
        # Neither fully ready — still return gemini (clearer error than a half-local path).
        return GeminiBackend()

    cls = BACKENDS.get(choice)
    if cls is None:
        raise ValueError(f"Unknown backend {choice!r}. Options: {sorted(BACKENDS)} or 'auto'.")
    return cls()


__all__ = [
    "Backend",
    "BackendConfig",
    "BackendUnavailable",
    "GeminiBackend",
    "LocalEnsembleBackend",
    "BACKENDS",
    "resolve_backend",
]
