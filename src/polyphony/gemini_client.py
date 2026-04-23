"""Build a google-genai Client, picking API-key mode over Vertex when possible.

If `$GEMINI_API_KEY` or `$GOOGLE_API_KEY` is set we talk to the Gemini
Generative Language API directly — no GCP project, no ADC, no Vertex. Otherwise
we fall back to Vertex AI, which needs a project with the Vertex AI API
enabled and Application Default Credentials configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def gemini_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


@dataclass(frozen=True)
class GeminiClientInfo:
    client: Any
    mode: str  # "api_key" or "vertex"
    project: str | None  # None in api_key mode
    location: str | None  # None in api_key mode

    def describe(self) -> str:
        if self.mode == "api_key":
            return "Gemini direct API"
        return f"Vertex (project={self.project}, location={self.location})"


def make_gemini_client(
    project: str | None,
    location: str | None,
    default_location: str,
) -> GeminiClientInfo:
    """Return a configured `google.genai.Client` plus which path it's using.

    Raises `RuntimeError` with actionable guidance if neither mode can be set up.
    """
    from google import genai

    api_key = gemini_api_key()
    if api_key:
        return GeminiClientInfo(
            client=genai.Client(api_key=api_key),
            mode="api_key",
            project=None,
            location=None,
        )

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "No Gemini credentials found. Either export $GEMINI_API_KEY "
            "(or $GOOGLE_API_KEY) to use the direct Gemini API, or pass "
            "--project / export $GOOGLE_CLOUD_PROJECT (plus `gcloud auth "
            "application-default login`) to use Vertex AI."
        )
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or default_location
    return GeminiClientInfo(
        client=genai.Client(vertexai=True, project=project, location=location),
        mode="vertex",
        project=project,
        location=location,
    )
