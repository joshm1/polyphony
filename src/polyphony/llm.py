"""Text-only LLM calls (diarization, reconciler, paragraphize, ASR correction) via pydantic-ai.

The model is any pydantic-ai model string (`provider:model`), a bare provider
name (`anthropic`, which expands to that provider's default model), `auto`, or
`none`. `auto` uses the first provider whose API key is set; with no key it
resolves to no LLM, and every text-side pass is skipped: speakers come from the
audio diarizer alone and confidence reflects that single signal.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

MODEL_ENV = "POLYPHONY_LLM_MODEL"
AUTO = "auto"
NO_LLM = "none"

# `auto` picks the first provider whose key is set, in this order.
_API_KEY_PROVIDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("openai", ("OPENAI_API_KEY",)),
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("google", ("GOOGLE_API_KEY", "GEMINI_API_KEY")),
)

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "openai:gpt-6-sol",
    "anthropic": "anthropic:claude-sonnet-5",
    "google": "google:gemini-3.8-flash",
    # ChatGPT subscription via the credentials `codex login` writes to ~/.codex/auth.json.
    "openai-codex": "openai-codex:gpt-6-sol",
}


class LLMUnavailable(RuntimeError):
    """The configured model can't be constructed (missing credentials, unknown provider, missing extra)."""


def resolve_llm_model(choice: str | None = None) -> str | None:
    """Turn a `--llm-model` value into a pydantic-ai model string, or None for no LLM.

    An unset choice falls back to `$POLYPHONY_LLM_MODEL`, then to `auto`.
    """
    choice = (choice or os.environ.get(MODEL_ENV) or AUTO).strip()
    if choice == NO_LLM:
        return None
    if choice == AUTO:
        for provider, env_vars in _API_KEY_PROVIDERS:
            if any(os.environ.get(var) for var in env_vars):
                return PROVIDER_DEFAULT_MODELS[provider]
        return None
    return PROVIDER_DEFAULT_MODELS.get(choice, choice)


def check_llm(model: str | None) -> None:
    """Fail fast before a long pipeline run if `model` can't be built. Makes no API call."""
    if model is None:
        return
    try:
        infer_model(model)
    except (UserError, ImportError, ValueError) as e:
        raise LLMUnavailable(f"LLM model {model!r} unavailable: {e}") from e


def run_structured[T: BaseModel](
    prompt: str, output_type: type[T], model: str, tools: Sequence[Callable[..., str]] = ()
) -> T:
    """Prompt → validated `output_type`, optionally letting the model call plain-function `tools`.

    pydantic-ai retries on schema violations.
    """
    try:
        agent = Agent(model, output_type=output_type, tools=tools)
    except (UserError, ImportError, ValueError) as e:
        raise LLMUnavailable(f"LLM model {model!r} unavailable: {e}") from e
    return agent.run_sync(prompt).output
