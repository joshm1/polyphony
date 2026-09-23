"""Text-only LLM calls (diarization, reconciler, paragraphize, ASR correction) via pydantic-ai.

The model is any pydantic-ai model string. The default runs on a ChatGPT/Codex
subscription through the `openai-codex` provider, which reads the credentials
`codex login` writes to `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`).
Override with `--llm-model` / `$POLYPHONY_LLM_MODEL`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

DEFAULT_LLM_MODEL = os.environ.get("POLYPHONY_LLM_MODEL") or "openai-codex:gpt-6-sol"


class LLMUnavailable(RuntimeError):
    """The configured model can't be constructed (missing credentials, unknown provider, missing extra)."""


def check_llm(model: str) -> None:
    """Fail fast before a long pipeline run if `model` can't be built. Makes no API call."""
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
