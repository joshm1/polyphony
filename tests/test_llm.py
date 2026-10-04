"""LLM model resolution: API-key auto-detection, bare providers, and the no-LLM fallback."""

from __future__ import annotations

import pytest

from polyphony.llm import MODEL_ENV, PROVIDER_DEFAULT_MODELS, LLMUnavailable, check_llm, resolve_llm_model

_KEY_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", MODEL_ENV)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_auto_without_any_key_means_no_llm():
    assert resolve_llm_model() is None
    assert resolve_llm_model("auto") is None


@pytest.mark.parametrize(
    ("env_var", "provider"),
    [("OPENAI_API_KEY", "openai"), ("ANTHROPIC_API_KEY", "anthropic"), ("GEMINI_API_KEY", "google")],
)
def test_auto_picks_the_provider_whose_key_is_set(monkeypatch: pytest.MonkeyPatch, env_var: str, provider: str):
    monkeypatch.setenv(env_var, "k")
    assert resolve_llm_model("auto") == PROVIDER_DEFAULT_MODELS[provider]


def test_auto_prefers_openai_when_several_keys_are_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert resolve_llm_model() == PROVIDER_DEFAULT_MODELS["openai"]


def test_explicit_choice_beats_env_and_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv(MODEL_ENV, "anthropic")
    assert resolve_llm_model() == PROVIDER_DEFAULT_MODELS["anthropic"]
    assert resolve_llm_model("none") is None
    assert resolve_llm_model("ollama:qwen3") == "ollama:qwen3"
    assert resolve_llm_model("openai-codex") == "openai-codex:gpt-6-sol"


def test_env_can_disable_the_llm(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv(MODEL_ENV, "none")
    assert resolve_llm_model() is None


def test_check_llm_accepts_no_llm_and_rejects_unknown_providers():
    check_llm(None)
    with pytest.raises(LLMUnavailable):
        check_llm("no-such-provider:model")
