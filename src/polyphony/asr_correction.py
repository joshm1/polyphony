"""Flag likely ASR errors and suggest corrections via an LLM.

Runs after diarization as an independent third pass. The LLM sees the full
transcript in order (so it has context like "this is a cooking podcast")
and emits a list of suspect spans with suggested replacements + confidence.

We never mutate the transcript automatically — corrections flow through the
review playground: user accepts/rejects per span, then copies a prompt that
applies accepted corrections to the draft markdown.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured
from .types import Chunk


@dataclass
class WordFlag:
    chunk_idx: int
    original: str
    suggested: str
    confidence: int
    reason: str = ""
    alternatives: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class _Flag(BaseModel):
    chunk_id: int
    original: str = Field(description="Exact span as it appears in the chunk, including capitalization and spacing.")
    suggested: str = Field(description="Single best correction.")
    alternatives: list[str] = Field(default_factory=list, max_length=3, description="Other plausible options.")
    confidence: int = Field(ge=0, le=100, description="How sure you are this is a TRANSCRIPTION ERROR.")
    reason: str = Field(default="", description="≤20 words pointing at the specific contextual signal.")


class _Flags(BaseModel):
    flags: list[_Flag] = Field(description="One entry per suspected error; empty if none.")


def flag_asr_errors(
    chunks: list[Chunk],
    model: str,
    context_hint: str | None = None,
    source_audio: Path | None = None,
) -> list[WordFlag]:
    """Ask the LLM to find likely Whisper transcription errors. Returns [] on failure."""
    if not chunks:
        return []

    numbered = "\n".join(f"[{c.idx}] {c.text}" for c in chunks)
    context_line = (
        f"Context: {context_hint.strip()}"
        if context_hint
        else "Context: general audio — infer domain from the transcript itself."
    )

    prompt = f"""You are auditing a Whisper ASR transcript for likely transcription errors.

{context_line}

Your job: spot words/phrases that don't fit the context — sound-alike mistakes
(e.g. "SaaS" → "sauce"), homophones, misheard technical terms, common wrong-word
substitutions. Ignore correct-but-unusual proper nouns, rare jargon, or stylistic
choices. When in doubt, don't flag.

Example flag:
  chunk_id=42, original="B2B sauce", suggested="B2B SaaS",
  alternatives=["software as a service"], confidence=92,
  reason="discussion of software business models; 'sauce' doesn't fit"

Rules:
- 'original' must be the exact span as it appears in the chunk. You may span multi-word phrases.
- 'alternatives' holds 0-3 other plausible options.
- 'confidence' is 0-100 — how sure you are this is a TRANSCRIPTION ERROR
  (not a stylistic issue). Use:
    90-100 — high confidence: context clearly rules out the transcribed word
    70-89  — confident but some ambiguity
    50-69  — plausible error, borderline
    below 50 — don't flag

Chunks:
{numbered}
"""

    # Cache keyed on audio + model + exact prompt (chunk listing + context) so iterations skip the LLM call.
    prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if source_audio is not None:
        from .cache import load_asr_flags

        cached = load_asr_flags(source_audio, model, prompt_digest)
        if cached is not None:
            return cached

    logger.info(f"ASR correction pass via {model} on {len(chunks)} chunks…")
    try:
        result = run_structured(prompt, _Flags, model)
    except Exception as e:
        logger.error(f"ASR correction LLM call failed ({model}): {e}")
        return []

    flags = _to_word_flags(result.flags, {c.idx for c in chunks})
    if source_audio is not None:
        from .cache import save_asr_flags

        save_asr_flags(source_audio, model, prompt_digest, flags)
    return flags


def _to_word_flags(entries: list[_Flag], valid_ids: set[int]) -> list[WordFlag]:
    flags = [
        WordFlag(
            chunk_idx=f.chunk_id,
            original=f.original,
            suggested=f.suggested,
            confidence=f.confidence,
            reason=f.reason,
            alternatives=f.alternatives,
        )
        for f in entries
        if f.chunk_id in valid_ids
    ]
    logger.info(f"ASR correction flagged {len(flags)} span(s).")
    return flags
