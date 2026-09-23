"""Ensemble reconciliation of two diarizations, with per-chunk 0-100 confidence.

Given audio-diarizer labels (pyannote / AssemblyAI) and LLM labels per
chunk, produce a final label per chunk plus a 0-100 confidence score. For
chunks where the two already agree, confidence is 100 and we skip the
reconciler. For the rest, we ask the LLM to choose, giving it full surrounding context + both
candidate labels, and have it emit a confidence score based on how clear
the evidence was.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured
from .types import Chunk, ChunkLabel


def reconcile(
    chunks: list[Chunk],
    audio: list[int | None],
    llm: list[int | None],
    names: list[str] | None,
    model: str,
) -> list[ChunkLabel]:
    """Merge two candidate diarizations into one with per-chunk confidence."""
    assert len(chunks) == len(audio) == len(llm), "label arrays must align with chunks"

    # Trivial cases resolved locally; reconciler only sees real disagreements.
    labels: list[ChunkLabel | None] = [None] * len(chunks)
    disputed: list[int] = []  # chunk indexes needing the reconciler

    for i, ch in enumerate(chunks):
        p, c = audio[i], llm[i]
        if p is not None and c is not None and p == c:
            labels[i] = ChunkLabel(ch, p, c, final=p, confidence=100, note="both backends agreed")
        elif p is None and c is None:
            # Nobody had a view — default to speaker 1 with low confidence; reconciler can adjust.
            disputed.append(i)
        elif p is None:
            labels[i] = ChunkLabel(ch, None, c, final=c, confidence=70, note="audio diarizer silent; used LLM")
        elif c is None:
            labels[i] = ChunkLabel(ch, p, None, final=p, confidence=70, note="LLM silent; used audio diarizer")
        else:
            disputed.append(i)

    if not disputed:
        logger.info("No disagreements; skipping reconciler.")
        return [lbl for lbl in labels if lbl is not None]  # type: ignore[misc]

    logger.info(f"Reconciler needed for {len(disputed)}/{len(chunks)} chunks.")
    decisions = _run_reconciler(chunks, audio, llm, names, disputed, model)

    for i in disputed:
        ch = chunks[i]
        p, c = audio[i], llm[i]
        dec = decisions.get(i)
        if dec is None:
            # Fallback: prefer the audio diarizer when both present, else whichever isn't None, else 1.
            final = p if p is not None else c if c is not None else 1
            labels[i] = ChunkLabel(
                ch,
                p,
                c,
                final=final,
                confidence=25,
                note="reconciler omitted this chunk; fell back heuristically",
            )
        else:
            labels[i] = ChunkLabel(
                ch,
                p,
                c,
                final=dec.speaker,
                confidence=dec.confidence,
                note=dec.reason or "",
            )

    return [lbl for lbl in labels if lbl is not None]  # type: ignore[misc]


class _Decision(BaseModel):
    id: int = Field(description="DISPUTED chunk id, exactly as shown in brackets.")
    speaker: int = Field(ge=1, description="1-indexed speaker id.")
    confidence: int = Field(ge=0, le=100)
    reason: str = Field(default="", description="≤15 words pointing at the specific signal.")


class _Decisions(BaseModel):
    decisions: list[_Decision] = Field(description="One entry per DISPUTED chunk id.")


def _run_reconciler(
    chunks: list[Chunk],
    audio: list[int | None],
    llm: list[int | None],
    names: list[str] | None,
    disputed: list[int],
    model: str,
) -> dict[int, _Decision]:
    if names:
        speaker_block = "\n".join(f"  Speaker {i} = {name}" for i, name in enumerate(names, start=1))
    else:
        speaker_block = "  Speakers are unnamed — use 1-indexed ids."

    # Present the whole transcript so the reconciler has full context —
    # surrounding agreed turns are the strongest signal for the disputes.
    lines: list[str] = []
    disputed_set = set(disputed)
    for i, ch in enumerate(chunks):
        p, c = audio[i], llm[i]
        if i in disputed_set:
            marker = f"DISPUTED audio={p} llm={c}"
        else:
            agreed = p if p == c else p or c
            marker = f"agreed={agreed}" if agreed is not None else "agreed=?"
        lines.append(f"[{i} {marker}] {ch.text}")
    full_listing = "\n".join(lines)

    prompt = f"""You are the tie-breaker in an ensemble diarization of a Whisper ASR transcript.

Two backends labeled each chunk:
  - audio = voice-based diarization (good at voice changes)
  - llm   = text-based reasoning (good at role cues and semantics)

They agreed on most chunks. Your job: for each DISPUTED chunk, pick the correct
speaker id using surrounding context, and rate your own confidence 0-100.

Speakers:
{speaker_block}

Confidence rubric (be honest — low confidence triggers human review):
  100 — only used for agreed chunks (not your job)
  80-99 — clear from context (role cues, names, explicit Q/A structure)
  60-79 — leaning one way but some ambiguity
  40-59 — genuinely unclear; coin flip with weak evidence
  0-39  — no real evidence; guessing

Full transcript (ALL chunks, with agreed labels as context and DISPUTED chunks to resolve):
{full_listing}
"""

    logger.info(f"Running reconciler via {model} on {len(disputed)} disputes…")
    try:
        result = run_structured(prompt, _Decisions, model)
    except Exception as e:
        logger.error(f"Reconciler LLM call failed ({model}): {e}")
        return {}
    return _decisions_by_chunk(result.decisions, disputed_set)


def _decisions_by_chunk(decisions: list[_Decision], disputed: set[int]) -> dict[int, _Decision]:
    # Answers for non-disputed chunks would silently override agreed labels' notes; drop them.
    return {d.id: d for d in decisions if d.id in disputed}
