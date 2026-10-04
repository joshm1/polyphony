"""Ensemble reconciliation of two diarizations, with per-chunk 0-100 confidence.

Given audio-diarizer labels (pyannote / AssemblyAI) and LLM labels per
chunk, produce a final label per chunk plus a 0-100 confidence score:

  100     both diarizers agreed
  ≤70     only one diarizer had a view; scaled down by how much of the chunk's
          speech its winning speaker owns, when the audio diarizer reports that
  reconciler-rated
          the two disagreed and an LLM tie-breaker chose, seeing the full
          transcript with both candidate labels
  25      no usable decision; fell back heuristically

Without an LLM (`model=None`) the text side is silent, so every chunk lands
in the single-signal band and nothing reaches the tie-breaker.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured
from .types import Chunk, ChunkLabel

AGREED_CONFIDENCE = 100
SINGLE_SIGNAL_CONFIDENCE = 70
FALLBACK_CONFIDENCE = 25


def reconcile(
    chunks: list[Chunk],
    audio: list[int | None],
    llm: list[int | None],
    names: list[str] | None,
    model: str | None,
    audio_purity: list[float | None] | None = None,
) -> list[ChunkLabel]:
    """Merge two candidate diarizations into one with per-chunk confidence.

    `audio_purity` is the share of each chunk's speech owned by the audio
    diarizer's winning speaker (see `diarize.pyannote_chunk_purity`).
    """
    if not len(chunks) == len(audio) == len(llm):
        raise ValueError("label arrays must align with chunks")
    if audio_purity is not None and len(audio_purity) != len(chunks):
        raise ValueError("audio_purity must align with chunks")

    labels: dict[int, ChunkLabel] = {}
    disputed: list[int] = []  # chunk indexes needing the reconciler

    for i, ch in enumerate(chunks):
        a, t = audio[i], llm[i]
        if a is not None and t is not None and a == t:
            labels[i] = ChunkLabel(ch, a, t, final=a, confidence=AGREED_CONFIDENCE, note="both backends agreed")
        elif a is not None and t is None:
            purity = audio_purity[i] if audio_purity is not None else None
            note = "LLM silent; used audio diarizer" if model is not None else "audio diarizer only (no LLM)"
            if purity is not None and purity < 1:
                note += f"; {round(purity * 100)}% of chunk speech is this speaker"
            labels[i] = ChunkLabel(ch, a, None, final=a, confidence=single_signal_confidence(purity), note=note)
        elif a is None and t is not None:
            labels[i] = ChunkLabel(
                ch, None, t, final=t, confidence=SINGLE_SIGNAL_CONFIDENCE, note="audio diarizer silent; used LLM"
            )
        else:
            # Real disagreement, or neither side had a view.
            disputed.append(i)

    decisions: dict[int, _Decision] = {}
    if disputed and model is not None:
        logger.info(f"Reconciler needed for {len(disputed)}/{len(chunks)} chunks.")
        decisions = _run_reconciler(chunks, audio, llm, names, disputed, model)
    elif not disputed:
        logger.info("No disagreements; skipping reconciler.")

    for i in disputed:
        ch = chunks[i]
        a, t = audio[i], llm[i]
        dec = decisions.get(i)
        if dec is None:
            # Prefer the audio diarizer when both present, else whichever isn't None, else speaker 1.
            final = a if a is not None else t if t is not None else 1
            note = (
                "reconciler omitted this chunk; fell back heuristically"
                if model is not None
                else "no diarizer labeled this chunk; defaulted to speaker 1"
            )
            labels[i] = ChunkLabel(ch, a, t, final=final, confidence=FALLBACK_CONFIDENCE, note=note)
        else:
            labels[i] = ChunkLabel(ch, a, t, final=dec.speaker, confidence=dec.confidence, note=dec.reason)

    out = [labels[i] for i in range(len(chunks))]
    if audio_purity is not None:
        for lbl, purity in zip(out, audio_purity, strict=True):
            lbl.audio_purity = purity
    return out


def single_signal_confidence(purity: float | None) -> int:
    """Confidence when only the audio diarizer spoke: capped below agreement, lower for mixed-speaker chunks."""
    if purity is None:
        return SINGLE_SIGNAL_CONFIDENCE
    return round(SINGLE_SIGNAL_CONFIDENCE * max(0.0, min(1.0, purity)))


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
