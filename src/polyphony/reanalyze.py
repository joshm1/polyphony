"""Rerun the text-side LLM passes on an existing sidecar with new names / context hint.

Audio-side work (ASR, voice diarization) is reused from the sidecar, so this
costs only LLM calls. Review state carries over: speaker overrides are keyed
by chunk id (unchanged), reviewer-added corrections are kept, and decisions
on LLM-suggested corrections follow the same (chunk, span) when it's
suggested again.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from .asr_correction import flag_asr_errors
from .diarize import diarize_llm
from .paragraphize import paragraphize
from .playground import flag_dict
from .reconcile import reconcile
from .review import labels_from_payload, overrides_of
from .speaker_names import identify_speakers
from .types import ChunkLabel


def resolve_names(
    labels: list[ChunkLabel],
    names: list[str],
    candidates: list[str],
    model: str,
    context_hint: str | None,
) -> list[str]:
    """Explicit per-speaker names win; the LLM places `candidates` on the rest."""
    fixed = {i: n.strip() for i, n in enumerate(names, start=1) if n.strip()}
    guessed = identify_speakers(labels, [c.strip() for c in candidates if c.strip()], fixed, model, context_hint)
    by_id = fixed | guessed
    n_speakers = max([*by_id, *(lbl.effective_speaker for lbl in labels)], default=0)
    return [by_id.get(i, "") for i in range(1, n_speakers + 1)]


def merge_flags(
    old_flags: list[dict[str, Any]],
    old_decisions: dict[str, dict[str, str]],
    new_flags: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """New LLM flags + reviewer-added ones, with explicit decisions re-keyed to the new indexes.

    A "suggested" decision is the default, so it isn't carried — the new
    suggestion applies instead.
    """
    carried = {
        (f["chunk_idx"], f["original"]): old_decisions[str(i)]
        for i, f in enumerate(old_flags)
        if str(i) in old_decisions and old_decisions[str(i)]["kind"] != "suggested"
    }
    merged = new_flags + [f for f in old_flags if f.get("userAdded")]
    decisions = {
        str(i): carried[(f["chunk_idx"], f["original"])]
        for i, f in enumerate(merged)
        if (f["chunk_idx"], f["original"]) in carried
    }
    return merged, decisions


def reanalyze(
    data: dict[str, Any],
    audio_path: Path,
    names: list[str],
    candidate_names: list[str],
    context_hint: str | None,
) -> dict[str, Any]:
    """Return an updated sidecar payload. `data` must already carry the reviewer's current state."""
    model = data["llm_model"]
    overrides = overrides_of(data)
    base = labels_from_payload(data)
    reviewed = [replace(lbl, override=overrides.get(lbl.chunk.idx)) for lbl in base]
    context_hint = (context_hint or "").strip() or None

    resolved = resolve_names(reviewed, names, candidate_names, model, context_hint)
    name_list = resolved if any(resolved) else None

    # Gemini labels speakers from audio + text in one call, so there's no
    # separate text signal to redo; everything else has an audio signal to cross-check.
    if any(lbl.audio is not None for lbl in base):
        chunks = [lbl.chunk for lbl in base]
        audio = [lbl.audio for lbl in base]
        n_speakers = len({a for a in audio if a is not None})
        llm = diarize_llm(chunks, name_list, model, expected_speakers=max(2, n_speakers, len(resolved)))
        base = reconcile(chunks, audio, llm, name_list, model=model)
    else:
        logger.info("No audio-diarizer labels in sidecar; keeping speaker labels, redoing text passes only.")

    reviewed = [replace(lbl, override=overrides.get(lbl.chunk.idx)) for lbl in base]
    breaks = paragraphize(reviewed, model, context_hint=context_hint, source_audio=audio_path)
    new_flags = flag_asr_errors([lbl.chunk for lbl in base], model, context_hint=context_hint, source_audio=audio_path)
    review = data.get("review") or {}
    flags, decisions = merge_flags(
        data.get("asr_flags") or [], review.get("word_decisions") or {}, [flag_dict(f) for f in new_flags]
    )

    updated = dict(data)
    updated["names"] = resolved
    updated["context_hint"] = context_hint
    updated["paragraph_breaks"] = breaks if breaks is not None else data.get("paragraph_breaks")
    updated["asr_flags"] = flags
    updated["review"] = {"overrides": review.get("overrides") or {}, "word_decisions": decisions}
    updated["chunks"] = [
        {**c, "llm": lbl.llm, "final": lbl.final, "confidence": lbl.confidence, "note": lbl.note}
        for c, lbl in zip(data["chunks"], base, strict=True)
    ]
    return updated
