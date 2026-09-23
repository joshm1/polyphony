"""Speaker diarization — two backends that each emit per-chunk speaker labels.

Both backends return `list[int | None]` of the same length as the Whisper
chunks, where `int` is a 1-indexed speaker id and `None` means "this backend
couldn't label this chunk" (e.g. pyannote saw silence, or the LLM omitted it).

Per-chunk alignment is the key design choice: it makes the two backends
directly comparable without fuzzy text matching in the reconciler.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from .cache import load_pyannote, save_pyannote
from .llm import run_structured
from .types import Chunk, PyannoteSegment

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"


# ---------------- pyannote ----------------


def get_hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        if val := os.environ.get(var):
            return val
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token() or None
    except Exception:
        return None


class PyannoteUnavailable(RuntimeError):
    """Raised when pyannote can't be loaded (missing token, unaccepted license, etc.)."""


def _load_pyannote_pipeline(token: str):
    """Load the diarization pipeline, raising PyannoteUnavailable with a clear message on failure."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise PyannoteUnavailable(f"pyannote.audio import failed: {e}") from e

    # newer pyannote/huggingface_hub renamed `use_auth_token` → `token`; try both.
    try:
        pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL_ID, token=token)
    except TypeError:
        try:
            pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL_ID, use_auth_token=token)
        except Exception as e:
            raise PyannoteUnavailable(_license_hint(e)) from e
    except Exception as e:
        raise PyannoteUnavailable(_license_hint(e)) from e

    if pipeline is None:
        raise PyannoteUnavailable(
            f"Pipeline.from_pretrained({PYANNOTE_MODEL_ID!r}) returned None — "
            "this almost always means the HF token is valid but the license hasn't been accepted. "
            "Accept BOTH:\n"
            f"  - https://huggingface.co/{PYANNOTE_MODEL_ID}\n"
            "  - https://huggingface.co/pyannote/segmentation-3.0\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-community-1"
        )
    return pipeline


def _license_hint(e: Exception) -> str:
    return (
        f"Could not load {PYANNOTE_MODEL_ID}: {e}. "
        "Accept BOTH licenses with the same HF account as $HF_TOKEN:\n"
        f"  - https://huggingface.co/{PYANNOTE_MODEL_ID}\n"
        "  - https://huggingface.co/pyannote/segmentation-3.0"
    )


def preflight_pyannote() -> None:
    """Fail fast if pyannote isn't ready — runs in seconds, before Whisper's 10-min transcribe.

    Raises PyannoteUnavailable on any failure. Callers can catch and downgrade
    gracefully or bail depending on whether pyannote was explicitly requested.
    """
    token = get_hf_token()
    if not token:
        raise PyannoteUnavailable(
            "No HF token found. Set one via `export HF_TOKEN=hf_...` "
            "(create at https://huggingface.co/settings/tokens) and accept licenses:\n"
            f"  - https://huggingface.co/{PYANNOTE_MODEL_ID}\n"
            "  - https://huggingface.co/pyannote/segmentation-3.0\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-community-1"
        )
    logger.info("Preflight: loading pyannote pipeline to verify access…")
    _load_pyannote_pipeline(token)
    logger.info("Preflight: pyannote pipeline loaded successfully.")


def _run_pyannote(audio_path: Path, device: str, source_audio: Path | None = None) -> list[PyannoteSegment]:
    cache_key_path = source_audio or audio_path
    cached = load_pyannote(cache_key_path, PYANNOTE_MODEL_ID)
    if cached is not None:
        return cached

    token = get_hf_token()
    if not token:
        logger.warning("No HF token found — pyannote skipped.")
        return []

    try:
        pipeline = _load_pyannote_pipeline(token)
    except PyannoteUnavailable as e:
        logger.warning(str(e))
        return []

    import contextlib

    import torch

    # pyannote on MPS currently crashes on some ops; fall back to CPU for diarization only.
    diar_device = "cpu" if device == "mps" else device
    with contextlib.suppress(Exception):
        pipeline.to(torch.device(diar_device))

    logger.info(f"Diarizing {audio_path.name} on {diar_device}…")
    result = pipeline(str(audio_path))

    # pyannote 3.0 returned an Annotation directly; 3.1+ wraps it in a
    # DiarizeOutput (Annotation lives on `.speaker_diarization`). Support both.
    annotation = getattr(result, "speaker_diarization", result)

    segments: list[PyannoteSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(PyannoteSegment(start=turn.start, end=turn.end, speaker=speaker))

    logger.info(f"Pyannote found {len({s.speaker for s in segments})} speaker(s), {len(segments)} turn(s).")
    save_pyannote(cache_key_path, PYANNOTE_MODEL_ID, segments)
    return segments


def pyannote_per_chunk_labels(chunks: list[Chunk], segments: list[PyannoteSegment]) -> list[int | None]:
    """Assign each chunk the pyannote speaker it overlaps most with (1-indexed)."""
    if not segments:
        return [None] * len(chunks)

    # Stable speaker id: first-appearance order → 1, 2, 3, …
    order: list[str] = []
    for seg in segments:
        if seg.speaker not in order:
            order.append(seg.speaker)
    label_of = {name: i + 1 for i, name in enumerate(order)}

    labels: list[int | None] = []
    for ch in chunks:
        best_label: str | None = None
        best_overlap = 0.0
        for seg in segments:
            overlap = max(0.0, min(ch.end, seg.end) - max(ch.start, seg.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = seg.speaker
        labels.append(label_of[best_label] if best_label else None)
    return labels


def diarize_pyannote(
    audio_path: Path, chunks: list[Chunk], device: str, source_audio: Path | None = None
) -> list[int | None]:
    """End-to-end pyannote: raw segments → per-chunk labels."""
    segments = _run_pyannote(audio_path, device, source_audio=source_audio)
    return pyannote_per_chunk_labels(chunks, segments)


# ---------------- LLM (text-side reasoning) ----------------


class _SpeakerAssignment(BaseModel):
    id: int = Field(description="Chunk id, exactly as shown in brackets.")
    speaker: int = Field(ge=1, description="1-indexed speaker id.")


class _SpeakerAssignments(BaseModel):
    labels: list[_SpeakerAssignment] = Field(description="Exactly one entry per chunk id.")


def diarize_llm(
    chunks: list[Chunk],
    names: list[str] | None,
    model: str,
    expected_speakers: int = 2,
) -> list[int | None]:
    """Ask the LLM to label each chunk by speaker from the text alone.

    Output is keyed by chunk id so it aligns back to chunks exactly, with no
    fuzzy text matching. Any failure yields all-None, which the reconciler
    treats as "text side silent" and falls back to pyannote.
    """
    if not chunks:
        return []

    if names:
        speaker_block = "\n".join(f"  Speaker {i} = {name}" for i, name in enumerate(names, start=1))
    else:
        speaker_block = "\n".join(f"  Speaker {i} (unknown name)" for i in range(1, expected_speakers + 1))
    max_speaker = max(2, len(names) if names else expected_speakers)
    numbered = "\n".join(f"[{c.idx}] {c.text}" for c in chunks)

    prompt = f"""Label each chunk of the Whisper ASR transcript below with its speaker.

Speakers (use 1-indexed ids in your output):
{speaker_block}

Rules:
- Exactly one entry per chunk id.
- Valid speaker values are 1..{max_speaker}.

Heuristics:
- The person asking questions and the person answering them are usually different speakers.
- Very short chunks ("yeah", "right", "mm-hm") are usually reactions — often the
  opposite speaker of the previous long turn.
- Names mentioned in the text are strong clues (a speaker addressing someone by name is not that person).
- If you're genuinely unsure, pick the most plausible speaker; the reconciler will flag low-confidence turns.

Chunks:
{numbered}
"""

    logger.info(f"LLM diarization on {len(chunks)} chunks via {model}…")
    try:
        result = run_structured(prompt, _SpeakerAssignments, model)
    except Exception as e:
        logger.error(f"LLM diarization failed ({model}): {e}")
        return [None] * len(chunks)
    return _labels_by_chunk(result.labels, len(chunks), max_speaker)


def _labels_by_chunk(entries: list[_SpeakerAssignment], expected_len: int, max_speaker: int) -> list[int | None]:
    labels: list[int | None] = [None] * expected_len
    for entry in entries:
        if 0 <= entry.id < expected_len and entry.speaker <= max_speaker:
            labels[entry.id] = entry.speaker

    missing = sum(1 for x in labels if x is None)
    if missing:
        logger.warning(f"LLM labeled {expected_len - missing}/{expected_len} chunks; rest = None.")
    return labels
