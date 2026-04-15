"""Speaker diarization — two backends that each emit per-chunk speaker labels.

Both backends return `list[int | None]` of the same length as the Whisper
chunks, where `int` is a 1-indexed speaker id and `None` means "this backend
couldn't label this chunk" (e.g. pyannote saw silence, or claude omitted it).

Per-chunk alignment is the key design choice: it makes the two backends
directly comparable without fuzzy text matching in the reconciler.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from loguru import logger

from .cache import load_pyannote, save_pyannote
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


# ---------------- claude (via claude -p + /transcript-diarize skill) ----------------

_JSON_ARRAY_RE = re.compile(r"\[\s*{.*?}\s*]", re.DOTALL)


def diarize_claude(
    chunks: list[Chunk],
    names: list[str] | None,
    expected_speakers: int = 2,
    claude_cwd: Path | None = None,
) -> list[int | None]:
    """Shell out to `claude -p` and ask it to label each chunk by speaker.

    Asks for strict JSON output so we can align back to chunks exactly,
    with no fuzzy text matching.
    """
    if shutil.which("claude") is None:
        raise RuntimeError(
            "`claude` CLI not found on PATH. Install Claude Code or switch to --diarize pyannote / --diarize none."
        )

    if not chunks:
        return []

    speaker_block_lines = []
    if names:
        for i, name in enumerate(names, start=1):
            speaker_block_lines.append(f"  Speaker {i} = {name}")
    else:
        for i in range(1, expected_speakers + 1):
            speaker_block_lines.append(f"  Speaker {i} (unknown name)")
    speaker_block = "\n".join(speaker_block_lines)

    numbered = "\n".join(f"[{c.idx}] {c.text}" for c in chunks)

    prompt = f"""Use the /transcript-diarize skill (from the audio-transcripts plugin in the
joshm1 Claude marketplace) to label each chunk of the Whisper ASR output below with a speaker.

Speakers (use 1-indexed ids in your output):
{speaker_block}

Output STRICT JSON — a single array with one object per chunk, in the same order:
  [{{"id": 0, "speaker": 1}}, {{"id": 1, "speaker": 2}}, ...]

Rules:
- Exactly one entry per chunk id.
- Valid speaker values are 1..{max(2, len(names) if names else expected_speakers)}.
- Output ONLY the JSON. No preamble, no commentary, no markdown fences.

Heuristics the skill should apply:
- Interviewer asks questions; interviewee answers.
- Very short chunks ("yeah", "right", "mm-hm") are usually reactions — often the
  opposite speaker of the previous long turn.
- Names mentioned in the text are strong clues ("so, Neville, tell me…").
- If you're genuinely unsure, pick the most plausible speaker; the reconciler will flag low-confidence turns.

Chunks:
{numbered}
"""

    logger.info(
        f"Claude diarization on {len(chunks)} chunks via `claude -p`{f' (cwd={claude_cwd})' if claude_cwd else ''}…"
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", "--permission-mode", "auto"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(claude_cwd) if claude_cwd else None,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"claude CLI failed (exit {e.returncode}): {e.stderr}")
        raise

    return _parse_claude_json(proc.stdout, len(chunks))


def _parse_claude_json(raw: str, expected_len: int) -> list[int | None]:
    """Extract and normalize the per-chunk label array from claude's output."""
    raw = raw.strip()
    # Tolerate markdown fences or surrounding prose by grabbing the first
    # top-level JSON array of objects.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            logger.error(f"Claude output had no parseable JSON array:\n{raw[:500]}")
            return [None] * expected_len
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after regex extract: {e}")
            return [None] * expected_len

    if not isinstance(data, list):
        logger.error(f"Claude JSON was not a list: {type(data).__name__}")
        return [None] * expected_len

    labels: list[int | None] = [None] * expected_len
    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        spk = entry.get("speaker")
        if isinstance(idx, int) and 0 <= idx < expected_len and isinstance(spk, int) and spk >= 1:
            labels[idx] = spk

    missing = sum(1 for x in labels if x is None)
    if missing:
        logger.warning(f"Claude labeled {expected_len - missing}/{expected_len} chunks; rest = None.")
    return labels
