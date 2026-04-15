"""Whisper ASR via HF transformers pipeline."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from .cache import load_whisper, save_whisper
from .types import Chunk

WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"


def transcribe(audio_path: Path, device: str, dtype, source_audio: Path | None = None) -> list[Chunk]:
    """Run Whisper-large-v3-turbo with chunk-level timestamps.

    We request chunk-level rather than word-level timestamps because
    chunk timestamps are markedly more stable across transformers
    versions and align cleanly with pyannote turns (which are also
    coarser than words).

    The cache is keyed on the ORIGINAL audio path (via `source_audio`) —
    not the temp 16kHz wav we actually transcribe — so rerunning on the
    same user-provided file hits the cache even though the wav path changes.
    """
    cache_key_path = source_audio or audio_path
    cached = load_whisper(cache_key_path, WHISPER_MODEL_ID)
    if cached is not None:
        return cached

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    logger.info(f"Loading Whisper ({WHISPER_MODEL_ID}) on {device}…")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )

    logger.info(f"Transcribing {audio_path.name}…")
    result = pipe(str(audio_path), return_timestamps=True)

    chunks: list[Chunk] = []
    for c in result.get("chunks", []):
        ts = c.get("timestamp")
        if not ts or ts[0] is None or ts[1] is None:
            continue
        text = (c.get("text") or "").strip()
        if text:
            chunks.append(Chunk(idx=len(chunks), start=float(ts[0]), end=float(ts[1]), text=text))

    # Fallback if the pipeline only returned top-level text.
    if not chunks and result.get("text"):
        chunks.append(Chunk(idx=0, start=0.0, end=0.0, text=result["text"].strip()))

    logger.info(f"Whisper produced {len(chunks)} chunk(s).")
    save_whisper(cache_key_path, WHISPER_MODEL_ID, chunks)
    return chunks
