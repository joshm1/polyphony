"""On-disk cache for expensive pipeline stages (Whisper transcription, pyannote diarization).

Key insight: Whisper on ~1h of audio takes 11+ min on MPS; pyannote on CPU
takes 30-40 min. Losing that work to a downstream bug is painful, so we
serialize each stage's output to ~/.cache/polyphony/<key>/{whisper,pyannote}.json.

Cache key = sha256 of absolute audio path + model id + file (mtime, size).
That key's cheap to compute (no reading of audio bytes) and invalidates
automatically if the file changes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from loguru import logger

from .types import Chunk, PyannoteSegment

CACHE_ROOT = Path(os.environ.get("POLYPHONY_CACHE_DIR") or Path.home() / ".cache" / "polyphony")


def _cache_key(audio_path: Path, model_id: str) -> str:
    stat = audio_path.stat()
    payload = f"{audio_path.resolve()}|{model_id}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _dir_for(audio_path: Path, model_id: str) -> Path:
    key = _cache_key(audio_path, model_id)
    d = CACHE_ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- Whisper ----------


def load_whisper(audio_path: Path, model_id: str) -> list[Chunk] | None:
    path = _dir_for(audio_path, model_id) / "whisper.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Whisper cache read failed ({e}); will recompute.")
        return None
    chunks = [Chunk(idx=c["idx"], start=c["start"], end=c["end"], text=c["text"]) for c in data]
    logger.info(f"Loaded {len(chunks)} Whisper chunks from cache: {path}")
    return chunks


def save_whisper(audio_path: Path, model_id: str, chunks: list[Chunk]) -> None:
    path = _dir_for(audio_path, model_id) / "whisper.json"
    payload = [{"idx": c.idx, "start": c.start, "end": c.end, "text": c.text} for c in chunks]
    path.write_text(json.dumps(payload))
    logger.info(f"Saved Whisper cache: {path}")


# ---------- Pyannote ----------


def load_pyannote(audio_path: Path, model_id: str) -> list[PyannoteSegment] | None:
    path = _dir_for(audio_path, model_id) / "pyannote.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Pyannote cache read failed ({e}); will recompute.")
        return None
    segs = [PyannoteSegment(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in data]
    logger.info(f"Loaded {len(segs)} pyannote segments from cache: {path}")
    return segs


def save_pyannote(audio_path: Path, model_id: str, segments: list[PyannoteSegment]) -> None:
    path = _dir_for(audio_path, model_id) / "pyannote.json"
    payload = [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments]
    path.write_text(json.dumps(payload))
    logger.info(f"Saved pyannote cache: {path}")


# ---------- ASR correction ----------


def load_asr_flags(audio_path: Path):
    # Defer import to avoid a circular import at module load.
    from .asr_correction import WordFlag

    path = _dir_for(audio_path, "asr-correction-v1") / "flags.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"ASR flags cache read failed ({e}); will recompute.")
        return None
    flags = [
        WordFlag(
            chunk_idx=f["chunk_idx"],
            original=f["original"],
            suggested=f["suggested"],
            confidence=f["confidence"],
            reason=f.get("reason", ""),
            alternatives=f.get("alternatives") or [],
        )
        for f in data
    ]
    logger.info(f"Loaded {len(flags)} ASR flag(s) from cache: {path}")
    return flags


def save_asr_flags(audio_path: Path, flags) -> None:
    path = _dir_for(audio_path, "asr-correction-v1") / "flags.json"
    payload = [f.as_dict() for f in flags]
    path.write_text(json.dumps(payload))
    logger.info(f"Saved ASR flags cache: {path}")
