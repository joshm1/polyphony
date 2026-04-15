"""Shapes the review payload that the React frontend consumes.

The frontend lives in `frontend/` and gets the payload via `GET /api/data`
from the FastAPI server in `serve.py`. `playground_payload` is the single
source of truth for that contract — keep it in sync with
`frontend/src/types.ts::PolyphonyData`.
"""

from pathlib import Path

from .asr_correction import WordFlag
from .types import ChunkLabel


def playground_payload(
    labels: list[ChunkLabel],
    names: list[str],
    audio_name: str,
    transcript_path: Path,
    asr_flags: list[WordFlag],
    audio_url: str | None = None,
) -> dict:
    """Assemble the JSON payload the React app reads on load.

    `audio_url` is left empty at write-time — the serve layer injects
    `/api/audio` when it loads the sidecar, so the same file can be
    consumed by future backends without hard-coding a path.
    """
    return {
        "audio": audio_name,
        "audio_url": audio_url,
        "transcript_path": str(transcript_path),
        "names": list(names),
        "chunks": [
            {
                "idx": lbl.chunk.idx,
                "start": round(lbl.chunk.start, 2),
                "end": round(lbl.chunk.end, 2),
                "text": lbl.chunk.text,
                "pyannote": lbl.pyannote,
                "claude": lbl.claude,
                "final": lbl.final,
                "confidence": lbl.confidence,
                "note": lbl.note,
            }
            for lbl in labels
        ],
        "asr_flags": [
            {
                "chunk_idx": f.chunk_idx,
                "original": f.original,
                "suggested": f.suggested,
                "alternatives": f.alternatives,
                "confidence": f.confidence,
                "reason": f.reason,
            }
            for f in asr_flags
        ],
    }
