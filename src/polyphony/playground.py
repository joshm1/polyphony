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
    paragraph_breaks: list[int] | None,
    review_threshold: int,
    backend: str,
    llm_model: str,
    context_hint: str | None,
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
        # Indexed by speaker id - 1; "" = not named yet.
        "names": list(names),
        "review_threshold": review_threshold,
        # How the text-side passes ran, so `POST /api/reanalyze` can rerun them the same way.
        "backend": backend,
        "llm_model": llm_model,
        "context_hint": context_hint,
        # LLM-chosen break-after chunk ids; lets a review re-render paragraphs without another LLM call.
        "paragraph_breaks": paragraph_breaks,
        # Reviewer decisions, filled in by `POST /api/apply`.
        "review": {"overrides": {}, "word_decisions": {}},
        "chunks": [
            {
                "idx": lbl.chunk.idx,
                "start": round(lbl.chunk.start, 2),
                "end": round(lbl.chunk.end, 2),
                "text": lbl.chunk.text,
                "audio": lbl.audio,
                "llm": lbl.llm,
                "final": lbl.final,
                "confidence": lbl.confidence,
                "note": lbl.note,
            }
            for lbl in labels
        ],
        "asr_flags": [flag_dict(f) for f in asr_flags],
    }


def flag_dict(f: WordFlag) -> dict:
    return {
        "chunk_idx": f.chunk_idx,
        "original": f.original,
        "suggested": f.suggested,
        "alternatives": f.alternatives,
        "confidence": f.confidence,
        "reason": f.reason,
    }
