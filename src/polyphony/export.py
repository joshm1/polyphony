"""Export the reviewed transcript from a sidecar as JSON, CSV, SRT, WebVTT, or markdown.

Every format carries the review: speaker overrides and accepted word
corrections are applied, speaker names come from the sidecar. JSON and CSV
are per chunk and keep the audit trail alongside the corrected text: the
original ASR text, both diarizers' labels, the pipeline's confidence and
rationale, and whether a reviewer changed the speaker or the words.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Literal

from .review import apply_review, labels_from_payload, reviewed_labels
from .transcript import display_name
from .types import ChunkLabel

ExportFormat = Literal["json", "csv", "srt", "vtt", "md"]
EXPORT_FORMATS: tuple[ExportFormat, ...] = ("json", "csv", "srt", "vtt", "md")

_CSV_COLUMNS = (
    "idx",
    "start",
    "end",
    "speaker_id",
    "speaker",
    "text",
    "original_text",
    "confidence",
    "speaker_overridden",
    "text_corrected",
    "audio_label",
    "llm_label",
    "note",
)


def export_reviewed(data: dict[str, Any], fmt: ExportFormat) -> str:
    """Render a sidecar payload, with its review applied, in `fmt`."""
    if fmt == "md":
        return apply_review(data)[0]
    labels, _ = reviewed_labels(data)
    names: list[str] = data.get("names") or []
    if fmt == "srt":
        return _srt(labels, names)
    if fmt == "vtt":
        return _vtt(labels, names)
    rows = _rows(data, labels, names)
    if fmt == "csv":
        return _csv(rows)
    return json.dumps(
        {
            "audio": data.get("audio"),
            "backend": data.get("backend"),
            "llm_model": data.get("llm_model"),
            "review_threshold": data.get("review_threshold"),
            "speakers": [{"id": i, "name": display_name(i, names)} for i in sorted({r["speaker_id"] for r in rows})],
            "chunks": rows,
        },
        indent=2,
        ensure_ascii=False,
    )


def _rows(data: dict[str, Any], labels: list[ChunkLabel], names: list[str]) -> list[dict[str, Any]]:
    original_text = {lbl.chunk.idx: lbl.chunk.text for lbl in labels_from_payload(data)}
    return [
        {
            "idx": lbl.chunk.idx,
            "start": lbl.chunk.start,
            "end": lbl.chunk.end,
            "speaker_id": lbl.effective_speaker,
            "speaker": display_name(lbl.effective_speaker, names),
            "text": lbl.chunk.text,
            "original_text": original_text[lbl.chunk.idx],
            "confidence": lbl.confidence,
            "speaker_overridden": lbl.override is not None and lbl.override != lbl.final,
            "text_corrected": lbl.chunk.text != original_text[lbl.chunk.idx],
            "audio_label": lbl.audio,
            "llm_label": lbl.llm,
            "note": lbl.note,
        }
        for lbl in labels
    ]


def _csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _cues(labels: list[ChunkLabel]) -> list[ChunkLabel]:
    # Zero-length cues (a Whisper fallback chunk with no timing) can't be displayed.
    return [lbl for lbl in labels if lbl.chunk.end > lbl.chunk.start and lbl.chunk.text.strip()]


def _srt(labels: list[ChunkLabel], names: list[str]) -> str:
    blocks = [
        f"{n}\n{_timestamp(lbl.chunk.start, ',')} --> {_timestamp(lbl.chunk.end, ',')}\n"
        f"{display_name(lbl.effective_speaker, names)}: {lbl.chunk.text.strip()}\n"
        for n, lbl in enumerate(_cues(labels), start=1)
    ]
    return "\n".join(blocks)


def _vtt(labels: list[ChunkLabel], names: list[str]) -> str:
    blocks = ["WEBVTT\n"]
    for lbl in _cues(labels):
        # `<v Name>` is WebVTT's voice span: players and editors can show or style the speaker.
        speaker = display_name(lbl.effective_speaker, names).replace(">", "")
        text = lbl.chunk.text.strip().replace("&", "&amp;").replace("<", "&lt;").replace("-->", "->")
        blocks.append(f"{_timestamp(lbl.chunk.start, '.')} --> {_timestamp(lbl.chunk.end, '.')}\n<v {speaker}>{text}\n")
    return "\n".join(blocks)


def _timestamp(seconds: float, ms_separator: str) -> str:
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{ms:03d}"
