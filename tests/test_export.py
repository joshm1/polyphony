"""Exports carry the review (speaker overrides, word corrections, names) plus the audit trail."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from polyphony.cli import main
from polyphony.export import export_reviewed


def _sidecar() -> dict[str, Any]:
    def chunk(idx: int, start: float, end: float, text: str, final: int, confidence: int) -> dict[str, Any]:
        return {
            "idx": idx,
            "start": start,
            "end": end,
            "text": text,
            "audio": final,
            "llm": None,
            "final": final,
            "confidence": confidence,
            "note": "audio diarizer only (no LLM)",
        }

    return {
        "audio": "talk.m4a",
        "backend": "local",
        "llm_model": None,
        "names": ["Ana", ""],
        "review_threshold": 70,
        "paragraph_breaks": None,
        "chunks": [
            chunk(0, 0.0, 2.5, "We sell sauce.", 1, 70),
            chunk(1, 2.5, 3661.25, "Since <when> & why?", 1, 30),
        ],
        "asr_flags": [{"chunk_idx": 0, "original": "sauce", "suggested": "SaaS", "confidence": 90}],
        "review": {"overrides": {"1": 2}, "word_decisions": {}},
    }


def test_json_keeps_corrected_and_original_text_with_audit_fields() -> None:
    out = json.loads(export_reviewed(_sidecar(), "json"))
    first, second = out["chunks"]
    assert (first["text"], first["original_text"], first["text_corrected"]) == ("We sell SaaS.", "We sell sauce.", True)
    assert (second["speaker_id"], second["speaker"], second["speaker_overridden"]) == (2, "Speaker 2", True)
    # Pipeline confidence is preserved so the export shows what the reviewer had to fix.
    assert second["confidence"] == 30
    assert out["speakers"] == [{"id": 1, "name": "Ana"}, {"id": 2, "name": "Speaker 2"}]


def test_csv_has_one_row_per_chunk() -> None:
    rows = list(csv.DictReader(io.StringIO(export_reviewed(_sidecar(), "csv"))))
    assert [r["speaker"] for r in rows] == ["Ana", "Speaker 2"]
    assert rows[0]["text"] == "We sell SaaS."


def test_srt_numbers_cues_and_uses_comma_milliseconds() -> None:
    assert export_reviewed(_sidecar(), "srt") == (
        "1\n00:00:00,000 --> 00:00:02,500\nAna: We sell SaaS.\n\n"
        "2\n00:00:02,500 --> 01:01:01,250\nSpeaker 2: Since <when> & why?\n"
    )


def test_vtt_uses_voice_spans_and_escapes_markup() -> None:
    assert export_reviewed(_sidecar(), "vtt") == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\n<v Ana>We sell SaaS.\n\n"
        "00:00:02.500 --> 01:01:01.250\n<v Speaker 2>Since &lt;when> &amp; why?\n"
    )


def test_markdown_matches_the_reviewed_note() -> None:
    assert export_reviewed(_sidecar(), "md") == "Ana: We sell SaaS.\n\nSpeaker 2: Since <when> & why?\n"


def test_cli_export_finds_the_sidecar_next_to_the_audio(tmp_path: Path):
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")
    (tmp_path / "talk.local.transcript.polyphony.json").write_text(json.dumps(_sidecar()))

    result = CliRunner().invoke(main, ["export", str(audio), "--format", "srt"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "talk.local.transcript.reviewed.srt").read_text().startswith("1\n00:00:00,000")

    stdout = CliRunner().invoke(main, ["export", str(audio), "-f", "csv", "-o", "-"])
    assert stdout.output.startswith("idx,start,end,speaker_id,speaker,text")


def test_cli_export_without_sidecar_explains_what_to_run(tmp_path: Path):
    audio = tmp_path / "talk.m4a"
    audio.write_bytes(b"")
    result = CliRunner().invoke(main, ["export", str(audio)])
    assert result.exit_code != 0
    assert "polyphony transcribe" in result.output
