"""Tests for applying review decisions to a sidecar payload."""

from __future__ import annotations

from pathlib import Path

from polyphony.review import apply_corrections, apply_review, resolve_corrections, reviewed_path


def _chunk(idx: int, text: str, final: int, confidence: int = 100) -> dict:
    return {
        "idx": idx,
        "start": float(idx),
        "end": float(idx + 1),
        "text": text,
        "audio": final,
        "llm": final,
        "final": final,
        "confidence": confidence,
        "note": "",
    }


def _flag(chunk_idx: int, original: str, suggested: str) -> dict:
    return {"chunk_idx": chunk_idx, "original": original, "suggested": suggested, "confidence": 90}


def test_reviewed_path_sits_next_to_raw_transcript():
    sidecar = Path("/x/a b.assemblyai.transcript.polyphony.json")
    assert reviewed_path(sidecar) == Path("/x/a b.assemblyai.transcript.reviewed.md")


def test_undecided_flag_takes_suggestion_and_original_is_skipped():
    flags = [_flag(0, "sauce", "SaaS"), _flag(0, "B2B", "b2b"), _flag(1, "x", "y")]
    decisions = {"1": {"kind": "original", "value": "B2B"}, "2": {"kind": "custom", "value": "z"}}
    assert resolve_corrections(flags, decisions) == {0: [("sauce", "SaaS")], 1: [("x", "z")]}


def test_apply_corrections_never_rematches_replaced_text():
    text, missed = apply_corrections("a cat sat", [("cat", "cat cat"), ("sat", "stood"), ("dog", "wolf")])
    assert text == "a cat cat stood"
    assert missed == ["dog"]


def test_apply_review_overrides_speaker_and_clears_flag():
    data = {
        "names": ["Sam", "Dan"],
        "review_threshold": 70,
        "paragraph_breaks": None,
        "chunks": [_chunk(0, "Hello.", 1), _chunk(1, "We sell sauce.", 1, confidence=40)],
        "asr_flags": [_flag(1, "sauce", "SaaS")],
        "review": {"overrides": {"1": 2}, "word_decisions": {}},
    }
    markdown, skipped = apply_review(data)
    assert markdown == "Sam: Hello.\n\nDan: We sell SaaS.\n"
    assert skipped == []


def test_apply_review_keeps_flag_on_unreviewed_low_confidence_turn():
    data = {
        "names": [],
        "chunks": [_chunk(0, "Hi.", 1, confidence=40)],
        "asr_flags": [],
    }
    markdown, _ = apply_review(data)
    assert "⚠️ Speaker 1: Hi." in markdown
