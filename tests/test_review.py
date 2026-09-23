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


# ---------- reanalyze: name resolution + flag merging ----------


def test_resolve_names_keeps_explicit_and_fills_rest_from_llm(monkeypatch):
    from polyphony import reanalyze
    from polyphony.types import Chunk, ChunkLabel

    seen = {}

    def fake_identify(labels, candidates, fixed, model, context_hint):
        seen.update(candidates=candidates, fixed=fixed)
        return {2: "Guest"}

    monkeypatch.setattr(reanalyze, "identify_speakers", fake_identify)
    labels = [
        ChunkLabel(Chunk(i, i, i + 1, "x"), audio=s, llm=s, final=s, confidence=100) for i, s in enumerate([1, 2, 3])
    ]
    names = reanalyze.resolve_names(labels, ["Host", ""], ["  Guest ", ""], "m", None)
    assert names == ["Host", "Guest", ""]
    assert seen == {"candidates": ["Guest"], "fixed": {1: "Host"}}


def test_merge_flags_carries_explicit_decisions_and_user_flags():
    from polyphony.reanalyze import merge_flags

    old = [_flag(1, "sauce", "SaaS"), _flag(2, "x", "y"), {**_flag(3, "a", "b"), "userAdded": True}]
    decisions = {"0": {"kind": "original", "value": "sauce"}, "1": {"kind": "suggested", "value": "y"}}
    new = [_flag(2, "x", "z"), _flag(1, "sauce", "sass")]
    flags, carried = merge_flags(old, decisions, new)
    assert [(f["chunk_idx"], f["original"]) for f in flags] == [(2, "x"), (1, "sauce"), (3, "a")]
    # "keep original" follows its span to the new index; the default "suggested" choice is not carried.
    assert carried == {"1": {"kind": "original", "value": "sauce"}}
