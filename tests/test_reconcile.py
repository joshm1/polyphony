"""Tests for the pure-logic reconciliation + transcript-rendering paths.

Anything that shells out to `claude -p` is mocked at the boundary; we only
exercise the local resolution rules, the reconciler JSON parser, and the
LLM-paragraph round-trip validator. The real reconciler and paragraphize
behavior are integration-tested by running polyphony on real audio.
"""

from __future__ import annotations

from polyphony.paragraphize import _apply_breaks_to_turns, _parse_break_ids
from polyphony.reconcile import _parse_reconciler_json, reconcile
from polyphony.transcript import _group_into_turns
from polyphony.types import Chunk, ChunkLabel


def _ch(idx: int, text: str = "x") -> Chunk:
    return Chunk(idx=idx, start=float(idx), end=float(idx + 1), text=text)


def test_full_agreement_skips_reconciler():
    chunks = [_ch(0), _ch(1), _ch(2)]
    out = reconcile(chunks, [1, 2, 1], [1, 2, 1], names=None)
    assert [lbl.final for lbl in out] == [1, 2, 1]
    assert all(lbl.confidence == 100 for lbl in out)
    assert all(lbl.note == "both backends agreed" for lbl in out)


def test_one_silent_backend_uses_other_with_medium_confidence():
    chunks = [_ch(0), _ch(1)]
    # pyannote silent on chunk 0; claude silent on chunk 1
    out = reconcile(chunks, [None, 2], [1, None], names=None)
    assert out[0].final == 1
    assert out[0].confidence == 70
    assert "claude" in out[0].note
    assert out[1].final == 2
    assert out[1].confidence == 70
    assert "pyannote" in out[1].note


def test_parse_reconciler_json_basic():
    raw = '[{"id": 0, "speaker": 1, "confidence": 88, "reason": "asks question"}]'
    decisions = _parse_reconciler_json(raw)
    assert 0 in decisions
    assert decisions[0].speaker == 1
    assert decisions[0].confidence == 88
    assert decisions[0].reason == "asks question"


def test_parse_reconciler_json_tolerates_surrounding_prose():
    raw = 'Sure, here you go:\n\n```json\n[{"id": 5, "speaker": 2, "confidence": 60}]\n```\n'
    decisions = _parse_reconciler_json(raw)
    assert decisions[5].speaker == 2
    assert decisions[5].confidence == 60


def test_parse_reconciler_json_skips_malformed_entries():
    raw = '[{"id": 0, "speaker": 1}, {"id": "bad", "speaker": 2}, {"speaker": 3}]'
    decisions = _parse_reconciler_json(raw)
    # Only the well-formed entry survives.
    assert list(decisions.keys()) == [0]


def test_parse_reconciler_json_returns_empty_on_garbage():
    assert _parse_reconciler_json("not json at all") == {}
    assert _parse_reconciler_json("") == {}


def test_parse_reconciler_json_defaults_bad_confidence_to_50():
    raw = '[{"id": 0, "speaker": 1, "confidence": "very high"}]'
    decisions = _parse_reconciler_json(raw)
    assert decisions[0].confidence == 50


# ---------- transcript._parse_break_ids + _apply_breaks_to_turns ----------


def _lbl(idx: int, final: int, text: str = "x") -> ChunkLabel:
    ch = Chunk(idx=idx, start=float(idx), end=float(idx + 1), text=text)
    return ChunkLabel(chunk=ch, pyannote=final, claude=final, final=final, confidence=100)


def test_parse_break_ids_accepts_clean_list():
    raw = '{"break_after_chunk": [3, 7, 12]}'
    assert _parse_break_ids(raw, valid_ids={3, 7, 12, 99}) == [3, 7, 12]


def test_parse_break_ids_filters_out_of_range():
    raw = '{"break_after_chunk": [3, 999, 12, -1]}'
    assert _parse_break_ids(raw, valid_ids={3, 12}) == [3, 12]


def test_parse_break_ids_dedupes_and_sorts():
    raw = '{"break_after_chunk": [7, 3, 7, 12, 3]}'
    assert _parse_break_ids(raw, valid_ids={3, 7, 12}) == [3, 7, 12]


def test_parse_break_ids_returns_empty_list_for_no_breaks():
    # Valid output meaning "nothing needs breaking" — not a failure.
    raw = '{"break_after_chunk": []}'
    assert _parse_break_ids(raw, valid_ids={0, 1}) == []


def test_parse_break_ids_rejects_wrong_shape():
    assert _parse_break_ids("[3, 7]", valid_ids={3, 7}) is None
    assert _parse_break_ids('{"foo": [3]}', valid_ids={3}) is None
    assert _parse_break_ids("not json", valid_ids={3}) is None


# Paragraphs here are long enough (~140c each) not to trip the stranded-tail
# merge that runs after `_apply_breaks_to_turns`; see tests below for that
# post-process specifically.
_LONG_A = (
    "This is a paragraph long enough to stand on its own, without getting "
    "merged back into whatever came before it by the stranded-tail guard."
)
_LONG_B = (
    "And this is the second paragraph, again plenty long to avoid being "
    "treated as a short stranded punchline that the post-process merges."
)
_LONG_C = (
    "A third paragraph in case we need one, just as long as the others "
    "so the test stays isolated from length-based post-processing behavior."
)


def test_apply_breaks_single_turn_multiple_paragraphs():
    labels = [_lbl(0, 1, _LONG_A), _lbl(1, 1, _LONG_B), _lbl(2, 1, _LONG_C)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [0])
    assert paras == [[_LONG_A, f"{_LONG_B} {_LONG_C}"]]


def test_apply_breaks_ignores_turn_boundary_breaks():
    labels = [_lbl(0, 1, _LONG_A), _lbl(1, 2, _LONG_B), _lbl(2, 2, _LONG_C)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [0])
    assert paras == [[_LONG_A], [f"{_LONG_B} {_LONG_C}"]]


def test_apply_breaks_no_breaks_joins_chunks():
    labels = [_lbl(0, 1, _LONG_A), _lbl(1, 1, _LONG_B), _lbl(2, 1, _LONG_C)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [])
    assert paras == [[f"{_LONG_A} {_LONG_B} {_LONG_C}"]]


# ---------- transcript._polish_paragraphs (stranded merge + oversize split) ----------


def test_polish_merges_stranded_trailing_fragment():
    # Long setup + short punchline → merged into one paragraph.
    setup = _LONG_A
    punchline = "We made it up."  # 14 chars, way under the 120c threshold
    labels = [_lbl(0, 1, setup), _lbl(1, 1, punchline)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [0])
    assert paras == [[f"{setup} {punchline}"]]


def test_polish_leaves_long_trailing_paragraph_alone():
    # Trailing paragraph > 120c → not stranded, kept as its own paragraph.
    labels = [_lbl(0, 1, _LONG_A), _lbl(1, 1, _LONG_B)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [0])
    assert paras == [[_LONG_A, _LONG_B]]


def test_polish_splits_oversized_paragraph_at_discourse_marker():
    # One 1500+ char paragraph with a clear ". And then " in the middle →
    # gets split there, producing two paragraphs.
    left_half = "Something about the history of accounting goes here, " * 12
    right_half = "something about the modern state of accounting goes here, " * 12
    text = left_half.rstrip(", ") + ". And then " + right_half.rstrip(", ") + "."
    labels = [_lbl(0, 1, text)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [])
    assert len(paras[0]) >= 2
    assert all(len(p) <= 1100 for p in paras[0]), f"got lengths {[len(p) for p in paras[0]]}"


def test_polish_leaves_oversized_without_marker_alone():
    # No discourse markers at all → we don't mid-sentence split.
    text = ("word " * 300).strip()  # ~1500 chars, no periods/markers
    labels = [_lbl(0, 1, text)]
    turns = _group_into_turns(labels, review_threshold=100)
    paras = _apply_breaks_to_turns(turns, [])
    assert paras == [[text]]
