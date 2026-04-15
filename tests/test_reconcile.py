"""Tests for the pure-logic reconciliation + transcript-rendering paths.

Anything that shells out to `claude -p` is mocked at the boundary; we only
exercise the local resolution rules, the reconciler JSON parser, and the
LLM-paragraph round-trip validator. The real reconciler and paragraphize
behavior are integration-tested by running polyphony on real audio.
"""

from __future__ import annotations

from polyphony.reconcile import _parse_reconciler_json, reconcile
from polyphony.transcript import _parse_paragraphs
from polyphony.types import Chunk


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


# ---------- transcript._parse_paragraphs (LLM response validator) ----------


def test_parse_paragraphs_accepts_matching_round_trip():
    raw = '[{"id": 0, "paragraphs": ["Hello.", "World."]}]'
    result = _parse_paragraphs(raw, ["Hello. World."])
    assert result == [["Hello.", "World."]]


def test_parse_paragraphs_rejects_wrong_turn_count():
    raw = '[{"id": 0, "paragraphs": ["One."]}]'
    assert _parse_paragraphs(raw, ["One.", "Two."]) is None


def test_parse_paragraphs_rejects_word_drift():
    # Claude "cleaned up" the text — must be rejected, not silently accepted.
    raw = '[{"id": 0, "paragraphs": ["Hi there, how are you?"]}]'
    assert _parse_paragraphs(raw, ["hi there how are you"]) is None


def test_parse_paragraphs_tolerates_whitespace_normalization():
    # Extra spaces between paragraphs vs the input — should still match.
    raw = '[{"id": 0, "paragraphs": ["First paragraph.", "Second paragraph."]}]'
    result = _parse_paragraphs(raw, ["First paragraph.   Second paragraph."])
    assert result == [["First paragraph.", "Second paragraph."]]


def test_parse_paragraphs_tolerates_markdown_fences():
    # A belt-and-suspenders check: Claude sometimes wraps JSON in ```json
    # despite the "no fences" rule. The regex fallback should still find it.
    raw = '```json\n[{"id": 0, "paragraphs": ["Hi."]}]\n```'
    assert _parse_paragraphs(raw, ["Hi."]) == [["Hi."]]


def test_parse_paragraphs_rejects_non_list_response():
    raw = '{"paragraphs": ["Hi."]}'
    assert _parse_paragraphs(raw, ["Hi."]) is None


def test_parse_paragraphs_rejects_empty_paragraphs():
    raw = '[{"id": 0, "paragraphs": []}]'
    assert _parse_paragraphs(raw, ["Hi."]) is None
