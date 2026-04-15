"""Tests for the pure-logic reconciliation paths.

Anything that shells out to `claude -p` is mocked at the boundary; we only
exercise the local resolution rules and the JSON parser. The real reconciler
behavior is integration-tested by running polyphony on real audio.
"""

from __future__ import annotations

from polyphony.reconcile import _parse_reconciler_json, reconcile
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
