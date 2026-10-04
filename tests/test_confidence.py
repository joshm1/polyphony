# pyright: reportPrivateUsage=false
"""The ensemble → confidence → review-flag path, end to end, with the LLM stubbed or absent."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from polyphony.backends.assemblyai import Word, words_to_chunks
from polyphony.diarize import (
    _SpeakerAssignment,
    _SpeakerAssignments,
    diarize_llm,
    pyannote_chunk_purity,
    pyannote_per_chunk_labels,
)
from polyphony.playground import playground_payload
from polyphony.reconcile import (
    AGREED_CONFIDENCE,
    FALLBACK_CONFIDENCE,
    SINGLE_SIGNAL_CONFIDENCE,
    _Decision,
    _Decisions,
    reconcile,
    single_signal_confidence,
)
from polyphony.review import apply_review, labels_from_payload
from polyphony.transcript import build_transcript
from polyphony.types import Chunk, PyannoteSegment


def _ch(idx: int, start: float, end: float, text: str = "x") -> Chunk:
    return Chunk(idx=idx, start=start, end=end, text=text)


def _returns(value: object) -> Callable[..., Any]:
    def stub(*_args: object, **_kwargs: object) -> object:
        return value

    return stub


def _no_llm_calls(*_args: object, **_kwargs: object) -> Any:
    raise AssertionError("no LLM call expected")


# ---------- pyannote purity ----------


def test_purity_is_share_of_speech_owned_by_majority_speaker() -> None:
    segments = [
        PyannoteSegment(0.0, 3.0, "SPEAKER_00"),
        PyannoteSegment(3.0, 4.0, "SPEAKER_01"),
    ]
    chunks = [_ch(0, 0.0, 2.0), _ch(1, 2.0, 4.0), _ch(2, 10.0, 11.0)]
    assert pyannote_chunk_purity(chunks, segments) == [1.0, 0.5, None]


def test_purity_ignores_silence_inside_a_chunk() -> None:
    # One speaker talking for 1s of a 4s chunk is still a pure chunk.
    segments = [PyannoteSegment(0.0, 1.0, "SPEAKER_00")]
    assert pyannote_chunk_purity([_ch(0, 0.0, 4.0)], segments) == [1.0]


def test_single_signal_confidence_scales_and_clamps() -> None:
    assert single_signal_confidence(None) == SINGLE_SIGNAL_CONFIDENCE
    assert single_signal_confidence(1.0) == SINGLE_SIGNAL_CONFIDENCE
    assert single_signal_confidence(0.5) == 35
    assert single_signal_confidence(1.5) == SINGLE_SIGNAL_CONFIDENCE
    assert single_signal_confidence(-1.0) == 0


# ---------- reconcile without an LLM ----------


def test_no_llm_uses_audio_labels_and_never_calls_the_reconciler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polyphony.reconcile.run_structured", _no_llm_calls)
    chunks = [_ch(0, 0, 1), _ch(1, 1, 2), _ch(2, 2, 3)]
    out = reconcile(chunks, [1, 2, None], [None, None, None], names=None, model=None, audio_purity=[1.0, 0.6, None])

    assert [lbl.final for lbl in out] == [1, 2, 1]
    assert [lbl.confidence for lbl in out] == [SINGLE_SIGNAL_CONFIDENCE, 42, FALLBACK_CONFIDENCE]
    assert out[0].note == "audio diarizer only (no LLM)"
    assert out[1].note == "audio diarizer only (no LLM); 60% of chunk speech is this speaker"
    assert "no diarizer labeled this chunk" in out[2].note


def test_no_llm_disagreement_falls_back_to_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only reachable when labels came from elsewhere (e.g. a sidecar); the reconciler must still not run.
    monkeypatch.setattr("polyphony.reconcile.run_structured", _no_llm_calls)
    out = reconcile([_ch(0, 0, 1)], [1], [2], names=None, model=None)
    assert (out[0].final, out[0].confidence) == (1, FALLBACK_CONFIDENCE)


def test_reconcile_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        reconcile([_ch(0, 0, 1)], [1, 2], [1], names=None, model=None)
    with pytest.raises(ValueError, match="audio_purity"):
        reconcile([_ch(0, 0, 1)], [1], [1], names=None, model=None, audio_purity=[])


def test_agreement_beats_low_purity() -> None:
    # Two independent signals agreeing outranks the audio diarizer's own mixed-chunk doubt.
    out = reconcile([_ch(0, 0, 1)], [1], [1], names=None, model=None, audio_purity=[0.4])
    assert out[0].confidence == AGREED_CONFIDENCE


def test_reconciler_only_sees_disputes_and_keeps_chunk_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(prompt: str, *_args: object, **_kwargs: object) -> _Decisions:
        seen.append(prompt)
        return _Decisions(decisions=[_Decision(id=1, speaker=1, confidence=55, reason="weak cue")])

    monkeypatch.setattr("polyphony.reconcile.run_structured", fake)
    chunks = [_ch(0, 0, 1, "Hi."), _ch(1, 1, 2, "Yeah."), _ch(2, 2, 3, "So.")]
    out = reconcile(chunks, [1, 2, 2], [1, 1, None], names=["Ana", "Ben"], model="test")

    assert [lbl.chunk.idx for lbl in out] == [0, 1, 2]
    assert [(lbl.final, lbl.confidence) for lbl in out] == [(1, 100), (1, 55), (2, 70)]
    assert len(seen) == 1
    assert "[1 DISPUTED audio=2 llm=1] Yeah." in seen[0]
    assert "Speaker 1 = Ana" in seen[0]


# ---------- the text diarizer ----------


def test_diarize_llm_without_model_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polyphony.diarize.run_structured", _no_llm_calls)
    assert diarize_llm([_ch(0, 0, 1), _ch(1, 1, 2)], None, None) == [None, None]


def test_diarize_llm_partial_answer_leaves_gaps_for_the_reconciler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polyphony.diarize.run_structured",
        _returns(_SpeakerAssignments(labels=[_SpeakerAssignment(id=0, speaker=2)])),
    )
    assert diarize_llm([_ch(0, 0, 1), _ch(1, 1, 2)], None, "test") == [2, None]


def test_diarize_llm_failure_is_silent_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> Any:
        raise RuntimeError("rate limited")

    monkeypatch.setattr("polyphony.diarize.run_structured", boom)
    assert diarize_llm([_ch(0, 0, 1)], None, "test") == [None]


# ---------- end to end ----------


def _words() -> list[Word]:
    spoken = [
        ("So", "A"), ("what", "A"), ("happened?", "A"),
        ("We", "B"), ("shipped", "B"), ("it.", "B"),
        ("Really?", "A"),
        ("Yes.", "B"),
    ]  # fmt: skip
    return [Word(text=t, start=i * 0.5, end=i * 0.5 + 0.4, speaker=s) for i, (t, s) in enumerate(spoken)]


def test_pipeline_flags_disputed_turns_and_review_clears_them(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks, audio = words_to_chunks(_words())
    assert audio == [1, 2, 1, 2]

    # The text side agrees except on "Really?", which it gives to speaker 2.
    monkeypatch.setattr(
        "polyphony.diarize.run_structured",
        _returns(_SpeakerAssignments(labels=[_SpeakerAssignment(id=i, speaker=s) for i, s in enumerate([1, 2, 2, 2])])),
    )
    monkeypatch.setattr(
        "polyphony.reconcile.run_structured",
        _returns(_Decisions(decisions=[_Decision(id=2, speaker=2, confidence=45, reason="reads as a reply")])),
    )
    names = ["Ana", "Ben"]
    labels = reconcile(chunks, audio, diarize_llm(chunks, names, "test"), names, model="test")

    markdown = build_transcript(labels, names, review_threshold=70)
    assert markdown == (
        "Ana: So what happened?\n\n"
        "<!-- review needed (min confidence 45): chunk 2: reads as a reply -->\n\n"
        "⚠️ Ben: We shipped it. Really? Yes.\n"
    )

    # The reviewer gives "Really?" back to Ana; the reviewed note has no flags left.
    payload = playground_payload(
        labels=labels,
        names=names,
        audio_name="a.m4a",
        transcript_path=Path("a.md"),
        asr_flags=[],
        paragraph_breaks=None,
        review_threshold=70,
        backend="assemblyai",
        llm_model="test",
        context_hint=None,
    )
    payload["review"] = {"overrides": {"2": 1}, "word_decisions": {}}
    reviewed, _ = apply_review(payload)
    assert reviewed == "Ana: So what happened?\n\nBen: We shipped it.\n\nAna: Really?\n\nBen: Yes.\n"


def test_pipeline_without_llm_still_produces_reviewable_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("polyphony.diarize.run_structured", _no_llm_calls)
    monkeypatch.setattr("polyphony.reconcile.run_structured", _no_llm_calls)
    chunks, audio = words_to_chunks(_words())
    labels = reconcile(chunks, audio, diarize_llm(chunks, None, None), None, model=None)

    assert [lbl.final for lbl in labels] == [1, 2, 1, 2]
    assert {lbl.confidence for lbl in labels} == {SINGLE_SIGNAL_CONFIDENCE}
    # A stricter threshold flags every single-signal turn for review.
    assert build_transcript(labels, None, review_threshold=80).count("⚠️") == 4


def test_label_and_purity_agree_on_the_majority_speaker():
    # A has the single longest segment, but B has more speech in total: B is the label, and purity is B's share.
    segments = [
        PyannoteSegment(0.0, 2.0, "A"),
        PyannoteSegment(2.0, 3.5, "B"),
        PyannoteSegment(3.5, 5.0, "B"),
    ]
    chunk = [_ch(0, 0.0, 5.0)]
    assert pyannote_per_chunk_labels(chunk, segments) == [2]
    assert pyannote_chunk_purity(chunk, segments) == [0.6]


def test_reanalysis_keeps_purity_through_the_sidecar(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("polyphony.diarize.run_structured", _no_llm_calls)
    labels = reconcile([_ch(0, 0, 1)], [1], [None], None, model=None, audio_purity=[0.5])
    payload = playground_payload(
        labels=labels,
        names=[],
        audio_name="a.m4a",
        transcript_path=Path("a.md"),
        asr_flags=[],
        paragraph_breaks=None,
        review_threshold=70,
        backend="local",
        llm_model=None,
        context_hint=None,
    )
    restored = labels_from_payload(payload)
    purity = [lbl.audio_purity for lbl in restored]
    again = reconcile([lbl.chunk for lbl in restored], [1], [None], None, model=None, audio_purity=purity)
    assert again[0].confidence == labels[0].confidence == 35
