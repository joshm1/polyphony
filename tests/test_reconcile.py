"""Tests for the pure-logic reconciliation + transcript-rendering paths.

LLM calls are mocked at the `run_structured` boundary; we only exercise the
local resolution rules, the post-validation filtering of LLM output, and the
paragraph round-trip validator. The real reconciler and paragraphize
behavior are integration-tested by running polyphony on real audio.
"""

from __future__ import annotations

from polyphony.asr_correction import _Flag, _to_word_flags
from polyphony.backends.assemblyai import Word, words_to_chunks
from polyphony.diarize import _labels_by_chunk, _SpeakerAssignment
from polyphony.paragraphize import _apply_breaks_to_turns, _clean_break_ids
from polyphony.reconcile import _Decision, _Decisions, _decisions_by_chunk, reconcile
from polyphony.transcript import _group_into_turns
from polyphony.types import Chunk, ChunkLabel


def _ch(idx: int, text: str = "x") -> Chunk:
    return Chunk(idx=idx, start=float(idx), end=float(idx + 1), text=text)


def test_full_agreement_skips_reconciler():
    chunks = [_ch(0), _ch(1), _ch(2)]
    out = reconcile(chunks, [1, 2, 1], [1, 2, 1], names=None, model="test")
    assert [lbl.final for lbl in out] == [1, 2, 1]
    assert all(lbl.confidence == 100 for lbl in out)
    assert all(lbl.note == "both backends agreed" for lbl in out)


def test_one_silent_backend_uses_other_with_medium_confidence():
    chunks = [_ch(0), _ch(1)]
    # audio diarizer silent on chunk 0; LLM silent on chunk 1
    out = reconcile(chunks, [None, 2], [1, None], names=None, model="test")
    assert out[0].final == 1
    assert out[0].confidence == 70
    assert "LLM" in out[0].note
    assert out[1].final == 2
    assert out[1].confidence == 70
    assert "audio" in out[1].note


def test_decisions_by_chunk_drops_non_disputed_ids():
    decisions = [
        _Decision(id=0, speaker=1, confidence=88, reason="asks question"),
        _Decision(id=3, speaker=2, confidence=60),
    ]
    out = _decisions_by_chunk(decisions, disputed={0})
    assert list(out) == [0]
    assert out[0].reason == "asks question"


def test_reconciler_failure_falls_back_heuristically(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("polyphony.reconcile.run_structured", boom)
    chunks = [_ch(0), _ch(1)]
    out = reconcile(chunks, [1, 2], [2, None], names=None, model="test")
    assert out[0].final == 1  # audio diarizer preferred when both present
    assert out[0].confidence == 25
    assert out[1].confidence == 70


def test_reconciler_applies_llm_decisions(monkeypatch):
    monkeypatch.setattr(
        "polyphony.reconcile.run_structured",
        lambda *_a, **_k: _Decisions(decisions=[_Decision(id=0, speaker=2, confidence=81, reason="answers Q")]),
    )
    out = reconcile([_ch(0)], [1], [2], names=None, model="test")
    assert (out[0].final, out[0].confidence, out[0].note) == (2, 81, "answers Q")


def test_llm_labels_drop_out_of_range_entries():
    entries = [
        _SpeakerAssignment(id=0, speaker=1),
        _SpeakerAssignment(id=1, speaker=5),  # beyond max_speaker
        _SpeakerAssignment(id=9, speaker=2),  # no such chunk
    ]
    assert _labels_by_chunk(entries, expected_len=3, max_speaker=2) == [1, None, None]


def test_asr_flags_drop_unknown_chunks():
    entries = [
        _Flag(chunk_id=4, original="sauce", suggested="SaaS", confidence=92),
        _Flag(chunk_id=99, original="x", suggested="y", confidence=90),
    ]
    flags = _to_word_flags(entries, valid_ids={4})
    assert [(f.chunk_idx, f.suggested) for f in flags] == [(4, "SaaS")]


# ---------- paragraphize._clean_break_ids + _apply_breaks_to_turns ----------


def _lbl(idx: int, final: int, text: str = "x") -> ChunkLabel:
    ch = Chunk(idx=idx, start=float(idx), end=float(idx + 1), text=text)
    return ChunkLabel(chunk=ch, audio=final, llm=final, final=final, confidence=100)


def test_clean_break_ids_filters_dedupes_and_sorts():
    assert _clean_break_ids([7, 3, 999, 7, -1, 12], valid_ids={3, 7, 12}) == [3, 7, 12]


def test_clean_break_ids_allows_no_breaks():
    # Valid output meaning "nothing needs breaking" — not a failure.
    assert _clean_break_ids([], valid_ids={0, 1}) == []


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


# ---------- backends.assemblyai.words_to_chunks ----------


def _w(text: str, speaker: str, t: float) -> Word:
    return Word(text=text, start=t, end=t + 0.5, speaker=speaker)


def test_words_to_chunks_splits_on_sentence_end_and_speaker_change():
    words = [
        _w("Hi", "B", 0), _w("there.", "B", 1), _w("How", "B", 2), _w("are", "B", 3),
        _w("Good", "A", 4), _w("thanks!", "A", 5),
    ]  # fmt: skip
    chunks, labels = words_to_chunks(words)
    assert [c.text for c in chunks] == ["Hi there.", "How are", "Good thanks!"]
    # Speaker ids follow first appearance, not AssemblyAI's letters.
    assert labels == [1, 1, 2]
    assert [c.idx for c in chunks] == [0, 1, 2]
    assert (chunks[1].start, chunks[1].end) == (2, 3.5)


# ---------- backends.resolve_backend ----------


def test_auto_prefers_assemblyai_when_key_set(monkeypatch):
    from polyphony.backends import resolve_backend

    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    assert resolve_backend("auto").name == "assemblyai"
    monkeypatch.delenv("ASSEMBLYAI_API_KEY")
    assert resolve_backend("auto").name == "gemini"
