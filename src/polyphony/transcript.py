"""Render ChunkLabels into the final markdown transcript."""

from __future__ import annotations

from .types import ChunkLabel


def display_name(speaker_id: int, names: list[str] | None) -> str:
    if names and 1 <= speaker_id <= len(names):
        return names[speaker_id - 1]
    return f"Speaker {speaker_id}"


def build_transcript(
    labels: list[ChunkLabel],
    names: list[str] | None,
    review_threshold: int = 70,
) -> str:
    """Fold labeled chunks into speaker turns, flagging low-confidence turns.

    A turn is a run of consecutive same-speaker chunks. We flag a turn with
    a ⚠️ marker if any chunk inside it scored below `review_threshold`, and
    include an HTML comment on the preceding line showing the candidates
    and reason so a reader (or follow-up Claude pass) can see the evidence.
    """
    lines: list[str] = []
    current_speaker: int | None = None
    current_parts: list[str] = []
    current_flagged = False
    current_min_conf = 100
    current_notes: list[str] = []

    def flush():
        nonlocal current_speaker, current_parts, current_flagged, current_min_conf, current_notes
        if current_speaker is None:
            return
        name = display_name(current_speaker, names)
        text = " ".join(current_parts).strip()
        prefix = "⚠️ " if current_flagged else ""
        if current_flagged:
            summary = "; ".join(n for n in current_notes if n)[:240] or "low-confidence turn"
            lines.append(f"<!-- review needed (min confidence {current_min_conf}): {summary} -->")
        lines.append(f"{prefix}{name}: {text}")
        current_speaker = None
        current_parts = []
        current_flagged = False
        current_min_conf = 100
        current_notes = []

    for lbl in labels:
        spk = lbl.effective_speaker
        if spk != current_speaker:
            flush()
            current_speaker = spk
        current_parts.append(lbl.chunk.text)
        if lbl.confidence < review_threshold:
            current_flagged = True
        current_min_conf = min(current_min_conf, lbl.confidence)
        if lbl.note:
            current_notes.append(f"chunk {lbl.chunk.idx}: {lbl.note}")

    flush()
    return "\n\n".join(lines) + "\n"
