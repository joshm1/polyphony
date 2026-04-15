"""Ensemble reconciliation of two diarizations, with per-chunk 0-100 confidence.

Given pyannote-labels and claude-labels per chunk, produce a final label
per chunk plus a 0-100 confidence score. For chunks where the two backends
already agree, confidence is 100 and we skip the reconciler. For the rest,
we ask `claude -p` to choose, giving it full surrounding context + both
candidate labels, and have it emit a confidence score based on how clear
the evidence was.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from .types import Chunk, ChunkLabel


def reconcile(
    chunks: list[Chunk],
    pyannote: list[int | None],
    claude: list[int | None],
    names: list[str] | None,
    claude_cwd: Path | None = None,
) -> list[ChunkLabel]:
    """Merge two candidate diarizations into one with per-chunk confidence."""
    assert len(chunks) == len(pyannote) == len(claude), "label arrays must align with chunks"

    # Trivial cases resolved locally; reconciler only sees real disagreements.
    labels: list[ChunkLabel | None] = [None] * len(chunks)
    disputed: list[int] = []  # chunk indexes needing the reconciler

    for i, ch in enumerate(chunks):
        p, c = pyannote[i], claude[i]
        if p is not None and c is not None and p == c:
            labels[i] = ChunkLabel(ch, p, c, final=p, confidence=100, note="both backends agreed")
        elif p is None and c is None:
            # Nobody had a view — default to speaker 1 with low confidence; reconciler can adjust.
            disputed.append(i)
        elif p is None:
            labels[i] = ChunkLabel(ch, None, c, final=c, confidence=70, note="pyannote silent; used claude")
        elif c is None:
            labels[i] = ChunkLabel(ch, p, None, final=p, confidence=70, note="claude silent; used pyannote")
        else:
            disputed.append(i)

    if not disputed:
        logger.info("No disagreements; skipping reconciler.")
        return [lbl for lbl in labels if lbl is not None]  # type: ignore[misc]

    logger.info(f"Reconciler needed for {len(disputed)}/{len(chunks)} chunks.")
    decisions = _run_reconciler(chunks, pyannote, claude, names, disputed, claude_cwd)

    for i in disputed:
        ch = chunks[i]
        p, c = pyannote[i], claude[i]
        dec = decisions.get(i)
        if dec is None:
            # Fallback: prefer pyannote when both present, else whichever isn't None, else 1.
            final = p if p is not None else c if c is not None else 1
            labels[i] = ChunkLabel(
                ch,
                p,
                c,
                final=final,
                confidence=25,
                note="reconciler omitted this chunk; fell back heuristically",
            )
        else:
            labels[i] = ChunkLabel(
                ch,
                p,
                c,
                final=dec.speaker,
                confidence=max(0, min(100, dec.confidence)),
                note=dec.reason or "",
            )

    return [lbl for lbl in labels if lbl is not None]  # type: ignore[misc]


@dataclass
class _Decision:
    speaker: int
    confidence: int
    reason: str


_JSON_ARRAY_RE = re.compile(r"\[\s*{.*?}\s*]", re.DOTALL)


def _run_reconciler(
    chunks: list[Chunk],
    pyannote: list[int | None],
    claude: list[int | None],
    names: list[str] | None,
    disputed: list[int],
    claude_cwd: Path | None = None,
) -> dict[int, _Decision]:
    if shutil.which("claude") is None:
        logger.warning("`claude` CLI missing; reconciler disabled, disputes fall back heuristically.")
        return {}

    speaker_lines = []
    if names:
        for i, name in enumerate(names, start=1):
            speaker_lines.append(f"  Speaker {i} = {name}")
    else:
        speaker_lines.append("  Speakers are unnamed — use 1-indexed ids.")
    speaker_block = "\n".join(speaker_lines)

    # Present the whole transcript so the reconciler has full context —
    # surrounding agreed turns are the strongest signal for the disputes.
    lines: list[str] = []
    disputed_set = set(disputed)
    for i, ch in enumerate(chunks):
        p, c = pyannote[i], claude[i]
        if i in disputed_set:
            marker = f"DISPUTED py={p} cl={c}"
        else:
            agreed = p if p == c else p or c
            marker = f"agreed={agreed}" if agreed is not None else "agreed=?"
        lines.append(f"[{i:04d} {marker}] {ch.text}")
    full_listing = "\n".join(lines)

    prompt = f"""You are the tie-breaker in an ensemble diarization of a Whisper ASR transcript.

Two backends labeled each chunk:
  - pyannote (audio-based voice fingerprinting, good at voice changes)
  - claude   (text-based reasoning, good at role cues and semantics)

They agreed on most chunks. Your job: for each DISPUTED chunk, pick the correct
speaker id using surrounding context, and rate your own confidence 0-100.

Speakers:
{speaker_block}

Strict output: a JSON array, one entry per DISPUTED chunk id, in any order:
  [{{"id": 0042, "speaker": 1, "confidence": 85, "reason": "responding to Josh's question about architecture"}}, ...]

Confidence rubric (be honest — low confidence triggers human review):
  100 — only used for agreed chunks (not your job)
  80-99 — clear from context (role cues, names, explicit Q/A structure)
  60-79 — leaning one way but some ambiguity
  40-59 — genuinely unclear; coin flip with weak evidence
  0-39  — no real evidence; guessing

Rules:
- Reasoning in 'reason' should be short (≤15 words) and point at the specific signal.
- Output ONLY the JSON array. No preamble, no markdown fences.

Full transcript (ALL chunks, with agreed labels as context and DISPUTED chunks to resolve):
{full_listing}
"""

    logger.info(
        f"Running reconciler via `claude -p` on {len(disputed)} disputes{f' (cwd={claude_cwd})' if claude_cwd else ''}…"
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", "--permission-mode", "auto"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(claude_cwd) if claude_cwd else None,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Reconciler claude call failed (exit {e.returncode}): {e.stderr}")
        return {}

    return _parse_reconciler_json(proc.stdout)


def _parse_reconciler_json(raw: str) -> dict[int, _Decision]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            logger.error(f"Reconciler produced no parseable JSON:\n{raw[:500]}")
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Reconciler JSON parse failed: {e}")
            return {}

    if not isinstance(data, list):
        return {}

    out: dict[int, _Decision] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        spk = entry.get("speaker")
        conf = entry.get("confidence", 50)
        reason = entry.get("reason", "")
        if isinstance(idx, int) and isinstance(spk, int) and spk >= 1:
            try:
                conf_int = int(conf)
            except (TypeError, ValueError):
                conf_int = 50
            out[idx] = _Decision(speaker=spk, confidence=conf_int, reason=str(reason))
    return out
