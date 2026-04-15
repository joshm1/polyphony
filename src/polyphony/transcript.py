"""Render ChunkLabels into the final markdown transcript.

Paragraph breaks inside a speaker turn come from `claude -p` with Haiku.
We ship the whole labeled transcript to Claude in one call; it returns a
list of paragraph strings per turn; we write those directly into the
markdown. No clever algorithm — an LLM does the paragraphing better than
any regex ever could, Haiku is fast and cheap, and this is a personal
tool so we can afford the dependency.

If the `claude` CLI isn't on PATH or the call fails, each speaker turn
renders as a single paragraph (chunks joined with spaces). That's the
legacy "wall of text" behavior; it's not pretty but it's predictable
and requires no extra dependencies.
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

PARAGRAPHIZE_MODEL = "claude-haiku-4-5-20251001"

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*]")
_WHITESPACE = re.compile(r"\s+")


def display_name(speaker_id: int, names: list[str] | None) -> str:
    if names and 1 <= speaker_id <= len(names):
        return names[speaker_id - 1]
    return f"Speaker {speaker_id}"


@dataclass
class _Turn:
    speaker: int
    chunks: list[Chunk]
    min_conf: int
    flagged: bool
    notes: list[str]


def _group_into_turns(labels: list[ChunkLabel], review_threshold: int) -> list[_Turn]:
    turns: list[_Turn] = []
    cur: _Turn | None = None
    for lbl in labels:
        spk = lbl.effective_speaker
        if cur is None or cur.speaker != spk:
            cur = _Turn(speaker=spk, chunks=[], min_conf=100, flagged=False, notes=[])
            turns.append(cur)
        cur.chunks.append(lbl.chunk)
        cur.min_conf = min(cur.min_conf, lbl.confidence)
        if lbl.confidence < review_threshold:
            cur.flagged = True
        if lbl.note:
            cur.notes.append(f"chunk {lbl.chunk.idx}: {lbl.note}")
    return turns


def _join_turn_text(chunks: list[Chunk]) -> str:
    return " ".join(ch.text.strip() for ch in chunks if ch.text.strip())


def paragraphize_via_claude(
    labels: list[ChunkLabel],
    names: list[str] | None,
    context_hint: str | None = None,
    claude_cwd: Path | None = None,
    source_audio: Path | None = None,
) -> list[list[str]] | None:
    """Ask `claude -p --model <haiku>` to split each speaker turn into paragraphs.

    Returns a list aligned with `_group_into_turns`; each inner list is
    the paragraphs for that turn. Returns None if claude isn't on PATH,
    the call fails, or the response doesn't round-trip to the input text.
    Callers should pass the result straight to `build_transcript`; if it's
    None, the markdown falls back to one-paragraph-per-turn.
    """
    turns = _group_into_turns(labels, review_threshold=100)
    if not turns:
        return []

    if source_audio is not None:
        from .cache import load_paragraphs

        cached = load_paragraphs(source_audio, len(turns))
        if cached is not None:
            return cached

    if shutil.which("claude") is None:
        logger.warning("`claude` CLI missing; skipping paragraphize pass.")
        return None

    turn_texts = [_join_turn_text(t.chunks) for t in turns]
    turn_names = [display_name(t.speaker, names) for t in turns]
    prompt = _build_prompt(turn_texts, turn_names, context_hint)

    logger.info(
        f"Paragraphizing {len(turns)} turn(s) via `claude -p` ({PARAGRAPHIZE_MODEL})"
        f"{f' (cwd={claude_cwd})' if claude_cwd else ''}…"
    )
    try:
        proc = subprocess.run(
            ["claude", "-p", "--permission-mode", "auto", "--model", PARAGRAPHIZE_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(claude_cwd) if claude_cwd else None,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Paragraphize claude call failed (exit {e.returncode}): {e.stderr}")
        return None

    result = _parse_paragraphs(proc.stdout, turn_texts)
    if result is None:
        return None

    if source_audio is not None:
        from .cache import save_paragraphs

        save_paragraphs(source_audio, result)
    return result


def _build_prompt(turn_texts: list[str], turn_names: list[str], context_hint: str | None) -> str:
    turns_json = json.dumps(
        [{"id": i, "speaker": n, "text": t} for i, (n, t) in enumerate(zip(turn_names, turn_texts, strict=True))],
        ensure_ascii=False,
    )
    context_line = (
        f"Context: {context_hint.strip()}" if context_hint else "Context: infer the domain from the transcript itself."
    )
    return f"""You are splitting a speaker-labeled transcript into readable paragraphs.

{context_line}

For each turn below, decide where paragraph breaks belong at natural topic
or breath boundaries. A short turn (1-2 sentences) is one paragraph. Long
monologues should become 3-6 sentences per paragraph, roughly 300-500
characters each.

STRICT RULES:
- Do NOT rewrite, paraphrase, clean up grammar, fix punctuation, re-case,
  or alter any words. You are ONLY deciding where to add paragraph breaks.
- When the paragraphs of a turn are joined with a single space, the result
  MUST equal the input text of that turn (whitespace-normalized). If you
  drop or add words the task fails.
- Keep the same turn order the input used; emit exactly one JSON entry
  per input turn.

Output strict JSON — a single array of objects, one per input turn, in order:
  [
    {{"id": 0, "paragraphs": ["first paragraph of turn 0.", "second paragraph."]}},
    {{"id": 1, "paragraphs": ["turn 1 as one paragraph."]}}
  ]

Output ONLY the JSON array. No preamble, no markdown fences, no commentary.

TURNS (JSON):
{turns_json}
"""


def _parse_paragraphs(raw: str, turn_texts: list[str]) -> list[list[str]] | None:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            logger.error(f"Paragraphize returned no parseable JSON:\n{raw[:500]}")
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"Paragraphize JSON parse failed: {e}")
            return None

    if not isinstance(data, list) or len(data) != len(turn_texts):
        logger.error(
            f"Paragraphize returned {len(data) if isinstance(data, list) else type(data).__name__} "
            f"entries; expected {len(turn_texts)}. Discarding."
        )
        return None

    out: list[list[str]] = []
    for entry, original in zip(data, turn_texts, strict=True):
        if not isinstance(entry, dict):
            logger.error("Paragraphize entry was not a dict; discarding.")
            return None
        paragraphs = entry.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            logger.error("Paragraphize entry had no 'paragraphs' list; discarding.")
            return None
        paragraphs = [str(p).strip() for p in paragraphs if str(p).strip()]
        if not paragraphs:
            logger.error("Paragraphize entry had only empty paragraphs; discarding.")
            return None
        joined = _WHITESPACE.sub(" ", " ".join(paragraphs)).strip()
        expected = _WHITESPACE.sub(" ", original).strip()
        if joined != expected:
            logger.error(
                "Paragraphize round-trip mismatch on a turn — Claude may have edited words; discarding this run."
            )
            return None
        out.append(paragraphs)
    logger.info(f"Paragraphize: {len(out)} turn(s) split into {sum(len(p) for p in out)} paragraph(s).")
    return out


def build_transcript(
    labels: list[ChunkLabel],
    names: list[str] | None,
    review_threshold: int = 70,
    per_turn_paragraphs: list[list[str]] | None = None,
) -> str:
    """Fold labeled chunks into speaker turns and render as markdown.

    If `per_turn_paragraphs` is provided (same length as the turn list),
    we use it verbatim. Otherwise each turn collapses to a single
    space-joined paragraph — the legacy wall-of-text shape. Call
    `paragraphize_via_claude` from the pipeline first for the good shape.
    """
    turns = _group_into_turns(labels, review_threshold=review_threshold)

    if per_turn_paragraphs is not None and len(per_turn_paragraphs) != len(turns):
        logger.warning(
            f"per_turn_paragraphs length mismatch ({len(per_turn_paragraphs)} vs {len(turns)}); "
            "using single-paragraph-per-turn fallback."
        )
        per_turn_paragraphs = None

    lines: list[str] = []
    for i, t in enumerate(turns):
        name = display_name(t.speaker, names)
        if per_turn_paragraphs is not None:
            paragraphs = per_turn_paragraphs[i]
        else:
            text = _join_turn_text(t.chunks)
            paragraphs = [text] if text else []
        prefix = "⚠️ " if t.flagged else ""
        if t.flagged:
            summary = "; ".join(n for n in t.notes if n)[:240] or "low-confidence turn"
            lines.append(f"<!-- review needed (min confidence {t.min_conf}): {summary} -->")
        if paragraphs:
            head, *tail = paragraphs
            lines.append(f"{prefix}{name}: {head}")
            lines.extend(tail)

    return "\n\n".join(lines) + "\n"
