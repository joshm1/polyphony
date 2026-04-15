"""Flag likely ASR errors and suggest corrections via `claude -p`.

Runs after diarization as an independent third pass. Claude sees the full
transcript in order (so it has context like "this is a software interview")
and emits a list of suspect spans with suggested replacements + confidence.

We never mutate the transcript automatically — corrections flow through the
review playground: user accepts/rejects per span, then copies a prompt that
applies accepted corrections to the draft markdown.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from .types import Chunk


@dataclass
class WordFlag:
    chunk_idx: int
    original: str
    suggested: str
    confidence: int
    reason: str = ""
    alternatives: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


_JSON_ARRAY_RE = re.compile(r"\[\s*{.*?}\s*]|\[\s*]", re.DOTALL)


def flag_asr_errors(
    chunks: list[Chunk],
    context_hint: str | None = None,
    claude_cwd: Path | None = None,
    source_audio: Path | None = None,
) -> list[WordFlag]:
    """Ask Claude to find likely Whisper transcription errors. Returns [] on failure."""
    # Cache keyed on original audio path so iterations skip the ~30-60s Claude call.
    if source_audio is not None:
        from .cache import load_asr_flags, save_asr_flags

        cached = load_asr_flags(source_audio)
        if cached is not None:
            return cached

    if shutil.which("claude") is None:
        logger.warning("`claude` CLI missing; skipping ASR correction pass.")
        return []
    if not chunks:
        return []

    numbered = "\n".join(f"[{c.idx}] {c.text}" for c in chunks)
    context_line = (
        f"Context: {context_hint.strip()}"
        if context_hint
        else "Context: general audio — infer domain from the transcript itself."
    )

    prompt = f"""You are auditing a Whisper ASR transcript for likely transcription errors.

{context_line}

Your job: spot words/phrases that don't fit the context — sound-alike mistakes
(e.g. "SaaS" → "sauce"), homophones, misheard technical terms, common wrong-word
substitutions. Ignore correct-but-unusual proper nouns, rare jargon, or stylistic
choices. When in doubt, don't flag.

Output STRICT JSON — a single array, one object per flag (empty array if none):
  [{{"chunk_id": 42, "original": "B2B sauce", "suggested": "B2B SaaS",
    "alternatives": ["software as a service"], "confidence": 92,
    "reason": "software engineering interview context; 'sauce' doesn't fit"}}]

Rules:
- 'original' must be the exact span as it appears in the chunk, including
  capitalization and spacing. You may span multi-word phrases.
- 'suggested' is your single best correction; 'alternatives' are other
  plausible options (0-3 items, optional).
- 'confidence' is 0-100 — how sure you are this is a TRANSCRIPTION ERROR
  (not a stylistic issue). Use:
    90-100 — high confidence: context clearly rules out the transcribed word
    70-89  — confident but some ambiguity
    50-69  — plausible error, borderline
    below 50 — don't flag
- 'reason' ≤ 20 words, pointing at the specific contextual signal.
- Output ONLY the JSON array. No preamble, no markdown fences.

Chunks:
{numbered}
"""

    logger.info(
        f"ASR correction pass via `claude -p` on {len(chunks)} chunks{f' (cwd={claude_cwd})' if claude_cwd else ''}…"
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
        logger.error(f"ASR correction claude call failed (exit {e.returncode}): {e.stderr}")
        return []

    flags = _parse_flags(proc.stdout, {c.idx for c in chunks})
    if source_audio is not None:
        from .cache import save_asr_flags

        save_asr_flags(source_audio, flags)
    return flags


def _parse_flags(raw: str, valid_ids: set[int]) -> list[WordFlag]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            logger.error(f"ASR correction produced no parseable JSON:\n{raw[:500]}")
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            logger.error(f"ASR correction JSON parse failed: {e}")
            return []

    if not isinstance(data, list):
        return []

    flags: list[WordFlag] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("chunk_id")
        orig = entry.get("original")
        sugg = entry.get("suggested")
        if not (isinstance(idx, int) and idx in valid_ids and isinstance(orig, str) and isinstance(sugg, str)):
            continue
        try:
            conf = int(entry.get("confidence", 50))
        except (TypeError, ValueError):
            conf = 50
        alts = entry.get("alternatives") or []
        if not isinstance(alts, list):
            alts = []
        alts = [str(a) for a in alts if isinstance(a, (str, int, float))][:3]
        flags.append(
            WordFlag(
                chunk_idx=idx,
                original=orig,
                suggested=sugg,
                confidence=max(0, min(100, conf)),
                reason=str(entry.get("reason", "")),
                alternatives=alts,
            )
        )

    logger.info(f"ASR correction flagged {len(flags)} span(s).")
    return flags
