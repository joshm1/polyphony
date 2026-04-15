"""Gemini 2.5 (Vertex AI) backend: one multimodal call does everything.

Sends the audio + a structured prompt to Gemini and receives JSON-shaped
speaker turns back. Gemini handles ASR and speaker attribution in one shot,
using content context (role cues, names, conversational flow) as well as
voice features. Typically cheaper and faster than the local ensemble,
especially for interview-style audio.

Uploads large files to GCS (anything above ~15 MB) because Vertex inline-data
limits apply per request. Auth reuses Application Default Credentials.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..types import Chunk, ChunkLabel
from .base import Backend, BackendConfig, BackendUnavailable

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-east1"
INLINE_BYTE_LIMIT = 15 * 1024 * 1024
# Opus at this bitrate preserves speech fidelity well past what ASR needs while
# giving roughly 3x compression vs 60kbps AAC — a 60min m4a at ~30MB shrinks
# to ~10MB, comfortably under the inline cap.
COMPRESS_BITRATE = "24k"

_JSON_ARRAY_RE = re.compile(r"\[\s*{.*?}\s*]|\[\s*]", re.DOTALL)


@dataclass
class _Turn:
    speaker: int
    text: str
    start_sec: float | None
    end_sec: float | None
    confidence: int


class GeminiBackend(Backend):
    name = "gemini"

    def preflight(self, cfg: BackendConfig) -> None:
        # Cheap checks only — ADC isn't verified here because a bad-credential
        # failure surfaces quickly at the first API call anyway, and actively
        # probing ADC costs a round-trip we don't need.
        try:
            import google.genai  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"google-genai not installed: {e}") from e

    def run(self, audio_path: Path, cfg: BackendConfig) -> list[ChunkLabel]:
        from google import genai
        from google.genai import types

        project = cfg.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise BackendUnavailable(
                "No GCP project set. Pass --project or export "
                "$GOOGLE_CLOUD_PROJECT (a project with the Vertex AI API enabled "
                "and Application Default Credentials configured via `gcloud auth "
                "application-default login`)."
            )
        location = cfg.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION
        model = cfg.model or DEFAULT_MODEL

        client = genai.Client(vertexai=True, project=project, location=location)

        # Compress-if-needed lives inside a temp dir so the reduced file
        # auto-cleans. Keep the full Gemini call inside the `with` so any
        # future change that needs the local file (streaming, retries) still
        # sees it.
        with tempfile.TemporaryDirectory(prefix="polyphony_gemini_") as td:
            audio_part = self._audio_part(
                audio_path,
                project,
                cfg.gcs_bucket,
                types,
                Path(td),
            )

            prompt = self._build_prompt(cfg.names, cfg.context_hint)
            # Use Vertex's response_schema to _constrain_ the output shape
            # instead of relying on prompt discipline. This eliminates the
            # "Gemini made up a JSON syntax" class of parse errors we kept
            # hitting on long responses.
            #
            # max_output_tokens must also accommodate an hour of speech as
            # JSON — roughly 9k words + overhead ≈ 30-40k tokens. The
            # default of ~8k silently truncates long transcripts.
            turn_schema = {
                "type": "OBJECT",
                "properties": {
                    "speaker": {"type": "INTEGER"},
                    "text": {"type": "STRING"},
                    "start_sec": {"type": "NUMBER"},
                    "end_sec": {"type": "NUMBER"},
                    "confidence": {"type": "INTEGER"},
                },
                "required": ["speaker", "text"],
            }
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema={"type": "ARRAY", "items": turn_schema},
                max_output_tokens=65536,
            )
            contents = [
                types.Content(
                    role="user",
                    parts=[audio_part, types.Part.from_text(text=prompt)],
                )
            ]

            logger.info(f"Calling Gemini {model} on Vertex (project={project}, location={location})…")
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        raw = (response.text or "").strip()
        if not raw:
            raise RuntimeError("Gemini returned an empty response.")

        try:
            turns = self._parse_turns(raw)
        except Exception:
            # Dump the raw payload so we can actually inspect whatever
            # surprise Gemini produced — otherwise the error is useless
            # because the first 500 chars rarely show the problem.
            dump_path = Path(tempfile.gettempdir()) / "polyphony_gemini_last_response.json"
            dump_path.write_text(raw)
            logger.error(f"Failed to parse Gemini response; raw output dumped to {dump_path}")
            raise
        logger.info(f"Gemini returned {len(turns)} turn(s).")
        return self._turns_to_labels(turns)

    # -------- helpers --------

    def _audio_part(
        self,
        audio_path: Path,
        project: str,
        bucket: str | None,
        types,
        tmpdir: Path,
    ):
        """Return a Part for the audio, compressing first if it won't fit inline.

        Strategy: always prefer the inline path (no network upload, no GCS
        dependency). If the original file is over the inline cap, transcode
        to a low-bitrate Opus mono copy — speech is extremely compressible
        and the ASR quality bar is well below audiophile. Only fall back to
        a GCS upload if an explicit bucket is configured AND compression
        still can't shrink the file under the cap.
        """
        size = audio_path.stat().st_size
        mime = _guess_mime(audio_path)

        if size <= INLINE_BYTE_LIMIT:
            logger.info(f"Sending {audio_path.name} inline ({size / 1024 / 1024:.1f} MB).")
            return types.Part.from_bytes(mime_type=mime, data=audio_path.read_bytes())

        # Too big to send raw — transcode to Opus 24kbps mono and try again.
        compressed = tmpdir / "audio_compressed.ogg"
        _compress_audio(audio_path, compressed)
        csize = compressed.stat().st_size
        ratio = size / csize if csize else 1
        logger.info(
            f"Compressed {audio_path.name}: {size / 1024 / 1024:.1f} MB → {csize / 1024 / 1024:.1f} MB ({ratio:.1f}x)."
        )
        if csize <= INLINE_BYTE_LIMIT:
            return types.Part.from_bytes(
                mime_type="audio/ogg",
                data=compressed.read_bytes(),
            )

        # Compression wasn't enough (very long audio) — last-resort GCS path.
        if not bucket:
            raise RuntimeError(
                f"Audio is {size / 1024 / 1024:.1f} MB and even after compression "
                f"to Opus 24kbps mono it's {csize / 1024 / 1024:.1f} MB, still "
                f"> {INLINE_BYTE_LIMIT // 1024 // 1024} MB inline cap. For audio this "
                "long, either chunk it locally or provide --gcs-bucket / "
                "$POLYPHONY_GCS_BUCKET so polyphony can upload."
            )
        uri = _upload_to_gcs(compressed, bucket, project)
        return types.Part.from_uri(file_uri=uri, mime_type="audio/ogg")

    def _build_prompt(self, names: list[str] | None, context_hint: str | None) -> str:
        speaker_lines = []
        if names:
            for i, name in enumerate(names, start=1):
                speaker_lines.append(f"  Speaker {i} = {name}")
        else:
            speaker_lines.append("  Unknown speakers — number them 1, 2, … in order of first appearance.")
        speaker_block = "\n".join(speaker_lines)
        context_line = (
            f"Context: {context_hint.strip()}" if context_hint else "Context: infer from the audio content itself."
        )
        # Prompt is now shape-agnostic — the response_schema on the API call
        # forces Gemini to emit a JSON array of objects. We just focus the
        # model on the *task* (what to transcribe, how to pick a speaker,
        # how to self-rate confidence).
        return f"""Transcribe and diarize the attached audio. Emit each speaker turn in chronological order.

{context_line}

Speakers:
{speaker_block}

Per-turn fields:
- speaker (1-indexed int matching the list above)
- text: VERBATIM what was said — include disfluencies, don't summarize / clean up / correct
- start_sec / end_sec: approximate audio timestamps (best effort)
- confidence (0-100): YOUR OWN certainty about the speaker assignment
    90-100 — clearly that speaker (voice + context both agree)
    60-89  — confident but some ambiguity
    below 60 — genuinely uncertain; mark honestly so human review catches them

A "turn" is a continuous utterance by one speaker. Short interjections ("right",
"yeah") get their own turn.
"""

    def _parse_turns(self, raw: str) -> list[_Turn]:
        """Parse Gemini's output — prefers positional arrays, tolerates dict form.

        Positional: [[speaker, text, start, end, confidence], ...]
        Dict (legacy): [{"speaker": 1, "text": "...", ...}, ...]
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            m = _JSON_ARRAY_RE.search(raw)
            if not m:
                raise RuntimeError(f"Gemini output had no parseable JSON array:\n{raw[:500]}") from e
            data = json.loads(m.group(0))

        if not isinstance(data, list):
            raise RuntimeError(f"Gemini output was not a JSON array: {type(data).__name__}")

        turns: list[_Turn] = []
        for entry in data:
            turn = _turn_from_entry(entry)
            if turn is not None:
                turns.append(turn)
        return turns

    def _turns_to_labels(self, turns: list[_Turn]) -> list[ChunkLabel]:
        """Adapt Gemini's turn-structured output into the ChunkLabel shape the rest of polyphony expects."""
        labels: list[ChunkLabel] = []
        for i, t in enumerate(turns):
            chunk = Chunk(
                idx=i,
                start=t.start_sec if t.start_sec is not None else 0.0,
                end=t.end_sec if t.end_sec is not None else 0.0,
                text=t.text.strip(),
            )
            labels.append(
                ChunkLabel(
                    chunk=chunk,
                    pyannote=None,  # not applicable
                    claude=t.speaker,  # Gemini fulfilling the text-signal role
                    final=t.speaker,
                    confidence=t.confidence,
                    note="gemini single-call",
                )
            )
        return labels


def _turn_from_entry(entry) -> _Turn | None:
    """Coerce a single Gemini output entry into a _Turn. Accepts positional arrays and dicts."""
    if isinstance(entry, list) and len(entry) >= 2:
        # positional: [speaker, text, start?, end?, confidence?]
        spk = entry[0]
        text = entry[1]
        start = entry[2] if len(entry) > 2 else None
        end = entry[3] if len(entry) > 3 else None
        conf = entry[4] if len(entry) > 4 else 80
    elif isinstance(entry, dict):
        spk = entry.get("speaker")
        text = entry.get("text")
        start = entry.get("start_sec")
        end = entry.get("end_sec")
        conf = entry.get("confidence", 80)
    else:
        return None

    if not (isinstance(spk, int) and spk >= 1 and isinstance(text, str)):
        return None
    try:
        conf_int = int(conf)
    except (TypeError, ValueError):
        conf_int = 80
    return _Turn(
        speaker=spk,
        text=text,
        start_sec=float(start) if isinstance(start, (int, float)) and start else None,
        end_sec=float(end) if isinstance(end, (int, float)) and end else None,
        confidence=max(0, min(100, conf_int)),
    )


def _guess_mime(audio_path: Path) -> str:
    mime, _ = mimetypes.guess_type(audio_path.name)
    if mime:
        return mime
    ext = audio_path.suffix.lower()
    return {
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(ext, "audio/mpeg")


def _upload_to_gcs(audio_path: Path, bucket: str, project: str) -> str:
    if shutil.which("gcloud") is None:
        raise RuntimeError("gcloud CLI missing; needed to upload large audio to GCS for Vertex.")
    dest = f"gs://{bucket}/polyphony/{audio_path.name}"
    logger.info(f"Uploading {audio_path.name} → {dest} (via gcloud storage)…")
    subprocess.run(
        ["gcloud", "storage", "cp", "--project", project, str(audio_path), dest],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return dest


def _compress_audio(src: Path, dst: Path) -> None:
    """Transcode src → dst as Opus mono at COMPRESS_BITRATE.

    Speech preserves well at these rates — 16 kHz sample rate matches what
    most ASR pipelines already downsample to, and Opus is purpose-built for
    speech efficiency. We apply ffmpeg VBR but clamp to a target bitrate so
    the output size is predictable.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; needed to compress audio for inline upload.")
    logger.info(f"Compressing {src.name} → Opus {COMPRESS_BITRATE} mono…")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libopus",
            "-b:a",
            COMPRESS_BITRATE,
            "-application",
            "voip",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
