"""AssemblyAI backend: hosted ASR + voice diarization, cross-checked by the LLM.

AssemblyAI returns word-level text, timing, and a speaker tag in one call.
Its speaker tags fill the audio-signal slot pyannote fills in the local
backend, so the same text-side LLM diarization + reconciler run on top and
every chunk gets a real ensemble confidence — without Whisper, pyannote, or a
Hugging Face token.

Words are regrouped into sentence-sized chunks (split on speaker change or
sentence-final punctuation) so the LLM and review UI work at the same
granularity as Whisper chunks, instead of AssemblyAI's multi-minute utterances.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..cache import load_assemblyai_words, save_assemblyai_words
from ..diarize import diarize_llm
from ..llm import LLMUnavailable, check_llm
from ..reconcile import reconcile
from ..types import Chunk, ChunkLabel
from .base import Backend, BackendConfig, BackendUnavailable

API_KEY_ENV = "ASSEMBLYAI_API_KEY"
_SENTENCE_END = (".", "?", "!")


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # seconds
    end: float
    speaker: str  # opaque AssemblyAI tag, e.g. "A"


class AssemblyAIBackend(Backend):
    name = "assemblyai"

    def preflight(self, cfg: BackendConfig) -> None:
        try:
            import assemblyai  # noqa: F401
        except ImportError as e:
            raise BackendUnavailable(f"assemblyai SDK not installed: {e}") from e
        if not os.environ.get(API_KEY_ENV):
            raise BackendUnavailable(f"${API_KEY_ENV} is not set (https://www.assemblyai.com/app/api-keys).")
        try:
            check_llm(cfg.llm_model)
        except LLMUnavailable as e:
            raise BackendUnavailable(str(e)) from e

    def run(self, audio_path: Path, cfg: BackendConfig) -> list[ChunkLabel]:
        speech_model = cfg.model or "default"
        words = load_assemblyai_words(audio_path, speech_model)
        if words is None:
            words = _transcribe(audio_path, cfg)
            save_assemblyai_words(audio_path, speech_model, words)
        if not words:
            raise RuntimeError("AssemblyAI returned no words.")

        chunks, audio_labels = words_to_chunks(words)
        n_speakers = len(set(audio_labels))
        logger.info(f"AssemblyAI: {len(words)} words → {len(chunks)} chunks, {n_speakers} speaker(s).")

        llm_labels = diarize_llm(chunks, cfg.names, cfg.llm_model, expected_speakers=max(2, n_speakers))
        return reconcile(chunks, audio_labels, llm_labels, cfg.names, model=cfg.llm_model)


def _transcribe(audio_path: Path, cfg: BackendConfig) -> list[Word]:
    import assemblyai as aai

    aai.settings.api_key = os.environ[API_KEY_ENV]
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=len(cfg.names) if cfg.names else None,
        speech_models=[cfg.model] if cfg.model else None,
    )
    logger.info(f"Uploading {audio_path.name} to AssemblyAI and transcribing…")
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
    return [
        Word(text=w.text, start=w.start / 1000, end=w.end / 1000, speaker=w.speaker or "?")
        for w in transcript.words or []
    ]


def words_to_chunks(words: list[Word]) -> tuple[list[Chunk], list[int | None]]:
    """Group words into sentence-sized single-speaker chunks.

    Speaker tags map to 1-indexed ids in order of first appearance, matching
    the `--names` convention.
    """
    speaker_ids: dict[str, int] = {}
    chunks: list[Chunk] = []
    labels: list[int | None] = []
    buf: list[Word] = []

    def flush() -> None:
        if not buf:
            return
        speaker = buf[0].speaker
        speaker_ids.setdefault(speaker, len(speaker_ids) + 1)
        chunks.append(Chunk(idx=len(chunks), start=buf[0].start, end=buf[-1].end, text=" ".join(w.text for w in buf)))
        labels.append(speaker_ids[speaker])
        buf.clear()

    for w in words:
        if buf and w.speaker != buf[0].speaker:
            flush()
        buf.append(w)
        if w.text.endswith(_SENTENCE_END):
            flush()
    flush()
    return chunks, labels
