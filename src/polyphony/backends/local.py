"""Local ensemble backend: Whisper + pyannote + LLM diarize + reconciler.

This is the strongest-correctness path — two independent diarization signals
(audio-based pyannote, text-based LLM) reconciled by a third LLM pass that
emits per-chunk confidence. Audio never leaves the machine; only transcript
text goes to the LLM.

Tradeoff: slow. Whisper on MPS ≈ 11 min for 1h audio; pyannote on CPU ≈ 30 min.
Cached on disk so iteration is fast after the first pass.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from loguru import logger

from ..audio import convert_to_wav, pick_device
from ..diarize import (
    PyannoteUnavailable,
    diarize_llm,
    diarize_pyannote,
    preflight_pyannote,
)
from ..llm import LLMUnavailable, check_llm
from ..reconcile import reconcile
from ..types import ChunkLabel
from ..whisper import transcribe
from .base import Backend, BackendConfig, BackendUnavailable


class LocalEnsembleBackend(Backend):
    name = "local"

    def preflight(self, cfg: BackendConfig) -> None:
        try:
            preflight_pyannote()
        except PyannoteUnavailable as e:
            raise BackendUnavailable(str(e)) from e
        try:
            check_llm(cfg.llm_model)
        except LLMUnavailable as e:
            raise BackendUnavailable(str(e)) from e

    def run(self, audio_path: Path, cfg: BackendConfig) -> list[ChunkLabel]:
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "audio_16k_mono.wav"
            logger.info(f"Converting {audio_path.name} → 16kHz mono wav…")
            convert_to_wav(audio_path, wav_path)

            device_str, dtype = pick_device(cfg.device)
            chunks = transcribe(wav_path, device_str, dtype, source_audio=audio_path)
            if not chunks:
                raise RuntimeError("Whisper returned no transcription.")

            pyannote_labels = diarize_pyannote(
                wav_path,
                chunks,
                device_str,
                source_audio=audio_path,
            )
            llm_labels = diarize_llm(chunks, cfg.names, cfg.llm_model)

        return reconcile(
            chunks,
            pyannote_labels,
            llm_labels,
            cfg.names,
            model=cfg.llm_model,
        )
