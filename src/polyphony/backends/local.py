"""Local ensemble backend: Whisper + pyannote + Claude diarize + reconciler.

This is the strongest-correctness path — two independent diarization signals
(audio-based pyannote, text-based Claude) reconciled by a third Claude pass
that emits per-chunk confidence. No external API for transcription or
diarization; only Claude CLI calls.

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
    diarize_claude,
    diarize_pyannote,
    preflight_pyannote,
)
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
            claude_labels = diarize_claude(
                chunks,
                cfg.names,
                claude_cwd=cfg.claude_cwd,
            )

        return reconcile(
            chunks,
            pyannote_labels,
            claude_labels,
            cfg.names,
            claude_cwd=cfg.claude_cwd,
        )
