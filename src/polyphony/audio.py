"""Audio preprocessing: normalize input to the format both Whisper and pyannote prefer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000


def convert_to_wav(src: Path, dst: Path) -> None:
    """Transcode any audio file to 16 kHz mono 16-bit PCM WAV via ffmpeg.

    Both Whisper and pyannote prefer 16 kHz mono; doing the conversion
    once upfront avoids silent resampling surprises downstream and makes
    the rest of the pipeline format-agnostic about the original input.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install it (e.g. `brew install ffmpeg`).")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def pick_device(requested: str | None) -> tuple[str, object]:
    """Return (device_string, torch_dtype) with safe defaults per platform.

    MPS (Apple Silicon) has partial fp16 support that trips Whisper in
    practice, so we stay on fp32 there. CUDA gets fp16. CPU stays fp32.
    """
    import torch  # imported lazily so the CLI is fast even without torch warm

    if requested and requested != "auto":
        device = requested
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return device, dtype
