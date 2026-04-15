"""Click CLI wiring the pipeline together.

Orchestration only — backend-specific transcription/diarization is delegated
to the adapters in polyphony.backends.*. This file stays focused on:
  1. reading CLI flags into a BackendConfig
  2. running the chosen backend (preflight → run)
  3. optional Claude ASR-error correction pass
  4. writing the transcript markdown + review HTML
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from loguru import logger

from .asr_correction import flag_asr_errors
from .backends import BackendUnavailable, resolve_backend
from .backends.base import BackendConfig
from .serve import dump_labels_sidecar, serve_review
from .transcript import build_transcript

# `claude -p` inherits the working directory's skills, so $POLYPHONY_CLAUDE_CWD
# lets you point Claude at a directory whose project-level skills you want
# loaded for diarization/ASR-correction passes. Defaults to the current dir.
DEFAULT_CLAUDE_CWD = Path(os.environ.get("POLYPHONY_CLAUDE_CWD") or Path.cwd())


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """polyphony — audio → multi-speaker transcript with review playground."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command(name="transcribe")
@click.argument(
    "audio_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Transcript output path (default: <audio-stem>.<backend>.transcript.md next to the audio).",
)
@click.option(
    "--names",
    default=None,
    help="Comma-separated speaker names in order of first appearance (e.g. 'Josh,Neville').",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "local", "gemini"]),
    default="auto",
    show_default=True,
    help=(
        "Transcription+diarization strategy. 'local' = Whisper+pyannote+Claude ensemble "
        "(slow, offline, most robust). 'gemini' = one Gemini 2.5 call on Vertex AI "
        "(fast, cheap, needs ADC). 'auto' picks gemini when available."
    ),
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "cuda:0", "mps"]),
    default="auto",
    help="Torch device for the local backend. Ignored by gemini.",
)
@click.option(
    "--context-hint",
    default=None,
    help="Short description of the recording (e.g. 'software engineering interview') — "
    "helps the backend infer domain-specific terms.",
)
@click.option(
    "--review-threshold",
    type=int,
    default=70,
    show_default=True,
    help="Turns with min-chunk confidence below this get a ⚠️ marker + review HTML entry.",
)
@click.option(
    "--no-asr-correction",
    is_flag=True,
    help="Skip the Claude ASR-error-flagging pass.",
)
@click.option(
    "--claude-cwd",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_CLAUDE_CWD,
    show_default=True,
    help="Directory to run `claude -p` from so it picks up local skills. Override via $POLYPHONY_CLAUDE_CWD.",
)
@click.option(
    "--project",
    default=None,
    help="GCP project for Gemini backend. Falls back to $GOOGLE_CLOUD_PROJECT.",
)
@click.option(
    "--location",
    default=None,
    help="GCP region for Gemini backend. Falls back to $GOOGLE_CLOUD_LOCATION then us-east1.",
)
@click.option(
    "--model",
    default=None,
    help="Gemini model id (default: gemini-2.5-flash).",
)
@click.option(
    "--gcs-bucket",
    default=lambda: os.environ.get("POLYPHONY_GCS_BUCKET"),
    help="GCS bucket for audio >15MB when using --backend gemini (or set $POLYPHONY_GCS_BUCKET).",
)
def transcribe_cmd(
    audio_path: Path,
    output: Path | None,
    names: str | None,
    backend: str,
    device: str,
    context_hint: str | None,
    review_threshold: int,
    no_asr_correction: bool,
    claude_cwd: Path,
    project: str | None,
    location: str | None,
    model: str | None,
    gcs_bucket: str | None,
):
    """Transcribe AUDIO_PATH and emit a transcript + `polyphony serve`-ready sidecar."""
    logger.remove()
    logger.add(sys.stderr, format="<dim>{time:HH:mm:ss}</dim> <level>{message}</level>")

    name_list = [n.strip() for n in names.split(",")] if names else None
    selected_backend = resolve_backend(backend)
    logger.info(f"Using backend: {selected_backend.name} (from --backend={backend})")

    if output is None:
        stem = audio_path.with_suffix("").name
        output = audio_path.with_name(f"{stem}.{selected_backend.name}.transcript.md")

    cfg = BackendConfig(
        names=name_list,
        context_hint=context_hint,
        device=device,
        claude_cwd=claude_cwd,
        project=project,
        location=location,
        model=model,
        gcs_bucket=gcs_bucket,
    )

    try:
        selected_backend.preflight(cfg)
    except BackendUnavailable as e:
        raise click.ClickException(str(e)) from e

    labels = selected_backend.run(audio_path, cfg)

    transcript_md = build_transcript(labels, name_list, review_threshold=review_threshold)
    output.write_text(transcript_md)
    logger.info(f"Wrote transcript: {output} ({len(transcript_md)} chars)")

    asr_flags = []
    if not no_asr_correction:
        asr_flags = flag_asr_errors(
            [lbl.chunk for lbl in labels],
            context_hint=context_hint,
            claude_cwd=claude_cwd,
            source_audio=audio_path,
        )

    # Dump the sidecar so `polyphony serve` can hydrate the React UI later
    # without re-running any part of the pipeline.
    sidecar_path = output.with_suffix(".polyphony.json")
    dump_labels_sidecar(sidecar_path, labels, asr_flags, name_list, audio_path, output)

    flagged = sum(1 for lbl in labels if lbl.confidence < review_threshold)
    logger.info(
        f'{flagged}/{len(labels)} chunks below conf {review_threshold}. Review: `polyphony serve "{audio_path}"`'
    )


@main.command(name="serve")
@click.argument(
    "audio_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--sidecar",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to the .polyphony.json sidecar (default: infer from audio name).",
)
@click.option("--port", type=int, default=8787, show_default=True, help="HTTP port (auto-bumps if busy).")
@click.option("--no-open", is_flag=True, help="Don't auto-open the browser.")
def serve_cmd(audio_path: Path, sidecar: Path | None, port: int, no_open: bool):
    """Serve the review playground over HTTP with seekable audio playback."""
    logger.remove()
    logger.add(sys.stderr, format="<dim>{time:HH:mm:ss}</dim> <level>{message}</level>")

    if sidecar is None:
        # Try known defaults — the transcribe command writes
        # <audio-stem>.<backend>.transcript.polyphony.json.
        base = audio_path.with_suffix("")
        for suffix in (".gemini.transcript.polyphony.json", ".local.transcript.polyphony.json"):
            candidate = base.with_name(f"{base.name}{suffix}")
            if candidate.exists():
                sidecar = candidate
                break
        if sidecar is None:
            raise click.ClickException(
                "No .polyphony.json sidecar found next to the audio. "
                "Run `polyphony transcribe` first, or pass --sidecar explicitly."
            )

    serve_review(audio_path, sidecar, port=port, open_browser=not no_open)
