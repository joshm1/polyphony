"""Click CLI wiring the pipeline together.

Orchestration only — backend-specific transcription/diarization is delegated
to the adapters in polyphony.backends.*. This file stays focused on:
  1. reading CLI flags into a BackendConfig
  2. running the chosen backend (preflight → run)
  3. optional LLM ASR-error correction pass
  4. writing the transcript markdown + review HTML
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from loguru import logger

from .asr_correction import flag_asr_errors
from .backends import BACKENDS, BackendUnavailable, resolve_backend
from .backends.base import BackendConfig
from .export import EXPORT_FORMATS, ExportFormat, export_reviewed
from .llm import LLMUnavailable, check_llm, resolve_llm_model
from .paragraphize import paragraphize, paragraphs_for
from .serve import dump_labels_sidecar, serve_review
from .transcript import build_transcript


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context) -> None:
    """polyphony — diarization you can audit.

    Transcribe a recording into a speaker-attributed transcript with per-chunk
    confidence, review and correct it in the browser, then export the corrected data.
    """
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
    help="Comma-separated speaker names in order of first appearance (e.g. 'Host,Guest').",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "local", "gemini", "assemblyai"]),
    default="auto",
    show_default=True,
    help=(
        "Transcription+diarization strategy. 'local' = Whisper+pyannote+LLM ensemble "
        "(slow, offline, most robust). 'gemini' = one Gemini 2.5 call (fast, cheap; uses "
        "$GEMINI_API_KEY if set, else Vertex AI via ADC + $GOOGLE_CLOUD_PROJECT). "
        "'assemblyai' = AssemblyAI ASR + voice diarization cross-checked by the LLM (needs "
        "$ASSEMBLYAI_API_KEY). 'auto' picks assemblyai when $ASSEMBLYAI_API_KEY is set, else gemini."
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
    help="Short description of the recording (e.g. 'cooking podcast') — helps the backend infer domain-specific terms.",
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
    help="Skip the LLM ASR-error-flagging pass.",
)
@click.option(
    "--no-paragraphize",
    is_flag=True,
    help="Skip the LLM paragraph-splitting pass (each speaker turn renders as one wall of text).",
)
@click.option(
    "--llm-model",
    default=None,
    show_default="auto",
    help="LLM for the text-side passes (speaker cross-check, tie-breaking, paragraphs, ASR correction): "
    "a pydantic-ai `provider:model` string, a bare provider (openai, anthropic, google, openai-codex) for its "
    "default model, 'auto' (first of $OPENAI_API_KEY / $ANTHROPIC_API_KEY / $GOOGLE_API_KEY that is set), or "
    "'none' to skip every LLM pass. Defaults to $POLYPHONY_LLM_MODEL, else auto.",
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
    help="Gemini model id (default: gemini-2.5-flash), or AssemblyAI speech model (default: AssemblyAI's).",
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
    no_paragraphize: bool,
    llm_model: str | None,
    project: str | None,
    location: str | None,
    model: str | None,
    gcs_bucket: str | None,
):
    """Transcribe AUDIO_PATH and emit a transcript + `polyphony serve`-ready sidecar."""
    logger.remove()
    logger.add(sys.stderr, format="<dim>{time:HH:mm:ss}</dim> <level>{message}</level>")

    name_list = [n.strip() for n in names.split(",")] if names else None
    resolved_llm = resolve_llm_model(llm_model)
    if resolved_llm is None:
        logger.warning(
            "No LLM configured: speakers come from the audio diarizer alone, and paragraphing + ASR correction "
            "are skipped. Set $OPENAI_API_KEY / $ANTHROPIC_API_KEY / $GOOGLE_API_KEY or pass --llm-model."
        )
    else:
        logger.info(f"Text-side LLM: {resolved_llm}")
    selected_backend = resolve_backend(backend)
    logger.info(f"Using backend: {selected_backend.name} (from --backend={backend})")

    if output is None:
        stem = audio_path.with_suffix("").name
        output = audio_path.with_name(f"{stem}.{selected_backend.name}.transcript.md")

    cfg = BackendConfig(
        names=name_list,
        context_hint=context_hint,
        device=device,
        llm_model=resolved_llm,
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

    paragraph_breaks = None
    if not no_paragraphize:
        paragraph_breaks = paragraphize(
            labels,
            resolved_llm,
            context_hint=context_hint,
            source_audio=audio_path,
        )

    transcript_md = build_transcript(
        labels,
        name_list,
        review_threshold=review_threshold,
        per_turn_paragraphs=paragraphs_for(labels, paragraph_breaks) if paragraph_breaks is not None else None,
    )
    output.write_text(transcript_md)
    logger.info(f"Wrote transcript: {output} ({len(transcript_md)} chars)")

    asr_flags = []
    if not no_asr_correction:
        asr_flags = flag_asr_errors(
            [lbl.chunk for lbl in labels],
            resolved_llm,
            context_hint=context_hint,
            source_audio=audio_path,
        )

    # Dump the sidecar so `polyphony serve` can hydrate the React UI later
    # without re-running any part of the pipeline.
    sidecar_path = output.with_suffix(".polyphony.json")
    dump_labels_sidecar(
        sidecar_path,
        labels,
        asr_flags,
        name_list,
        audio_path,
        output,
        paragraph_breaks=paragraph_breaks,
        review_threshold=review_threshold,
        backend=selected_backend.name,
        llm_model=resolved_llm,
        context_hint=context_hint,
    )

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
@click.option(
    "--vault",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    envvar="POLYPHONY_VAULT",
    default=None,
    help="Notes vault (e.g. Obsidian) the UI can file this recording into. Or set $POLYPHONY_VAULT.",
)
@click.option(
    "--llm-model",
    default=None,
    help="LLM for re-analysis, summaries, and vault filing (same values as `transcribe --llm-model`). "
    "Defaults to the model recorded in the sidecar, else $POLYPHONY_LLM_MODEL, else auto.",
)
def serve_cmd(
    audio_path: Path, sidecar: Path | None, port: int, no_open: bool, vault: Path | None, llm_model: str | None
):
    """Serve the review playground over HTTP with seekable audio playback."""
    logger.remove()
    logger.add(sys.stderr, format="<dim>{time:HH:mm:ss}</dim> <level>{message}</level>")

    if sidecar is None:
        sidecar = find_sidecar(audio_path)

    model = resolve_llm_model(llm_model) if llm_model else None
    try:
        check_llm(model)
    except LLMUnavailable as e:
        raise click.ClickException(str(e)) from e
    serve_review(audio_path, sidecar, port=port, open_browser=not no_open, vault=vault, llm_model=model)


@main.command(name="export")
@click.argument(
    "source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-f",
    "--format",
    "fmt",
    type=click.Choice(EXPORT_FORMATS),
    default="json",
    show_default=True,
    help="json/csv: per-chunk rows with the audit trail (original text, both diarizers' labels, confidence, "
    "what the reviewer changed). srt/vtt: speaker-labeled captions. md: the reviewed transcript.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output path, or '-' for stdout (default: <transcript>.reviewed.<format> next to the sidecar).",
)
def export_cmd(source: Path, fmt: ExportFormat, output: Path | None) -> None:
    """Export the reviewed transcript for SOURCE (an audio file or its .polyphony.json sidecar)."""
    sidecar = source if source.name.endswith(".polyphony.json") else find_sidecar(source)
    data = json.loads(sidecar.read_text())
    rendered = export_reviewed(data, fmt)
    if output is not None and str(output) == "-":
        click.echo(rendered, nl=False)
        return
    if output is None:
        output = sidecar.with_name(sidecar.name.removesuffix(".polyphony.json") + f".reviewed.{fmt}")
    output.write_text(rendered)
    click.echo(f"Wrote {output}", err=True)


def find_sidecar(audio_path: Path) -> Path:
    """The `<audio-stem>.<backend>.transcript.polyphony.json` that `transcribe` wrote next to the audio."""
    base = audio_path.with_suffix("")
    for backend in BACKENDS:
        candidate = base.with_name(f"{base.name}.{backend}.transcript.polyphony.json")
        if candidate.exists():
            return candidate
    raise click.ClickException(
        f"No .polyphony.json sidecar found next to {audio_path.name}. "
        "Run `polyphony transcribe` first, or pass the sidecar path explicitly."
    )
