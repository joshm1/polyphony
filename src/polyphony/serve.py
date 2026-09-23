"""FastAPI server for the polyphony review playground.

Serves a Vite-built React bundle (committed under `polyphony/static/`) and a
small JSON API the app fetches on mount. Split by path prefix:

  GET /                    → static index.html (the React entrypoint)
  GET /assets/*            → bundled JS/CSS/source-maps
  GET /api/data            → the review payload (ChunkLabels + ASR flags)
  GET /api/audio           → the source audio with HTTP Range support
  POST /api/apply          → persist review decisions, write the reviewed transcript
  POST /api/reanalyze      → rerun the LLM passes with new names / context hint
  POST /api/vault/propose  → LLM-suggested vault folder + name for this recording
  POST /api/vault/move     → apply the review, then move all files into the vault

The React bundle has no knowledge of server-side substitution: it just
does `fetch('/api/data')` on startup. That keeps the built HTML fully
static and lets us swap in new audio without rebuilding the frontend.
"""

# NOTE: no `from __future__ import annotations` — FastAPI inspects
# `Request` type annotations at runtime to wire up DI, and PEP 563
# stringified annotations defeat that.

import json
import socket
from importlib.resources import files
from pathlib import Path
from typing import Literal

import click
from loguru import logger
from pydantic import BaseModel

from .asr_correction import WordFlag
from .filing import move_recording, obsidian_url, planned_moves, propose_location
from .llm import DEFAULT_LLM_MODEL
from .playground import playground_payload
from .reanalyze import reanalyze
from .review import apply_review, reviewed_path
from .types import ChunkLabel

# The built frontend ships as a package resource. We resolve the path at
# import time so FastAPI's StaticFiles can mount the directory.
STATIC_DIR = Path(str(files(__package__) / "static"))


class ReviewFlag(BaseModel):
    chunk_idx: int
    original: str
    suggested: str
    alternatives: list[str] = []
    confidence: int
    reason: str = ""
    userAdded: bool | None = None  # noqa: N815 — mirrors the frontend's WordFlag field


class WordDecision(BaseModel):
    kind: Literal["suggested", "alternative", "custom", "original"]
    value: str


class ApplyRequest(BaseModel):
    overrides: dict[int, int]  # chunk idx → speaker id
    asr_flags: list[ReviewFlag]  # includes reviewer-added flags
    word_decisions: dict[int, WordDecision]  # asr_flags index → decision
    names: list[str] | None = None  # indexed by speaker id - 1; "" = unnamed; None = keep current


class VaultMoveRequest(ApplyRequest):
    folder: str  # relative to the vault root
    basename: str


class ReanalyzeRequest(ApplyRequest):
    names: list[str]
    candidate_names: list[str]  # any order; the LLM places them on unnamed speakers
    context_hint: str | None


def _find_free_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in {start}..{start + 50}")


def _get_tailscale_ip() -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(
            ["tailscale", "ip", "-4"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip().splitlines()[0] if out.strip() else None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def serve_review(
    audio_path: Path,
    labels_path: Path,
    port: int = 8787,
    open_browser: bool = True,
    vault: Path | None = None,
) -> None:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    if not audio_path.exists():
        raise click.ClickException(f"Audio file not found: {audio_path}")
    if not labels_path.exists():
        raise click.ClickException(f"Labels sidecar not found: {labels_path}")
    if not STATIC_DIR.exists() or not (STATIC_DIR / "index.html").exists():
        raise click.ClickException(
            f"Frontend bundle missing at {STATIC_DIR}. Run `cd frontend && bun run build` to regenerate it."
        )

    # Load the sidecar once; we serve it back verbatim plus the audio_url
    # the client needs to point its <audio> element at.
    data = json.loads(labels_path.read_text())
    data["audio_url"] = "/api/audio"
    # Sidecars written before these fields existed.
    data.setdefault("llm_model", DEFAULT_LLM_MODEL)
    data.setdefault("context_hint", None)
    data.setdefault("review", {"overrides": {}, "word_decisions": {}})

    app = FastAPI(title=f"polyphony review — {audio_path.name}")

    # /assets/* → built bundles (JS, CSS, source maps, static media).
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="assets",
    )

    @app.get("/api/data")
    def api_data() -> JSONResponse:
        return JSONResponse({**data, "vault": str(vault) if vault else None})

    @app.get("/api/audio")
    def api_audio(request: Request):
        """Stream the audio file with HTTP Range support so <audio> can seek."""
        file_size = audio_path.stat().st_size
        range_header = request.headers.get("range")
        mime = _guess_audio_mime(audio_path)

        if not range_header:
            return StreamingResponse(
                _file_iter(audio_path, 0, file_size - 1),
                media_type=mime,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(file_size),
                },
            )

        try:
            units, _, ranges = range_header.partition("=")
            if units.strip().lower() != "bytes":
                raise ValueError
            start_s, _, end_s = ranges.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
        except ValueError as e:
            raise HTTPException(status_code=416, detail="malformed Range header") from e
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Range out of bounds")

        length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        return StreamingResponse(
            _file_iter(audio_path, start, end),
            status_code=206,
            media_type=mime,
            headers=headers,
        )

    def take_review_state(req: ApplyRequest) -> None:
        if req.names is not None:
            data["names"] = req.names
        data["asr_flags"] = [f.model_dump(exclude_none=True) for f in req.asr_flags]
        data["review"] = {
            "overrides": {str(k): v for k, v in req.overrides.items()},
            "word_decisions": {str(k): d.model_dump() for k, d in req.word_decisions.items()},
        }

    def write_outputs() -> tuple[Path, list[dict]]:
        markdown, skipped = apply_review(data)
        out = reviewed_path(labels_path)
        out.write_text(markdown)
        labels_path.write_text(json.dumps({k: v for k, v in data.items() if k not in ("audio_url", "vault")}, indent=2))
        logger.info(f"Applied review → {out} ({len(skipped)} correction(s) skipped)")
        return out, skipped

    @app.post("/api/apply")
    def api_apply(req: ApplyRequest) -> JSONResponse:
        take_review_state(req)
        out, skipped = write_outputs()
        return JSONResponse({"reviewed_path": str(out), "skipped": skipped})

    # Sync handler on purpose: FastAPI runs it in a worker thread, so the
    # minute-long LLM calls don't block audio streaming.
    @app.post("/api/reanalyze")
    def api_reanalyze(req: ReanalyzeRequest) -> JSONResponse:
        take_review_state(req)
        data.update(reanalyze(data, audio_path, req.names, req.candidate_names, req.context_hint))
        out, skipped = write_outputs()
        return JSONResponse({"reviewed_path": str(out), "skipped": skipped, "data": data})

    def require_vault() -> Path:
        if vault is None:
            raise HTTPException(status_code=400, detail="No vault configured: pass --vault or set $POLYPHONY_VAULT.")
        return vault

    # Sync handler: the LLM browses the vault via tools, which takes a while.
    @app.post("/api/vault/propose")
    def api_vault_propose(req: ApplyRequest) -> JSONResponse:
        root = require_vault()
        take_review_state(req)
        markdown, _ = apply_review(data)
        proposal = propose_location(
            root, audio_path, markdown, data.get("names") or [], data.get("context_hint"), data["llm_model"]
        )
        return JSONResponse(
            {**proposal.model_dump(), "files": [str(src.name) for src, _ in planned_moves(audio_path, root, "x")]}
        )

    @app.post("/api/vault/move")
    def api_vault_move(req: VaultMoveRequest) -> JSONResponse:
        nonlocal audio_path, labels_path
        root = require_vault()
        if labels_path.resolve() not in {src for src, _ in planned_moves(audio_path.resolve(), root, "x")}:
            detail = f"Sidecar {labels_path} isn't next to the audio; move it there first."
            raise HTTPException(status_code=409, detail=detail)
        take_review_state(req)
        write_outputs()
        try:
            moves = move_recording(audio_path, root, req.folder, req.basename)
        except (ValueError, FileExistsError) as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        renamed = {src.name: dst for src, dst in moves}
        audio_path = moves[0][1]
        labels_path = renamed[labels_path.name]
        # The moved sidecar was rewritten with new paths; keep serving from it.
        data.update(json.loads(labels_path.read_text()))
        note = reviewed_path(labels_path)
        return JSONResponse(
            {
                "moved": [str(dst) for _, dst in moves],
                "note_path": str(note),
                "obsidian_url": obsidian_url(root, note),
            }
        )

    # Everything else → the React entrypoint. Declared last so explicit
    # routes (/assets/*, /api/*) win first.
    @app.get("/{_path:path}")
    def spa(_path: str) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    port = _find_free_port(port)
    tailscale_ip = _get_tailscale_ip()

    logger.info(f"Review server: http://localhost:{port}/")
    if tailscale_ip:
        logger.info(f"Tailscale:     http://{tailscale_ip}:{port}/")
    logger.info("Press Ctrl-C to stop.")

    if open_browser:
        import webbrowser

        webbrowser.open(f"http://localhost:{port}/")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def _file_iter(path: Path, start: int, end: int, chunk_size: int = 1024 * 256):
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def _guess_audio_mime(path: Path) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    return {
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "audio/mpeg")


def dump_labels_sidecar(
    path: Path,
    labels: list[ChunkLabel],
    asr_flags: list[WordFlag],
    names: list[str] | None,
    audio_path: Path,
    transcript_path: Path,
    paragraph_breaks: list[int] | None,
    review_threshold: int,
    backend: str,
    llm_model: str,
    context_hint: str | None,
) -> None:
    """Persist the review payload next to the transcript for `polyphony serve` to consume later."""
    data = playground_payload(
        labels=labels,
        names=names or [],
        audio_name=audio_path.name,
        transcript_path=transcript_path,
        asr_flags=asr_flags,
        paragraph_breaks=paragraph_breaks,
        review_threshold=review_threshold,
        backend=backend,
        llm_model=llm_model,
        context_hint=context_hint,
    )
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Wrote review sidecar: {path}")
