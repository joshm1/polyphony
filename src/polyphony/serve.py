"""FastAPI server for the polyphony review playground.

Serves a Vite-built React bundle (committed under `polyphony/static/`) and a
small JSON API the app fetches on mount. Split by path prefix:

  GET /                    → static index.html (the React entrypoint)
  GET /assets/*            → bundled JS/CSS/source-maps
  GET /api/data            → the review payload (ChunkLabels + ASR flags)
  GET /api/audio           → the source audio with HTTP Range support

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

import click
from loguru import logger

from .asr_correction import WordFlag
from .cache import load_asr_flags
from .playground import playground_payload
from .types import ChunkLabel

# The built frontend ships as a package resource. We resolve the path at
# import time so FastAPI's StaticFiles can mount the directory.
STATIC_DIR = Path(str(files(__package__) / "static"))


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

    app = FastAPI(title=f"polyphony review — {audio_path.name}")

    # /assets/* → built bundles (JS, CSS, source maps, static media).
    app.mount(
        "/assets",
        StaticFiles(directory=str(STATIC_DIR / "assets")),
        name="assets",
    )

    @app.get("/api/data")
    def api_data() -> JSONResponse:
        return JSONResponse(data)

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
) -> None:
    """Persist the review payload next to the transcript for `polyphony serve` to consume later."""
    data = playground_payload(
        labels=labels,
        names=names or [],
        audio_name=audio_path.name,
        transcript_path=transcript_path,
        asr_flags=asr_flags,
    )
    path.write_text(json.dumps(data, indent=2))
    logger.info(f"Wrote review sidecar: {path}")


def load_asr_flags_or_empty(audio_path: Path) -> list[WordFlag]:
    return load_asr_flags(audio_path) or []
