# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

polyphony: audio → speaker-attributed transcript (`.md`) + review sidecar (`.polyphony.json`) + a React review UI served by `polyphony serve`. See README.md for user-facing usage, env vars, and the pipeline diagram.

## Commands

Python (uv, Python 3.12):

```bash
uv sync                                   # install
uv run pytest                             # all tests
uv run pytest tests/test_reconcile.py::test_name   # single test
uv run ruff check src tests               # lint (CI)
uv run ruff format --check src tests      # format check (CI)
uv run polyphony transcribe examples/demo.m4a --names "Sam,Daniel" --backend local
uv run polyphony serve examples/demo.m4a  # review UI on :8787 (auto-bumps port)
```

Frontend (`frontend/`, bun + Vite + React 18 + Biome):

```bash
cd frontend
bun install
bun run typecheck    # tsc --noEmit
bun run lint         # biome check src (CI)
bun run format       # biome format --write src
bun run build        # → ../src/polyphony/static/ (committed)
bun run dev          # vite dev server; proxies /api → localhost:8787, so run `polyphony serve` alongside
```

Pre-commit: `uvx pre-commit run --all-files` (ruff, tsc, biome, frontend build).

## Architecture

**Backend adapter seam.** `backends/base.py` defines `Backend` (`preflight` + `run`) returning `list[ChunkLabel]`. Everything downstream is backend-agnostic. `backends/__init__.py` registers backends; `auto` prefers AssemblyAI when `$ASSEMBLYAI_API_KEY` is set, else Gemini whenever `google.genai` imports. `BackendConfig` is one shared bag of knobs — each backend ignores the ones it doesn't use. New backend = subclass, implement both methods, register in `BACKENDS`, add to the `--backend` click choice in `cli.py`.

- **local** (`backends/local.py`): ffmpeg → 16 kHz wav → Whisper (`whisper.py`) → two independent diarizations: pyannote (`diarize.py`, audio) and the LLM (`diarize.py::diarize_llm`, text) → `reconcile.py`. Agreements get confidence 100; one-side-silent gets 70; only disputes go to a reconciler LLM call.
- **assemblyai** (`backends/assemblyai.py`): AssemblyAI word-level ASR + speaker tags (cached) → sentence-sized chunks → the same `diarize_llm` + `reconcile` ensemble as local, with AssemblyAI's speakers in the `audio` slot. Needs `$ASSEMBLYAI_API_KEY`.
- **gemini** (`backends/gemini.py`): single multimodal call returning JSON turns, converted to `ChunkLabel`s (`audio` is `None`; Gemini's speaker fills `llm`). Oversized audio is Opus-compressed, then GCS upload as a last resort (Vertex only).

**Post-backend stages** (orchestrated in `cli.py::transcribe_cmd`): `paragraphize.py` (LLM picks break-after chunk IDs, stored in the sidecar; never re-emits text, falls back to one paragraph per turn on any failure) → `transcript.py` renders markdown with ⚠️ on turns below `--review-threshold` → `asr_correction.py` (LLM flags likely Whisper mis-hears) → `serve.dump_labels_sidecar` writes the sidecar.

**LLM calls.** Text-only stages (text diarization, reconciler, paragraphize, ASR correction) go through `llm.py::run_structured` — a pydantic-ai `Agent` with a Pydantic `output_type`, so there's no JSON scraping. Model = `--llm-model` / `$POLYPHONY_LLM_MODEL`, default `openai-codex:gpt-6-sol` (ChatGPT subscription via `codex login` credentials in `~/.codex/auth.json`). The Codex backend rejects audio input, so it can't replace Whisper or the Gemini backend. Local backend preflight calls `check_llm`; at runtime the stages degrade gracefully (log and fall back) on any LLM error. The Gemini backend's client comes from `gemini_client.make_gemini_client`: `$GEMINI_API_KEY`/`$GOOGLE_API_KEY` wins over Vertex ADC.

**Cache** (`cache.py`): per-stage JSON under `~/.cache/polyphony/<key>/`, key = sha256(abs path + stage/model id + mtime + size). Stage ids like `asr-correction-v2` / `paragraphize-v3` are embedded in the key — bump the version string when changing a stage's prompt or output shape so stale caches invalidate.

**Frontend ↔ Python contract.** `playground.py::playground_payload` is the single source of truth for the `/api/data` JSON; keep it in sync with `frontend/src/types.ts::PolyphonyData`. `serve.py` is FastAPI serving `static/index.html`, `/assets/*`, `/api/data`, `/api/audio` (HTTP Range for seeking), `POST /api/apply`, which persists review decisions + speaker names into the sidecar and writes `<stem>.transcript.reviewed.md` via `review.py` (deterministic re-render from chunks; the raw transcript is never modified), and `POST /api/reanalyze` (`reanalyze.py`), which reruns the text-side LLM passes on the stored chunks with new names/context hint — `speaker_names.py` maps unassigned candidate names to speaker ids — and carries review state over. The sidecar records `backend`, `llm_model`, and `context_hint` so re-analysis matches the original run. With `--vault`, `POST /api/vault/propose` lets the LLM pick a folder + name via read-only, vault-confined tools (`filing.py`), and `POST /api/vault/move` applies the review then moves the audio + every `<stem>.*.transcript*` sibling, rebinding the server to the new paths. `serve.py` intentionally omits `from __future__ import annotations` — FastAPI needs real annotations for DI.

## Gotchas

- The built bundle in `src/polyphony/static/` is committed and shipped in the wheel. Any `frontend/` change must be rebuilt and committed; CI fails on drift. Vite emits fixed filenames (`assets/main.js`, no hashes) on purpose.
- Tests cover only pure logic (`reconcile`, label parsing, `pyannote_per_chunk_labels`); Whisper/pyannote/LLM/Gemini paths are verified by running on real audio (e.g. `examples/demo.m4a`).
- Ruff: line length 120, rules `E,F,I,UP,B,SIM`.
- Local backend needs `ffmpeg`, `HF_TOKEN` with both pyannote licenses accepted, and working credentials for `--llm-model` (`codex login` for the default).
