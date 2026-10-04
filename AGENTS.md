# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

polyphony: audio → speaker-attributed transcript (`.md`) + review sidecar (`.polyphony.json`) + a React review UI served by `polyphony serve`. See README.md for user-facing usage, env vars, and the pipeline diagram.

## Commands

Python (uv, Python 3.12):

```bash
uv sync --all-extras                      # install, including the `local` extra (torch, pyannote, transformers)
uv run pytest                             # all tests
uv run pyright                            # strict type check (CI)
uv run pytest tests/test_reconcile.py::test_name   # single test
uv run ruff check src tests               # lint (CI)
uv run ruff format --check src tests      # format check (CI)
uv run polyphony transcribe examples/demo.m4a --names "Sam,Daniel" --backend local
uv run polyphony serve examples/demo.m4a  # review UI on :8787 (auto-bumps port)
uv run polyphony export examples/demo.m4a -f csv -o -   # reviewed data as json/csv/srt/vtt/md
```

Frontend (`frontend/`, bun + Vite + React 18 + oxlint + oxfmt):

```bash
cd frontend
bun install
bun run check:lint   # oxlint --type-aware (CI)
bun run check:fmt    # oxfmt --check (CI)
bun run check:tsc    # tsc over tsconfig.json (src) and tsconfig.node.json (vite.config.ts) (CI)
bun run fix          # oxfmt --write + oxlint --fix
bun run build        # → ../src/polyphony/static/ (committed)
bun run dev          # vite dev server; proxies /api → localhost:8787, so run `polyphony serve` alongside
```

Pre-commit: `uvx pre-commit run --all-files` (ruff, tsc, oxlint, oxfmt, frontend build).

## TypeScript standards

Applies to `frontend/`. `tsconfig.base.json`, `.oxlintrc.json` and `.oxfmtrc.json` are copied from the shared standards in joshm1/apps; keep them in step with that source rather than tuning them here.

- `tsconfig.json` and `tsconfig.node.json` extend `tsconfig.base.json` (strict, `noUncheckedIndexedAccess`, `exactOptionalPropertyTypes`, `noImplicitOverride`, `verbatimModuleSyntax`, `erasableSyntaxOnly`, `allowJs: false`). They add only `include` and `types`; never change a base flag or switch off a check `strict` enables. `check:tsc` type-checks every tsconfig.
- No `any`, `@ts-ignore`, `@ts-nocheck`, `as unknown as`, or non-null `!` outside tests, and no lint-disable comments. Narrow with checks or a schema instead of asserting with `as`.
- Parse with Zod at every boundary (`fetch` responses, storage, URL params, `postMessage`), then trust the types inside. Server payloads use plain `z.object` (they gain fields over time); schemas live in `src/types.ts` next to the types they produce, which mirror `playground.py::playground_payload`. Never hand-roll `typeof`/`in` guards on external data.
- Model variants as discriminated unions and end exhaustive switches with `assertNever`. Shared helpers (`assertNever`, `Result`, branded ids) live in `src/lib/`, mirroring joshm1/apps `packages/types`; add one only when code needs it.
- Return a `Result` for expected failures the caller must handle; throw for unexpected ones, and throw only `Error`s.
- Use top-level `import type` for type-only imports.
- Production code never imports test doubles or fixtures (enforced by lint). Do not write tests that restate what the type checker already proves.
- Guidance no tool enforces yet: prefer named exports, keeping default exports for what a tool loads by default (configs); add no barrel files.

## Architecture

**Backend adapter seam.** `backends/base.py` defines `Backend` (`preflight` + `run`) returning `list[ChunkLabel]`. Everything downstream is backend-agnostic. `backends/__init__.py` registers backends; `auto` prefers AssemblyAI when `$ASSEMBLYAI_API_KEY` is set, then Gemini when it has credentials, else local. `BackendConfig` is one shared bag of knobs — each backend ignores the ones it doesn't use. New backend = subclass, implement both methods, register in `BACKENDS`, add to the `--backend` click choice in `cli.py`.

- **local** (`backends/local.py`): ffmpeg → 16 kHz wav → Whisper (`whisper.py`) → two independent diarizations: pyannote (`diarize.py`, audio) and the LLM (`diarize.py::diarize_llm`, text) → `reconcile.py`. Agreements get confidence 100; one-side-silent gets ≤70 (scaled by `pyannote_chunk_purity` when the audio side spoke); only disputes go to a reconciler LLM call; no usable decision gets 25. Needs the `local` extra; its preflight says so when torch/transformers/pyannote are missing.
- **assemblyai** (`backends/assemblyai.py`): AssemblyAI word-level ASR + speaker tags (cached) → sentence-sized chunks → the same `diarize_llm` + `reconcile` ensemble as local, with AssemblyAI's speakers in the `audio` slot. Needs `$ASSEMBLYAI_API_KEY`.
- **gemini** (`backends/gemini.py`): single multimodal call returning JSON turns, converted to `ChunkLabel`s (`audio` is `None`; Gemini's speaker fills `llm`). Oversized audio is Opus-compressed, then GCS upload as a last resort (Vertex only).

**Post-backend stages** (orchestrated in `cli.py::transcribe_cmd`): `paragraphize.py` (LLM picks break-after chunk IDs, stored in the sidecar; never re-emits text, falls back to one paragraph per turn on any failure) → `transcript.py` renders markdown with ⚠️ on turns below `--review-threshold` → `asr_correction.py` (LLM flags likely Whisper mis-hears) → `serve.dump_labels_sidecar` writes the sidecar.

**LLM calls.** Text-only stages (text diarization, reconciler, paragraphize, ASR correction) go through `llm.py::run_structured` — a pydantic-ai `Agent` with a Pydantic `output_type`, so there's no JSON scraping. `llm.py::resolve_llm_model` turns `--llm-model` / `$POLYPHONY_LLM_MODEL` into a model string or `None`: `auto` (default) = first of `$OPENAI_API_KEY` / `$ANTHROPIC_API_KEY` / `$GOOGLE_API_KEY` set, a bare provider = its default in `PROVIDER_DEFAULT_MODELS` (`openai-codex` = ChatGPT subscription via `codex login`), `none` = no LLM. With `None`, every text stage is skipped (diarize_llm returns all-None, reconcile never calls the tie-breaker, paragraphize/ASR correction return nothing) and serve's LLM endpoints return 400; the UI disables those buttons when `llm_model` is null. Local backend preflight calls `check_llm`; at runtime the stages degrade gracefully (log and fall back) on any LLM error. The Gemini backend's client comes from `gemini_client.make_gemini_client`: `$GEMINI_API_KEY`/`$GOOGLE_API_KEY` wins over Vertex ADC.

**Cache** (`cache.py`): per-stage JSON under `~/.cache/polyphony/<key>/`, key = sha256(abs path + stage/model id + mtime + size). Stage ids like `asr-correction-v2` / `paragraphize-v3` are embedded in the key — bump the version string when changing a stage's prompt or output shape so stale caches invalidate.

**Export** (`export.py`, `polyphony export`): renders a sidecar with its review applied (`review.py::reviewed_labels`) as json/csv (per chunk, with the audit trail: original text, both labels, confidence, what the reviewer changed), srt/vtt (speaker-labeled cues), or md (same as Apply).

**Frontend ↔ Python contract.** `playground.py::playground_payload` is the single source of truth for the `/api/data` JSON; keep it in sync with `frontend/src/types.ts::PolyphonyData`. `serve.py` is FastAPI serving `static/index.html`, `/assets/*`, `/api/data`, `/api/audio` (HTTP Range for seeking), `POST /api/apply`, which persists review decisions + speaker names into the sidecar and writes `<stem>.transcript.reviewed.md` via `review.py` (deterministic re-render from chunks; the raw transcript is never modified), and `POST /api/reanalyze` (`reanalyze.py`), which reruns the text-side LLM passes on the stored chunks with new names/context hint — `speaker_names.py` maps unassigned candidate names to speaker ids — and carries review state over. The sidecar records `backend`, `llm_model`, and `context_hint` so re-analysis matches the original run. With `--vault`, `POST /api/vault/propose` lets the LLM pick a folder + name via read-only, vault-confined tools (`filing.py`), and `POST /api/vault/move` applies the review then moves the audio + every `<stem>.*.transcript*` sibling, rebinding the server to the new paths. `serve.py` intentionally omits `from __future__ import annotations` — FastAPI needs real annotations for DI.

## Gotchas

- The built bundle in `src/polyphony/static/` is committed and shipped in the wheel. Any `frontend/` change must be rebuilt and committed; CI fails on drift. Vite emits fixed filenames (`assets/main.js`, no hashes) on purpose.
- Tests cover pure logic and the ensemble → confidence → review → export path with `run_structured` monkeypatched (or `model=None`); Whisper/pyannote/real-LLM/Gemini paths are verified by running on real audio (e.g. `examples/demo.m4a`).
- Pyright runs in strict mode over `src` and `tests`. Untyped third-party boundaries (transformers, pyannote, google-genai, assemblyai) use narrow casts or line-level `# pyright: ignore[rule]`.
- Ruff: line length 120, rules `E,F,I,UP,B,SIM`.
- Local backend needs the `local` extra, `ffmpeg`, and `HF_TOKEN` with both pyannote licenses accepted. An LLM is optional everywhere.
