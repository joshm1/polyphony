# polyphony

Audio → multi-speaker transcript with **ensemble diarization**, per-chunk confidence scores, and an interactive HTML review playground.

**Your choice of where audio goes.** The `local` backend never sends audio off your machine — only transcript text goes to the LLM for the text-side passes. The `gemini` backend talks to your own GCP project; the `assemblyai` backend uploads audio to AssemblyAI. Built for people who want a transcript of a private conversation and control over who sees it.

Three backends, one pipeline:

- **`local`** — Whisper + pyannote + LLM diarize + reconciler. Slow, audio stays local, most robust. Two independent diarization signals (audio-based and text-based) get reconciled by a third pass that emits per-chunk 0–100 confidence.
- **`assemblyai`** — AssemblyAI hosted ASR + voice diarization, cross-checked by the same LLM diarize + reconciler ensemble. Fast (~2 min for 20 min of audio), no Whisper/pyannote/HF token; needs `$ASSEMBLYAI_API_KEY`.
- **`gemini`** — one Gemini 2.5 call on Vertex AI does ASR and speaker attribution at once. Fast, cheap, needs ADC.

Why the ensemble: pyannote is excellent at voice changes but struggles with short interjections; LLM diarization is excellent at role cues ("so tell me about your background…") but blind to actual voice. Combine them and the disagreements are exactly the chunks worth flagging for human review.

---

## Demo

A 90-second m4a of two friends bantering about LLMs, sycophancy, and AI code review lives at [`examples/demo.m4a`](examples/demo.m4a). The generated transcript + review sidecar ship with the repo:

- [`examples/demo.local.transcript.md`](examples/demo.local.transcript.md) — speaker-attributed markdown with `⚠️` markers on low-confidence turns
- [`examples/demo.local.transcript.polyphony.json`](examples/demo.local.transcript.polyphony.json) — the review sidecar that feeds the React review UI

Regenerate either yourself:

```bash
polyphony transcribe examples/demo.m4a --names "Sam,Daniel" --backend local
```

Then open the interactive review UI (React + Vite, served over HTTP so the `<audio>` element can seek the source file):

```bash
polyphony serve examples/demo.m4a   # opens localhost:8787 in your browser
```

Drag the confidence slider; click a speaker name to override; select text in any chunk to propose a word correction; hit **Apply** to write `<stem>.transcript.reviewed.md` next to the raw transcript (the raw file is never modified, and your decisions are saved in the sidecar so a reload resumes them). **Speakers & context** lets you name speakers (typed names apply instantly) or list names in any order for the LLM to match, and set a context hint; **Re-analyze with LLM** reruns the text-side passes with them (~1–2 min) while keeping your overrides and corrections. With `--vault PATH` (or `$POLYPHONY_VAULT`), **Apply** also files the recording: the LLM suggests where it belongs in your notes vault (it browses folder/file names, not contents, and names the recording from the conversation), you confirm or edit the folder and name, and **Apply & move** writes the reviewed transcript and moves the audio + all transcripts + sidecar into their own folder, `<folder>/YYYY-MM-DD Short Title/`. Once the recording is in the vault, Apply writes in place. **Copy apply prompt** remains for handing edits to an AI agent instead. Keyboard nav: `j`/`k`, `1`-`9`, `/`, `Tab`, `space`, `p`.

---

## Install

```bash
git clone https://github.com/joshm1/polyphony
cd polyphony
uv sync
```

System dependencies:

- **`ffmpeg`** — for audio normalization (`brew install ffmpeg`)
- **LLM for the text-side passes** (local backend diarization + reconciler, and ASR correction on every backend) — runs through [pydantic-ai](https://ai.pydantic.dev). The default `openai-codex:gpt-6-sol` bills your ChatGPT/Codex subscription: install the [Codex CLI](https://github.com/openai/codex) and run `codex login` once. Any other pydantic-ai model string works via `--llm-model` / `$POLYPHONY_LLM_MODEL` (install the matching `pydantic-ai-slim` extra).
- **Hugging Face token** — for pyannote. Create at <https://huggingface.co/settings/tokens> and accept BOTH:
  - <https://huggingface.co/pyannote/speaker-diarization-3.1>
  - <https://huggingface.co/pyannote/segmentation-3.0>
- **Google Cloud ADC** — for the Gemini backend (`gcloud auth application-default login`)

Set up your shell env (one option):

```bash
cp mise.toml.example mise.toml
# edit mise.toml, fill in your HF_TOKEN source
mise trust
```

Or just `export HF_TOKEN=hf_…` however you like.

---

## Usage

```bash
# Auto-pick the backend (AssemblyAI if $ASSEMBLYAI_API_KEY is set, else Gemini)
polyphony transcribe recording.m4a --names "Host,Guest"

# Force AssemblyAI (hosted ASR + voice diarization, cross-checked by the LLM)
polyphony transcribe recording.m4a --backend assemblyai --names "Host,Guest"

# Force the local ensemble backend
polyphony transcribe recording.m4a --backend local --names "Host,Guest"

# Force Gemini
polyphony transcribe recording.m4a --backend gemini --project my-gcp-project

# Domain hint helps the LLM passes with domain-specific terms
polyphony transcribe talk.m4a --context-hint "cooking podcast about regional barbecue styles"

# Tighter review threshold (default 70 — turns below this get ⚠️ flagged)
polyphony transcribe talk.m4a --review-threshold 85

# Open an existing transcript + audio in the browser with seekable playback
polyphony serve recording.m4a
```

All flags: `polyphony --help` and `polyphony transcribe --help`.

Outputs land next to the audio: `<stem>.<backend>.transcript.md` and `<stem>.<backend>.transcript.polyphony.json` (the sidecar `polyphony serve` reads).

---

## Architecture

```
                    ┌──────────────┐
        audio ──────▶  ffmpeg 16k  ──────┐
                    └──────────────┘     │
                                         ▼
                              ┌────────────────────┐
                              │  Whisper-large-v3  │
                              │  (chunk-level ts)  │
                              └─────────┬──────────┘
                                        │
                                        ▼  per-chunk text + timing
        ┌───────────────────────────────┴───────────────────────────────┐
        │                                                                │
        ▼                                                                ▼
┌────────────────┐                                            ┌────────────────────┐
│   pyannote     │  speaker by voice fingerprint              │  LLM (pydantic-ai) │
│   3.1 audio    │                                            │  text-side reasoning│
└───────┬────────┘                                            └─────────┬──────────┘
        │ chunk → speaker (or None)                                     │ chunk → speaker
        │                                                                │
        └───────────────────────┐               ┌────────────────────────┘
                                ▼               ▼
                           ┌─────────────────────────┐
                           │   reconciler (LLM)       │
                           │  picks final + 0-100     │
                           │     confidence           │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │  asr_correction (LLM)    │
                           │   flag wrong-word ASR    │
                           │   errors with reasons    │
                           └────────────┬─────────────┘
                                        │
                ┌───────────────────────┴───────────────────────┐
                ▼                                               ▼
        ┌───────────────┐                              ┌──────────────────┐
        │ transcript.md │                              │ polyphony.json   │
        │  ⚠️ on low-   │                              │  (sidecar)        │
        │  confidence    │                              └────────┬─────────┘
        └───────────────┘                                       │
                                                                 ▼
                                                      ┌──────────────────┐
                                                      │ polyphony serve  │
                                                      │ React + Vite UI  │
                                                      │ accept / reject  │
                                                      │ Apply → reviewed │
                                                      └──────────────────┘
```

The Gemini backend collapses the entire upper half into one multimodal API call; the markdown + sidecar + UI stages are backend-agnostic.

---

## Module map

| File | Purpose |
| --- | --- |
| [`cli.py`](src/polyphony/cli.py) | Click CLI; picks a backend, runs it, writes the transcript + sidecar + `polyphony serve` entrypoint |
| [`backends/base.py`](src/polyphony/backends/base.py) | `Backend` ABC and `BackendConfig`; what new backends implement |
| [`backends/local.py`](src/polyphony/backends/local.py) | Whisper + pyannote + LLM diarize + reconciler |
| [`backends/assemblyai.py`](src/polyphony/backends/assemblyai.py) | AssemblyAI ASR + speaker tags → sentence-sized chunks, then LLM diarize + reconciler |
| [`backends/gemini.py`](src/polyphony/backends/gemini.py) | Single Gemini 2.5 multimodal call on Vertex AI; auto-compresses with Opus when over the inline limit |
| [`whisper.py`](src/polyphony/whisper.py) | HF transformers Whisper-large-v3-turbo with chunk-level timestamps |
| [`diarize.py`](src/polyphony/diarize.py) | pyannote pipeline + LLM text diarization; per-chunk speaker labels |
| [`llm.py`](src/polyphony/llm.py) | pydantic-ai wrapper for the text-side LLM passes (structured output, model selection) |
| [`reconcile.py`](src/polyphony/reconcile.py) | Merges two diarizations into one with confidence scores |
| [`asr_correction.py`](src/polyphony/asr_correction.py) | Optional pass that flags likely Whisper errors (e.g. "SaaS" → "sauce") |
| [`transcript.py`](src/polyphony/transcript.py) | Folds labeled chunks into speaker turns; renders markdown |
| [`playground.py`](src/polyphony/playground.py) | Shapes the JSON payload the React review UI consumes (single source of truth for the wire contract) |
| [`serve.py`](src/polyphony/serve.py) | FastAPI server: serves the React bundle under [`static/`](src/polyphony/static) + the audio with HTTP range support |
| [`static/`](src/polyphony/static) | Built React bundle (committed; rebuilt from [`frontend/`](frontend) by the pre-commit hook) |
| [`cache.py`](src/polyphony/cache.py) | On-disk cache for the slow stages — Whisper + pyannote can take 40 min on a CPU |
| [`audio.py`](src/polyphony/audio.py) | `ffmpeg` wrapper to normalize anything to 16 kHz mono 16-bit PCM |

The review UI itself is a small React/Vite app under [`frontend/`](frontend). Python-only contributors never need to touch it — the built bundle is committed under `src/polyphony/static/` and pre-commit rebuilds it whenever frontend sources change.

---

## When to use which backend

| Situation | Backend |
| --- | --- |
| Audio must stay on your machine | `local` |
| Fast, with the audio + text ensemble and per-chunk confidence | `assemblyai` |
| Hour-long audio, want results in <60s, willing to call out | `gemini` |
| Quality matters more than speed | `local` (the ensemble catches more errors) |
| Just trying it out | `assemblyai` (one API key + `codex login`) |

Cost: a 1-hour audio call on Gemini 2.5 Flash via Vertex is currently a few cents; AssemblyAI bills per audio hour. The LLM passes run on your ChatGPT subscription by default. The local backend is free if you ignore your laptop fan.

---

## Configuration

Environment variables:

| Var | Purpose |
| --- | --- |
| `ASSEMBLYAI_API_KEY` | AssemblyAI API key (assemblyai backend) |
| `HF_TOKEN` | Hugging Face token with pyannote license accepted (local backend) |
| `GOOGLE_CLOUD_PROJECT` | GCP project for Vertex AI (Gemini backend) |
| `GOOGLE_CLOUD_LOCATION` | Defaults to `us-east1` |
| `POLYPHONY_GCS_BUCKET` | Bucket for audio >15MB on the Gemini backend (only needed for very long audio) |
| `POLYPHONY_LLM_MODEL` | pydantic-ai model for the text-side LLM passes (default `openai-codex:gpt-6-sol`) |
| `CODEX_HOME` | Where `codex login` stored `auth.json` (default `~/.codex`) |
| `POLYPHONY_VAULT` | Notes vault (e.g. Obsidian) that `polyphony serve` can file recordings into |
| `POLYPHONY_CACHE_DIR` | Override the cache location (default `~/.cache/polyphony`) |

---

## Caching

The slow stages (Whisper transcription, pyannote diarization, ASR-correction LLM pass) get cached at `~/.cache/polyphony/<sha>/` keyed on the audio path + file mtime + size. Iterating on prompts or rendering is fast after the first pass; touching the audio file invalidates everything.

---

## Tests

```bash
uv run pytest
```

Coverage is intentionally narrow — pure-logic bits (`reconcile`, label parsing, `pyannote_per_chunk_labels`). Anything that hits Whisper, pyannote, the LLM, or Vertex is integration-tested by running on real audio.

---

## License

MIT — see [LICENSE](LICENSE).
