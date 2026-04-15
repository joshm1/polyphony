# polyphony

Audio → multi-speaker transcript with **ensemble diarization**, per-chunk confidence scores, and an interactive HTML review playground.

**Local-first.** The `local` backend runs entirely on your machine — no audio ever leaves your laptop. The optional `gemini` backend talks only to your own GCP project, never to a vendor's hosted service. Built for people who want a transcript of a private conversation, not 60 days of indexing in someone else's database.

Two backends, one pipeline:

- **`local`** — Whisper + pyannote + Claude diarize + reconciler. Slow, offline, most robust. Two independent diarization signals (audio-based and text-based) get reconciled by a third pass that emits per-chunk 0–100 confidence.
- **`gemini`** — one Gemini 2.5 call on Vertex AI does ASR and speaker attribution at once. Fast, cheap, needs ADC.

Why the ensemble: pyannote is excellent at voice changes but struggles with short interjections; LLM diarization is excellent at role cues ("Neville, tell me about…") but blind to actual voice. Combine them and the disagreements are exactly the chunks worth flagging for human review.

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

Drag the confidence slider; click a speaker name to override; select text in any chunk to propose a word correction; copy the apply-prompt to roll your edits back into the markdown via Claude. Keyboard nav: `j`/`k`, `1`-`9`, `/`, `Tab`, `space`, `p`.

---

## Install

```bash
git clone https://github.com/joshm1/polyphony
cd polyphony
uv sync
```

System dependencies:

- **`ffmpeg`** — for audio normalization (`brew install ffmpeg`)
- **`claude`** CLI — for the local backend's text-side diarization (`npm install -g @anthropic-ai/claude-code`). The text-side diarizer expects the `/transcript-diarize` skill from the [`audio-transcripts`](https://github.com/joshm1/joshm1-claude-plugins) plugin:
  ```
  /plugin marketplace add joshm1/claude-eng-toolkit
  /plugin install audio-transcripts@claude-eng-toolkit
  ```
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
# Auto-pick the backend (Gemini if ADC is available, else local)
polyphony transcribe interview.m4a --names "Josh,Neville"

# Force the local ensemble backend
polyphony transcribe interview.m4a --backend local --names "Josh,Neville"

# Force Gemini
polyphony transcribe interview.m4a --backend gemini --project my-gcp-project

# Domain hint helps both backends with technical terms
polyphony transcribe talk.m4a --context-hint "software engineering interview about distributed systems"

# Tighter review threshold (default 70 — turns below this get ⚠️ flagged)
polyphony transcribe talk.m4a --review-threshold 85

# Open an existing transcript + audio in the browser with seekable playback
polyphony serve interview.m4a
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
│   pyannote     │  speaker by voice fingerprint              │  claude -p         │
│   3.1 audio    │                                            │  text-side reasoning│
└───────┬────────┘                                            └─────────┬──────────┘
        │ chunk → speaker (or None)                                     │ chunk → speaker
        │                                                                │
        └───────────────────────┐               ┌────────────────────────┘
                                ▼               ▼
                           ┌─────────────────────────┐
                           │   reconciler (claude)    │
                           │  picks final + 0-100     │
                           │     confidence           │
                           └────────────┬─────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │  asr_correction (claude) │
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
                                                      │ + apply prompt   │
                                                      └──────────────────┘
```

The Gemini backend collapses the entire upper half into one multimodal API call; the markdown + sidecar + UI stages are backend-agnostic.

---

## Module map

| File | Purpose |
| --- | --- |
| [`cli.py`](src/polyphony/cli.py) | Click CLI; picks a backend, runs it, writes the transcript + sidecar + `polyphony serve` entrypoint |
| [`backends/base.py`](src/polyphony/backends/base.py) | `Backend` ABC and `BackendConfig`; what new backends implement |
| [`backends/local.py`](src/polyphony/backends/local.py) | Whisper + pyannote + Claude diarize + reconciler |
| [`backends/gemini.py`](src/polyphony/backends/gemini.py) | Single Gemini 2.5 multimodal call on Vertex AI; auto-compresses with Opus when over the inline limit |
| [`whisper.py`](src/polyphony/whisper.py) | HF transformers Whisper-large-v3-turbo with chunk-level timestamps |
| [`diarize.py`](src/polyphony/diarize.py) | pyannote pipeline + Claude shellout; per-chunk speaker labels |
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
| You need it offline, or no GCP project handy | `local` |
| Hour-long audio, want results in <60s, willing to call out | `gemini` |
| Quality matters more than speed | `local` (the ensemble catches more errors) |
| Just trying it out | `gemini` (simplest preflight) |

Cost: a 1-hour audio call on Gemini 2.5 Flash via Vertex is currently a few cents. The local backend is free if you ignore your laptop fan.

---

## Configuration

Environment variables:

| Var | Purpose |
| --- | --- |
| `HF_TOKEN` | Hugging Face token with pyannote license accepted (local backend) |
| `GOOGLE_CLOUD_PROJECT` | GCP project for Vertex AI (Gemini backend) |
| `GOOGLE_CLOUD_LOCATION` | Defaults to `us-east1` |
| `POLYPHONY_GCS_BUCKET` | Bucket for audio >15MB on the Gemini backend (only needed for very long audio) |
| `POLYPHONY_CLAUDE_CWD` | Where to run `claude -p` from — set if you want a project's local Claude skills loaded |
| `POLYPHONY_CACHE_DIR` | Override the cache location (default `~/.cache/polyphony`) |

---

## Caching

The slow stages (Whisper transcription, pyannote diarization, ASR-correction Claude pass) get cached at `~/.cache/polyphony/<sha>/` keyed on the audio path + file mtime + size. Iterating on prompts or rendering is fast after the first pass; touching the audio file invalidates everything.

---

## Tests

```bash
uv run pytest
```

Coverage is intentionally narrow — pure-logic bits (`reconcile`, label parsing, `pyannote_per_chunk_labels`). Anything that hits Whisper, pyannote, Claude, or Vertex is integration-tested by running on real audio.

---

## License

MIT — see [LICENSE](LICENSE).
