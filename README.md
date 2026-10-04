# polyphony

**Diarization you can audit.** polyphony turns a recording into a speaker-attributed transcript where every chunk carries a 0–100 confidence score and the reason behind it. You review the doubtful parts in a browser with the audio one click away, fix speakers and misheard words, and export the corrected data as JSON, CSV, SRT, WebVTT, or markdown.

Most transcription tools hand you a confident-looking transcript and leave you to find the mistakes. polyphony shows you where it isn't sure, so a misattributed quote doesn't make it into a story, a paper, or your show notes.

Built for:

- **Journalists** checking who actually said a quote before it runs.
- **Researchers** coding interviews, who need corrected, per-utterance data in a spreadsheet.
- **Podcasters** who want accurate speaker-labeled captions and show notes.

---

## Per-chunk confidence

Each chunk (roughly a sentence) is labeled by two independent signals: an **audio diarizer** that tells voices apart (pyannote, or AssemblyAI's speaker tags) and a **text diarizer**, an LLM that reads the conversation for cues like who asks and who answers. Where they agree, you can trust the label. Where they don't, that's exactly where a human should look.

| Confidence | Meaning |
| --- | --- |
| **100** | Both signals agreed. |
| **≤ 70** | Only one signal had a view. With pyannote, this drops further when a chunk straddles a speaker change (e.g. 42 when 60% of the chunk's speech is the labeled speaker). |
| **0–99, rated** | The signals disagreed, and an LLM tie-breaker picked a speaker, rated its own certainty, and gave a short reason. |
| **25** | Nothing usable; fell back to a default. |

Turns with any chunk below `--review-threshold` (default 70) are marked `⚠️` in the transcript, preceded by an HTML comment giving the lowest score and the reasons:

```markdown
<!-- review needed (min confidence 45): chunk 12: reads as a reply to the question -->

⚠️ Daniel: We shipped it. Really? Yes.
```

A separate LLM pass flags likely ASR mis-hears ("B2B sauce" → "B2B SaaS") with a suggested fix, alternatives, and its own confidence. Nothing is changed until you accept it.

## The review UI

```bash
polyphony serve interview.m4a
```

Opens a local web app (default `http://localhost:8787`) with the transcript and seekable audio.

- **Speakers view.** Drag the confidence slider to show only turns at or below a score, play any turn or chunk from its ▶ button, and click a speaker name to reassign it. Select text to propose your own correction.
- **Words view.** Step through suggested ASR fixes with `j`/`k`; `1` takes the suggestion, `2`–`9` an alternative, `0` keeps the original, `/` types your own.
- **Speakers & context.** Name speakers, or list names in any order and let the LLM match them. Add a context hint ("city council meeting on zoning") and **Re-analyze** to rerun the text passes; your overrides and corrections carry over.
- **Apply** writes `<name>.transcript.reviewed.md` next to the raw transcript. The raw transcript is never modified, and every decision is saved in the `.polyphony.json` sidecar, so a reload picks up where you left off.
- **Summary** adds a summary, decisions, and action items to the top of the reviewed note.

## Export the corrected data

```bash
polyphony export interview.m4a --format csv     # → interview.local.transcript.reviewed.csv
polyphony export interview.m4a --format srt -o captions.srt
polyphony export interview.m4a --format json -o - | jq '.chunks[] | select(.speaker_overridden)'
```

| Format | Contents |
| --- | --- |
| `json`, `csv` | One row per chunk: start/end, speaker, corrected text, **original ASR text**, confidence, both diarizers' labels, the rationale, and whether a reviewer changed the speaker or the words. The audit trail travels with the data. |
| `srt`, `vtt` | Speaker-labeled captions. WebVTT uses `<v Speaker>` voice tags, which players and caption editors understand. |
| `md` | The reviewed transcript, same as **Apply**. |

Exports reflect the review: speaker overrides, accepted corrections, and speaker names are applied.

---

## Install

polyphony is a command-line tool. Install it from GitHub with [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/):

```bash
uv tool install git+https://github.com/joshm1/polyphony
# or
pipx install git+https://github.com/joshm1/polyphony
# or run once without installing
uvx --from git+https://github.com/joshm1/polyphony polyphony --help
```

To run everything on your own machine with Whisper + pyannote, add the `local` extra. It pulls in PyTorch and model runtimes, several GB in total:

```bash
uv tool install "polyphony[local] @ git+https://github.com/joshm1/polyphony"
```

You also need [`ffmpeg`](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg`, `apt install ffmpeg`).

## Quick start

```bash
export ASSEMBLYAI_API_KEY=...   # speech-to-text + voice diarization
export OPENAI_API_KEY=...       # or ANTHROPIC_API_KEY / GOOGLE_API_KEY, for the text-side checks

polyphony transcribe interview.m4a --names "Host,Guest" --context-hint "local news interview about the transit levy"
polyphony serve interview.m4a
polyphony export interview.m4a --format csv
```

`transcribe` writes `<name>.<backend>.transcript.md` and the `<name>.<backend>.transcript.polyphony.json` sidecar next to the audio. Names are listed in order of first appearance.

---

## Backends

| Backend | Speech-to-text and voice diarization | Audio leaves your machine? | Needs |
| --- | --- | --- | --- |
| `assemblyai` | AssemblyAI, cross-checked by the text diarizer | Yes, to AssemblyAI | `$ASSEMBLYAI_API_KEY` |
| `local` | Whisper large-v3-turbo + pyannote 3.1, cross-checked by the text diarizer | **No** | the `local` extra, a Hugging Face token |
| `gemini` | One Gemini call does both | Yes, to Google | `$GEMINI_API_KEY`, or Vertex AI credentials |

`--backend auto` (the default) uses AssemblyAI when `$ASSEMBLYAI_API_KEY` is set, then Gemini when it has credentials, and otherwise `local`. The Gemini backend rates its own confidence per turn but has no second signal to cross-check, so `assemblyai` and `local` give the more meaningful scores.

The `local` backend needs a [Hugging Face token](https://huggingface.co/settings/tokens) in `$HF_TOKEN` with both pyannote licenses accepted: [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). Expect roughly 10 minutes of Whisper per hour of audio on Apple Silicon and longer for pyannote on CPU; both results are cached.

## Choosing the LLM

The text-side passes (text diarization, tie-breaking, paragraphing, ASR-error flags, summaries) run through [pydantic-ai](https://ai.pydantic.dev), so any provider it supports works. Set `--llm-model` or `$POLYPHONY_LLM_MODEL`:

| Value | Uses |
| --- | --- |
| `auto` (default) | The first of `$OPENAI_API_KEY`, `$ANTHROPIC_API_KEY`, `$GOOGLE_API_KEY` (or `$GEMINI_API_KEY`) that is set |
| `openai`, `anthropic`, `google` | That provider's default model |
| `openai-codex` | A ChatGPT subscription, via the credentials `codex login` stores |
| `provider:model` | Any pydantic-ai model string, e.g. `anthropic:claude-opus-5` or `ollama:qwen3` (set `$OLLAMA_BASE_URL`) for a model on your own hardware |
| `none` | No LLM at all |

**Without an LLM**, polyphony still works: speakers come from the audio diarizer alone and every chunk is capped at 70, lower where pyannote saw a speaker change inside the chunk. Paragraphing and ASR-error flags are skipped, and the UI disables re-analysis and summaries. Raise `--review-threshold` (e.g. 75) to send every turn through review. Combined with `--backend local`, nothing leaves your machine.

The text passes never receive audio: they see transcript text, speaker names, and your context hint (vault filing also sees your vault's folder and file names).

---

## Configuration

| Variable | Purpose |
| --- | --- |
| `POLYPHONY_LLM_MODEL` | Default for `--llm-model` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` | LLM provider keys picked up by `auto` (`GEMINI_API_KEY` also counts for Google) |
| `ASSEMBLYAI_API_KEY` | AssemblyAI backend |
| `HF_TOKEN` | Hugging Face token for pyannote (`local` backend) |
| `GEMINI_API_KEY` | Gemini backend via the Gemini API; without it, Vertex AI via ADC |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | Gemini backend on Vertex AI (location defaults to `us-east1`) |
| `POLYPHONY_GCS_BUCKET` | Bucket for audio too large to send inline to Gemini on Vertex AI |
| `POLYPHONY_VAULT` | Notes vault (e.g. Obsidian) that `polyphony serve` can file reviewed recordings into |
| `POLYPHONY_CACHE_DIR` | Cache location (default `~/.cache/polyphony`) |

Slow stages (Whisper, pyannote, AssemblyAI, and each LLM pass) are cached per audio file, keyed on its path, size, and modification time. Re-running with different names or a different threshold is fast.

All options: `polyphony transcribe --help`, `polyphony serve --help`, `polyphony export --help`.

---

## Example

[`examples/demo.m4a`](examples/demo.m4a) is a 90-second synthetic two-person conversation (regenerate it with [`examples/generate_demo.sh`](examples/generate_demo.sh) on macOS). Try the full loop on it:

```bash
polyphony transcribe examples/demo.m4a --names "Sam,Daniel"
polyphony serve examples/demo.m4a
```

## Development

```bash
uv sync --all-extras
uv run pytest
uv run pyright
uv run ruff check src tests && uv run ruff format --check src tests
```

The review UI is a React app in [`frontend/`](frontend); its built bundle is committed under `src/polyphony/static/` so installs don't need Node. See [AGENTS.md](AGENTS.md) for the architecture.

## License

MIT. See [LICENSE](LICENSE).
