import { z } from 'zod'

// Schemas for the JSON the Python backend serves. `PolyphonyData` must stay in sync with
// `polyphony.playground.playground_payload` plus the fields `serve.py::client_payload` adds.

export const ChunkSchema = z.object({
  idx: z.number(),
  start: z.number(),
  end: z.number(),
  text: z.string(),
  // Absent in sidecars written before the backend-agnostic field names.
  audio: z.number().nullable().default(null),
  llm: z.number().nullable().default(null),
  final: z.number(),
  confidence: z.number(),
  note: z.string(),
  audio_purity: z.number().nullish(), // pyannote only; absent in older sidecars
})
export type Chunk = z.infer<typeof ChunkSchema>

export const WordFlagSchema = z.object({
  chunk_idx: z.number(),
  original: z.string(),
  suggested: z.string(),
  alternatives: z.array(z.string()),
  confidence: z.number(),
  reason: z.string(),
  userAdded: z.boolean().optional(),
})
export type WordFlag = z.infer<typeof WordFlagSchema>

export const WordDecisionSchema = z.object({
  kind: z.enum(['suggested', 'alternative', 'custom', 'original']),
  value: z.string(),
})
export type WordDecision = z.infer<typeof WordDecisionSchema>

export const ReviewStateSchema = z.object({
  overrides: z.record(z.string(), z.number()).default({}), // chunk idx → speaker id
  word_decisions: z.record(z.string(), WordDecisionSchema).default({}), // asr_flags index → decision
})

export const TranscriptSummarySchema = z.object({
  tldr: z.string(),
  key_points: z.array(z.string()),
  decisions: z.array(z.string()),
  action_items: z.array(z.object({ owner: z.string().nullable(), task: z.string() })),
  model: z.string(),
  created_at: z.string(),
})
export type TranscriptSummary = z.infer<typeof TranscriptSummarySchema>

export const PolyphonyDataSchema = z.object({
  audio: z.string(),
  audio_url: z.string().nullable(),
  transcript_path: z.string(),
  names: z.array(z.string()), // indexed by speaker id - 1; "" = unnamed
  // Recorded for the server's own re-runs; absent in older sidecars.
  review_threshold: z.number().optional(),
  backend: z.string().optional(),
  paragraph_breaks: z.array(z.number()).nullish(),
  llm_model: z.string().nullable(), // null = no LLM configured; LLM-only actions are disabled
  context_hint: z.string().nullable(),
  vault: z.string().nullable(), // injected by the server; null when no vault is configured
  filed: z.boolean(), // the recording already sits in its own folder inside the vault
  summary: TranscriptSummarySchema.nullish(),
  summary_stale: z.boolean(), // the reviewed transcript changed since the summary was generated
  review: ReviewStateSchema.nullish().transform((r) => r ?? { overrides: {}, word_decisions: {} }),
  chunks: z.array(ChunkSchema),
  asr_flags: z
    .array(WordFlagSchema)
    .nullish()
    .transform((f) => f ?? []),
})
export type PolyphonyData = z.infer<typeof PolyphonyDataSchema>

export const ApplyResultSchema = z.object({
  reviewed_path: z.string(),
  skipped: z.array(z.object({ chunk_idx: z.number(), original: z.string() })),
})
export type ApplyResult = z.infer<typeof ApplyResultSchema>

export const ReanalyzeResultSchema = ApplyResultSchema.extend({ data: PolyphonyDataSchema })

export const SummaryResultSchema = z.object({
  summary: TranscriptSummarySchema,
  summary_stale: z.boolean(),
})

export const VaultProposalSchema = z.object({
  folder: z.string(),
  basename: z.string(),
  reason: z.string(),
  files: z.array(z.string()),
})
export type VaultProposal = z.infer<typeof VaultProposalSchema>

export const VaultMoveResultSchema = z.object({
  moved: z.array(z.string()),
  note_path: z.string(),
  obsidian_url: z.string(),
})

// A status line that survives App remounting after the server replaces the data.
export interface Notice {
  message: string
  href?: string
  linkText?: string
}

export type TextToken =
  | { type: 'text'; at: number; value: string }
  | { type: 'correction'; at: number; original: string; replacement: string; user: boolean }

export interface Turn {
  speaker: number
  chunks: [Chunk, ...Chunk[]]
  minConf: number
  overridden: boolean
}

export type PopoverState =
  | { open: false }
  | { open: true; chunkIdx: number; original: string; value: string; top: number; left: number }
