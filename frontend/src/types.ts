// Shape of the JSON payload the Python backend injects into index.html.
// Must stay in sync with `polyphony.playground.playground_payload` in Python.

export interface Chunk {
  idx: number;
  start: number;
  end: number;
  text: string;
  audio: number | null;
  llm: number | null;
  final: number;
  confidence: number;
  note: string;
}

export interface WordFlag {
  chunk_idx: number;
  original: string;
  suggested: string;
  alternatives: string[];
  confidence: number;
  reason: string;
  userAdded?: boolean;
}

export interface ReviewState {
  overrides: Record<string, number>; // chunk idx → speaker id
  word_decisions: Record<string, WordDecision>; // asr_flags index → decision
}

export interface PolyphonyData {
  audio: string;
  audio_url: string | null;
  transcript_path: string;
  names: string[]; // indexed by speaker id - 1; "" = unnamed
  review_threshold: number;
  backend: string;
  llm_model: string;
  context_hint: string | null;
  vault: string | null; // injected by the server; null when no vault is configured
  paragraph_breaks: number[] | null;
  review: ReviewState;
  chunks: Chunk[];
  asr_flags: WordFlag[];
}

export interface ApplyResult {
  reviewed_path: string;
  skipped: { chunk_idx: number; original: string }[];
}

export interface ReanalyzeResult extends ApplyResult {
  data: PolyphonyData;
}

export interface VaultProposal {
  folder: string;
  basename: string;
  reason: string;
  files: string[];
}

export interface VaultMoveResult {
  moved: string[];
  note_path: string;
  obsidian_url: string;
}

// A status line that survives App remounting after the server replaces the data.
export interface Notice {
  message: string;
  href?: string;
  linkText?: string;
}

export type WordDecisionKind = "suggested" | "alternative" | "custom" | "original";

export interface WordDecision {
  kind: WordDecisionKind;
  value: string;
}

export interface TextToken {
  type: "text" | "correction";
  value?: string;
  original?: string;
  replacement?: string;
  user?: boolean;
}

export interface Turn {
  speaker: number;
  chunks: Chunk[];
  minConf: number;
  overridden: boolean;
}

export interface PopoverState {
  open: boolean;
  chunkIdx: number | null;
  original: string;
  value: string;
  top: number;
  left: number;
}
