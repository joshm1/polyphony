// Shape of the JSON payload the Python backend injects into index.html.
// Must stay in sync with `polyphony.playground.playground_payload` in Python.

export interface Chunk {
  idx: number;
  start: number;
  end: number;
  text: string;
  pyannote: number | null;
  claude: number | null;
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

export interface PolyphonyData {
  audio: string;
  audio_url: string | null;
  transcript_path: string;
  names: string[];
  chunks: Chunk[];
  asr_flags: WordFlag[];
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
