import {
  type ChangeEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  Chunk,
  PolyphonyData,
  PopoverState,
  TextToken,
  Turn,
  WordDecision,
  WordFlag,
} from "./types";

const clampConf = (c: number) => Math.max(0, Math.min(100, c));
const confColor = (c: number) => `hsl(${(clampConf(c) * 1.2).toFixed(0)}, 70%, 50%)`;

function fmtTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

interface AppProps {
  data: PolyphonyData;
}

export default function App({ data }: AppProps) {
  const { names, audio_url: audioUrl, audio: audioName, transcript_path: transcriptPath } = data;

  // ---------- state ----------
  const [chunks] = useState<Chunk[]>(data.chunks);
  const [flags, setFlags] = useState<WordFlag[]>([...(data.asr_flags ?? [])]);
  const [mode, setMode] = useState<"speakers" | "words">("speakers");
  const [threshold, setThreshold] = useState<number>(100);
  const [overrides, setOverrides] = useState<Map<number, number>>(new Map());
  const [wordDecisions, setWordDecisions] = useState<Map<number, WordDecision>>(new Map());
  const [focusedWordIdx, setFocusedWordIdx] = useState<number>(flags.length > 0 ? 0 : -1);
  const [playingChunkIdx, setPlayingChunkIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [copyStatus, setCopyStatus] = useState(false);
  const [popover, setPopover] = useState<PopoverState>({
    open: false,
    chunkIdx: null,
    original: "",
    value: "",
    top: 0,
    left: 0,
  });

  const playerRef = useRef<HTMLAudioElement | null>(null);
  const autoStopAtRef = useRef<number | null>(null);
  const popoverInputRef = useRef<HTMLInputElement | null>(null);

  // ---------- derived / helpers ----------

  const totalSpeakers = useMemo(
    () => Math.max(2, ...chunks.map((c) => c.final), names.length),
    [chunks, names.length],
  );

  const speakerName = useCallback(
    (id: number) => (id >= 1 && id <= names.length ? names[id - 1] : `Speaker ${id}`),
    [names],
  );

  const chunkById = useCallback((idx: number) => chunks.find((c) => c.idx === idx), [chunks]);

  // Collect active correction spans per chunk (ASR flags + user manual edits).
  const correctionsByChunk = useMemo(() => {
    const map = new Map<number, { original: string; replacement: string; user: boolean }[]>();
    flags.forEach((f, i) => {
      const dec =
        wordDecisions.get(i) ?? ({ kind: "suggested", value: f.suggested } as WordDecision);
      if (dec.kind === "original") return;
      if (!map.has(f.chunk_idx)) map.set(f.chunk_idx, []);
      map.get(f.chunk_idx)?.push({
        original: f.original,
        replacement: dec.value,
        user: !!f.userAdded,
      });
    });
    return map;
  }, [flags, wordDecisions]);

  const renderChunkTokens = useCallback(
    (chunk: Chunk): TextToken[] => {
      const corrections = correctionsByChunk.get(chunk.idx) ?? [];
      if (corrections.length === 0) return [{ type: "text", value: chunk.text }];
      const tokens: TextToken[] = [];
      let remaining = chunk.text;
      const pending = corrections.map((c) => ({ ...c }));
      while (remaining.length > 0 && pending.length > 0) {
        let bestIdx = -1;
        let bestAt = Number.POSITIVE_INFINITY;
        pending.forEach((c, i) => {
          const at = remaining.indexOf(c.original);
          if (at >= 0 && at < bestAt) {
            bestAt = at;
            bestIdx = i;
          }
        });
        if (bestIdx < 0) break;
        const c = pending.splice(bestIdx, 1)[0];
        if (bestAt > 0) tokens.push({ type: "text", value: remaining.slice(0, bestAt) });
        tokens.push({
          type: "correction",
          original: c.original,
          replacement: c.replacement,
          user: c.user,
        });
        remaining = remaining.slice(bestAt + c.original.length);
      }
      if (remaining) tokens.push({ type: "text", value: remaining });
      return tokens;
    },
    [correctionsByChunk],
  );

  const turns = useMemo<Turn[]>(() => {
    const result: Turn[] = [];
    let cur: Turn | null = null;
    for (const ch of chunks) {
      const spk = overrides.has(ch.idx) ? (overrides.get(ch.idx) as number) : ch.final;
      if (!cur || cur.speaker !== spk) {
        if (cur) result.push(cur);
        cur = {
          speaker: spk,
          chunks: [ch],
          minConf: ch.confidence,
          overridden: overrides.has(ch.idx),
        };
      } else {
        cur.chunks.push(ch);
        cur.minConf = Math.min(cur.minConf, ch.confidence);
        if (overrides.has(ch.idx)) cur.overridden = true;
      }
    }
    if (cur) result.push(cur);
    return result;
  }, [chunks, overrides]);

  const visibleTurns = useMemo(
    () => turns.filter((t) => t.minConf <= threshold),
    [turns, threshold],
  );

  const lowConfTurnCount = useMemo(() => turns.filter((t) => t.minConf < 70).length, [turns]);

  const decisionOf = useCallback(
    (i: number): WordDecision =>
      wordDecisions.get(i) ?? { kind: "suggested", value: flags[i].suggested },
    [wordDecisions, flags],
  );

  const decidedWordCount = useMemo(
    () => Array.from(wordDecisions.values()).filter((d) => d.kind !== "suggested").length,
    [wordDecisions],
  );

  // ---------- mutations ----------

  const setSpeaker = useCallback((ch: Chunk, s: number) => {
    setOverrides((prev) => {
      const m = new Map(prev);
      if (s === ch.final && m.has(ch.idx)) m.delete(ch.idx);
      else m.set(ch.idx, s);
      return m;
    });
  }, []);

  const chooseDecision = useCallback((i: number, kind: WordDecision["kind"], value: string) => {
    setWordDecisions((prev) => {
      const m = new Map(prev);
      m.set(i, { kind, value });
      return m;
    });
  }, []);

  const chooseSuggested = useCallback(
    (i: number) => chooseDecision(i, "suggested", flags[i].suggested),
    [chooseDecision, flags],
  );
  const chooseAlternative = useCallback(
    (i: number, alt: string) => chooseDecision(i, "alternative", alt),
    [chooseDecision],
  );
  const keepOriginal = useCallback(
    (i: number) => chooseDecision(i, "original", flags[i].original),
    [chooseDecision, flags],
  );

  const advanceFocus = useCallback(() => {
    setFocusedWordIdx((idx) => Math.min(flags.length - 1, idx + 1));
  }, [flags.length]);

  // ---------- audio ----------

  const playFromChunk = useCallback(
    (chunk: Chunk, opts?: { untilEnd?: boolean }) => {
      if (!audioUrl || !playerRef.current) return;
      const untilEnd = opts?.untilEnd ?? true;
      setPlayingChunkIdx(chunk.idx);
      autoStopAtRef.current = untilEnd && chunk.end > chunk.start ? chunk.end + 0.2 : null;
      playerRef.current.currentTime = Math.max(0, chunk.start - 0.1);
      playerRef.current.play();
    },
    [audioUrl],
  );

  const playFromTurn = useCallback(
    (turn: Turn) => {
      if (!audioUrl) return;
      playFromChunk(turn.chunks[0], { untilEnd: false });
      autoStopAtRef.current = turn.chunks[turn.chunks.length - 1].end + 0.2;
    },
    [audioUrl, playFromChunk],
  );

  const onTimeupdate = useCallback(() => {
    const el = playerRef.current;
    if (!el) return;
    setCurrentTime(el.currentTime);
    if (autoStopAtRef.current !== null && el.currentTime >= autoStopAtRef.current) {
      el.pause();
      autoStopAtRef.current = null;
    }
  }, []);

  const onPause = useCallback(() => {
    setPlayingChunkIdx(null);
    setPlaying(false);
  }, []);
  const onPlay = useCallback(() => setPlaying(true), []);
  const onLoadedMetadata = useCallback(() => {
    setDuration(playerRef.current?.duration ?? 0);
  }, []);

  const scrubPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  const togglePlay = useCallback(() => {
    const el = playerRef.current;
    if (!el) return;
    if (el.paused) {
      autoStopAtRef.current = null;
      el.play();
    } else {
      el.pause();
    }
  }, []);

  const seekFromClick = useCallback(
    (e: ReactMouseEvent<HTMLDivElement>) => {
      const el = playerRef.current;
      if (!el || !duration) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      el.currentTime = Math.max(0, Math.min(duration, ratio * duration));
      autoStopAtRef.current = null;
    },
    [duration],
  );

  // ---------- popover ----------

  const onMouseupSpeakers = useCallback(() => {
    // Defer so the selection range stabilizes.
    setTimeout(() => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      const selected = sel.toString().trim();
      if (!selected) return;
      const container = range.commonAncestorContainer;
      const parentEl = container.nodeType === 1 ? (container as Element) : container.parentElement;
      const textEl = parentEl?.closest<HTMLElement>(".text[data-chunk-idx]");
      if (!textEl?.dataset.chunkIdx) return;
      const chunkIdx = Number.parseInt(textEl.dataset.chunkIdx, 10);
      const chunk = chunkById(chunkIdx);
      if (!chunk || !chunk.text.includes(selected)) return;
      const rect = range.getBoundingClientRect();
      setPopover({
        open: true,
        chunkIdx,
        original: selected,
        value: selected,
        top: rect.bottom + window.scrollY + 6,
        left: Math.max(12, Math.min(rect.left, window.innerWidth - 340)),
      });
      setTimeout(() => popoverInputRef.current?.select(), 0);
    }, 10);
  }, [chunkById]);

  const cancelPopover = useCallback(() => setPopover((p) => ({ ...p, open: false })), []);

  const acceptPopover = useCallback(() => {
    setPopover((p) => {
      if (!p.open) return p;
      const replacement = p.value.trim();
      if (!replacement || replacement === p.original) return { ...p, open: false };
      const newFlag: WordFlag = {
        chunk_idx: p.chunkIdx as number,
        original: p.original,
        suggested: replacement,
        alternatives: [],
        confidence: 100,
        reason: "manual correction",
        userAdded: true,
      };
      setFlags((fs) => {
        const next = [...fs, newFlag];
        setWordDecisions((prev) => {
          const m = new Map(prev);
          m.set(next.length - 1, { kind: "suggested", value: replacement });
          return m;
        });
        return next;
      });
      window.getSelection()?.removeAllRanges();
      return { ...p, open: false };
    });
  }, []);

  const onPopoverKeydown = useCallback(
    (e: ReactKeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") {
        e.preventDefault();
        acceptPopover();
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancelPopover();
      }
      e.stopPropagation();
    },
    [acceptPopover, cancelPopover],
  );

  // Close popover on outside click.
  useEffect(() => {
    if (!popover.open) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest(".word-popover")) return;
      cancelPopover();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [popover.open, cancelPopover]);

  // ---------- keyboard ----------

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const active = document.activeElement as HTMLElement | null;
      if (active?.tagName === "INPUT") return;

      if (e.key === "Tab") {
        e.preventDefault();
        setMode((m) => (m === "speakers" ? "words" : "speakers"));
        return;
      }
      if (audioUrl && e.key === " ") {
        e.preventDefault();
        if (playerRef.current) {
          if (playerRef.current.paused) playerRef.current.play();
          else playerRef.current.pause();
        }
        return;
      }
      if (mode !== "words" || flags.length === 0) return;

      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        advanceFocus();
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setFocusedWordIdx((i) => Math.max(0, i - 1));
      } else if (e.key === "/") {
        e.preventDefault();
        document.querySelector<HTMLInputElement>(".word-card.focused input")?.focus();
      } else if (e.key === "p" && audioUrl) {
        e.preventDefault();
        const f = flags[focusedWordIdx];
        const ch = chunkById(f.chunk_idx);
        if (ch) playFromChunk(ch);
      } else if (e.key === "a" || e.key === "Enter") {
        e.preventDefault();
        chooseSuggested(focusedWordIdx);
        advanceFocus();
      } else if (e.key === "0") {
        e.preventDefault();
        keepOriginal(focusedWordIdx);
        advanceFocus();
      } else if (/^[1-9]$/.test(e.key)) {
        e.preventDefault();
        const f = flags[focusedWordIdx];
        const n = Number.parseInt(e.key, 10);
        if (n === 1) chooseSuggested(focusedWordIdx);
        else if (f.alternatives[n - 2]) chooseAlternative(focusedWordIdx, f.alternatives[n - 2]);
        else return;
        advanceFocus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [
    audioUrl,
    mode,
    flags,
    focusedWordIdx,
    advanceFocus,
    chooseSuggested,
    chooseAlternative,
    keepOriginal,
    chunkById,
    playFromChunk,
  ]);

  // Scroll focused card into view when it changes. The effect body
  // doesn't literally read focusedWordIdx (it just queries the DOM), but
  // we want it to fire exactly when focus changes — so keep the dep.
  // biome-ignore lint/correctness/useExhaustiveDependencies: focusedWordIdx is the trigger, not a value read in the body
  useEffect(() => {
    if (mode !== "words") return;
    const el = document.querySelector<HTMLElement>(".word-card.focused");
    el?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [mode, focusedWordIdx]);

  // ---------- apply prompt ----------

  const buildApplyPrompt = useCallback((): string => {
    const overrideList = Array.from(overrides.entries())
      .sort(([a], [b]) => a - b)
      .map(([idx, spk]) => `- chunk ${idx} → speaker ${spk} (${speakerName(spk)})`);

    const wordCorrections: string[] = [];
    flags.forEach((f, i) => {
      const d = wordDecisions.get(i) ?? ({ kind: "suggested", value: f.suggested } as WordDecision);
      if (d.kind === "original") return;
      const suffix = d.kind === "suggested" ? " (default suggestion)" : "";
      wordCorrections.push(`- chunk ${f.chunk_idx}: "${f.original}" → "${d.value}"${suffix}`);
    });

    if (overrideList.length === 0 && wordCorrections.length === 0) {
      return `No overrides or corrections selected — the draft transcript at \`${transcriptPath}\` is accepted as-is.`;
    }
    const parts: string[] = [
      "Apply the following overrides to the polyphony draft transcript at:",
      `  ${transcriptPath}`,
      "",
    ];
    if (overrideList.length) {
      parts.push("Speaker overrides (chunk index → speaker):", ...overrideList, "");
    }
    if (wordCorrections.length) {
      parts.push("Word corrections (chunk index, original → replacement):", ...wordCorrections, "");
    }
    parts.push(
      "For each speaker override: reassign that chunk, recoalesce consecutive same-speaker chunks, and remove ⚠️ markers + 'review needed' HTML comments from fully-confirmed turns.",
      "For each word correction: find the exact 'original' span in the chunk's text and replace with the replacement, preserving whitespace and punctuation.",
      "",
      "Output the final corrected transcript file in place.",
    );
    return parts.join("\n");
  }, [overrides, flags, wordDecisions, speakerName, transcriptPath]);

  const copyApplyPrompt = useCallback(async () => {
    const text = buildApplyPrompt();
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopyStatus(true);
    window.setTimeout(() => setCopyStatus(false), 1500);
  }, [buildApplyPrompt]);

  // ---------- keyhint bar ----------

  const keyhint = useMemo<string[]>(() => {
    const playHints: string[] = [];
    if (audioUrl) {
      playHints.push("<kbd>space</kbd> play/pause");
      if (mode === "words") playHints.push("<kbd>p</kbd> play focused");
    }
    if (mode === "speakers") {
      return [
        "<kbd>drag slider</kbd> filter by confidence",
        "<kbd>click name</kbd> override speaker",
        "<kbd>select text</kbd> propose correction",
        "<kbd>Tab</kbd> switch view",
        ...playHints,
      ];
    }
    return [
      "<kbd>j</kbd>/<kbd>↓</kbd> next",
      "<kbd>k</kbd>/<kbd>↑</kbd> prev",
      "<kbd>1</kbd> suggested · <kbd>2-9</kbd> alt · <kbd>0</kbd> keep",
      "<kbd>/</kbd> custom",
      "<kbd>Tab</kbd> switch view",
      ...playHints,
    ];
  }, [audioUrl, mode]);

  const statsSpeakers = `${visibleTurns.length}/${turns.length} turns · ${overrides.size} override${overrides.size === 1 ? "" : "s"}`;
  const statsWords =
    flags.length === 0
      ? ""
      : `${focusedWordIdx + 1}/${flags.length} · ${decidedWordCount} custom decision${decidedWordCount === 1 ? "" : "s"}`;

  // ---------- render ----------

  return (
    <>
      <header>
        <div className="brand">
          <h1>polyphony</h1>
          <span className="subtitle" title={transcriptPath}>
            {audioName} · {transcriptPath}
          </span>
        </div>
        <div className="tabs">
          <button
            type="button"
            className={`tab ${mode === "speakers" ? "active" : ""}`}
            onClick={() => setMode("speakers")}
          >
            Speakers
            {lowConfTurnCount > 0 && <span className="tab-badge">{lowConfTurnCount}</span>}
          </button>
          <button
            type="button"
            className={`tab ${mode === "words" ? "active" : ""}`}
            onClick={() => setMode("words")}
          >
            Words
            {flags.length > 0 && <span className="tab-badge">{flags.length}</span>}
          </button>
        </div>

        {mode === "speakers" ? (
          <div className="controls">
            <div className="control-group">
              <label htmlFor="threshold">conf ≤</label>
              <input
                id="threshold"
                type="range"
                min="0"
                max="100"
                value={threshold}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  setThreshold(Number.parseInt(e.target.value, 10))
                }
              />
              <span className="threshold-value">{threshold}</span>
            </div>
            <div className="stats">{statsSpeakers}</div>
            <button type="button" className="primary" onClick={copyApplyPrompt}>
              Copy apply prompt
            </button>
            <span className={`copy-status${copyStatus ? " visible" : ""}`}>Copied!</span>
          </div>
        ) : (
          <div className="controls">
            <div className="stats">{statsWords}</div>
            <button type="button" className="primary" onClick={copyApplyPrompt}>
              Copy apply prompt
            </button>
            <span className={`copy-status${copyStatus ? " visible" : ""}`}>Copied!</span>
          </div>
        )}
      </header>

      <main>
        {mode === "speakers" ? (
          <div onMouseUp={onMouseupSpeakers}>
            {visibleTurns.map((turn) => {
              const first = turn.chunks[0];
              const last = turn.chunks[turn.chunks.length - 1];
              const isPlaying = turn.chunks.some((c) => c.idx === playingChunkIdx);
              return (
                <div
                  key={first.idx}
                  className={`turn${isPlaying ? " now-playing" : ""}`}
                  data-chunk-idx={first.idx}
                >
                  <div className="turn-header">
                    <div>
                      <span className={`speaker-name${turn.overridden ? " overridden" : ""}`}>
                        <span
                          className="conf-dot"
                          style={{ background: confColor(turn.minConf) }}
                        />
                        {speakerName(turn.speaker)}
                      </span>
                      <span className="meta" style={{ marginLeft: 12 }}>
                        <span className="time">
                          {fmtTime(first.start)}–{fmtTime(last.end)}
                        </span>
                        <span>conf {turn.minConf}</span>
                      </span>
                    </div>
                    {audioUrl && (
                      <button
                        type="button"
                        className="play-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          playFromTurn(turn);
                        }}
                      >
                        {fmtTime(first.start)}
                      </button>
                    )}
                  </div>

                  {turn.chunks.map((ch, ci) => (
                    <div
                      key={ch.idx}
                      style={
                        ci > 0
                          ? {
                              marginTop: 8,
                              paddingTop: 8,
                              borderTop: "1px dashed var(--border)",
                            }
                          : undefined
                      }
                    >
                      <div className="text" data-chunk-idx={ch.idx}>
                        {renderChunkTokens(ch).map((tok, i) =>
                          tok.type === "text" ? (
                            // biome-ignore lint/suspicious/noArrayIndexKey: token list is rebuilt atomically
                            <span key={i}>{tok.value}</span>
                          ) : (
                            // biome-ignore lint/suspicious/noArrayIndexKey: token list is rebuilt atomically
                            <span key={i}>
                              <span className="correction-orig">{tok.original}</span>
                              <span className="correction-arrow">→</span>
                              <span className={`correction-new${tok.user ? " user" : ""}`}>
                                {tok.replacement}
                              </span>
                            </span>
                          ),
                        )}
                      </div>
                      <div className="meta">
                        <span
                          className={`candidates${ch.pyannote != null && ch.claude != null && ch.pyannote !== ch.claude ? " disagree" : ""}`}
                        >
                          py={ch.pyannote ?? "∅"} cl={ch.claude ?? "∅"} → {ch.final} (conf{" "}
                          {ch.confidence})
                        </span>
                        {ch.note && <span className="note">· {ch.note}</span>}
                      </div>
                      <div className="actions">
                        {Array.from({ length: totalSpeakers }, (_, k) => k + 1).map((s) => {
                          const current = overrides.has(ch.idx)
                            ? (overrides.get(ch.idx) as number)
                            : ch.final;
                          return (
                            <button
                              type="button"
                              key={s}
                              className={current === s ? "active" : ""}
                              onClick={() => setSpeaker(ch, s)}
                            >
                              {speakerName(s)}
                            </button>
                          );
                        })}
                        {audioUrl && turn.chunks.length > 1 && (
                          <button
                            type="button"
                            className="play-btn"
                            onClick={() => playFromChunk(ch)}
                          >
                            {fmtTime(ch.start)}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        ) : (
          <div>
            {flags.length === 0 ? (
              <div className="empty">No suspected ASR errors flagged. 🎉</div>
            ) : (
              flags.map((flag, i) => {
                const dec = decisionOf(i);
                const chunk = chunkById(flag.chunk_idx);
                const ctxBefore = chunk?.text.includes(flag.original)
                  ? chunk.text.split(flag.original)[0]
                  : null;
                const ctxAfter = chunk?.text.includes(flag.original)
                  ? chunk.text.split(flag.original).slice(1).join(flag.original)
                  : null;
                const isPlaying = chunk?.idx === playingChunkIdx;
                // Composite key: chunk_idx + original span is stable across
                // reorders. A plain index would conflate entries when the
                // user adds a manual correction in the middle of the list.
                const cardKey = `${flag.chunk_idx}:${flag.original}`;
                return (
                  <div
                    key={cardKey}
                    className={`word-card${i === focusedWordIdx ? " focused" : ""}${dec.kind === "original" ? " resolved" : ""}${isPlaying ? " now-playing" : ""}`}
                    data-word-chunk={flag.chunk_idx}
                    // biome-ignore lint/a11y/useSemanticElements: the card contains nested buttons + inputs; wrapping it in a real <button> produces invalid HTML (interactive elements can't nest). div + role=button is the correct ARIA pattern here.
                    role="button"
                    tabIndex={0}
                    onClick={() => setFocusedWordIdx(i)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        setFocusedWordIdx(i);
                      }
                    }}
                  >
                    <div className="word-context">
                      {ctxBefore !== null ? (
                        <>
                          {ctxBefore}
                          <mark>{flag.original}</mark>
                          {ctxAfter}
                        </>
                      ) : (
                        chunk?.text || ""
                      )}
                    </div>
                    <div className="word-transform">
                      <span className="word-original">{flag.original}</span>
                      <span className="word-arrow">→</span>
                      <span
                        className={`word-suggested${dec.kind === "alternative" ? " chosen-alt" : ""}${dec.kind === "custom" ? " chosen-custom" : ""}`}
                      >
                        {dec.kind === "original" ? "(kept original)" : dec.value}
                      </span>
                      {audioUrl && chunk && (
                        <button
                          type="button"
                          className="play-btn"
                          style={{ marginLeft: "auto" }}
                          onClick={(e) => {
                            e.stopPropagation();
                            playFromChunk(chunk);
                          }}
                        >
                          {fmtTime(chunk.start)}
                        </button>
                      )}
                    </div>
                    {flag.reason && <div className="word-reason">{flag.reason}</div>}
                    <div className="word-alts">
                      <button
                        type="button"
                        className={
                          dec.kind === "suggested" && dec.value === flag.suggested ? "chosen" : ""
                        }
                        onClick={() => chooseSuggested(i)}
                      >
                        <span className="shortcut">1</span>
                        {flag.suggested}
                      </button>
                      {flag.alternatives.map((alt, k) => (
                        <button
                          type="button"
                          key={alt}
                          className={
                            dec.kind === "alternative" && dec.value === alt ? "chosen" : ""
                          }
                          onClick={() => chooseAlternative(i, alt)}
                        >
                          <span className="shortcut">{k + 2}</span>
                          {alt}
                        </button>
                      ))}
                      <button
                        type="button"
                        className={`orig${dec.kind === "original" ? " chosen" : ""}`}
                        onClick={() => keepOriginal(i)}
                      >
                        <span className="shortcut">0</span>keep original
                      </button>
                    </div>
                    <div className="word-custom">
                      <input
                        type="text"
                        placeholder="type a custom correction and press Enter"
                        defaultValue={dec.kind === "custom" ? dec.value : ""}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            const v = e.currentTarget.value.trim();
                            if (v) chooseDecision(i, "custom", v);
                          } else if (e.key === "Escape") {
                            e.currentTarget.blur();
                          }
                          e.stopPropagation();
                        }}
                      />
                    </div>
                    <div className="word-meta">
                      <span>
                        <span
                          className="conf-dot"
                          style={{ background: confColor(flag.confidence) }}
                        />
                        confidence {flag.confidence}
                      </span>
                      {chunk && (
                        <span>
                          chunk {flag.chunk_idx} · {fmtTime(chunk.start)}
                        </span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}
      </main>

      <div className="bottombar">
        {audioUrl && (
          <>
            <audio
              ref={playerRef}
              className="hidden-audio"
              src={audioUrl}
              preload="metadata"
              onTimeUpdate={onTimeupdate}
              onPause={onPause}
              onPlay={onPlay}
              onLoadedMetadata={onLoadedMetadata}
            >
              <track kind="captions" />
            </audio>
            <div className="mini-player">
              <button
                type="button"
                onClick={togglePlay}
                title={playing ? "Pause (space)" : "Play (space)"}
              >
                {playing ? "⏸" : "▶"}
              </button>
              <div className="scrub" onClick={seekFromClick} onKeyDown={() => {}}>
                <div className="scrub-fill" style={{ width: `${scrubPercent}%` }} />
              </div>
              <span className="time">
                {fmtTime(currentTime)} / {fmtTime(duration)}
              </span>
            </div>
          </>
        )}
        <div className="keybar">
          {keyhint.map((h, i) => (
            // biome-ignore lint/security/noDangerouslySetInnerHtml: hint strings are code-generated
            // biome-ignore lint/suspicious/noArrayIndexKey: static array
            <span key={i} dangerouslySetInnerHTML={{ __html: h }} />
          ))}
        </div>
      </div>

      {popover.open && (
        <div className="word-popover" style={{ top: popover.top, left: popover.left }}>
          <div className="label">
            Replace <code>&ldquo;{popover.original}&rdquo;</code> with:
          </div>
          <input
            ref={popoverInputRef}
            type="text"
            value={popover.value}
            onChange={(e) => setPopover((p) => ({ ...p, value: e.target.value }))}
            onKeyDown={onPopoverKeydown}
            spellCheck
            autoComplete="off"
          />
          <div className="row">
            <span className="hint">Enter = accept · Esc = cancel</span>
            <button type="button" onClick={cancelPopover}>
              Cancel
            </button>
            <button type="button" className="primary" onClick={acceptPopover}>
              Correct
            </button>
          </div>
        </div>
      )}
    </>
  );
}
