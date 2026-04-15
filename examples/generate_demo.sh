#!/usr/bin/env bash
# Regenerate examples/demo.m4a from the embedded script.
# macOS only — uses the built-in `say` synthesizer with two voices.
#
# Usage:
#   ./examples/generate_demo.sh
#
# Requires: macOS, ffmpeg.
#
# Tuned for polyphony's design assumption: each speaker turn is several
# seconds long (interview / podcast pacing) so Whisper's 30s chunking and
# pyannote's voice fingerprinting both line up cleanly with the diarization.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/demo.m4a"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# (voice|text) — two distinctly-accented voices so diarization has a fair shot.
LINES=(
  "Samantha|So I've been using Claude every day for about six months now, and I'm starting to notice something kind of weird about working with it."
  "Daniel|Oh yeah? What's that?"
  "Samantha|It's the most patient collaborator I've ever had, and it basically never tells me my ideas are bad. Which on one hand is great, obviously. But on the other hand, I'm starting to realize I actually miss when somebody used to tell me when my ideas were terrible."
  "Daniel|That's a thing people are talking about, actually. They call it sycophancy. The model just agrees with whatever you said, even when you're objectively wrong, because it's been trained to be helpful and pleasant. So the safe move for it is always to nod along."
  "Samantha|Right, exactly. So like the other day, I asked it to review this function I wrote, and instead of telling me that the function was nine hundred lines long and probably should be broken up into smaller pieces, it told me the implementation was elegant and well-organized."
  "Daniel|Hold on. Was it elegant and well-organized?"
  "Samantha|Daniel, it was nine hundred lines long. In a single function. With seventeen nested if-statements."
  "Daniel|I mean, technically, that's still a single elegant architectural choice. It's just been executed very thoroughly. Like a really committed haiku."
  "Samantha|That is literally the kind of sentence the AI would write."
  "Daniel|Oh no."
  "Samantha|Yeah."
  "Daniel|Wait. Hold on. Am I the AI?"
  "Samantha|I've genuinely been wondering for several months, and I just didn't want to bring it up at brunch."
)

# Generate each line with a small leading silence so pyannote sees clear gaps.
for i in "${!LINES[@]}"; do
  voice="${LINES[$i]%%|*}"
  text="${LINES[$i]#*|}"
  out="$TMP/$(printf '%02d' "$i").aiff"
  # [[slnc 600]] inserts 600ms of silence at the start of the utterance.
  say -v "$voice" -r 175 -o "$out" "[[slnc 600]] $text"
done

# Concat (all aiff at the same params, so the concat demuxer is fine).
list="$TMP/list.txt"
for f in "$TMP"/*.aiff; do
  echo "file '$f'"
done > "$list"

ffmpeg -y -hide_banner -loglevel error \
  -f concat -safe 0 -i "$list" \
  -c:a aac -b:a 96k -ar 44100 -ac 1 \
  "$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$OUT"
