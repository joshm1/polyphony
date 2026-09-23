"""File a recording and its polyphony outputs into a notes vault (e.g. Obsidian).

The LLM proposes a folder + base name by browsing the vault's folder/file
names through read-only tools confined to the vault; the reviewer confirms
or edits before anything moves. Moving renames the audio and every
`<stem>.*.transcript*` sibling together, so `polyphony serve` still finds
the sidecar next to the audio afterwards.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from glob import escape
from pathlib import Path
from urllib.parse import quote

from loguru import logger
from pydantic import BaseModel, Field

from .llm import run_structured

_MAX_LISTING = 300
_MAX_SEARCH_HITS = 50
_MAX_TRANSCRIPT_CHARS = 12_000


class FilingProposal(BaseModel):
    folder: str = Field(description="Folder relative to the vault root. Prefer an existing folder.")
    basename: str = Field(
        description="File name without extension: the vault's naming convention applied to what the recording is about."
    )
    reason: str = Field(description="≤25 words: why this folder and name.")


def inside_vault(vault: Path, relative: str) -> Path:
    """Resolve `relative` under `vault`, rejecting anything that escapes it."""
    root = vault.resolve()
    target = (root / relative.strip().strip("/")).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{relative!r} is outside the vault")
    return target


def clean_basename(name: str) -> str:
    name = re.sub(r'[/\\:*?"<>|]', "-", name).strip().strip(".")
    if not name:
        raise ValueError("file name is empty")
    return name


def recording_date(audio_path: Path) -> datetime:
    """When the audio was recorded: container metadata if present, else file creation time."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=creation_time", "-of", "json", str(audio_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        created = json.loads(out).get("format", {}).get("tags", {}).get("creation_time")
        if created:
            return datetime.fromisoformat(created.replace("Z", "+00:00")).astimezone()
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    st = audio_path.stat()
    return datetime.fromtimestamp(getattr(st, "st_birthtime", st.st_mtime)).astimezone()


def _vault_tools(vault: Path):
    root = vault.resolve()

    def list_folder(path: str = "") -> str:
        """List the subfolders (suffixed with /) and files in a vault folder.

        `path` is relative to the vault root; "" is the root.
        """
        try:
            folder = inside_vault(root, path)
        except ValueError as e:
            return str(e)
        if not folder.is_dir():
            return f"Not a folder: {path!r}"
        entries = sorted((e for e in folder.iterdir() if not e.name.startswith(".")), key=lambda e: e.name.lower())
        names = [f"{e.name}/" for e in entries if e.is_dir()] + [e.name for e in entries if e.is_file()]
        more = f"\n…and {len(names) - _MAX_LISTING} more" if len(names) > _MAX_LISTING else ""
        return "\n".join(names[:_MAX_LISTING]) + more or "(empty)"

    def search_vault(query: str) -> str:
        """Find vault folders and files whose path contains `query` (case-insensitive).

        Returns vault-relative paths.
        """
        needle = query.lower()
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            rel_dir = Path(dirpath).relative_to(root)
            for name in [*(f"{d}/" for d in dirnames), *filenames]:
                rel = str(rel_dir / name) if str(rel_dir) != "." else name
                if needle in rel.lower():
                    hits.append(rel)
                    if len(hits) >= _MAX_SEARCH_HITS:
                        return "\n".join(hits) + "\n…(more results truncated; refine the query)"
        return "\n".join(hits) or "(no matches)"

    return [list_folder, search_vault]


def propose_location(
    vault: Path,
    audio_path: Path,
    transcript_markdown: str,
    names: list[str],
    context_hint: str | None,
    model: str,
) -> FilingProposal:
    date = recording_date(audio_path)
    named = ", ".join(n for n in names if n) or "(not named)"
    excerpt = transcript_markdown[:_MAX_TRANSCRIPT_CHARS]
    truncated = "\n…(transcript truncated)" if len(transcript_markdown) > _MAX_TRANSCRIPT_CHARS else ""
    prompt = f"""Pick where a recording's transcript belongs in the user's notes vault, and what to name it.

Explore the vault with the tools before deciding: look at the top-level folders,
search for folders related to the conversation's subject, people, or organization,
and look at how similar recordings or notes nearby are named.

Rules:
- Prefer an existing folder. Only propose a new folder when nothing fits, and then
  only one level under the closest existing folder.
- If similar recordings nearby live in a dedicated subfolder, use that pattern.
- Build the name from the recording date and what the conversation is about (who
  and what, in a few words), formatted like its neighbors (date format,
  separators, casing). If nearby files share no convention, use
  "YYYY-MM-DD <short description>".
- The name must not include a file extension.

Recording date: {date:%Y-%m-%d %H:%M}
Speakers: {named}
Context hint: {context_hint or "(none)"}

Transcript:
{excerpt}{truncated}
"""
    logger.info(f"Proposing a vault location via {model}…")
    return run_structured(prompt, FilingProposal, model, tools=_vault_tools(vault))


def planned_moves(audio_path: Path, dest_dir: Path, basename: str) -> list[tuple[Path, Path]]:
    """The audio plus every `<stem>.*.transcript*` sibling, renamed to `basename`."""
    stem = audio_path.stem
    siblings = sorted(audio_path.parent.glob(f"{escape(stem)}.*.transcript*"))
    moves = [(audio_path, dest_dir / f"{basename}{audio_path.suffix}")]
    moves += [(p, dest_dir / f"{basename}{p.name[len(stem) :]}") for p in siblings]
    return moves


def move_recording(audio_path: Path, vault: Path, folder: str, basename: str) -> list[tuple[Path, Path]]:
    """Move + rename everything; rewrite paths inside moved sidecars. Raises FileExistsError on any clash."""
    dest_dir = inside_vault(vault, folder)
    moves = planned_moves(audio_path.resolve(), dest_dir, clean_basename(basename))
    if clashes := [dst for _, dst in moves if dst.exists()]:
        raise FileExistsError(f"Already exists: {', '.join(str(c) for c in clashes)}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        shutil.move(src, dst)
        logger.info(f"Moved {src} → {dst}")

    new_audio = moves[0][1]
    for _, dst in moves:
        if dst.name.endswith(".polyphony.json"):
            data = json.loads(dst.read_text())
            data["audio"] = new_audio.name
            # `transcribe` writes the raw transcript as the sidecar's name with `.polyphony.json` → `.md`.
            raw = dst.with_name(dst.name.removesuffix(".polyphony.json") + ".md")
            if raw.exists():
                data["transcript_path"] = str(raw)
            dst.write_text(json.dumps(data, indent=2))
    return moves


def obsidian_url(vault: Path, note: Path) -> str:
    rel = note.resolve().relative_to(vault.resolve())
    return f"obsidian://open?vault={quote(vault.resolve().name)}&file={quote(str(rel))}"
