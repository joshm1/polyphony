"""Tests for moving a recording + its outputs into a vault."""

from __future__ import annotations

import json

import pytest

from polyphony.filing import clean_basename, inside_vault, move_recording, obsidian_url


def _recording(tmp_path):
    src = tmp_path / "Downloads"
    src.mkdir()
    audio = src / "Voice Memo [x].m4a"  # glob metacharacters must not break sibling matching
    audio.write_bytes(b"audio")
    raw = src / "Voice Memo [x].assemblyai.transcript.md"
    raw.write_text("raw")
    (src / "Voice Memo [x].assemblyai.transcript.reviewed.md").write_text("reviewed")
    (src / "Voice Memo [x].assemblyai.transcript.polyphony.json").write_text(
        json.dumps({"audio": audio.name, "transcript_path": str(raw)})
    )
    (src / "unrelated.md").write_text("stay")
    vault = tmp_path / "Vault"
    vault.mkdir()
    return audio, vault


def test_inside_vault_rejects_escape(tmp_path):
    assert inside_vault(tmp_path, "a/b") == (tmp_path / "a/b").resolve()
    with pytest.raises(ValueError):
        inside_vault(tmp_path, "../elsewhere")


def test_clean_basename_strips_path_characters():
    assert clean_basename(" 2026-09-23 Visit: follow/up ") == "2026-09-23 Visit- follow-up"
    with pytest.raises(ValueError):
        clean_basename(" . ")


def test_move_recording_renames_everything_and_rewrites_sidecar(tmp_path):
    audio, vault = _recording(tmp_path)
    moves = move_recording(audio, vault, "Health/Visits", "2026-09-23 Follow-up")
    dest = vault / "Health" / "Visits"
    assert sorted(p.name for p in dest.iterdir()) == [
        "2026-09-23 Follow-up.assemblyai.transcript.md",
        "2026-09-23 Follow-up.assemblyai.transcript.polyphony.json",
        "2026-09-23 Follow-up.assemblyai.transcript.reviewed.md",
        "2026-09-23 Follow-up.m4a",
    ]
    assert len(moves) == 4
    assert (audio.parent / "unrelated.md").exists()
    sidecar = json.loads((dest / "2026-09-23 Follow-up.assemblyai.transcript.polyphony.json").read_text())
    assert sidecar["audio"] == "2026-09-23 Follow-up.m4a"
    assert sidecar["transcript_path"] == str(dest / "2026-09-23 Follow-up.assemblyai.transcript.md")


def test_move_recording_refuses_to_overwrite(tmp_path):
    audio, vault = _recording(tmp_path)
    (vault / "Notes").mkdir()
    (vault / "Notes" / "Taken.m4a").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        move_recording(audio, vault, "Notes", "Taken")
    assert audio.exists()  # nothing moved


def test_obsidian_url_is_vault_relative(tmp_path):
    note = tmp_path / "Health" / "a b.md"
    assert obsidian_url(tmp_path, note) == f"obsidian://open?vault={tmp_path.name}&file=Health/a%20b.md"
