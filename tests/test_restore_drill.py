"""Tests for the disaster-recovery restore drill."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core.runtime.backup_restore import (
    BACKUP_INCLUDED_DIRS,
    aura_home,
    perform_backup,
    perform_restore,
)
from core.runtime.restore_drill import (
    DrillReport,
    _fingerprint_tree,
    perform_drill,
)


# ---------------------------------------------------------------------------
# fingerprinting
# ---------------------------------------------------------------------------
def test_fingerprint_tree_is_stable(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    fp1 = _fingerprint_tree(tmp_path)
    fp2 = _fingerprint_tree(tmp_path)
    assert fp1 == fp2
    assert "a.txt" in fp1


def test_fingerprint_changes_on_edit(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello", encoding="utf-8")
    before = _fingerprint_tree(tmp_path)["a.txt"]
    f.write_text("hello!", encoding="utf-8")
    after = _fingerprint_tree(tmp_path)["a.txt"]
    assert before != after


# ---------------------------------------------------------------------------
# round-trip drill
# ---------------------------------------------------------------------------
def test_perform_drill_full_roundtrip_succeeds(tmp_path: Path):
    home = tmp_path / "home"
    backup = tmp_path / "backup"
    report = perform_drill(home_override=home, backup_target_override=backup)
    assert isinstance(report, DrillReport)
    assert report.ok is True, report.mismatches
    assert report.file_count > 0
    assert report.fingerprint_before == report.fingerprint_after
    assert report.mismatches == []
    assert Path(report.snapshot).exists()


def test_drill_report_serializes(tmp_path: Path):
    home = tmp_path / "home"
    backup = tmp_path / "backup"
    report = perform_drill(home_override=home, backup_target_override=backup)
    payload = report.to_dict()
    assert payload["ok"] is True
    assert "fingerprint_before" in payload
    assert "fingerprint_after" in payload
    assert "elapsed_seconds" in payload


def test_drill_uses_ephemeral_dirs_when_not_supplied(tmp_path: Path, monkeypatch):
    # Don't pass overrides; the drill should still complete and not
    # touch the real ~/.aura because seeded contents go into a fresh
    # tempdir.  We just check that ok=True and some files were
    # included so we know the path actually ran.
    report = perform_drill()
    assert report.ok is True
    assert report.file_count > 0
    # Cleanup the ephemeral homes the drill made.
    shutil.rmtree(report.home, ignore_errors=True)


# ---------------------------------------------------------------------------
# detection of corruption
# ---------------------------------------------------------------------------
def test_drill_detects_post_restore_corruption(tmp_path: Path, monkeypatch):
    """Force a content_mismatch by truncating a restored file before
    fingerprint compare.  The drill cannot do this on its own (it's a
    happy-path round trip), so we exercise the comparator directly."""
    from core.runtime.restore_drill import (
        _fingerprint_hash,
        _fingerprint_tree,
        DrillReport,
    )

    home = tmp_path / "home"
    home.mkdir()
    (home / "a.txt").write_text("good", encoding="utf-8")
    before = _fingerprint_tree(home)
    before_hash = _fingerprint_hash(before)

    # Simulate a "restore that lost a byte"
    (home / "a.txt").write_text("god", encoding="utf-8")
    after = _fingerprint_tree(home)
    after_hash = _fingerprint_hash(after)

    assert before_hash != after_hash
    mismatches = []
    for k in sorted(set(before) | set(after)):
        if before.get(k) != after.get(k):
            mismatches.append(k)
    assert mismatches == ["a.txt"]


# ---------------------------------------------------------------------------
# AURA_HOME override
# ---------------------------------------------------------------------------
def test_aura_home_respects_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AURA_HOME", str(tmp_path / "elsewhere"))
    assert aura_home() == (tmp_path / "elsewhere").resolve()


def test_perform_backup_honours_aura_home(monkeypatch, tmp_path: Path):
    home = tmp_path / "isolated_home"
    monkeypatch.setenv("AURA_HOME", str(home))
    (home / "state").mkdir(parents=True)
    (home / "state" / "x.json").write_text("{}", encoding="utf-8")

    target = tmp_path / "backups"
    info = perform_backup(target=target)
    assert info["ok"] is True
    assert "state" in info["included"]
    assert Path(info["snapshot"]).exists()


def test_backup_restore_round_trip_via_aura_home(monkeypatch, tmp_path: Path):
    home = tmp_path / "h"
    monkeypatch.setenv("AURA_HOME", str(home))
    (home / "state").mkdir(parents=True)
    (home / "state" / "vitality.json").write_text(
        '{"vitality": 0.5}', encoding="utf-8"
    )
    target = tmp_path / "b"
    info = perform_backup(target=target)
    snapshot = Path(info["snapshot"])

    # Wipe the live state then restore.
    shutil.rmtree(home / "state")
    assert not (home / "state" / "vitality.json").exists()

    restored = perform_restore(snapshot=snapshot)
    assert restored["ok"] is True
    assert (home / "state" / "vitality.json").exists()
    body = (home / "state" / "vitality.json").read_text(encoding="utf-8")
    assert "0.5" in body
