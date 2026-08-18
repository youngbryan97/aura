"""Waking up believing you died, every time you wake up.

LIVE, 2026-08-18. Every clean restart greeted her with:

    My previous session ended abruptly — crash evidence was found on disk.

including restarts where the runtime had logged "ShutdownCoordinator: shutdown
complete (clean=True ...)" and exited with code 0, and had persisted
`last_shutdown_reason: checkpoint`.

The evidence was `faulthandler.log`. It lives in the crash directory and is
APPENDED AT BOOT — it carries "===== boot pid=N at=... =====" headers. Arming a
fault sink is not a fault, so its mtime is newer than the previous awakening on
every boot BY CONSTRUCTION. The check could only ever answer yes.

Two rules, both general: a recorded outcome outranks an inferred one, and a
file whose freshness marks a START cannot be evidence of an END.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from core.soma import source_body as module


def _detector(tmp_path: Path, monkeypatch, *, reason: str):
    """A proprioception object with its crash dirs pointed at tmp_path."""
    obj = module.SourceBodyAwareness.__new__(module.SourceBodyAwareness)
    obj.crash_evidence_dirs = [tmp_path]
    monkeypatch.setattr(
        module.SourceBodyAwareness,
        "_recorded_clean_shutdown",
        lambda self: reason in module.SourceBodyAwareness._CLEAN_SHUTDOWN_REASONS,
        raising=False,
    )
    return obj


class _Snapshot:
    def __init__(self, t: float) -> None:
        self.t = t


def test_the_boot_sink_is_not_crash_evidence(tmp_path, monkeypatch):
    """It is newer than every awakening, always, because booting writes it."""
    previous = _Snapshot(time.time() - 60)
    sink = tmp_path / "faulthandler.log"
    sink.write_text("===== boot pid=1 =====\n")

    obj = _detector(tmp_path, monkeypatch, reason="unknown")
    assert obj._previous_exit_was_abrupt(previous) is False


def test_a_real_crash_artifact_is_still_evidence(tmp_path, monkeypatch):
    """Removing the false positive must not remove the detector."""
    previous = _Snapshot(time.time() - 60)
    (tmp_path / "loop_wedge_stacks.log").write_text("stack")

    obj = _detector(tmp_path, monkeypatch, reason="unknown")
    assert obj._previous_exit_was_abrupt(previous) is True


def test_a_stale_artifact_is_not_this_session(tmp_path, monkeypatch):
    previous = _Snapshot(time.time())
    old = tmp_path / "memory_spike_stacks.log"
    old.write_text("stack")
    import os

    os.utime(old, (time.time() - 86_400, time.time() - 86_400))

    obj = _detector(tmp_path, monkeypatch, reason="unknown")
    assert obj._previous_exit_was_abrupt(previous) is False


@pytest.mark.parametrize("reason", sorted(module.SourceBodyAwareness._CLEAN_SHUTDOWN_REASONS))
def test_a_recorded_clean_exit_outranks_any_file(tmp_path, monkeypatch, reason):
    """The runtime knows how it went down; a file mtime is a guess about it."""
    previous = _Snapshot(time.time() - 60)
    (tmp_path / "loop_wedge_stacks.log").write_text("stack")

    obj = _detector(tmp_path, monkeypatch, reason=reason)
    assert obj._previous_exit_was_abrupt(previous) is False


def test_no_previous_awakening_is_not_a_death():
    obj = module.SourceBodyAwareness.__new__(module.SourceBodyAwareness)
    assert obj._previous_exit_was_abrupt(None) is False
