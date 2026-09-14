"""A periodic backup measured from boot never runs on a runtime that restarts first.

The job registered with `last_run=now` at every boot, so the first backup was
a day after boot. Seventy-seven boots in the desktop log, never a day apart,
and not one "Creating backup" line. Measured from the newest archive on disk,
a runtime that restarts every few hours still backs up once a day.
"""

from __future__ import annotations

import os
import time

import pytest

from core.ops.backup import BackupManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    m = BackupManager()
    m.backup_dir = tmp_path / "backups"
    m.backup_dir.mkdir()
    m.data_dir = tmp_path / "data"
    m.data_dir.mkdir()
    return m


async def _registered(manager, monkeypatch):
    from core import scheduler as sched

    specs = []

    async def register(spec):
        specs.append(spec)

    monkeypatch.setattr(sched.scheduler, "register", register)
    await manager.on_start_async()
    return {s.name: s for s in specs}


@pytest.mark.asyncio
async def test_with_no_archive_the_first_backup_is_due_after_the_grace(manager, monkeypatch):
    before = time.monotonic()
    specs = await _registered(manager, monkeypatch)
    backup = specs["periodic_state_backup"]
    due_in = backup.last_run + backup.tick_interval - before
    assert 14 * 60 <= due_in <= 16 * 60, f"due in {due_in:.0f}s; expected about fifteen minutes"


@pytest.mark.asyncio
async def test_a_stale_archive_makes_the_backup_due_soon(manager, monkeypatch):
    old = manager.backup_dir / "aura_state_20260101_000000-1.tar.gz"
    old.write_bytes(b"x")
    two_days_ago = time.time() - 2 * 86400
    os.utime(old, (two_days_ago, two_days_ago))
    before = time.monotonic()
    specs = await _registered(manager, monkeypatch)
    backup = specs["periodic_state_backup"]
    due_in = backup.last_run + backup.tick_interval - before
    assert due_in <= 16 * 60, f"a two-day-old archive left the backup due in {due_in:.0f}s"


@pytest.mark.asyncio
async def test_a_fresh_archive_keeps_the_daily_cadence(manager, monkeypatch):
    fresh = manager.backup_dir / "aura_state_20260913_000000-1.tar.gz"
    fresh.write_bytes(b"x")
    an_hour_ago = time.time() - 3600
    os.utime(fresh, (an_hour_ago, an_hour_ago))
    before = time.monotonic()
    specs = await _registered(manager, monkeypatch)
    backup = specs["periodic_state_backup"]
    due_in = backup.last_run + backup.tick_interval - before
    assert 22 * 3600 <= due_in <= 23.5 * 3600, f"an hour-old archive left the backup due in {due_in / 3600:.1f}h"
