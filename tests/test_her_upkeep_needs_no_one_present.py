"""Her upkeep needs no person present (third day run, 6 November 2026).

The user anchor held back her periodic backup (the newest state backup was from 13 September) and queued her biological
sleep all day. Self-modification stays anchored.
"""
from __future__ import annotations

import asyncio

from core.runtime import background_policy


def _policy_spy(monkeypatch, seen):
    def fake(orchestrator=None, *, profile=None, allow_no_user_anchor=False, **kwargs):
        seen.append(allow_no_user_anchor)
        return "" if allow_no_user_anchor else "no_user_anchor"

    monkeypatch.setattr(background_policy, "background_activity_reason", fake)


def test_her_backup_is_not_held_for_want_of_a_person(monkeypatch):
    from core.ops import backup

    seen: list[bool] = []
    _policy_spy(monkeypatch, seen)
    monkeypatch.setattr(backup, "resolve_orchestrator", lambda: None)
    target = backup.BackupManager.__new__(backup.BackupManager)
    assert target._maintenance_block_reason() == ""
    assert seen == [True]


def test_her_sleep_runs_unattended_and_self_modification_does_not_come_through_here(monkeypatch):
    from core.maintenance import dream_coordinator as module

    seen: list[bool] = []
    _policy_spy(monkeypatch, seen)
    ran = []

    async def sleep_cycle():
        ran.append("slept")

    coordinator = module.DreamCoordinator()
    coordinator._last_run.pop("biological_sleep", None)
    assert asyncio.run(coordinator.run_if_due("biological_sleep", sleep_cycle, 7200, priority=2)) is True
    assert ran == ["slept"] and seen[-1] is True
    coordinator._last_run.pop("something_else", None)
    assert asyncio.run(coordinator.run_if_due("something_else", sleep_cycle, 7200)) is False
    assert seen[-1] is False
