"""Her consciousness audit runs, on the schedule it already had, and keeps its reports under her data directory.

Its scheduler had no caller, so the audit never ran in her app: agency's hourly self-development read "no audits
yet", and the opacity monitor the audit builds on first use was never built (13 November 2026, after the five organs
were switched on). Its reports went to a path relative to wherever the process stood.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.config import config


def test_reports_go_under_her_data_directory():
    from core.consciousness.unified_audit import ConsciousnessAuditSuite

    suite = ConsciousnessAuditSuite()
    assert Path(suite.REPORT_DIR) == Path(config.paths.data_dir) / "consciousness_reports"


def test_the_schedule_starts_a_running_audit_loop():
    from core.consciousness.unified_audit import ConsciousnessAuditSuite

    async def started() -> bool:
        suite = ConsciousnessAuditSuite()
        suite.schedule()
        task = suite._schedule_task
        alive = task is not None and not task.done()
        task.cancel()
        return alive

    assert asyncio.run(started())


def test_boot_starts_it():
    source = Path("core/orchestrator/mixins/boot/boot_cognitive.py").read_text(encoding="utf-8")
    assert "get_audit_suite().schedule()" in source
