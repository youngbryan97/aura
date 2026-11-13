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


def test_a_report_holding_numpy_values_is_written(tmp_path):
    """Her first scheduled audit was lost when a numpy bool reached json; the report is written now."""
    import json

    import numpy as np

    from core.consciousness.unified_audit import AuditReport

    report = AuditReport(
        timestamp=0.0, audit_id="audit_0", theory_results=[], consciousness_index=0.5, phi=0.1,
        phenomenal_active=True, global_workspace_ignited=False, free_energy=0.2, structural_opacity=0.3,
        causal_loop_active=False, summary="s", disclaimer="d",
        raw_metrics={"opacity_flag": np.bool_(True), "loop": np.float32(0.25), "trace": np.arange(3)},
    )
    path = tmp_path / "audit_0.json"
    asyncio.run(report.save_to_file(str(path)))
    raw = json.loads(path.read_text(encoding="utf-8"))["raw_metrics"]
    assert raw == {"opacity_flag": True, "loop": 0.25, "trace": [0, 1, 2]}


def test_a_full_audit_runs_and_saves(tmp_path):
    from core.consciousness.unified_audit import ConsciousnessAuditSuite

    suite = ConsciousnessAuditSuite()
    suite.REPORT_DIR = tmp_path
    report = asyncio.run(suite.run(save=True))
    assert (tmp_path / f"{report.audit_id}.json").exists()
    assert suite.get_trend().get("status") != "no audits yet"


def test_the_audit_reads_her_published_phi_and_the_synthesizers_own_keys(monkeypatch):
    """Her first audits read phi 0.000 beside a live 0.263, and PRI 0 from keys the synthesizer never had."""
    from core.consciousness import unified_audit
    from core.container import ServiceContainer
    from core.runtime import service_registry

    class PhiCore:
        current_phi = 0.26331
        history_length = 40

    class Synth:
        def get_snapshot(self):
            return {"pri": 0.42, "dominant_dimension": "precision", "history_depth": 12}

    monkeypatch.setattr(service_registry, "get_runtime_service",
                        lambda name, default=None: PhiCore() if name == "phi_core" else default)
    real_get = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get",
                        lambda name, default=None: Synth() if name == "qualia_synthesizer" else real_get(name, default))
    suite = unified_audit.ConsciousnessAuditSuite()
    iit, raw = asyncio.run(suite._gather_iit())
    assert iit.key_metric == 0.26331 and iit.criteria_met == [True, True, True] and raw["phi_samples"] == 40
    qualia, raw = asyncio.run(suite._gather_qualia())
    assert qualia.key_metric == 0.42 and qualia.criteria_met == [True, True, True]
    assert raw["dominant_dimension"] == "precision"
