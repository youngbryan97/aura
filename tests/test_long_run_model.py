import json

from tools.long_run_model.profiles import get_profile
from tools.long_run_model.registry import build_registry
from tools.long_run_model.report import write_report_bundle
from tools.long_run_model.simulate import run_forecast


def _checkpoint(summary, horizon):
    for item in summary.checkpoints:
        if item.horizon == horizon:
            return item
    raise AssertionError(f"missing checkpoint: {horizon}")


def test_build_registry_extracts_runtime_hardening_contracts():
    registry = build_registry()

    assert registry.hardware_model == "Apple Silicon M5 MacBook Pro"
    assert registry.total_ram_gb == 64.0
    assert registry.total_storage_gb == 2000.0
    assert registry.backup_wired is True
    assert registry.database_coordinator_wired is True
    assert registry.conversation_prune_scheduled is True
    assert registry.lock_watchdog_auto_repair is True
    assert registry.state_queue_repair_enabled is True
    assert registry.pending_initiative_cap == 10
    assert registry.active_goal_cap == 10
    assert registry.vector_soft_prune_threshold_days == 14
    assert registry.cognitive_ledger_prune_days == 7
    assert registry.source_signature["critical_supervision_audit"]["all_resolved"] is True
    assert registry.source_signature["hash"]


def test_run_forecast_supports_all_profiles():
    registry = build_registry()

    for name in ("stress_load", "mixed_daily", "idle_heavy"):
        summary = run_forecast(get_profile(name), ["24h", "72h"], registry)

        assert [item.horizon for item in summary.checkpoints] == ["24h", "72h"]
        assert summary.profile["name"] == name
        assert summary.registry["hardware_model"] == "Apple Silicon M5 MacBook Pro"


def test_run_forecast_surfaces_requested_retention_cliffs():
    registry = build_registry()
    summary = run_forecast(get_profile("stress_load"), ["7d", "14d", "31d"], registry)

    cliff_7d = {item.name: item.reached for item in _checkpoint(summary, "7d").cliff_summary}
    cliff_14d = {item.name: item.reached for item in _checkpoint(summary, "14d").cliff_summary}
    cliff_31d = {item.name: item.reached for item in _checkpoint(summary, "31d").cliff_summary}

    assert cliff_7d["ledger_prune_window"] is True
    assert cliff_7d["vector_soft_prune_window"] is False
    assert cliff_14d["vector_soft_prune_window"] is True
    assert cliff_31d["conversation_retention"] is True
    assert cliff_31d["vector_retention"] is True


def test_run_forecast_restart_reentry_pressure_changes_by_restart_kind():
    registry = build_registry()
    summary = run_forecast(get_profile("stress_load"), ["31d"], registry)

    continuity = _checkpoint(summary, "31d").organism_summary["continuity"]
    restarts = continuity["restarts"]

    assert len(restarts) == 2
    assert restarts[0]["kind"] == "graceful"
    assert restarts[1]["kind"] == "abrupt"
    assert restarts[1]["continuity_pressure"] > restarts[0]["continuity_pressure"]
    assert "abrupt_shutdown" in continuity["continuity_scar"]


def test_run_forecast_bounds_pressure_and_tracks_repairs():
    registry = build_registry()
    summary = run_forecast(get_profile("stress_load"), ["24h", "48h", "72h", "31d"], registry)
    by_horizon = {item.horizon: item for item in summary.checkpoints}

    for checkpoint in summary.checkpoints:
        pressure = checkpoint.pressure_summary
        assert pressure["queue_depth"] <= registry.state_commit_queue_maxsize
        assert pressure["rss_gb"] < registry.total_ram_gb
        assert pressure["lock_hold_age_s"] <= registry.lock_watchdog_threshold_s

    assert by_horizon["24h"].maintenance_summary["scheduled_repair_actions"] <= 12
    assert by_horizon["31d"].maintenance_summary["scheduled_repair_actions"] <= 24
    assert summary.risk_ledger == []
    assert summary.remediation_backlog == []


def test_run_forecast_keeps_social_budget_off_the_floor_under_stress_load():
    registry = build_registry()
    summary = run_forecast(get_profile("stress_load"), ["24h", "31d"], registry)

    assert _checkpoint(summary, "24h").organism_summary["motivation"]["social"] > 25.0
    assert _checkpoint(summary, "31d").organism_summary["motivation"]["social"] > 10.0


def test_write_report_bundle_emits_markdown_and_json(tmp_path):
    summary = run_forecast(get_profile("stress_load"), ["24h", "14d", "31d"], build_registry())
    outputs = write_report_bundle(summary, tmp_path)

    markdown = (tmp_path / "forecast_report.md").read_text(encoding="utf-8")
    summary_json = json.loads((tmp_path / "forecast_summary.json").read_text(encoding="utf-8"))
    risk_json = json.loads((tmp_path / "risk_ledger.json").read_text(encoding="utf-8"))
    remediation_json = json.loads((tmp_path / "remediation_backlog.json").read_text(encoding="utf-8"))

    assert outputs["markdown"].endswith("forecast_report.md")
    assert "24h" in markdown
    assert "31d" in markdown
    assert summary_json["checkpoints"][0]["horizon"] == "24h"
    assert isinstance(risk_json, list)
    assert isinstance(remediation_json, list)
