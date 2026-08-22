from __future__ import annotations

import json
import shutil
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel
from core.runtime.subprocess_gateway import get_subprocess_gateway


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before timeout")


def _fallback() -> dict[str, object]:
    return {"status": "booting", "healthy": False, "blockers": ["initializing"]}


def test_read_model_normalizes_retry_bounds():
    model = HealthSnapshotReadModel(
        lambda: {"status": "ok"},
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=-1.0,
            max_stale_s=-1.0,
            collection_timeout_s=-1.0,
            retry_base_s=-1.0,
            retry_max_s=-2.0,
        ),
    )

    assert model.config.refresh_interval_s == 0.05
    assert model.config.max_stale_s == 0.05
    assert model.config.collection_timeout_s == 0.05
    assert model.config.retry_base_s == 0.05
    assert model.config.retry_max_s == 0.05


def test_read_model_supports_named_snapshot_identity():
    model = HealthSnapshotReadModel(
        lambda: {"healthy": True},
        lambda: {"healthy": False},
        config=HealthReadModelConfig(
            refresh_interval_s=1.0,
            schema_version="aura.integrity.snapshot.v1",
            metadata_key="integrity_read_model",
            worker_name_prefix="AuraIntegritySnapshot",
            incident_prefix="integrity-refresh",
            log_label="Integrity snapshot",
        ),
    )

    assert model.start() is True
    _wait_until(lambda: model.read().get("healthy") is True)
    payload = model.read()

    assert "health_read_model" not in payload
    assert payload["integrity_read_model"]["schema_version"] == (
        "aura.integrity.snapshot.v1"
    )
    assert payload["integrity_read_model"]["fresh"] is True


def test_read_model_never_joins_blocked_collector_and_keeps_one_worker():
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(1.0)
        return {"status": "ok", "healthy": True, "blockers": []}

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.05,
            max_stale_s=0.1,
            collection_timeout_s=0.02,
            retry_base_s=0.01,
            retry_max_s=0.05,
        ),
    )

    started_at = time.monotonic()
    first = model.read()
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.05
    assert first["status"] == "booting"
    assert first["health_read_model"]["serving"] == "initializing"
    assert first["health_read_model"]["refresh_in_flight"] is True
    assert started.wait(0.2)

    for _ in range(20):
        assert model.read()["health_read_model"]["refresh_in_flight"] is True
    assert calls == 1

    time.sleep(0.06)
    timed_out = model.read()["health_read_model"]
    assert timed_out["refresh_timed_out"] is True
    assert timed_out["total_timeouts"] == 1
    assert timed_out["consecutive_failures"] == 1
    assert timed_out["incident_id"] == "health-refresh-000001"

    release.set()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    recovered = model.read()["health_read_model"]
    assert recovered["total_refreshes"] == 1
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_recovery"]["incident_id"] == "health-refresh-000001"
    assert recovered["last_recovery"]["failed_refreshes"] == 1


def test_close_and_restart_do_not_overlap_collectors():
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        current_call = calls
        if current_call == 1:
            first_started.set()
            assert release_first.wait(1.0)
        return {"status": "ok", "call": current_call}

    model = HealthSnapshotReadModel(collect, _fallback)
    assert model.start() is True
    assert first_started.wait(0.2)

    model.close()
    assert model.start() is False
    assert calls == 1

    release_first.set()
    _wait_until(lambda: model.read().get("call") == 2)
    assert calls == 2


def test_read_model_marks_old_success_expired_during_singleflight_refresh():
    block_refresh = threading.Event()
    refresh_started = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls > 1:
            refresh_started.set()
            assert block_refresh.wait(1.0)
        return {
            "status": "ok",
            "healthy": True,
            "connected": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "certification_ready": True,
            "runtime_revision": {
                "schema": "aura.runtime_revision.v2",
                "required": True,
                "verified": True,
                "revision_token": "a" * 64,
            },
            "required_probes": {"all_passed": True},
            "blockers": [],
            "readiness_contract": {
                "healthy": True,
                "system_ready": True,
                "conversation_ready": True,
                "runtime_probe_healthy": True,
                "certification_ready": True,
                "required_probes": {"all_passed": True},
                "blockers": [],
            },
            "boot": {
                "status": "ready",
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "required_probes": {"all_passed": True},
                "blockers": [],
            },
        }

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.05,
            max_stale_s=0.06,
            collection_timeout_s=0.2,
            retry_base_s=0.01,
            retry_max_s=0.05,
        ),
    )
    model.start()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    time.sleep(0.07)

    expired = model.read()
    assert refresh_started.wait(0.2)
    assert calls == 2
    assert expired["healthy"] is True
    assert expired["health_read_model"]["serving"] == "expired"
    assert expired["health_read_model"]["refresh_in_flight"] is True

    from interface.routes.system import _apply_health_read_model_truth

    truthful = _apply_health_read_model_truth(expired)
    assert truthful["status"] == "stale"
    assert truthful["healthy"] is False
    assert truthful["connected"] is False
    assert truthful["conversation_ready"] is False
    assert truthful["runtime_probe_healthy"] is False
    assert truthful["certification_ready"] is False
    assert truthful["required_probes"]["all_passed"] is False
    assert truthful["readiness_contract"]["healthy"] is False
    assert truthful["boot"]["ready"] is False
    assert truthful["blockers"][0] == "health_snapshot_expired"
    assert truthful["runtime_revision"]["verified"] is False
    assert truthful["runtime_revision"]["revision_token"] == ""
    assert truthful["runtime_revision"]["required"] is True
    assert truthful["runtime_revision"]["issues"] == ["health_snapshot_expired"]

    block_refresh.set()


def test_read_model_coalesces_failure_episode_and_reports_one_recovery():
    outcomes: list[object] = [RuntimeError("collector unavailable"), {"status": "ok"}]

    def collect() -> dict[str, object]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.01,
            max_stale_s=0.03,
            collection_timeout_s=0.1,
            retry_base_s=0.01,
            retry_max_s=0.02,
        ),
    )
    model.start()
    _wait_until(
        lambda: model.read()["health_read_model"]["consecutive_failures"] == 1
    )
    failed = model.read()["health_read_model"]
    assert failed["incident_id"] == "health-refresh-000001"
    assert failed["total_failures"] == 1

    time.sleep(0.02)
    model.request_refresh()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    recovered = model.read()["health_read_model"]
    assert recovered["incident_id"] is None
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_recovery"] == {
        "incident_id": "health-refresh-000001",
        "failed_refreshes": 1,
        "recovered_at_unix": recovered["last_recovery"]["recovered_at_unix"],
    }


@pytest.mark.asyncio
async def test_api_health_returns_initial_snapshot_while_collector_is_blocked(monkeypatch):
    from interface.routes import system as system_routes

    release = threading.Event()
    started = threading.Event()

    def collect() -> dict[str, object]:
        started.set()
        assert release.wait(1.0)
        return {"status": "ok", "healthy": True}

    model = HealthSnapshotReadModel(
        collect,
        system_routes._health_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=1.0,
            max_stale_s=2.0,
            collection_timeout_s=0.5,
        ),
    )
    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", model)
    monkeypatch.setattr(
        system_routes,
        "_restore_owner_session_from_request",
        lambda _request: None,
    )
    monkeypatch.setattr(
        system_routes,
        "_mark_runtime_service_progress",
        lambda _source: None,
    )

    started_at = time.monotonic()
    response = await system_routes.api_health(SimpleNamespace(headers={}))
    elapsed = time.monotonic() - started_at
    payload = json.loads(response.body)

    assert elapsed < 0.05
    assert started.wait(0.2)
    assert payload["status"] == "booting"
    assert payload["healthy"] is False
    assert payload["blockers"][0] == "health_snapshot_initializing"
    assert payload["health_read_model"]["refresh_in_flight"] is True
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-aura-health-generation"] == "0"
    assert response.headers["x-aura-health-serving"] == "initializing"
    release.set()


@pytest.mark.asyncio
async def test_readyz_uses_cached_canonical_readiness_without_inline_probe(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    required_probes = {
        group: {
            "ok": True,
            "components": {component: True for component in components},
        }
        for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    payload = {
        "status": "ok",
        "healthy": True,
        "ready": True,
        "connected": True,
        "uptime": 321.5,
        "blockers": [],
        "required_probes": required_probes,
        "readiness_contract": {
            "healthy": True,
            "system_ready": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "required_probes": required_probes,
            "blockers": [],
        },
        "health_read_model": {
            "expired": False,
            "snapshot_generation": 9,
            "age_s": 1.25,
            "serving": "fresh",
        },
        "runtime_revision": {
            "schema": "aura.runtime_revision.v2",
            "required": False,
            "verified": False,
            "source_verified": False,
            "revision_token": "",
        },
    }

    class ReadModel:
        def read(self):
            return payload

    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", ReadModel())
    monkeypatch.setattr(system_routes, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": False, "request": {"requested": False}},
    )

    started = time.monotonic()
    response = await system_routes.readyz(SimpleNamespace(headers={}))
    elapsed = time.monotonic() - started
    result = json.loads(response.body)

    assert elapsed < 0.05
    assert response.status_code == 200
    assert result == {
        "status": "ready",
        "ready": True,
        "issues": [],
        "uptime_s": 321.5,
        "conversation_ready": True,
        "runtime_probe_healthy": True,
        "required_probes_passed": True,
        "snapshot_generation": 9,
        "snapshot_age_s": 1.25,
        "serving": "fresh",
    }


@pytest.mark.asyncio
async def test_readyz_fails_closed_when_health_snapshot_is_expired(monkeypatch):
    from interface.routes import system as system_routes

    payload = system_routes._health_snapshot_fallback()
    payload["health_read_model"] = {
        "expired": True,
        "captured_at_unix": 1.0,
        "snapshot_generation": 3,
        "age_s": 31.0,
        "serving": "expired",
    }

    class ReadModel:
        def read(self):
            return payload

    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", ReadModel())
    monkeypatch.setattr(system_routes, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": False, "request": {"requested": False}},
    )

    response = await system_routes.readyz(SimpleNamespace(headers={}))
    result = json.loads(response.body)

    assert response.status_code == 503
    assert result["ready"] is False
    assert result["issues"][0] == "health_snapshot_expired"
    assert result["serving"] == "expired"


def test_public_health_route_applies_shutdown_truth_to_cached_success(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": True, "request": {"requested": True}},
    )
    payload = {
        "status": "ok",
        "healthy": True,
        "connected": True,
        "conversation_ready": True,
        "runtime_probe_healthy": True,
        "certification_ready": True,
        "required_probes": {"all_passed": True},
        "blockers": [],
        "readiness_contract": {
            "healthy": True,
            "system_ready": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "certification_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
        },
        "boot": {
            "status": "ready",
            "ready": True,
            "system_ready": True,
            "conversation_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
        },
    }

    result = system_routes._apply_current_shutdown_truth(payload)

    assert result["status"] == "stopping"
    assert result["healthy"] is False
    assert result["connected"] is False
    assert result["conversation_ready"] is False
    assert result["required_probes"]["all_passed"] is False
    assert result["blockers"][0] == "runtime_shutdown"
    assert result["shutdown"]["request"]["requested"] is True


@pytest.mark.asyncio
async def test_snapshot_collector_observes_without_constructing_services(monkeypatch):
    from interface.routes import system as system_routes

    def forbidden_get(*_args, **_kwargs):
        raise AssertionError("health snapshot collection must not construct a service")

    monkeypatch.setattr(system_routes.ServiceContainer, "get", forbidden_get)
    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "peek",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(system_routes, "get_runtime_state", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_integrity_report",
        lambda: {
            "healthy": False,
            "concerns": ["not_sampled"],
            "advisory": [],
        },
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [0.0] if percpu else 0.0,
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0),
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "disk_usage",
        lambda _path: SimpleNamespace(percent=0.0),
    )
    revision = {
        "schema": "aura.runtime_revision.v2",
        "required": True,
        "verified": True,
        "source_verified": True,
        "revision_token": "f" * 64,
        "expected_source_root_sha256": "b" * 64,
        "actual_source_root_sha256": "b" * 64,
        "expected_commit_sha": "a" * 40,
        "actual_commit_sha": "a" * 40,
        "expected_workspace_state_sha256": "c" * 64,
        "actual_workspace_state_sha256": "c" * 64,
        "expected_shell_assets_sha256": "d" * 64,
        "actual_shell_assets_sha256": "d" * 64,
        "capture_stable": True,
        "launch_mode": "signed_app",
        "issues": [],
    }
    monkeypatch.setattr(system_routes, "_runtime_revision_contract", lambda: revision)

    payload = await system_routes._collect_api_health_payload(
        allow_owner_loop_reads=False
    )

    assert isinstance(payload, dict)
    assert isinstance(payload["healthy"], bool)
    assert isinstance(payload["conversation_ready"], bool)
    assert isinstance(payload["readiness_contract"], dict)
    assert payload["runtime_revision"] == revision


def test_runtime_revision_contract_covers_exact_source_workspace_and_shell_identity(
    monkeypatch,
    tmp_path,
):
    from interface.routes import system as system_routes

    source_root = tmp_path / "source"
    source_root.mkdir()
    workspace = "b" * 64
    provenance = {
        "required": True,
        "verified": True,
        "source_verified": True,
        "launch_mode": "signed_app",
        "expected": {
            "source_root": str(source_root),
            "commit_sha": "a" * 40,
            "workspace_state_sha256": workspace,
        },
        "actual": {
            "source_root": str(source_root),
            "commit_sha": "a" * 40,
            "workspace_state_sha256": workspace,
        },
        "manifest": {"shell_assets_sha256": "c" * 64},
        "issues": [],
    }
    exact = system_routes._runtime_revision_from_provenance(
        provenance,
        shell_assets_sha256="c" * 64,
    )
    changed_shell = system_routes._runtime_revision_from_provenance(
        provenance,
        shell_assets_sha256="d" * 64,
    )
    missing_signed_shell = json.loads(json.dumps(provenance))
    missing_signed_shell["manifest"].pop("shell_assets_sha256")
    unsigned_shell = system_routes._runtime_revision_from_provenance(
        missing_signed_shell,
        shell_assets_sha256="c" * 64,
    )
    unstable_capture = system_routes._runtime_revision_from_provenance(
        provenance,
        shell_assets_sha256="c" * 64,
        capture_stable=False,
    )
    mismatched_workspace = json.loads(json.dumps(provenance))
    mismatched_workspace["actual"]["workspace_state_sha256"] = "e" * 64
    workspace_mismatch = system_routes._runtime_revision_from_provenance(
        mismatched_workspace,
        shell_assets_sha256="c" * 64,
    )
    mismatched_root = json.loads(json.dumps(provenance))
    mismatched_root["actual"]["source_root"] = str(tmp_path / "other-source")
    root_mismatch = system_routes._runtime_revision_from_provenance(
        mismatched_root,
        shell_assets_sha256="c" * 64,
    )
    mismatched_commit = json.loads(json.dumps(provenance))
    mismatched_commit["actual"]["commit_sha"] = "f" * 40
    commit_mismatch = system_routes._runtime_revision_from_provenance(
        mismatched_commit,
        shell_assets_sha256="c" * 64,
    )
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", None)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE_COLLECTED_AT", 0.0)

    fallback = system_routes._runtime_revision_fallback_contract()

    assert exact["verified"] is True
    assert exact["schema"] == "aura.runtime_revision.v2"
    assert len(exact["revision_token"]) == 64
    assert exact["expected_source_root_sha256"] == exact["actual_source_root_sha256"]
    assert exact["expected_workspace_state_sha256"] == workspace
    assert exact["actual_workspace_state_sha256"] == workspace
    assert exact["expected_shell_assets_sha256"] == "c" * 64
    assert exact["actual_shell_assets_sha256"] == "c" * 64
    assert exact["source_current"] is True
    # Shell, workspace and commit are MEASURED, so disagreeing with the
    # build-time manifest means the workspace moved on — reported as
    # source_current False, not as a failed identity. Requiring agreement made
    # every commit "unverified" and left the revision token permanently empty,
    # which is what forced a rebuild after each change.
    assert changed_shell["verified"] is True
    assert changed_shell["source_current"] is False
    assert len(changed_shell["revision_token"]) == 64
    assert unsigned_shell["source_current"] is False
    assert workspace_mismatch["verified"] is True
    assert workspace_mismatch["source_current"] is False
    # The token names the revision actually running.
    assert workspace_mismatch["actual_workspace_state_sha256"] == "e" * 64
    assert commit_mismatch["verified"] is True
    assert commit_mismatch["source_current"] is False
    assert commit_mismatch["actual_commit_sha"] == "f" * 40
    # Identity failures are unchanged: a different checkout, or a capture taken
    # while the workspace was being written, still fail.
    assert unstable_capture["verified"] is False
    assert "workspace_changed_during_revision_capture" in unstable_capture["issues"]
    assert root_mismatch["verified"] is False
    assert "source_root_identity_unverified" in root_mismatch["issues"]
    assert fallback == {
        "schema": "aura.runtime_revision.v2",
        "required": False,
        "verified": False,
        "source_verified": False,
        "revision_token": "",
        "expected_source_root_sha256": "",
        "actual_source_root_sha256": "",
        "expected_commit_sha": "",
        "actual_commit_sha": "",
        "expected_workspace_state_sha256": "",
        "actual_workspace_state_sha256": "",
        "expected_shell_assets_sha256": "",
        "actual_shell_assets_sha256": "",
        "capture_stable": False,
        "launch_mode": "unknown",
        "issues": ["runtime_revision_initializing"],
    }


def test_runtime_shell_asset_identity_changes_with_live_shell_bytes(tmp_path):
    from core.runtime import launch_provenance

    for relative in launch_provenance.RUNTIME_SHELL_ASSETS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"asset:{relative}\n", encoding="utf-8")

    before = launch_provenance.runtime_shell_assets_sha256(tmp_path)
    (tmp_path / "interface/static/aura.js").write_text(
        "asset:interface/static/aura.js\nchanged\n",
        encoding="utf-8",
    )
    after = launch_provenance.runtime_shell_assets_sha256(tmp_path)

    assert len(before) == 64
    assert len(after) == 64
    assert after != before


def test_runtime_revision_collection_rejects_workspace_change_during_capture(
    monkeypatch,
    tmp_path,
):
    from core.runtime import launch_provenance
    from interface.routes import system as system_routes

    root = tmp_path / "source"
    root.mkdir()
    expected_shell = "c" * 64

    def provenance(workspace: str) -> dict[str, object]:
        return {
            "required": True,
            "verified": True,
            "source_verified": True,
            "launch_mode": "signed_app",
            "expected": {
                "source_root": str(root),
                "commit_sha": "a" * 40,
                "workspace_state_sha256": "b" * 64,
            },
            "actual": {
                "source_root": str(root),
                "commit_sha": "a" * 40,
                "workspace_state_sha256": workspace,
            },
            "manifest": {"shell_assets_sha256": expected_shell},
            "issues": [],
        }

    observations = iter((provenance("b" * 64), provenance("d" * 64)))
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: next(observations),
    )
    monkeypatch.setattr(
        system_routes,
        "_runtime_shell_assets_sha256",
        lambda _root: expected_shell,
    )
    monkeypatch.setattr(
        system_routes,
        "_invalidate_launch_provenance_source_observation_cache",
        lambda: None,
    )

    result = system_routes._collect_runtime_revision_uncached()

    assert result["verified"] is False
    assert result["capture_stable"] is False
    assert result["revision_token"] == ""
    assert "workspace_changed_during_revision_capture" in result["issues"]
    # A workspace that CHANGED MID-CAPTURE is still an identity failure: the
    # measurement itself is incoherent. That is distinct from a workspace that
    # has simply moved on since the bundle was built, which is now reported as
    # source_current rather than failed.
    assert result["source_current"] is False


def test_runtime_revision_collection_rejects_shell_change_during_capture(
    monkeypatch,
    tmp_path,
):
    from core.runtime import launch_provenance
    from interface.routes import system as system_routes

    root = tmp_path / "source"
    root.mkdir()
    workspace = "b" * 64
    final_shell = "d" * 64
    provenance = {
        "required": True,
        "verified": True,
        "source_verified": True,
        "launch_mode": "signed_app",
        "expected": {
            "source_root": str(root),
            "commit_sha": "a" * 40,
            "workspace_state_sha256": workspace,
        },
        "actual": {
            "source_root": str(root),
            "commit_sha": "a" * 40,
            "workspace_state_sha256": workspace,
        },
        "manifest": {"shell_assets_sha256": final_shell},
        "issues": [],
    }
    shell_observations = iter(("c" * 64, final_shell))
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: provenance,
    )
    monkeypatch.setattr(
        system_routes,
        "_runtime_shell_assets_sha256",
        lambda _root: next(shell_observations),
    )
    monkeypatch.setattr(
        system_routes,
        "_invalidate_launch_provenance_source_observation_cache",
        lambda: None,
    )

    result = system_routes._collect_runtime_revision_uncached()

    assert result["expected_shell_assets_sha256"] == final_shell
    assert result["actual_shell_assets_sha256"] == final_shell
    assert result["capture_stable"] is False
    assert result["verified"] is False
    assert result["revision_token"] == ""
    assert "shell_assets_changed_during_revision_capture" in result["issues"]


def test_required_runtime_revision_failure_withholds_all_readiness_claims():
    from interface.routes import system as system_routes

    payload = {
        "status": "ok",
        "healthy": True,
        "ready": True,
        "connected": True,
        "system_ready": True,
        "launcher_ready": True,
        "proof_readiness_healthy": True,
        "certification_ready": True,
        "blockers": [],
        "runtime_revision": {
            "schema": "aura.runtime_revision.v2",
            "required": True,
            "verified": False,
            "revision_token": "",
        },
        "readiness_contract": {
            "healthy": True,
            "system_ready": True,
            "proof_readiness_healthy": True,
            "certification_ready": True,
            "blockers": [],
        },
        "boot": {
            "status": "ready",
            "ready": True,
            "system_ready": True,
            "launcher_ready": True,
            "proof_readiness_healthy": True,
            "certification_ready": True,
            "blockers": [],
        },
    }

    result = system_routes._apply_runtime_revision_truth(payload)

    assert result["healthy"] is False
    assert result["ready"] is False
    assert result["connected"] is False
    assert result["system_ready"] is False
    assert result["launcher_ready"] is False
    assert result["proof_readiness_healthy"] is False
    assert result["certification_ready"] is False
    assert result["blockers"][0] == "runtime_revision_unverified"
    assert result["readiness_contract"]["healthy"] is False
    assert result["readiness_contract"]["system_ready"] is False
    assert result["readiness_contract"]["proof_readiness_healthy"] is False
    assert result["readiness_contract"]["certification_ready"] is False
    assert result["boot"]["ready"] is False
    assert result["boot"]["system_ready"] is False
    assert result["boot"]["launcher_ready"] is False
    assert result["boot"]["proof_readiness_healthy"] is False
    assert result["boot"]["certification_ready"] is False

    payload["runtime_revision"] = {
        "schema": "aura.runtime_revision.v2",
        "required": False,
        "verified": False,
        "source_verified": False,
        "revision_token": "",
    }
    assert system_routes._apply_runtime_revision_truth(payload) is payload


def test_runtime_revision_blocker_rejects_malformed_verified_contracts(monkeypatch):
    from interface.routes import system as system_routes

    malformed = {
        "schema": "aura.runtime_revision.v2",
        "required": True,
        "verified": True,
        "source_verified": True,
        "capture_stable": True,
        "launch_mode": "signed_app",
        "revision_token": "a" * 64,
    }
    assert system_routes._runtime_revision_blocker(malformed) == (
        "runtime_revision_identity_invalid"
    )
    assert system_routes._runtime_revision_blocker(
        {**malformed, "schema": "aura.runtime_revision.v1"}
    ) == "runtime_revision_contract_invalid"
    assert system_routes._runtime_revision_blocker(None) == (
        "runtime_revision_contract_missing"
    )

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    direct = system_routes._runtime_revision_unavailable("", required=False)
    assert system_routes._runtime_revision_blocker(direct) == (
        "runtime_revision_required_contract_missing"
    )


def test_runtime_revision_blocker_recomputes_verified_identity_token(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.delenv("AURA_LAUNCHED_FROM_APP", raising=False)
    root = "a" * 64
    commit = "b" * 40
    workspace = "c" * 64
    shell = "d" * 64
    token = system_routes._runtime_revision_token(
        source_root_sha256=root,
        commit_sha=commit,
        workspace_state_sha256=workspace,
        shell_assets_sha256=shell,
    )
    contract = {
        "schema": "aura.runtime_revision.v2",
        "required": True,
        "verified": True,
        "source_verified": True,
        "capture_stable": True,
        "launch_mode": "signed_app",
        "revision_token": token,
        "expected_source_root_sha256": root,
        "actual_source_root_sha256": root,
        "expected_commit_sha": commit,
        "actual_commit_sha": commit,
        "expected_workspace_state_sha256": workspace,
        "actual_workspace_state_sha256": workspace,
        "expected_shell_assets_sha256": shell,
        "actual_shell_assets_sha256": shell,
    }
    assert system_routes._runtime_revision_blocker(contract) == ""
    assert system_routes._runtime_revision_blocker(
        {**contract, "revision_token": "e" * 64}
    ) == "runtime_revision_token_invalid"


def test_boot_health_contract_withholds_readiness_for_required_unverified_shell():
    from interface.routes import system as system_routes

    launch = {
        "schema": "aura.launch_provenance.v1",
        "required": True,
        "verified": True,
        "source_verified": True,
        "issues": [],
    }
    revision = {
        "schema": "aura.runtime_revision.v2",
        "required": True,
        "verified": False,
        "revision_token": "",
        "issues": ["shell_asset_identity_unverified"],
    }

    result, status_code = system_routes._attach_launch_provenance_contract(
        {
            "ready": True,
            "launcher_ready": True,
            "system_ready": True,
            "proof_readiness_healthy": True,
            "certification_ready": True,
            "checks": {},
            "blockers": [],
        },
        200,
        provenance=launch,
        runtime_revision=revision,
    )

    assert status_code == 503
    assert result["ready"] is False
    assert result["launcher_ready"] is False
    assert result["system_ready"] is False
    assert result["proof_readiness_healthy"] is False
    assert result["certification_ready"] is False
    assert result["checks"]["launch_provenance"] is True
    assert result["checks"]["runtime_revision"] is False
    assert result["blockers"] == ["runtime_revision_unverified"]
    assert result["boot_phase"] == "runtime_revision_failed"


def test_runtime_revision_public_projection_is_coarse_but_owner_keeps_diagnostics():
    from interface.routes import system as system_routes

    payload = {
        "launch_provenance": {
            "schema": "aura.launch_provenance.v1",
            "required": True,
            "verified": True,
            "actual": {"commit_sha": "c" * 40},
        },
        "boot": {
            "launch_provenance": {
                "schema": "aura.launch_provenance.v1",
                "required": True,
                "verified": True,
                "expected": {"source_root": "/private/source"},
            }
        },
        "runtime_revision": {
            "schema": "aura.runtime_revision.v2",
            "required": True,
            "verified": True,
            "revision_token": "a" * 64,
            "expected_source_root_sha256": "b" * 64,
            "actual_source_root_sha256": "b" * 64,
            "expected_commit_sha": "c" * 40,
            "actual_commit_sha": "c" * 40,
            "expected_workspace_state_sha256": "d" * 64,
            "actual_workspace_state_sha256": "d" * 64,
            "expected_shell_assets_sha256": "e" * 64,
            "actual_shell_assets_sha256": "e" * 64,
            "issues": [],
        }
    }

    public = system_routes._runtime_revision_response_projection(
        payload,
        include_diagnostics=False,
    )
    owner = system_routes._runtime_revision_response_projection(
        payload,
        include_diagnostics=True,
    )

    assert public["runtime_revision"] == {
        "schema": "aura.runtime_revision.v2",
        "required": True,
        "verified": True,
        "revision_token": "a" * 64,
        "status": "verified",
        "source_current": True,
        "blocker": "",
    }
    assert "expected_commit_sha" not in public["runtime_revision"]
    assert public["launch_provenance"] == {
        "schema": "aura.launch_provenance.v1",
        "required": True,
        "verified": True,
        "status": "verified",
        "source_current": True,
        "source_drift": [],
        "blocker": "",
    }
    assert "actual" not in public["launch_provenance"]
    assert public["boot"]["launch_provenance"] == {
        "schema": "aura.launch_provenance.v1",
        "required": True,
        "verified": True,
        "status": "verified",
        "source_current": True,
        "source_drift": [],
        "blocker": "",
    }
    assert "expected" not in public["boot"]["launch_provenance"]
    assert owner is payload
    assert owner["runtime_revision"]["expected_commit_sha"] == "c" * 40

    drifted_public = system_routes._runtime_revision_response_projection(
        {
            "launch_provenance": {
                "schema": "aura.launch_provenance.v1",
                "required": True,
                "verified": True,
                "source_current": False,
                "source_drift": ["commit_sha"],
                "verification_scope": "bundle_identity",
                "freshness_status": "drifted",
            },
            "runtime_revision": {
                "schema": "aura.runtime_revision.v2",
                "required": True,
                "verified": True,
                "source_current": False,
                "revision_token": "f" * 64,
            },
        },
        include_diagnostics=False,
    )
    assert drifted_public["launch_provenance"]["status"] == "verified_drifted"
    assert drifted_public["launch_provenance"]["source_current"] is False
    assert drifted_public["launch_provenance"]["source_drift"] == ["commit_sha"]
    assert drifted_public["runtime_revision"]["status"] == "verified_drifted"
    assert drifted_public["runtime_revision"]["source_current"] is False

    launch_only = system_routes._runtime_revision_response_projection(
        {"launch_provenance": payload["launch_provenance"]},
        include_diagnostics=False,
    )
    assert launch_only["launch_provenance"] == {
        "schema": "aura.launch_provenance.v1",
        "required": True,
        "verified": True,
        "status": "verified",
        "source_current": True,
        "source_drift": [],
        "blocker": "",
    }


@pytest.mark.asyncio
async def test_api_health_applies_provenance_projection_by_authenticated_surface(
    monkeypatch,
):
    from interface.routes import system as system_routes

    payload = {
        "status": "ok",
        "healthy": True,
        "blockers": [],
        "runtime_revision": {
            "schema": "aura.runtime_revision.v2",
            "required": True,
            "verified": True,
            "revision_token": "a" * 64,
            "expected_commit_sha": "b" * 40,
            "actual_commit_sha": "b" * 40,
            "issues": [],
        },
        "boot": {
            "launch_provenance": {
                "schema": "aura.launch_provenance.v1",
                "required": True,
                "verified": True,
                "expected": {"source_root": "/private/source"},
            }
        },
        "health_read_model": {
            "fresh": True,
            "expired": False,
            "snapshot_generation": 7,
            "captured_at_unix": 123.0,
            "serving": "fresh",
        },
    }

    class ReadModel:
        def read(self):
            return payload

    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", ReadModel())
    monkeypatch.setattr(system_routes, "_mark_runtime_service_progress", lambda _source: None)
    monkeypatch.setattr(
        system_routes,
        "_restore_owner_session_from_request",
        lambda _request: None,
    )
    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": False, "request": {"requested": False}},
    )
    surface = {"surface": "paired_device"}
    monkeypatch.setattr(system_routes, "request_access_profile", lambda _request: surface)

    public_response = await system_routes.api_health(SimpleNamespace(headers={}))
    public = json.loads(public_response.body)

    assert set(public["runtime_revision"]) == {
        "schema",
        "required",
        "verified",
        "revision_token",
        "status",
        "source_current",
        "blocker",
    }
    assert "expected_commit_sha" not in public["runtime_revision"]
    assert "expected" not in public["boot"]["launch_provenance"]

    surface["surface"] = "owner"
    owner_response = await system_routes.api_health(SimpleNamespace(headers={}))
    owner = json.loads(owner_response.body)

    assert owner["runtime_revision"]["expected_commit_sha"] == "b" * 40
    assert owner["boot"]["launch_provenance"]["expected"] == {
        "source_root": "/private/source"
    }


@pytest.mark.asyncio
async def test_api_boot_health_applies_provenance_projection_by_authenticated_surface(
    monkeypatch,
):
    from interface.routes import system as system_routes

    payload = {
        "status": "ready",
        "ready": True,
        "runtime_revision": {
            "schema": "aura.runtime_revision.v2",
            "required": True,
            "verified": True,
            "revision_token": "a" * 64,
            "expected_commit_sha": "b" * 40,
            "actual_commit_sha": "b" * 40,
            "expected_source_root": "/private/source",
            "actual_source_root": "/private/source",
            "issues": [],
        },
        "launch_provenance": {
            "schema": "aura.launch_provenance.v1",
            "required": True,
            "verified": True,
            "expected": {
                "commit_sha": "b" * 40,
                "source_root": "/private/source",
            },
            "actual": {
                "commit_sha": "b" * 40,
                "source_root": "/private/source",
            },
            "issues": [],
        },
    }
    surface = {"surface": "paired_device"}

    async def build_boot_health_payload_bounded(*, is_gui_proxy):
        assert is_gui_proxy is False
        return payload, 200

    monkeypatch.delenv("AURA_GUI_PROXY", raising=False)
    monkeypatch.setattr(
        system_routes,
        "_build_boot_health_payload_bounded",
        build_boot_health_payload_bounded,
    )
    monkeypatch.setattr(system_routes, "_mark_runtime_service_progress", lambda _source: None)
    monkeypatch.setattr(system_routes, "request_access_profile", lambda _request: surface)

    public_response = await system_routes.api_boot_health(SimpleNamespace(headers={}))
    public = json.loads(public_response.body)

    assert set(public["runtime_revision"]) == {
        "schema",
        "required",
        "verified",
        "revision_token",
        "status",
        "source_current",
        "blocker",
    }
    assert "expected_commit_sha" not in public["runtime_revision"]
    assert public["launch_provenance"] == {
        "schema": "aura.launch_provenance.v1",
        "required": True,
        "verified": True,
        "status": "verified",
        "source_current": True,
        "source_drift": [],
        "blocker": "",
    }

    surface["surface"] = "owner"
    owner_response = await system_routes.api_boot_health(SimpleNamespace(headers={}))
    owner = json.loads(owner_response.body)

    assert owner["runtime_revision"]["expected_commit_sha"] == "b" * 40
    assert owner["launch_provenance"]["expected"] == {
        "commit_sha": "b" * 40,
        "source_root": "/private/source",
    }


def test_runtime_revision_cache_has_short_negative_ttl_and_explicit_invalidation(
    monkeypatch,
):
    from interface.routes import system as system_routes

    assert (
        system_routes._RUNTIME_REVISION_UNVERIFIED_TTL_S
        < system_routes._RUNTIME_REVISION_VERIFIED_TTL_S
    )
    monkeypatch.setattr(
        system_routes,
        "_RUNTIME_REVISION_CACHE",
        {"verified": True, "revision_token": "a" * 64},
    )
    monkeypatch.setattr(
        system_routes,
        "_RUNTIME_REVISION_CACHE_COLLECTED_AT",
        123.0,
    )

    system_routes.invalidate_runtime_revision_cache()

    assert system_routes._RUNTIME_REVISION_CACHE is None
    assert system_routes._RUNTIME_REVISION_CACHE_COLLECTED_AT == 0.0

    stale = {"verified": True, "revision_token": "b" * 64}
    fresh = {"verified": True, "revision_token": "c" * 64, "issues": []}
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", stale)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE_COLLECTED_AT", 123.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_INVALIDATION_PENDING", False)
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_revision_uncached",
        lambda: fresh,
    )
    assert system_routes._RUNTIME_REVISION_LOCK.acquire(timeout=1.0)
    try:
        started = time.monotonic()
        system_routes.invalidate_runtime_revision_cache()
        assert time.monotonic() - started < 0.05
        assert system_routes._RUNTIME_REVISION_INVALIDATION_PENDING is True
        assert system_routes._RUNTIME_REVISION_CACHE is stale
    finally:
        system_routes._RUNTIME_REVISION_LOCK.release()

    assert system_routes._runtime_revision_contract()["revision_token"] == "c" * 64
    assert system_routes._RUNTIME_REVISION_INVALIDATION_PENDING is False


def test_every_runtime_revision_shell_asset_is_no_store_or_revision_addressed():
    from core.runtime import launch_provenance
    from interface import server

    for relative in launch_provenance.RUNTIME_SHELL_ASSETS:
        path = launch_provenance.runtime_shell_request_path(relative)
        policy = server._cache_policy_for_path(path)
        assert policy is not None, path
        assert policy["Cache-Control"].startswith("no-store"), path
        addressed = server._cache_policy_for_path(
            path,
            revision_addressed=True,
        )
        assert addressed["Cache-Control"].endswith("immutable"), path

    assert server._RUNTIME_REVISION_SHELL_PATHS == {
        launch_provenance.runtime_shell_request_path(relative)
        for relative in launch_provenance.RUNTIME_SHELL_ASSETS
    }

    assert server._cache_policy_for_path("/")["Cache-Control"].startswith("no-store")
    assert server._cache_policy_for_path("/data/uploads/private.png")[
        "Cache-Control"
    ].startswith("no-store")
    for future_shell_path in (
        "/static/mission_control.html",
        "/static/future-shell-module.js",
        "/static/future-shell-style.css",
        "/static/vendor/fonts/future-shell-font.woff2",
        "/static/future-shell-image.png",
    ):
        assert server._cache_policy_for_path(future_shell_path)[
            "Cache-Control"
        ].startswith("no-store")


@pytest.mark.asyncio
async def test_service_worker_response_grants_exact_root_scope():
    from interface import server

    response = SimpleNamespace(headers={}, status_code=200)

    async def call_next(_request):
        return response

    result = await server.add_cache_headers(
        SimpleNamespace(
            url=SimpleNamespace(path="/static/service-worker.js"),
            query_params={},
        ),
        call_next,
    )

    assert result is response
    assert response.headers["Service-Worker-Allowed"] == "/"
    assert response.headers["Cache-Control"].startswith("no-store")


@pytest.mark.asyncio
async def test_only_verified_snapshot_responses_receive_immutable_cache_policy():
    from interface import server

    revision = "a" * 64
    request = SimpleNamespace(
        url=SimpleNamespace(path="/static/aura.js"),
        query_params={"_aura_runtime": revision},
    )

    rejected = SimpleNamespace(
        headers=dict(server.NO_CACHE_HEADERS),
        status_code=409,
    )
    async def reject_next(_request):
        return rejected

    result = await server.add_cache_headers(request, reject_next)
    assert result.headers["Cache-Control"].startswith("no-store")

    verified = SimpleNamespace(
        headers={"X-Aura-Runtime-Revision": revision},
        status_code=200,
    )
    async def verified_next(_request):
        return verified

    result = await server.add_cache_headers(request, verified_next)
    assert result.headers["Cache-Control"].endswith("immutable")


def test_runtime_revision_retries_unverified_collection_after_worker_ttl(monkeypatch):
    from interface.routes import system as system_routes

    attempts: list[int] = []
    unverified = system_routes._runtime_revision_unavailable("resident_app_not_running")
    verified = dict(unverified)
    verified.update(
        {
            "verified": True,
            "source_verified": True,
            "revision_token": "a" * 64,
            "issues": [],
        }
    )

    def collect():
        attempts.append(len(attempts) + 1)
        return unverified if len(attempts) == 1 else verified

    # Patching time.monotonic replaces it for the WHOLE process, so an
    # exhaustible iterator here is a trap: any background thread, lock or
    # logger that reads the clock during this test consumes a scripted value
    # and the test fails with "generator raised StopIteration" — a failure
    # that measures what else the process happened to be doing. The clock is a
    # value this test sets explicitly instead.
    clock = {"now": 100.0}
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", None)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE_COLLECTED_AT", 0.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_UNVERIFIED_TTL_S", 2.0)
    monkeypatch.setattr(system_routes, "_collect_runtime_revision_uncached", collect)
    monkeypatch.setattr(system_routes.time, "monotonic", lambda: clock["now"])

    first = system_routes._runtime_revision_contract()
    clock["now"] = 101.0
    before_ttl = system_routes._runtime_revision_contract()
    clock["now"] = 102.1
    after_ttl = system_routes._runtime_revision_contract()

    assert first["verified"] is False
    assert before_ttl["verified"] is False
    assert after_ttl["verified"] is True
    assert attempts == [1, 2]


def test_runtime_revision_refreshes_verified_identity_after_ttl(monkeypatch):
    from interface.routes import system as system_routes

    attempts: list[int] = []

    def collect():
        attempts.append(len(attempts) + 1)
        return {
            **system_routes._runtime_revision_unavailable(""),
            "verified": True,
            "source_verified": True,
            "revision_token": str(attempts[-1]) * 64,
            "issues": [],
        }

    # Patching time.monotonic replaces it for the WHOLE process, so an
    # exhaustible iterator here is a trap: any background thread, lock or
    # logger that reads the clock during this test consumes a scripted value
    # and the test fails with "generator raised StopIteration" — a failure
    # that measures what else the process happened to be doing. The clock is a
    # value this test sets explicitly instead.
    clock = {"now": 100.0}
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", None)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE_COLLECTED_AT", 0.0)
    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_VERIFIED_TTL_S", 30.0)
    monkeypatch.setattr(system_routes, "_collect_runtime_revision_uncached", collect)
    monkeypatch.setattr(system_routes.time, "monotonic", lambda: clock["now"])

    first = system_routes._runtime_revision_contract()
    clock["now"] = 110.0
    before_ttl = system_routes._runtime_revision_contract()
    clock["now"] = 131.0
    after_ttl = system_routes._runtime_revision_contract()

    assert first["revision_token"] == "1" * 64
    assert before_ttl["revision_token"] == "1" * 64
    assert after_ttl["revision_token"] == "2" * 64
    assert attempts == [1, 2]


def test_runtime_revision_fallback_never_waits_for_collection_lock(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_RUNTIME_REVISION_CACHE", None)
    result: dict[str, object] = {}

    def read_fallback() -> None:
        result["fallback"] = system_routes._runtime_revision_fallback_contract()

    assert system_routes._RUNTIME_REVISION_LOCK.acquire(timeout=1.0)
    reader = threading.Thread(target=read_fallback)
    try:
        reader.start()
        reader.join(timeout=0.5)
        completed_without_lock = not reader.is_alive()
    finally:
        system_routes._RUNTIME_REVISION_LOCK.release()
        reader.join(timeout=1.0)

    assert completed_without_lock is True
    fallback = result["fallback"]
    assert isinstance(fallback, dict)
    assert fallback["verified"] is False
    assert fallback["issues"] == ["runtime_revision_collection_in_flight"]


def test_legacy_shell_health_poll_is_single_scheduled_incident_loop():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "interface/static/aura.js").read_text(encoding="utf-8")
    service_worker = (project_root / "interface/static/service-worker.js").read_text(
        encoding="utf-8"
    )

    assert "function scheduleHealthPoll" in source
    assert "function recordHealthPollFailure" in source
    assert "function recordHealthPollSuccess" in source
    assert "HEALTH_POLL_JITTER_RATIO" in source
    assert "HEALTH_POLL_REMINDER_MS" in source
    assert "setInterval(pollHealth" not in source
    assert "health endpoint unavailable; retaining last known state" in source
    assert "endpoint recovered after" in source
    assert "fetch('/api/health')" not in service_worker
    assert service_worker.count("fetch('/api/health/heartbeat')") == 2
    install_block = service_worker.split("self.addEventListener('install'", 1)[1].split(
        "self.addEventListener('activate'", 1
    )[0]
    assert "self.skipWaiting()" not in install_block


def test_legacy_shell_handoff_preserves_draft_active_and_queued_turns():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "interface/static/aura.js").read_text(encoding="utf-8")

    assert "function verifiedRuntimeRevision(payload)" in source
    assert "revision.schema !== 'aura.runtime_revision.v2'" in source
    assert "revision.verified !== true" in source
    assert "revision.required !== true" in source
    assert "function healthSnapshotRevisionEvidence(payload)" in source
    # This used to pin `metadata.fresh !== true`, which was the defect rather
    # than the contract: this same module serves stale-while-revalidate, so a
    # valid unexpired snapshot reads fresh=false most of the time and the
    # shell formed no revision evidence at all. An open desktop window then
    # ran a replaced shell indefinitely (measured live 2026-08-03, across four
    # runtime restarts). `expired` is the real "do not trust this" signal and
    # is what must be honoured — see tests/test_open_shell_reloads_for_new_assets.py.
    assert "metadata.fresh !== true" not in source
    assert "metadata.expired === true" in source
    assert "function runtimeRevisionEvidenceIsMonotonic" in source
    assert "function runtimeRevisionMarkerFromLocation" in source
    assert "RUNTIME_REVISION_RELOAD_LIMIT" in source
    assert "function reconcileRuntimeShellRevision(payload)" in source
    assert "if (reconcileRuntimeShellRevision(d)) return;" in source
    assert "'X-Idempotency-Key': item.idempotencyKey" in source
    assert "!visibleUserMessageMatches(msg)" in source
    assert "item.approvalResumeToken = item.turnId;" in source
    assert "void runChatRequest(item, { messageAlreadyRendered: true });" in source
    assert "function chatDeliveryDecision(" in source
    assert "swReloadTriggered = requestGuardedShellReload({" in source
    assert "nextUrl.searchParams.set('_aura_runtime', revision);" in source
    assert "revision.slice(0, 12)" not in source
    submit_block = source.split("$('chat-form').onsubmit", 1)[1].split(
        "async function appendMsg", 1
    )[0]
    assert submit_block.index("enqueueChatMessage(item)") < submit_block.index(
        "msgInput.value = '';"
    )
    assert submit_block.index("runChatRequest(item") < submit_block.index(
        "msgInput.value = '';"
    )

    handoff_start = source.index("function createChatIdempotencyKey()")
    handoff_end = source.index("function enqueueChatMessage", handoff_start)
    drain_start = source.index("function drainQueuedChatMessages()", handoff_end)
    drain_end = source.index("async function runChatRequest", drain_start)
    revision_start = source.index("function verifiedRuntimeRevision(payload)")
    revision_end = source.index("async function pollHealth()", revision_start)
    production_functions = "\n\n".join(
        (
            source[handoff_start:handoff_end].strip(),
            source[drain_start:drain_end].strip(),
            source[revision_start:revision_end].strip(),
        )
    )
    node = shutil.which("node")
    assert node is not None, "Node.js is required to validate the production shell logic"
    script = textwrap.dedent(
        f"""
        'use strict';
        const assert = require('node:assert/strict');
        const RUNTIME_REVISION_STORAGE_KEY = 'aura.runtime_revision';
        const RUNTIME_REVISION_RELOAD_STORAGE_KEY = 'aura.runtime_revision_reload';
        const RUNTIME_REVISION_RECORD_SCHEMA = 'aura.runtime_revision.client.v2';
        const RUNTIME_REVISION_RELOAD_LIMIT = 2;
        const CHAT_HANDOFF_STORAGE_KEY = 'aura.chat_handoff';
            const CHAT_HANDOFF_SCHEMA = 'aura.chat_handoff.v3';
            const CHAT_HANDOFF_ACCEPTED_SCHEMAS = new Set([CHAT_HANDOFF_SCHEMA]);
            const CHAT_HANDOFF_ACTIVE_REPLAY_MAX_WAIT_MS = 410000;
            const CHAT_HANDOFF_MAX_AGE_MS = 600000;
            const SERVICE_WORKER_REGISTRATION_RETRY_MAX_MS = 30000;
        const values = new Map();
        const setStorageItem = (key, value) => values.set(key, String(value));
        const sessionStorage = {{
            getItem: (key) => values.has(key) ? values.get(key) : null,
            setItem: setStorageItem,
            removeItem: (key) => values.delete(key),
        }};
        const composer = {{ value: '', style: {{}}, scrollHeight: 42 }};
        const resizeChatComposer = (input) => {{
            if (!input || !input.style) return;
            input.style.height = 'auto';
            input.style.height = `${{input.scrollHeight}}px`;
            input.style.overflowY = 'hidden';
        }};
        const $ = (id) => id === 'chat-input' ? composer : null;
        const replacements = [];
        const reloads = [];
        const timers = [];
        const sent = [];
        let uuidCounter = 0;
        const window = {{
            crypto: {{ randomUUID: () => `stable-${{++uuidCounter}}-0000-0000-000000000000` }},
            setTimeout: (callback, delay) => {{ timers.push([callback, delay]); return timers.length; }},
            clearTimeout: () => {{}},
            location: {{
                href: 'http://127.0.0.1:8000/?existing=1',
                replace: (url) => replacements.push(url),
                reload: () => reloads.push('reload'),
            }},
        }};
        const navigator = {{}};
        const console = {{ warn: () => {{}}, error: () => {{}} }};
        const state = {{
            activeChatRequest: null,
            chatSendQueue: [],
            chatDrainTimer: null,
            chatHandoffPending: false,
            deferredShellReload: null,
            waitingServiceWorker: null,
            runtimeRevision: null,
            runtimeRevisionGeneration: 0,
            runtimeRevisionCapturedAtUnix: 0,
                runtimeRevisionReloadAttempts: {{}},
                runtimeRevisionReloading: false,
                runtimeRevisionTrust: 'unknown',
                runtimeShellRetirementPromise: null,
                serviceWorkerRevision: null,
                serviceWorkerRegistrationTarget: null,
                serviceWorkerRegistrationPromise: null,
                serviceWorkerRegistrationEpoch: 0,
                serviceWorkerRegistrationFailures: 0,
                serviceWorkerRegistrationRetryAt: 0,
                accessProfile: {{
                    surface: 'owner',
                    handoff_scope: 'e'.repeat(64),
                }},
                conversationLane: null,
            isSubmitting: false,
        }};
        const laneHasActiveGeneration = (lane) => Boolean(lane && lane.active);
        async function runChatRequest(item, options) {{ sent.push([item, options]); }}

        {production_functions}

        const revisionPayload = (token, {{
            verified = true,
            required = true,
            fresh = true,
            expired = false,
            generation = 1,
            capturedAtUnix = 100,
        }} = {{}}) => ({{
            runtime_revision: {{
                schema: 'aura.runtime_revision.v2',
                required,
                verified,
                revision_token: token,
            }},
            health_read_model: {{
                fresh,
                expired,
                snapshot_generation: generation,
                captured_at_unix: capturedAtUnix,
            }},
        }});
        const revisionA = 'a'.repeat(64);
        const revisionB = 'b'.repeat(64);
        const activeKey = 'aura-chat-active-key-0001';
        const queuedKey = 'aura-chat-queued-key-0002';

        composer.value = 'complete draft\\nsecond line';
            state.activeChatRequest = normalizeChatQueueItem({{
                message: 'active turn', idempotencyKey: activeKey, rendered: true, queuedAt: 10,
                approvalResumeToken: 'f'.repeat(32),
            }});
        state.chatSendQueue = [normalizeChatQueueItem({{
            message: 'queued turn', idempotencyKey: queuedKey, rendered: true, queuedAt: 20,
        }})];
        assert.equal(persistChatHandoff({{ force: true }}), true);
        let handoff = JSON.parse(values.get(CHAT_HANDOFF_STORAGE_KEY));
        assert.equal(handoff.draft, composer.value);
            assert.equal(handoff.active.idempotencyKey, activeKey);
            assert.equal(handoff.queue[0].idempotencyKey, queuedKey);
            assert.equal(handoff.scope, `owner:${{'e'.repeat(64)}}`);
            assert.equal(Object.hasOwn(handoff.active, 'approvalResumeToken'), false);

        // A shell that CLAIMS a required revision and cannot prove it is
        // untrusted, and a heartbeat carrying no runtime_revision key at all
        // must inherit that verdict.
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionA, {{ verified: false }})), false);
        assert.equal(state.runtimeRevisionTrust, 'untrusted');
        assert.equal(runtimeRevisionPolicyBlocker({{}}), 'runtime_revision_unverified');

        // A STALE snapshot does not unprove a proven revision. This module
        // serves stale-while-revalidate, so fresh=false is the normal reading
        // of a perfectly valid snapshot; treating it as no evidence left an
        // open desktop window running a shell the runtime had replaced (live
        // 2026-08-03). First sighting binds the revision without reloading.
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionA, {{ fresh: false }})), false);
        assert.equal(state.runtimeRevisionTrust, 'trusted');
        assert.equal(state.runtimeRevision, revisionA);

        // An EXPIRED snapshot is neither proof nor disproof: it is not
        // authoritative, so it must leave the standing verdict alone.
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionA, {{ expired: true }})), false);
        assert.equal(state.runtimeRevisionTrust, 'trusted');
        assert.equal(replacements.length, 0);

        // A direct/source launch has no signed revision and never will: the
        // runtime says required === false, and the server's own blocker treats
        // that as no blocker. Calling it 'untrusted' made every keyless
        // heartbeat report runtime_revision_unverified, so the header badge
        // read RUNTIME_REVISION_UNVERIFIED while /api/health reported none.
        const sourceLaunch = revisionPayload('', {{ required: false, verified: false }});
        assert.equal(reconcileRuntimeShellRevision(sourceLaunch), false);
        assert.equal(state.runtimeRevisionTrust, 'not_required');
        assert.equal(runtimeRevisionPolicyBlocker(sourceLaunch), '');
        assert.equal(runtimeRevisionPolicyBlocker({{}}), '');
        assert.equal(replacements.length, 0);
        assert.equal(reloads.length, 0);

        // ...and it must not have quietly disarmed the signed-app check.
        assert.equal(
            runtimeRevisionPolicyBlocker(revisionPayload(revisionA, {{ verified: false }})),
            'runtime_revision_unverified',
        );

        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionA.toUpperCase())), false);
        assert.equal(state.runtimeRevision, revisionA);
        let revisionRecord = JSON.parse(values.get(RUNTIME_REVISION_STORAGE_KEY));
        assert.equal(revisionRecord.revision, revisionA);
        assert.equal(revisionRecord.generation, 1);
        assert.equal(revisionRecord.captured_at_unix, 100);
        assert.equal(replacements.length, 0);

        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionB, {{
            generation: 2, capturedAtUnix: 100,
        }})), false);
        assert.equal(state.runtimeRevision, revisionA);
        assert.equal(replacements.length, 0);

        assert.equal(runtimeRevisionEvidenceIsMonotonic(
            {{ revision: revisionA, generation: 1, capturedAtUnix: 200 }},
            {{ revision: revisionA, generation: 99, capturedAtUnix: 100 }},
        ), true);
        assert.equal(runtimeRevisionEvidenceIsMonotonic(
            {{ revision: revisionA, generation: 1, capturedAtUnix: 100 }},
            {{ revision: revisionA, generation: 99, capturedAtUnix: 100 }},
        ), false);

        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionB, {{
            generation: 2, capturedAtUnix: 110,
        }})), true);
        revisionRecord = JSON.parse(values.get(RUNTIME_REVISION_STORAGE_KEY));
        assert.equal(revisionRecord.revision, revisionB);
        assert.equal(revisionRecord.generation, 2);
        assert.equal(replacements.length, 1);
        assert.match(replacements[0], new RegExp(`_aura_runtime=${{revisionB}}`));
        handoff = JSON.parse(values.get(CHAT_HANDOFF_STORAGE_KEY));
        assert.equal(handoff.draft, 'complete draft\\nsecond line');
        assert.equal(handoff.active.idempotencyKey, activeKey);
        assert.equal(handoff.queue[0].idempotencyKey, queuedKey);
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionB, {{
            generation: 2, capturedAtUnix: 110,
        }})), false);
        assert.equal(replacements.length, 1);

        composer.value = '';
        state.activeChatRequest = null;
        state.chatSendQueue = [];
        state.runtimeRevision = null;
        state.runtimeRevisionReloading = false;
        state.chatHandoffPending = false;
        assert.equal(restoreChatHandoff(composer), true);
        assert.equal(composer.value, 'complete draft\\nsecond line');
        assert.equal(composer.style.height, '42px');
        assert.deepEqual(
            state.chatSendQueue.map(item => item.idempotencyKey),
            [activeKey, queuedKey],
        );
            assert.ok(state.chatSendQueue.every(item => item.rendered === false));
            assert.ok(state.chatSendQueue.every(item => item.approvalResumeToken === ''));
        assert.equal(state.chatSendQueue[0].resumePending, true);
        assert.equal(state.chatSendQueue[1].resumePending, false);

        state.conversationLane = {{ active: true }};
        drainQueuedChatMessages();
        assert.equal(sent.length, 1);
        assert.equal(sent[0][0].idempotencyKey, activeKey);
        assert.equal(sent[0][1].messageAlreadyRendered, false);
        assert.equal(state.chatSendQueue.length, 1);
        assert.equal(timers.length, 0);

        const activationMessages = [];
            const revisionWorker = {{
            scriptURL: `http://127.0.0.1:8000/static/service-worker.js?_aura_runtime=${{revisionB}}`,
                postMessage: (message) => activationMessages.push(message),
            }};
            state.serviceWorkerRegistrationTarget = revisionB;
            assert.equal(requestServiceWorkerActivation(revisionWorker, revisionB), true);
        assert.equal(requestServiceWorkerActivation(revisionWorker, revisionA), false);
        assert.deepEqual(activationMessages, [{{
            type: 'SKIP_WAITING', revision: revisionB,
        }}]);

        const revisionC = 'c'.repeat(64);
        composer.value = '';
        state.activeChatRequest = null;
        state.chatSendQueue = [];
        state.runtimeRevisionReloading = false;
        state.chatHandoffPending = false;
        sessionStorage.setItem = () => {{ throw new Error('storage blocked'); }};
            assert.equal(requestGuardedShellReload({{
            revision: revisionC,
            generation: 3,
            capturedAtUnix: 120,
            replaceUrl: `http://127.0.0.1:8000/?_aura_runtime=${{revisionC}}`,
            }}), true);
            assert.equal(reloads.length, 0);
            assert.equal(replacements.length, 2);
            assert.equal(state.deferredShellReload, null);

            window.location.href = `http://127.0.0.1:8000/?_aura_runtime=${{revisionC}}`;
            state.runtimeRevisionReloading = false;
        state.runtimeRevision = null;
        state.runtimeRevisionGeneration = 0;
        state.runtimeRevisionCapturedAtUnix = 0;
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionC, {{
            generation: 3, capturedAtUnix: 120,
        }})), false);
            assert.equal(replacements.length, 2);
        assert.equal(state.runtimeRevision, revisionC);
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionC, {{
            generation: 2, capturedAtUnix: 130,
        }})), false);
            assert.equal(state.runtimeRevisionGeneration, 2);

        const revisionD = 'd'.repeat(64);
        assert.equal(reconcileRuntimeShellRevision(revisionPayload(revisionD, {{
            generation: 2, capturedAtUnix: 110,
        }})), false);
            assert.equal(replacements.length, 2);

        sessionStorage.setItem = setStorageItem;
        state.runtimeRevisionReloadAttempts = {{}};
        values.delete(RUNTIME_REVISION_RELOAD_STORAGE_KEY);
        state.runtimeRevisionReloading = false;
        assert.equal(requestGuardedShellReload({{
            revision: revisionD, generation: 4, capturedAtUnix: 130,
        }}), true);
        state.runtimeRevisionReloading = false;
        assert.equal(requestGuardedShellReload({{
            revision: revisionD, generation: 4, capturedAtUnix: 130,
        }}), true);
        state.runtimeRevisionReloading = false;
        assert.equal(requestGuardedShellReload({{
            revision: revisionD, generation: 4, capturedAtUnix: 130,
        }}), false);
            assert.equal(reloads.length, 2);

            composer.value = 'must not cross principals';
            state.activeChatRequest = null;
            state.chatSendQueue = [];
            assert.equal(persistChatHandoff({{ force: true }}), true);
            state.accessProfile.handoff_scope = '9'.repeat(64);
            assert.equal(restoreChatHandoff(composer), false);
            assert.equal(values.has(CHAT_HANDOFF_STORAGE_KEY), false);

            state.accessProfile.handoff_scope = 'e'.repeat(64);
            composer.value = 'expired';
            assert.equal(persistChatHandoff({{ force: true }}), true);
            const expired = JSON.parse(values.get(CHAT_HANDOFF_STORAGE_KEY));
            expired.savedAt = Date.now() - CHAT_HANDOFF_MAX_AGE_MS - 1;
            values.set(CHAT_HANDOFF_STORAGE_KEY, JSON.stringify(expired));
            assert.equal(restoreChatHandoff(composer), false);
            assert.equal(values.has(CHAT_HANDOFF_STORAGE_KEY), false);
            """
    )
    completed = get_subprocess_gateway().run(
        [node, "-e", script],
        timeout=10,
        offline_tooling=True,
        source="certification_tooling:test_runtime_shell_revision",
        accelerator_capability="none",
    )
    assert completed.returncode == 0, completed.stderr
