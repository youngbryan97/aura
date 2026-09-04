import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest
import interface.routes.chat_preflight as _chat_preflight
from tests.chat_lane_support import patch_chat_lane


@pytest.fixture(autouse=True)
def _isolate_boot_health_cache():
    from interface.routes import system as system_routes

    system_routes._reset_boot_health_cache_for_test()
    try:
        yield
    finally:
        system_routes._reset_boot_health_cache_for_test()


@pytest.mark.asyncio
async def test_pneuma_status_rejects_dead_background_task(monkeypatch):
    from core.pneuma import pneuma as pneuma_module
    from interface.routes import subsystems

    class DeadPneuma:
        def get_state_dict(self):
            return {"online": False, "tick_count": 3}

    monkeypatch.setattr(pneuma_module, "get_pneuma", lambda: DeadPneuma())

    response = await subsystems.api_pneuma_status()

    assert response.status_code == 503
    assert json.loads(response.body)["online"] is False


@pytest.mark.asyncio
async def test_mhaf_status_rejects_dead_background_task(monkeypatch):
    from core.consciousness import mhaf_field, neologism_engine
    from interface.routes import subsystems

    class DeadMHAF:
        def get_state_dict(self):
            return {"online": False, "tick_count": 2}

    monkeypatch.setattr(mhaf_field, "get_mhaf", lambda: DeadMHAF())
    monkeypatch.setattr(neologism_engine, "get_neologism_engine", lambda: None)

    response = await subsystems.api_mhaf_status()

    assert response.status_code == 503
    assert json.loads(response.body)["online"] is False


def test_system_health_uses_task_liveness_for_pneuma_and_mhaf():
    from pathlib import Path

    from interface.routes import system as system_routes

    source = Path(system_routes.__file__).read_text(encoding="utf-8")

    assert 'pneuma_data["online"] = bool(runtime_state.get("online", False))' in source
    assert 'mhaf_data["online"] = bool(runtime_state.get("online", False))' in source


def test_conversation_lane_resilient_helper_contains_legacy_override_failure(monkeypatch):
    from interface.routes import system as system_routes

    def broken_legacy_override():
        failure = RuntimeError("legacy lane collector exploded")
        raise failure

    monkeypatch.setattr(system_routes, "_collect_conversation_lane_status", broken_legacy_override)

    lane = system_routes._collect_conversation_lane_status_resilient()

    assert lane["state"] == "degraded"
    assert lane["conversation_ready"] is False
    assert "legacy lane collector exploded" in lane["last_failure_reason"]


def test_stability_details_mark_missing_guardian_unhealthy(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready", "runtime_identity_ok": True},
    )

    details = system_routes._collect_stability_details()

    assert details["healthy"] is False
    assert details["status"] == "unavailable"
    assert details["active_issues"][0]["name"] == "stability_guardian"


def test_stability_details_do_not_default_missing_report_field_to_healthy(monkeypatch):
    from interface.routes import system as system_routes

    class Guardian:
        def get_latest_report(self):
            return {
                "checks": [{"name": "probe", "message": "missing boolean"}],
                "memory_pct": 12.0,
                "cpu_pct": 3.0,
            }

    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "peek",
        staticmethod(lambda name, default=None: Guardian() if name == "stability_guardian" else default),
    )
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready", "runtime_identity_ok": True},
    )

    details = system_routes._collect_stability_details()

    assert details["healthy"] is False
    assert details["status"] == "degraded"
    assert details["active_issues"][0]["name"] == "probe"


@pytest.mark.asyncio
async def test_telemetry_stream_emits_idle_heartbeat_and_unsubscribes(monkeypatch):
    from interface.routes import system as system_routes

    queue: asyncio.Queue = asyncio.Queue()
    unsubscribed = []

    class _Request:
        def __init__(self):
            self.checks = 0

        async def is_disconnected(self):
            self.checks += 1
            return self.checks > 2

    class _Bus:
        async def subscribe(self):
            return queue

        async def unsubscribe(self, subscribed_queue):
            unsubscribed.append(subscribed_queue)

    monkeypatch.setattr(system_routes.config.security, "internal_only_mode", False)
    monkeypatch.setattr(system_routes, "_SSE_IDLE_HEARTBEAT_S", 0.001)
    monkeypatch.setattr(system_routes, "broadcast_bus", _Bus())
    monkeypatch.setattr(
        system_routes,
        "runtime_heartbeat_payload",
        lambda kind="heartbeat": {
            "type": kind,
            "healthy": False,
            "runtime_probe_healthy": False,
            "transport_only": False,
            "required_probes": {"all_passed": False},
            "blockers": ["runtime_required_probes"],
        },
    )

    response = await system_routes.telemetry_stream(_Request())
    iterator = response.body_iterator
    first_event = await anext(iterator)
    heartbeat_event = await anext(iterator)
    await iterator.aclose()

    assert "event: telemetry" in first_event
    assert "event: heartbeat" in heartbeat_event
    heartbeat_payload = json.loads(heartbeat_event.split("data: ", 1)[1])
    assert heartbeat_payload["type"] == "heartbeat"
    assert heartbeat_payload["healthy"] is False
    assert heartbeat_payload["runtime_probe_healthy"] is False
    assert heartbeat_payload["transport_only"] is False
    assert heartbeat_payload["required_probes"]["all_passed"] is False
    assert "runtime_required_probes" in heartbeat_payload["blockers"]
    assert unsubscribed == [queue]


@pytest.mark.asyncio
async def test_ui_shell_error_route_logs_and_broadcasts_recovered_render_fault(monkeypatch):
    from interface.routes import system as system_routes

    published = []

    class _Bus:
        async def publish(self, message, priority=10):
            published.append((message, priority))

    monkeypatch.setattr(system_routes, "broadcast_bus", _Bus())

    response = await system_routes.api_ui_shell_error(
        {"error": "Cannot read properties of undefined", "component_stack": "App > NeuralFeed"}
    )

    body = json.loads(response.body.decode("utf-8"))
    assert body == {"ok": True}
    assert published
    message, priority = published[0]
    assert priority == 0
    assert message["kind"] == "log"
    assert message["level"] == "error"
    assert message["source"] == "Aura.Desktop.Shell"
    assert "Desktop shell render fault recovered" in message["message"]
    assert message["payload"]["component_stack"] == "App > NeuralFeed"


def test_websocket_runtime_heartbeat_requires_conversation_lane(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface import websocket_manager
    from interface.routes import chat as chat_routes

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True

    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
            "services": [
                {
                    "container_key": component,
                    "present": True,
                    "liveness": "ok",
                }
                for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
                for component in components
            ],
        },
    )
    patch_chat_lane(
        monkeypatch,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "failed",
            "last_failure_reason": "desktop_cognitive_engine_required_no_reply",
        },
    )
    monkeypatch.setattr(chat_routes, "_conversation_lane_is_standby", lambda _lane: False)

    payload = websocket_manager.runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert payload["runtime_probe_healthy"] is True
    assert payload["conversation_ready"] is False
    assert "conversation_ready" in payload["blockers"]
    assert "conversation_lane:failed" in payload["blockers"]


def test_websocket_runtime_heartbeat_treats_active_generation_as_working_not_healthy(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface import websocket_manager
    from interface.routes import chat as chat_routes

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True

    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "healthy": True,
            "status": "healthy",
            "required_probes": required_probes,
            "failures": {"critical": [], "important": [], "optional": []},
            "services": [
                {
                    "container_key": component,
                    "present": True,
                    "liveness": "ok",
                }
                for _group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
                for component in components
            ],
        },
    )
    patch_chat_lane(
        monkeypatch,
        "_collect_conversation_lane_status",
        lambda: {
            "conversation_ready": False,
            "state": "ready",
            "active_generations": 1,
            "last_failure_reason": "active_generation_in_flight",
        },
    )

    payload = websocket_manager.runtime_heartbeat_payload("heartbeat")

    assert payload["healthy"] is False
    assert payload["status"] == "working"
    assert payload["conversation_ready"] is False
    assert payload["conversation_busy"] is True
    assert "conversation_ready" not in payload["blockers"]
    assert "conversation_reason:active_generation_in_flight" not in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_fails_closed_when_required_probes_fail(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": False, "state": "failed"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": False,
                "system_ready": False,
                "conversation_ready": False,
                "boot_phase": "kernel_ready",
                "blockers": ["runtime_required_probes", "probe:scheduler"],
                "required_probes": {
                    "scheduler": {
                        "ok": False,
                        "components": {"scheduler": False},
                    },
                    "all_passed": False,
                },
            },
            503,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert payload["required_probes"]["scheduler"]["ok"] is False
    assert "runtime_required_probes" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_success_code_when_probe_group_missing(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "blockers": [],
                "required_probes": {
                    "all_passed": True,
                    "kernel": {"ok": True, "components": {"kernel_interface": True}},
                    "memory": {"ok": True, "components": {"state_repository": True}},
                    "scheduler": {"ok": True, "components": {"scheduler": True}},
                    "tool_governance": {"ok": True, "components": {"unified_will": True}},
                },
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["status"] == "unhealthy"
    assert "probe:inference" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_partial_probe_components(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "boot_phase": "kernel_ready",
                "blockers": [],
                "required_probes": {
                    "all_passed": True,
                    "kernel": {"ok": True, "components": {"kernel_interface": True}},
                    "inference": {
                        "ok": True,
                        "components": {"inference_gate": True, "llm_router": True},
                    },
                    "memory": {"ok": True, "components": {"state_repository": True}},
                    "scheduler": {"ok": True, "components": {"scheduler": True}},
                    "tool_governance": {
                        "ok": True,
                        "components": {
                            "unified_will": True,
                            "authority_gateway": True,
                            "capability_engine": True,
                        },
                    },
                },
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert "probe:memory" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_refuses_boot_blockers_even_when_required_probes_pass(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    system_routes._store_boot_health_cache({}, 503)

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": False, "state": "failed"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": False,
                "boot_phase": "conversation_failed",
                "blockers": ["conversation_ready", "conversation_failed"],
                "required_probes": required_probes,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["runtime_probe_healthy"] is True
    assert payload["status"] == "unhealthy"
    assert payload["required_probes"]["all_passed"] is True
    assert "conversation_failed" in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_drops_stale_conversation_blocker_when_lane_is_ready(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    system_routes._store_boot_health_cache({}, 503)

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": True,
                "system_ready": True,
                "conversation_ready": False,
                "boot_phase": "kernel_ready",
                "blockers": ["conversation_ready"],
                "required_probes": required_probes,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["healthy"] is True
    assert payload["conversation_ready"] is True
    assert payload["conversation_lane"]["state"] == "ready"
    assert "conversation_ready" not in payload["blockers"]


@pytest.mark.asyncio
async def test_runtime_heartbeat_treats_active_generation_as_working_not_healthy(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    system_routes._store_boot_health_cache({}, 503)

    required_probes = {
        group: {"ok": True, "components": {key: True for key in keys}}
        for group, keys in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    busy_lane = {
        "conversation_ready": False,
        "state": "ready",
        "active_generations": 1,
        "last_failure_reason": "active_generation_in_flight",
    }
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: busy_lane,
    )
    monkeypatch.setattr(
        system_routes,
        "build_boot_health_snapshot",
        lambda orch, rt, is_gui_proxy=False, conversation_lane=None: (
            {
                "ready": False,
                "system_ready": True,
                "conversation_ready": False,
                "conversation_busy": True,
                "boot_phase": "conversation_working",
                "blockers": [],
                "required_probes": required_probes,
            },
            200,
        ),
    )

    response = await system_routes.api_heartbeat()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["status"] == "working"
    assert payload["healthy"] is False
    assert payload["conversation_ready"] is False
    assert payload["conversation_busy"] is True
    assert payload["conversation_lane"]["state"] == "ready"
    assert "conversation_ready" not in payload["blockers"]


@pytest.mark.asyncio
async def test_boot_health_probe_times_out_instead_of_hanging_http_loop(monkeypatch):
    from interface.routes import system as system_routes

    degradations: list[tuple] = []

    def slow_health_snapshot(*, is_gui_proxy: bool):
        time.sleep(0.2)
        return ({"ready": True, "required_probes": {"all_passed": True}}, 200)

    system_routes._store_boot_health_cache({}, 503)
    system_routes._reset_health_probe_state_for_test()
    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 0.001)
    monkeypatch.setattr(system_routes, "_runtime_manifest_boot_health_payload", lambda _reason: None)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(system_routes, "_build_boot_health_payload_sync", slow_health_snapshot)
    monkeypatch.setattr(
        system_routes,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    started_at = time.perf_counter()
    payload, status_code = await system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
    elapsed = time.perf_counter() - started_at

    assert status_code == 503
    assert elapsed < 0.15
    assert payload["ready"] is False
    assert payload["required_probes"]["all_passed"] is False
    assert payload["blockers"] == ["health_probe_timeout"]
    assert payload["health_probe_runtime"]["consecutive_failures"] == 0
    assert payload["health_probe_runtime"]["total_timeouts"] == 1
    assert payload["health_probe_runtime"]["total_terminal_failures"] == 0
    assert payload["health_probe_runtime"]["escalated"] is False
    assert degradations == []
    await asyncio.sleep(0.25)
    system_routes._reset_health_probe_state_for_test()


@pytest.mark.asyncio
async def test_boot_health_does_not_start_full_probe_during_foreground_generation(
    monkeypatch,
):
    from interface.routes import system as system_routes

    system_routes._store_boot_health_cache(
        {
            "ready": True,
            "system_ready": True,
            "conversation_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
        },
        200,
    )
    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(system_routes, "_HEALTH_STALE_CACHE_TTL_S", 30.0)
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {
            "state": "ready",
            "conversation_ready": False,
            "active_generations": 1,
            "last_failure_reason": "active_generation_in_flight",
        },
    )
    monkeypatch.setattr(
        system_routes,
        "_start_or_join_health_probe",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreground generation must not launch a health sweep")
        ),
    )

    payload, status_code = await system_routes._build_boot_health_payload_bounded(
        is_gui_proxy=False
    )

    assert status_code == 200
    assert payload["ready"] is True
    assert payload["cache_status"] == "stale_while_revalidate"
    assert payload["cache_reason"] == "foreground_generation_active"
    assert payload["health_probe_runtime"]["total_timeouts"] == 0


def test_boot_health_cache_is_partitioned_by_runtime_surface(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 30.0)
    system_routes._store_boot_health_cache(
        {"ready": False, "surface": "runtime"},
        503,
        is_gui_proxy=False,
    )
    system_routes._store_boot_health_cache(
        {"ready": True, "surface": "gui_proxy"},
        200,
        is_gui_proxy=True,
    )

    runtime_payload = system_routes._fresh_boot_health_payload(is_gui_proxy=False)
    proxy_payload = system_routes._fresh_boot_health_payload(is_gui_proxy=True)

    assert runtime_payload is not None
    assert proxy_payload is not None
    assert runtime_payload[0]["surface"] == "runtime"
    assert runtime_payload[1] == 503
    assert proxy_payload[0]["surface"] == "gui_proxy"
    assert proxy_payload[1] == 200


@pytest.mark.asyncio
async def test_boot_health_probe_reports_single_flight_while_prior_probe_is_wedged(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    started = threading.Event()
    release = threading.Event()
    required_probes = {
        "all_passed": True,
        **{
            group: {"ok": True, "components": {component: True for component in components}}
            for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
        },
    }

    def slow_snapshot(*_args, **_kwargs):
        started.set()
        release.wait(0.3)
        return (
            {
                "ready": True,
                "system_ready": True,
                "required_probes": required_probes,
                "blockers": [],
                "conversation_ready": True,
            },
            200,
        )

    system_routes._store_boot_health_cache(
        {
            "ready": True,
            "system_ready": True,
            "launcher_ready": True,
            "required_probes": required_probes,
            "blockers": [],
            "conversation_ready": True,
        },
        200,
    )
    system_routes._reset_health_probe_state_for_test()
    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(system_routes, "_HEALTH_STALE_CACHE_TTL_S", 30.0)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(system_routes.ServiceContainer, "get", staticmethod(lambda _name, default=None: default))
    monkeypatch.setattr(system_routes, "_get_runtime_state_safe", lambda: {})
    monkeypatch.setattr(
        system_routes,
        "_collect_conversation_lane_status_resilient",
        lambda: {"conversation_ready": True, "state": "ready"},
    )
    monkeypatch.setattr(system_routes, "build_boot_health_snapshot", slow_snapshot)

    first_probe = asyncio.create_task(
        system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
    )
    assert await asyncio.to_thread(started.wait, 0.2)
    await asyncio.sleep(0.08)

    payload, status_code = await system_routes._build_boot_health_payload_bounded(
        is_gui_proxy=False
    )
    release.set()
    first_payload, first_status_code = await first_probe

    assert status_code == 200
    assert payload["ready"] is True
    assert payload["cache_status"] == "stale_while_revalidate"
    assert payload["cache_reason"] == "health_probe_in_flight"
    assert first_status_code == 200
    assert first_payload["ready"] is True
    assert first_payload["cache_status"] == "stale_while_revalidate"
    assert first_payload["cache_reason"] == "health_probe_timeout"
    assert payload["health_probe_runtime"]["total_contentions"] == 1
    assert payload["health_probe_runtime"]["consecutive_failures"] == 0
    system_routes._reset_health_probe_state_for_test()


@pytest.mark.asyncio
async def test_boot_health_poll_storm_does_not_escalate_one_slow_probe(monkeypatch):
    from interface.routes import system as system_routes

    degradations: list[tuple] = []

    def slow_health_snapshot(*, is_gui_proxy: bool):
        del is_gui_proxy
        time.sleep(0.15)
        return ({"ready": True, "required_probes": {"all_passed": True}}, 200)

    system_routes._store_boot_health_cache({}, 503)
    system_routes._reset_health_probe_state_for_test()
    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 0.001)
    monkeypatch.setattr(system_routes, "_runtime_manifest_boot_health_payload", lambda _reason: None)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_TIMEOUT_S", 0.01)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_DEGRADATION_THRESHOLD", 3)
    monkeypatch.setattr(system_routes, "_build_boot_health_payload_sync", slow_health_snapshot)
    monkeypatch.setattr(
        system_routes,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    payloads = [
        await system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
        for _ in range(3)
    ]

    assert [item[0]["health_probe_runtime"]["consecutive_failures"] for item in payloads] == [
        0,
        0,
        0,
    ]
    assert [item[0]["health_probe_runtime"]["total_timeouts"] for item in payloads] == [
        1,
        1,
        1,
    ]
    assert [item[0]["health_probe_runtime"]["total_contentions"] for item in payloads] == [
        0,
        1,
        2,
    ]
    assert payloads[-1][0]["health_probe_runtime"]["escalated"] is False
    assert degradations == []
    await asyncio.sleep(0.2)
    assert system_routes._health_probe_state_snapshot()["active"] is False
    system_routes._reset_health_probe_state_for_test()


@pytest.mark.asyncio
async def test_boot_health_escalation_counts_distinct_terminal_probe_failures(monkeypatch):
    from interface.routes import system as system_routes

    degradations: list[tuple] = []

    def failed_health_snapshot(*, is_gui_proxy: bool):
        del is_gui_proxy
        raise RuntimeError("probe failed independently")

    system_routes._store_boot_health_cache({}, 503)
    system_routes._reset_health_probe_state_for_test()
    monkeypatch.setattr(system_routes, "_HEALTH_CACHE_TTL_S", 0.0)
    monkeypatch.setattr(system_routes, "_runtime_manifest_boot_health_payload", lambda _reason: None)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(system_routes, "_HEALTH_PROBE_DEGRADATION_THRESHOLD", 2)
    monkeypatch.setattr(system_routes, "_build_boot_health_payload_sync", failed_health_snapshot)
    monkeypatch.setattr(
        system_routes,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    first = await system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
    await asyncio.sleep(0.02)
    second = await system_routes._build_boot_health_payload_bounded(is_gui_proxy=False)
    await asyncio.sleep(0.02)

    assert first[0]["health_probe_runtime"]["consecutive_failures"] == 1
    assert second[0]["health_probe_runtime"]["consecutive_failures"] == 2
    assert second[0]["health_probe_runtime"]["total_terminal_failures"] == 2
    assert second[0]["health_probe_runtime"]["escalated"] is True
    assert len(degradations) == 1
    assert "distinct terminal health-probe failures" in degradations[0][1]["action"]
    system_routes._reset_health_probe_state_for_test()


def test_runtime_manifest_health_fallback_preserves_required_probe_shape(tmp_path, monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    project_root = tmp_path
    artifact_root = project_root / "artifacts" / "current"
    artifact_root.mkdir(parents=True)
    (artifact_root / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_unix": time.time(),
                "readiness_snapshot": {
                    "ready": True,
                    "status": "healthy",
                    "required_probe_blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        system_routes,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=project_root)),
    )

    payload, status_code = system_routes._runtime_manifest_boot_health_payload(
        "health_probe_timeout"
    )

    assert status_code == 200
    assert payload["ready"] is True
    assert payload["cache_status"] == "manifest"
    assert payload["required_probes"]["all_passed"] is True
    for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items():
        assert payload["required_probes"][group]["ok"] is True
        assert payload["required_probes"][group]["components"] == {
            component: True for component in components
        }


def test_runtime_manifest_absence_before_emission_is_not_a_degradation(
    tmp_path,
    monkeypatch,
):
    from interface.routes import system as system_routes

    degradations: list[tuple] = []
    monkeypatch.setattr(
        system_routes,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=tmp_path)),
    )
    monkeypatch.setattr(
        system_routes,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    assert system_routes._runtime_manifest_boot_health_payload("cold_boot") is None
    assert degradations == []


def test_runtime_manifest_malformed_after_emission_remains_a_degradation(
    tmp_path,
    monkeypatch,
):
    from interface.routes import system as system_routes

    artifact_root = tmp_path / "artifacts" / "current"
    artifact_root.mkdir(parents=True)
    (artifact_root / "runtime_manifest.json").write_text("{not-json", encoding="utf-8")
    degradations: list[tuple] = []
    monkeypatch.setattr(
        system_routes,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=tmp_path)),
    )
    monkeypatch.setattr(
        system_routes,
        "record_degradation",
        lambda *args, **kwargs: degradations.append((args, kwargs)),
    )

    assert system_routes._runtime_manifest_boot_health_payload("probe_failed") is None
    assert len(degradations) == 1
    assert isinstance(degradations[0][0][1], json.JSONDecodeError)


def test_runtime_manifest_health_fallback_rejects_stale_ready_manifest(tmp_path, monkeypatch):
    from interface.routes import system as system_routes

    project_root = tmp_path
    artifact_root = project_root / "artifacts" / "current"
    artifact_root.mkdir(parents=True)
    (artifact_root / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "generated_at_unix": 1.0,
                "readiness_snapshot": {
                    "ready": True,
                    "status": "healthy",
                    "required_probe_blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        system_routes,
        "config",
        SimpleNamespace(paths=SimpleNamespace(project_root=project_root)),
    )
    monkeypatch.setattr(system_routes, "_HEALTH_MANIFEST_FALLBACK_TTL_S", 15.0)

    payload, status_code = system_routes._runtime_manifest_boot_health_payload(
        "health_probe_timeout"
    )

    assert status_code == 503
    assert payload["ready"] is False
    assert payload["cache_status"] == "manifest_stale"
    assert "health_manifest_stale" in payload["blockers"]


def test_boot_health_probe_single_flight_fails_closed_instead_of_stacking():
    from interface.routes import system as system_routes

    assert system_routes._HEALTH_PROBE_LOCK.acquire(False)
    try:
        with pytest.raises(TimeoutError, match="health_probe_already_running"):
            system_routes._build_boot_health_payload_sync(is_gui_proxy=False)
    finally:
        system_routes._HEALTH_PROBE_LOCK.release()
