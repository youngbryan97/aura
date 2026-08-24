import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.health.boot_status import build_boot_health_snapshot
from core.runtime.errors import get_degradation_tracker
from core.runtime.health_contract import (
    RUNTIME_CONTRACT,
    ServiceRequirement,
    ServiceTier,
    evaluate_health,
)


@pytest.fixture(autouse=True)
def isolated_runtime_contract_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()


def _service_for(requirement: ServiceRequirement, *, failing_key: str | None = None) -> object:
    if requirement.liveness_check is None:
        return SimpleNamespace()
    live = requirement.container_key != failing_key
    return SimpleNamespace(**{requirement.liveness_check: lambda live=live: live})


def _register_runtime_contract_services(
    *,
    tiers: set[ServiceTier],
    failing_key: str | None = None,
) -> None:
    for requirement in RUNTIME_CONTRACT:
        if requirement.tier in tiers:
            ServiceContainer.register_instance(
                requirement.container_key,
                _service_for(requirement, failing_key=failing_key),
            )


def test_boot_health_ready_for_kernel_mode():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["status_message"] == "Aura is awake."
    assert payload["ready"] is True
    assert payload["launcher_ready"] is True
    assert payload["boot_phase"] == "kernel_ready"
    assert payload["progress"] == 100
    assert payload["semver"]
    assert payload["version"].startswith("Aura Luna v")
    assert payload["checks"]["runtime_integrity"] is True
    assert payload["checks"]["runtime_contract_operational"] is True
    assert payload["checks"]["runtime_contract_healthy"] is True
    assert payload["runtime_contract"]["status"] == "healthy"
    assert payload["blockers"] == []


def test_boot_health_reports_booting_when_orchestrator_missing():
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        None,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": False, "state": "warming"},
    )

    assert status_code == 503
    assert payload["status"] == "booting"
    assert payload["status_message"] == "Starting Aura kernel…"
    assert payload["ready"] is False
    assert payload["boot_phase"] == "kernel_bootstrap"
    assert payload["progress"] == 14
    assert "orchestrator" in payload["blockers"]


def test_boot_health_proxy_mode_cannot_impersonate_runtime_health():
    payload, status_code = build_boot_health_snapshot(
        None,
        {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"},
        is_gui_proxy=True,
    )

    assert status_code == 503
    assert payload["status"] == "not_ready"
    assert payload["status_message"] == "Aura proxy is alive; canonical runtime is not ready."
    assert payload["ready"] is False
    assert payload["launcher_ready"] is False
    assert payload["mode"] == "gui_proxy"
    assert payload["boot_phase"] == "proxy_transport_only"
    assert payload["checks"]["runtime_required_probes"] is False
    assert payload["required_probes"]["all_passed"] is False
    assert "runtime_required_probes" in payload["blockers"]


def test_boot_health_separates_system_ready_from_conversation_ready():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": False, "state": "warming"},
    )

    assert status_code == 503
    assert payload["status"] == "warming"
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["system_ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["boot_phase"] == "conversation_warming"
    assert payload["status_message"] == "Warming local Cortex…"
    assert payload["progress"] == 78
    assert "conversation_ready" in payload["blockers"]


def test_boot_health_treats_active_foreground_generation_as_working_not_unhealthy():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "ready",
            "active_generations": 1,
            "last_failure_reason": "active_generation_in_flight",
        },
    )

    assert status_code == 200
    assert payload["status"] == "working"
    assert payload["system_ready"] is True
    # A functional lane actively answering a turn is READY. Reporting a busy
    # lane as not-ready made the desktop shell sit at "Connecting to runtime"
    # for the length of a long turn or a run of back-to-back turns (live,
    # July 2026). Readiness is Kubernetes-style: serving a request is ready.
    assert payload["ready"] is True
    assert payload["launcher_ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["conversation_busy"] is True
    assert payload["boot_phase"] == "conversation_working"
    assert payload["status_message"] == "Aura is answering through the live conversation lane."
    assert "conversation_ready" not in payload["blockers"]


def test_boot_health_warming_lane_busy_is_not_ready_unlike_serving_lane():
    """The two 'busy' states must not collapse: serving ⇒ ready, warming ⇒ not."""
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    warming, _ = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "spawning",
            "active_generations": 0,
            "warmup_in_flight": True,
        },
    )
    assert warming["conversation_busy"] is True
    assert warming["ready"] is False, "a lane busy *warming up* is not yet ready"
    assert warming["boot_phase"] == "conversation_working"


def test_boot_health_treats_handshaking_warmup_as_working():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "handshaking",
            "warmup_attempted": True,
            "warmup_in_flight": True,
            "readiness_blockers": ["visible_conversation_probe_missing", "warmup_in_flight"],
            "last_failure_reason": "visible_conversation_probe_missing",
        },
    )

    assert status_code == 200
    assert payload["status"] == "working"
    assert payload["system_ready"] is True
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["conversation_busy"] is True
    assert payload["boot_phase"] == "conversation_working"
    assert payload["status_message"] == "Aura is answering through the live conversation lane."
    assert "conversation_ready" not in payload["blockers"]


def test_boot_health_treats_cold_standby_lane_as_not_conversation_ready():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "cold",
            "warmup_attempted": False,
            "warmup_in_flight": False,
        },
    )

    assert status_code == 200
    assert payload["status"] == "warming"
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["system_ready"] is True
    assert payload["boot_phase"] == "conversation_warming"
    assert payload["conversation_ready"] is False
    assert payload["status_message"] == "Warming local Cortex…"
    assert "conversation_ready" in payload["blockers"]


def test_boot_health_safe_desktop_boot_does_not_fake_cold_conversation_ready(monkeypatch):
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "cold",
            "warmup_attempted": False,
            "warmup_in_flight": False,
        },
    )

    assert status_code == 200
    assert payload["system_ready"] is True
    assert payload["launcher_ready"] is True
    assert payload["ready"] is False
    assert payload["conversation_ready"] is False
    assert payload["boot_phase"] == "conversation_warming"
    assert "conversation_ready" in payload["blockers"]


def test_boot_health_reports_hard_conversation_failure():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={
            "conversation_ready": False,
            "state": "failed",
            "last_failure_reason": "local_runtime_unavailable:server_unreachable",
        },
    )

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["system_ready"] is True
    assert payload["conversation_ready"] is False
    assert payload["boot_phase"] == "conversation_failed"
    assert payload["status_message"] == (
        "Local Cortex is unavailable: Aura's managed backend failed during startup."
    )
    assert "conversation_failed" in payload["blockers"]


def test_boot_health_fails_closed_when_runtime_contract_is_not_operational():
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 503
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["checks"]["runtime_contract_operational"] is False
    assert payload["runtime_contract"]["status"] == "dead"
    assert "runtime_contract" in payload["blockers"]
    assert any(blocker.startswith("critical:") for blocker in payload["blockers"])


def test_boot_health_fails_closed_when_required_runtime_probe_fails():
    _register_runtime_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="scheduler",
    )
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 503
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["checks"]["runtime_required_probes"] is False
    assert payload["required_probes"]["scheduler"]["ok"] is False
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:scheduler" in payload["blockers"]


def test_boot_health_records_health_check_failure_as_structured_degradation():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=12,
        start_time=time.time() - 5,
    )

    class FailingHealthProbe:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> bool:
            self.calls += 1
            raise RuntimeError("orchestrator probe timed out")

    failing_health_check = FailingHealthProbe()
    orchestrator = SimpleNamespace(status=status, health_check=failing_health_check)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    records = get_degradation_tracker().recent(subsystem="boot_status")

    # A failing orchestrator health probe with a READY conversation lane and
    # an operational runtime contract is "degraded but conversational": the
    # desktop must connect (200) while the degradation stays fully visible.
    assert status_code == 200
    assert payload["status"] == "degraded"
    assert payload["boot_phase"] == "conversation_operational"
    assert payload["system_ready"] is False
    assert "healthy" in payload["blockers"]
    assert payload["launcher_ready"] is True
    assert payload["health_check_error"] == "orchestrator probe timed out"
    assert records
    assert records[-1].severity == "degraded"
    assert "failed closed" in records[-1].action
    assert failing_health_check.calls == 1


def test_runtime_health_contract_rejects_async_liveness_coroutine():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})

    async def async_ready() -> bool:
        return True

    ServiceContainer.register_instance(
        "inference_gate",
        SimpleNamespace(is_inference_ready=lambda: async_ready()),
    )

    report = evaluate_health().to_report()

    assert report["status_code"] == 503
    assert report["healthy"] is False
    assert report["required_probes"]["inference"]["ok"] is False
    failures = report["failures"]["critical"]
    inference_failure = next(item for item in failures if item["container_key"] == "inference_gate")
    assert "awaitable" in inference_failure["error"]


def test_runtime_health_contract_rejects_truthy_non_bool_liveness():
    _register_runtime_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance(
        "scheduler",
        SimpleNamespace(is_alive=lambda: "ready"),
    )

    report = evaluate_health().to_report()

    assert report["status_code"] == 503
    assert report["healthy"] is False
    assert report["required_probes"]["scheduler"]["ok"] is False
    failures = report["failures"]["critical"]
    scheduler_failure = next(item for item in failures if item["container_key"] == "scheduler")
    assert "unsupported liveness result type: str" in scheduler_failure["error"]


def test_boot_health_degraded_important_service_is_conversational_not_booting():
    """A conversational instance with ONE degraded important-tier service must
    present as 'degraded but conversational', never as perpetual 'booting'.

    Observed live (Jul 6): mind_tick unhealthy under event-loop load turned
    orchestrator `healthy` false, and the shell showed "booting, 48%" for 55
    minutes while chat worked the whole time.
    """
    _register_runtime_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="mind_tick",
    )
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=False,  # orchestrator health_check reflects the degradation
        last_error="",
        cycle_count=52_000,
        start_time=time.time() - 3300,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: False)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 200
    assert payload["boot_phase"] == "conversation_operational"
    assert payload["status"] == "degraded"
    assert payload["ready"] is True
    assert payload["launcher_ready"] is True
    # The degradation stays visible — honesty about health, honesty about readiness.
    assert payload["system_ready"] is False
    assert "healthy" in payload["blockers"]
    assert any(b.startswith("important:") for b in payload["blockers"])


def test_boot_health_warming_cortex_stays_conversational_via_fallback():
    """A failed critical inference probe must not look healthy even if the
    conversation lane reports ready. This keeps the boot/heartbeat surface from
    overstating live-chat readiness."""
    _register_runtime_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=5000,
        start_time=time.time() - 600,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 503
    assert payload["boot_phase"] == "kernel_warming"
    assert payload["status"] == "booting"
    assert payload["ready"] is False
    assert payload["launcher_ready"] is True
    assert payload["system_ready"] is False
    assert "runtime_contract" in payload["blockers"]
    assert "critical:inference_gate" in payload["blockers"]
    assert "runtime_required_probes" in payload["blockers"]
    assert "probe:inference" in payload["blockers"]
    assert any("inference" in b for b in payload["blockers"])


def test_boot_health_failing_memory_probe_still_blocks_conversation():
    """The fallback ladder cannot rescue a broken memory probe — that group
    must still gate conversational readiness."""
    _register_runtime_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="state_repository",  # a memory-group required probe
    )
    status = SimpleNamespace(
        initialized=True,
        running=True,
        healthy=True,
        last_error="",
        cycle_count=5000,
        start_time=time.time() - 600,
    )
    orchestrator = SimpleNamespace(status=status, health_check=lambda: True)
    runtime = {"state": {"process_id": 1234}, "sha256": "abc123", "signature": "sig"}

    payload, status_code = build_boot_health_snapshot(
        orchestrator,
        runtime,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert payload["boot_phase"] != "conversation_operational"
    assert payload["ready"] is False
    assert any("memory" in b or "state_repository" in b for b in payload["blockers"])


def test_api_health_status_never_regresses_to_booting_while_answering():
    """The public status word must not say "booting" on a runtime that serves.

    Every branch below "ok" required `service_ok`, which is
    boot_snapshot["system_ready"] — False whenever ANY important-tier service is
    degraded. So a degraded-but-conversational runtime fell through to
    "booting". Measured live: 52 minutes of uptime, chat answering normally,
    top-level status "booting", while the boot snapshot one layer down had
    already concluded status="degraded" phase="conversation_operational".
    """
    from interface.routes.system import _derive_api_health_status

    degraded_but_serving = {
        "status": "degraded",
        "boot_phase": "conversation_operational",
        "system_ready": False,
    }

    assert _derive_api_health_status(
        healthy_ready=False,
        service_ok=False,          # an important-tier service is degraded
        lane_is_standby=False,
        lane_state="ready",
        conversation_ready=True,   # ...but chat works
        conversation_busy=False,
        boot_snapshot=degraded_but_serving,
    ) == "degraded"

    # Mid-turn on that same degraded runtime is still degraded, not booting —
    # carried by the snapshot's own verdict, not by "busy".
    assert _derive_api_health_status(
        healthy_ready=False,
        service_ok=False,
        lane_is_standby=False,
        lane_state="ready",
        conversation_ready=False,
        conversation_busy=True,
        boot_snapshot=degraded_but_serving,
    ) == "degraded"

    # A COLD boot is busy while it warms the lane. That must still read
    # "booting" — caught on a genuine first start whose blockers still included
    # critical:inference_gate. Busy is not evidence of having served.
    assert _derive_api_health_status(
        healthy_ready=False,
        service_ok=False,
        lane_is_standby=False,
        lane_state="warming",
        conversation_ready=False,
        conversation_busy=True,
        boot_snapshot={"status": "booting", "boot_phase": "kernel_warming"},
    ) == "booting"

    # A genuine cold boot — nothing has served yet — must still say "booting".
    assert _derive_api_health_status(
        healthy_ready=False,
        service_ok=False,
        lane_is_standby=False,
        lane_state="cold",
        conversation_ready=False,
        conversation_busy=False,
        boot_snapshot={"status": "booting", "boot_phase": "kernel_bootstrap"},
    ) == "booting"

    # And the healthy ladder is unchanged.
    assert _derive_api_health_status(
        healthy_ready=True, service_ok=True, lane_is_standby=False,
        lane_state="ready", conversation_ready=True, conversation_busy=False,
    ) == "ok"
    for state, expected in (("failed", "unavailable"), ("recovering", "recovering")):
        assert _derive_api_health_status(
            healthy_ready=False, service_ok=True, lane_is_standby=False,
            lane_state=state, conversation_ready=False, conversation_busy=False,
        ) == expected
    assert _derive_api_health_status(
        healthy_ready=False, service_ok=True, lane_is_standby=True,
        lane_state="ready", conversation_ready=True, conversation_busy=False,
    ) == "standby"
    assert _derive_api_health_status(
        healthy_ready=False, service_ok=True, lane_is_standby=False,
        lane_state="ready", conversation_ready=False, conversation_busy=True,
    ) == "working"
    assert _derive_api_health_status(
        healthy_ready=False, service_ok=True, lane_is_standby=False,
        lane_state="warming", conversation_ready=False, conversation_busy=False,
    ) == "warming"
