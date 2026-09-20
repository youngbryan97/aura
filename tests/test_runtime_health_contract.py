from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.health.system_health import (
    get_full_health_report,
    get_health_v2,
    get_runtime_health_contract,
)
from core.runtime.health_contract import (
    HEALTH_CONTRACT_VERSION,
    REQUIRED_HEALTH_PROBE_GROUPS,
    RUNTIME_CONTRACT,
    HealthLevel,
    ServiceRequirement,
    ServiceTier,
    evaluate_health,
    log_health_report,
    required_probe_blockers,
    required_probe_groups_pass,
    required_probe_status,
    runtime_health_report,
)


@pytest.fixture(autouse=True)
def isolated_service_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def _service_for(requirement: ServiceRequirement, *, failing_key: str | None = None) -> object:
    if requirement.liveness_check is None:
        return SimpleNamespace()
    live = requirement.container_key != failing_key
    return SimpleNamespace(**{requirement.liveness_check: lambda live=live: live})


def _register_contract_services(
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


def test_runtime_contract_report_marks_all_required_tiers_healthy():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})

    report = runtime_health_report()

    assert report["contract_version"] == HEALTH_CONTRACT_VERSION
    assert report["status"] == HealthLevel.HEALTHY.value
    assert report["healthy"] is True
    assert report["operational"] is True
    assert report["status_code"] == 200
    assert report["tier_summary"]["critical"]["failed"] == 0
    assert report["tier_summary"]["important"]["failed"] == 0
    assert report["tier_summary"]["optional"]["missing"] > 0
    assert report["failures"]["critical"] == []


def test_runtime_contract_distinguishes_degraded_from_failed_runtime():
    _register_contract_services(tiers={ServiceTier.CRITICAL})

    verdict = evaluate_health()
    report = verdict.to_report()

    assert verdict.level == HealthLevel.DEGRADED
    assert report["healthy"] is False
    assert report["operational"] is True
    assert report["tier_summary"]["important"]["failed"] > 0
    assert report["failures"]["critical"] == []


def test_runtime_contract_fails_closed_on_critical_liveness_failure():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )

    report = runtime_health_report()

    assert report["status"] == HealthLevel.CRITICAL.value
    assert report["healthy"] is False
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert report["tier_summary"]["critical"]["liveness_failed"] == 1
    assert report["failures"]["critical"][0]["container_key"] == "inference_gate"
    assert report["failures"]["critical"][0]["liveness"] == "failed"


def test_runtime_health_log_severity_tracks_each_service_status(caplog):
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )
    caplog.set_level(logging.INFO, logger="Aura.HealthContract")

    verdict = log_health_report()

    assert verdict.level == HealthLevel.CRITICAL
    records_by_message = {record.message: record.levelname for record in caplog.records}
    assert records_by_message["Health: CRITICAL"] == "CRITICAL"
    assert any(
        level == "INFO" and "Kernel Interface" in message and "[✓]" in message
        for message, level in records_by_message.items()
    )
    assert any(
        level == "CRITICAL" and "InferenceGate" in message and "liveness FAIL" in message
        for message, level in records_by_message.items()
    )


def test_runtime_contract_fails_closed_when_required_liveness_method_is_missing():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance("inference_gate", SimpleNamespace())

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert report["operational"] is False
    assert report["failures"]["critical"][0]["container_key"] == "inference_gate"
    assert report["failures"]["critical"][0]["error"] == "missing liveness check: is_inference_ready()"
    assert probes["inference"]["ok"] is False
    assert probes["all_passed"] is False


def test_runtime_contract_reports_important_liveness_failures():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="event_bus",
    )

    report = runtime_health_report()

    assert report["status"] == HealthLevel.DEGRADED.value
    assert report["operational"] is True
    assert report["failures"]["important"][0]["container_key"] == "event_bus"
    assert report["failures"]["important"][0]["liveness"] == "failed"


def test_runtime_contract_reports_dead_when_no_critical_service_exists():
    report = runtime_health_report()

    assert report["status"] == HealthLevel.DEAD.value
    assert report["operational"] is False
    assert report["status_code"] == 503
    concrete_critical = [
        service
        for service in report["services"]
        if service["tier"] == ServiceTier.CRITICAL.value
        and service["container_key"] != "unified_memory_pressure"
    ]
    assert all(service["present"] is False for service in concrete_critical)


def test_runtime_health_endpoint_uses_contract_status_code():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="inference_gate",
    )

    response = asyncio.run(get_runtime_health_contract())
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["contract_version"] == HEALTH_CONTRACT_VERSION
    assert payload["failures"]["critical"][0]["container_key"] == "inference_gate"


def test_runtime_health_projects_shutdown_progress_and_never_reports_healthy():
    from core.runtime.shutdown_coordinator import request_shutdown

    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    request_shutdown("health-contract-test")

    report = runtime_health_report()
    response = asyncio.run(get_runtime_health_contract())
    payload = json.loads(response.body)

    assert report["pre_shutdown_status"] == HealthLevel.HEALTHY.value
    assert report["status"] == "stopping"
    assert report["healthy"] is False
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert report["required_probes"]["all_passed"] is False
    assert report["probe_blockers"][0] == "runtime_shutdown"
    assert report["shutdown"]["request"]["first_reason"] == "health-contract-test"
    assert response.status_code == 503
    assert payload["status"] == "stopping"


def test_full_health_report_fails_closed_with_runtime_contract():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="scheduler",
    )

    response = asyncio.run(get_full_health_report())
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["operational"] is False
    assert payload["required_probes"]["scheduler"]["ok"] is False
    assert payload["contract"]["failures"]["critical"][0]["container_key"] == "scheduler"


def test_tricorder_health_v2_cannot_override_failed_runtime_contract():
    _register_contract_services(
        tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT},
        failing_key="authority_gateway",
    )

    async def scan(_state):
        return {"scan": "ok"}

    ServiceContainer.register_instance(
        "tricorder",
        SimpleNamespace(healthy=True, scan=scan),
    )

    response = asyncio.run(get_health_v2())
    payload = json.loads(response.body)

    assert response.status_code == 503
    assert payload["healthy"] is False
    assert payload["legacy_status"] == "degraded"
    assert payload["runtime_contract"]["required_probes"]["tool_governance"]["ok"] is False


def test_compute_orchestrator_is_liveness_checked_by_runtime_health_contract():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key == "compute_orchestrator"
    }

    assert set(required) == {"compute_orchestrator"}
    assert required["compute_orchestrator"].tier == ServiceTier.IMPORTANT
    assert required["compute_orchestrator"].liveness_check == "is_alive"


def test_user_facing_runtime_services_have_explicit_liveness_checks():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key
        in {
            "output_gate",
            "cognitive_engine",
            "affect_engine",
            "database_coordinator",
            "drive_engine",
            "agency_core",
        }
    }

    assert required["output_gate"].tier == ServiceTier.CRITICAL
    assert required["output_gate"].liveness_check == "is_ready"
    assert required["cognitive_engine"].liveness_check == "is_ready"
    assert required["affect_engine"].liveness_check == "is_ready"
    assert required["database_coordinator"].liveness_check == "is_alive"
    assert required["drive_engine"].liveness_check == "is_alive"
    assert required["agency_core"].liveness_check == "is_alive"


def test_mind_tick_is_an_explicit_important_runtime_liveness_probe():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key == "mind_tick"
    }

    assert required["mind_tick"].tier == ServiceTier.IMPORTANT
    assert required["mind_tick"].liveness_check == "is_alive"


def test_consciousness_enrichment_services_have_explicit_liveness_checks():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key
        in {
            "synaptic_plasticity",
            "temporal_continuity",
            "attention_gate",
            "somatic_qualia",
        }
    }

    assert required["synaptic_plasticity"].tier == ServiceTier.OPTIONAL
    assert required["synaptic_plasticity"].liveness_check == "is_ready"
    assert required["temporal_continuity"].tier == ServiceTier.OPTIONAL
    assert required["temporal_continuity"].liveness_check == "is_ready"
    assert required["attention_gate"].tier == ServiceTier.OPTIONAL
    assert required["attention_gate"].liveness_check == "is_ready"
    assert required["somatic_qualia"].tier == ServiceTier.OPTIONAL
    assert required["somatic_qualia"].liveness_check == "is_ready"


def test_runtime_contract_requires_kernel_inference_memory_scheduler_and_tool_governance():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.tier == ServiceTier.CRITICAL
    }

    assert required["kernel_interface"].liveness_check == "is_ready"
    assert required["inference_gate"].liveness_check == "is_inference_ready"
    assert required["llm_router"].liveness_check == "is_ready"
    assert required["state_repository"].liveness_check == "is_initialized"
    assert required["memory_facade"].liveness_check == "is_ready"
    assert required["memory_write_gateway"].liveness_check == "is_ready"
    assert REQUIRED_HEALTH_PROBE_GROUPS["memory"] == (
        "state_repository",
        "memory_facade",
        "memory_write_gateway",
        "unified_memory_pressure",
        "external_memory_sentinel",
    )
    assert required["external_memory_sentinel"].liveness_check == "is_armed"
    assert required["scheduler"].liveness_check == "is_alive"
    assert required["unified_will"].liveness_check == "is_alive"
    assert required["authority_gateway"].liveness_check == "is_ready"
    assert required["capability_engine"].liveness_check == "is_ready"
    assert required["output_gate"].liveness_check == "is_ready"


def test_runtime_contract_fails_closed_on_critical_unified_memory_pressure(
    monkeypatch,
    resource_observer,
):
    gib = 1024**3
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    resource_observer.configure_memory(
        total_bytes=64 * gib,
        available_bytes=2 * gib,
        percent=96.0,
    )

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert report["healthy"] is False
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert "probe:memory" in report["probe_blockers"]
    assert probes["memory"]["ok"] is False
    assert probes["memory"]["components"]["unified_memory_pressure"] is False
    assert any(
        failure["container_key"] == "unified_memory_pressure"
        and "memory_pressure:96.0%" in failure["error"]
        for failure in report["failures"]["critical"]
    )


def test_runtime_contract_allows_high_noncritical_unified_memory_pressure(
    monkeypatch,
    resource_observer,
):
    gib = 1024**3
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    resource_observer.configure_memory(
        total_bytes=64 * gib,
        available_bytes=int(9.5 * gib),
        percent=86.0,
    )

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.HEALTHY.value
    assert report["healthy"] is True
    assert probes["memory"]["ok"] is True
    assert probes["memory"]["components"]["unified_memory_pressure"] is True


def test_required_probe_summary_fails_if_scheduler_or_tool_governance_is_unhealthy():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance("scheduler", SimpleNamespace(is_alive=lambda: False))

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert probes["scheduler"]["ok"] is False
    assert probes["tool_governance"]["ok"] is True
    assert probes["all_passed"] is False


def test_required_probe_summary_fails_if_llm_router_is_present_but_not_ready():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance("llm_router", SimpleNamespace())

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert probes["inference"]["ok"] is False
    assert probes["inference"]["components"]["llm_router"] is False
    assert probes["all_passed"] is False
    assert any(
        failure["container_key"] == "llm_router"
        for failure in report["failures"]["critical"]
    )


def test_health_aware_llm_router_readiness_requires_routable_endpoint():
    from core.brain.llm_health_router import HealthAwareLLMRouter

    router = HealthAwareLLMRouter()
    assert router.is_ready() is False

    router.register(
        "Cortex",
        url="http://127.0.0.1:9001",
        model="local-cortex",
        is_local=True,
        tier="local",
    )
    assert router.is_ready() is True

    for endpoint in router.endpoints.values():
        endpoint.trip_temporarily("unit_test")
    assert router.is_ready() is False


def test_observability_readiness_uses_runtime_health_contract():
    from core.observability.metrics import check_readiness

    ServiceContainer.clear()

    result = check_readiness()

    assert result["ready"] is False
    assert result["status"] == "not_ready"
    assert any(issue.startswith("runtime_contract:") for issue in result["issues"])
    assert any(issue.startswith("required_probes:") for issue in result["issues"])


def test_observability_readiness_uses_mind_tick_not_metabolic_pulse(monkeypatch):
    import time

    from core.observability import metrics as metrics_module

    collector = metrics_module.MetricsCollector()
    collector._boot_time = time.time() - 300.0
    collector._tick_count = 5
    collector._last_tick_time = time.time() - 45.0
    monkeypatch.setattr(metrics_module, "_metrics_instance", collector)
    ServiceContainer.register_instance(
        "mind_tick",
        SimpleNamespace(
            get_health_status=lambda: {
                "healthy": True,
                "tick_count": 150,
                "last_successful_tick_at": time.time() - 1.0,
                "last_loop_progress_at": time.time() - 0.5,
                "active_tick_stage": "idle",
            },
            is_alive=lambda: True,
        ),
    )
    monkeypatch.setattr(
        "core.runtime.health_contract.runtime_health_report",
        lambda: {
            "status": "healthy",
            "operational": True,
            "required_probes": {
                group: {
                    "ok": True,
                    "components": {component: True for component in components},
                }
                for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
            }
            | {"all_passed": True},
        },
    )

    result = metrics_module.check_readiness()

    assert result["ready"] is True
    assert result["tick_source"] == "mind_tick"
    assert result["tick_count"] == 150
    assert result["tick_age_s"] < 2.0
    assert result["metabolic_tick_count"] == 5
    assert result["metabolic_tick_age_s"] >= 45.0
    assert not any("last_tick_stale" in issue for issue in result["issues"])


def test_runtime_health_report_attributes_probe_and_total_latency(monkeypatch):
    from core.runtime import health_contract

    requirement = health_contract.RUNTIME_CONTRACT[0]
    service = SimpleNamespace()

    def liveness():
        time.sleep(0.01)
        return True

    setattr(service, str(requirement.liveness_check), liveness)
    monkeypatch.setattr(health_contract, "RUNTIME_CONTRACT", [requirement])
    monkeypatch.setattr(
        health_contract,
        "get_runtime_service",
        lambda key, default=None: service if key == requirement.container_key else default,
    )
    monkeypatch.setattr(
        health_contract,
        "_unified_memory_pressure_status",
        lambda: health_contract.ServiceStatus(
            health_contract.UNIFIED_MEMORY_PRESSURE_REQUIREMENT,
            True,
            True,
        ),
    )
    monkeypatch.setattr(
        health_contract,
        "_runtime_pressure_status",
        lambda: health_contract.ServiceStatus(
            health_contract.UNIFIED_RUNTIME_PRESSURE_REQUIREMENT,
            True,
            True,
        ),
    )

    report = health_contract.evaluate_health().to_report()
    measured = next(
        item
        for item in report["services"]
        if item["container_key"] == requirement.container_key
    )

    assert measured["duration_ms"] >= 8.0
    assert report["evaluation_duration_ms"] >= measured["duration_ms"]


def test_runtime_contract_rejects_deferred_inference_as_healthy():
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})
    ServiceContainer.register_instance(
        "inference_gate",
        SimpleNamespace(is_alive=lambda: True, is_inference_ready=lambda: False),
    )

    report = runtime_health_report()
    probes = required_probe_status(report)

    assert report["status"] == HealthLevel.CRITICAL.value
    assert report["operational"] is False
    assert report["failures"]["critical"][0]["container_key"] == "inference_gate"
    assert report["failures"]["critical"][0]["error"] == "is_inference_ready() returned False"
    assert probes["inference"]["ok"] is False


def test_required_probe_groups_require_explicit_liveness_ok():
    services = [
        {
            "container_key": component,
            "present": True,
            "liveness": "not_configured",
        }
        for components in REQUIRED_HEALTH_PROBE_GROUPS.values()
        for component in components
    ]

    probes = required_probe_status({"services": services})

    assert probes["all_passed"] is False
    assert all(
        probe["ok"] is False
        for group, probe in probes.items()
        if group != "all_passed"
    )
    assert required_probe_groups_pass(probes) is False


def test_required_probe_groups_reject_partial_or_forged_payloads():
    forged = {
        "all_passed": True,
        "kernel": {"ok": True, "components": {"kernel_interface": True}},
        "inference": {
            "ok": True,
            "components": {"inference_gate": True, "llm_router": True},
        },
        "memory": {
            "ok": True,
            "components": {"state_repository": True, "memory_facade": True},
        },
        "scheduler": {"ok": True, "components": {"scheduler": True}},
    }

    assert required_probe_groups_pass(forged) is False
    assert required_probe_blockers(forged) == [
        "runtime_required_probes",
        "probe:inference",
        "probe:memory",
        "probe:scheduler",
        "probe:tool_governance",
        "probe:workspace",
        "probe:attention",
        "probe:live_mind",
    ]


def test_required_probe_blockers_fail_closed_on_malformed_payloads():
    assert required_probe_blockers(None) == ["runtime_required_probes"]
    assert required_probe_blockers({"all_passed": True}) == [
        "runtime_required_probes",
        "probe:kernel",
        "probe:inference",
        "probe:memory",
        "probe:scheduler",
        "probe:tool_governance",
        "probe:workspace",
        "probe:attention",
        "probe:live_mind",
    ]


def test_health_report_healthy_requires_required_probe_groups(monkeypatch):
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})

    def forged_probe_status(_services):
        return {
            "all_passed": True,
            "kernel": {"ok": True, "components": {"kernel_interface": True}},
        }

    monkeypatch.setattr(
        "core.runtime.health_contract._required_probe_status_from_services",
        forged_probe_status,
    )

    report = runtime_health_report()

    assert report["status"] == HealthLevel.HEALTHY.value
    assert report["healthy"] is False
    assert report["operational"] is False
    assert report["status_code"] == 503
    assert report["probe_blockers"] == [
        "runtime_required_probes",
        "probe:inference",
        "probe:memory",
        "probe:scheduler",
        "probe:tool_governance",
        "probe:workspace",
        "probe:attention",
        "probe:live_mind",
    ]


def test_health_verdict_status_code_uses_required_probe_groups(monkeypatch):
    _register_contract_services(tiers={ServiceTier.CRITICAL, ServiceTier.IMPORTANT})

    def forged_probe_status(_services):
        return {
            "all_passed": True,
            "kernel": {"ok": True, "components": {"kernel_interface": True}},
        }

    monkeypatch.setattr(
        "core.runtime.health_contract._required_probe_status_from_services",
        forged_probe_status,
    )

    verdict = evaluate_health()

    assert verdict.level == HealthLevel.HEALTHY
    assert verdict.is_operational is True
    assert verdict.status_code == 503


class TestTheShapeItIsHolding:
    """The topology had one reader and it was an HTTP route.

    `_collect_morphogenesis_status` in interface/routes/system.py was the
    only thing that looked at it, so nothing reading this report could see
    the shape — including the mind's own self-knowledge. Asked live on
    2026-09-20 which of its cells were cut off, with `/api/health` carrying
    `components: 7, partitioned: true` at that moment, the answer was "I
    don't have any operational information regarding a system called
    'morphogenic cell'".
    """

    def _block(self, runtime):
        from core.runtime import service_registry
        from core.runtime.health_contract import (
            _integrity_of_control_and_measured_effect,
        )

        previous = service_registry._service_resolver
        service_registry.install_service_resolver(
            lambda name, default=None: runtime
            if name == "morphogenetic_runtime"
            else default
        )
        try:
            block: dict[str, object] = {}
            _integrity_of_control_and_measured_effect(block)
            return block
        finally:
            service_registry.install_service_resolver(previous)

    def test_the_partition_reaches_the_integrity_report(self) -> None:
        runtime = SimpleNamespace(
            status=lambda: {
                "topology": {"version": 2, "edges": 196, "components": 7},
                "registry": {"cells": 50, "organs": 45},
            }
        )

        shape = self._block(runtime)["the_shape_it_is_holding"]

        assert shape == {
            "cells": 50,
            "organs": 45,
            "bindings": 196,
            "topology_version": 2,
            "components": 7,
            "partitioned": True,
        }

    def test_one_component_is_not_partitioned(self) -> None:
        runtime = SimpleNamespace(
            status=lambda: {
                "topology": {"version": 3, "edges": 12, "components": 1},
                "registry": {"cells": 6, "organs": 0},
            }
        )

        shape = self._block(runtime)["the_shape_it_is_holding"]

        assert shape["partitioned"] is False
        assert shape["components"] == 1

    def test_no_runtime_is_silence_not_an_error(self) -> None:
        """Health must never raise at its caller, and must not invent a shape."""
        block = self._block(None)

        assert "the_shape_it_is_holding" not in block
        assert "the_shape_it_is_holding_error" not in block

    def test_a_runtime_that_raises_is_reported_not_swallowed(self) -> None:
        def _angry():
            raise RuntimeError("the governor is mid-transaction")

        block = self._block(SimpleNamespace(status=_angry))

        assert "the_shape_it_is_holding" not in block
        assert "the governor is mid-transaction" in block["the_shape_it_is_holding_error"]
