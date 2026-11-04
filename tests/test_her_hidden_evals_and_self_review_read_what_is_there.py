"""Her hidden evaluations and her self-review read the services and fields that exist.

On both day runs four hidden-eval scenarios read "unavailable" with every service they measure running: they asked
for service names nothing registers (continuous_substrate, whole_system_phi_service), an attribute no value node
carries (baseline_weight), and a world model that reports its surprise only through telemetry. And her self-review
flagged "low phi (phi=0.000, entropy=0.000)" on a field the feedback observer does not report: it says current_phi.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.architect import hidden_eval

pytestmark = pytest.mark.unit


def _registered(monkeypatch, services: dict):
    import core.runtime.service_registry as registry

    monkeypatch.setattr(registry, "get_runtime_service", lambda name, default=None: services.get(name, default))


def test_the_substrate_is_found_under_her_name(monkeypatch):
    _registered(monkeypatch, {"liquid_substrate": SimpleNamespace(x=np.array([0.3, -0.4]))})
    assert hidden_eval._probe_substrate_energy() == pytest.approx(0.5 / np.sqrt(2))


def test_phi_is_read_from_the_whole_system_estimate(monkeypatch):
    service = SimpleNamespace(latest=lambda: SimpleNamespace(phi_raw=0.42))
    _registered(monkeypatch, {"whole_system_phi": service})
    assert hidden_eval._probe_phi_value() == pytest.approx(0.42)


def test_value_drift_reads_where_each_value_started(monkeypatch):
    import core.adaptation.dynamic_value_graph as graph_module

    nodes = {"a": SimpleNamespace(weight=0.6, origin_weight=0.5), "b": SimpleNamespace(weight=0.5, origin_weight=0.5)}
    monkeypatch.setattr(graph_module, "get_dynamic_value_graph", lambda: SimpleNamespace(_nodes=nodes))
    assert hidden_eval._probe_value_drift() == pytest.approx(0.1)


def test_world_surprise_is_read_from_its_telemetry_channel(monkeypatch):
    import core.fsw.telemetry_dictionary as telemetry

    sample = SimpleNamespace(value=1.7)
    fake = SimpleNamespace(value=lambda name: sample if name == "ontogeny.world_surprise" else None)
    monkeypatch.setattr(telemetry, "get_telemetry", lambda: fake)
    _registered(monkeypatch, {})
    assert hidden_eval._probe_world_model_surprise() == pytest.approx(1.7)


def test_nothing_registered_is_still_unavailable_never_zero(monkeypatch):
    import core.fsw.telemetry_dictionary as telemetry

    monkeypatch.setattr(telemetry, "get_telemetry", lambda: SimpleNamespace(value=lambda name: None))
    _registered(monkeypatch, {})
    for probe in (hidden_eval._probe_substrate_energy, hidden_eval._probe_phi_value,
                  hidden_eval._probe_world_model_surprise):
        with pytest.raises(hidden_eval.ProbeUnavailable):
            probe()


def _review(loop_state: dict, caplog) -> list[str]:
    import logging

    from core.kernel.self_review import SelfReviewPhase
    from core.state.aura_state import AuraState

    phase = SelfReviewPhase(SimpleNamespace(loop_state=lambda: loop_state))
    phase._last_review_ts = 0.0
    with caplog.at_level(logging.DEBUG, logger=phase.__module__ if hasattr(phase, "__module__") else None):
        asyncio.run(phase.execute(AuraState.default()))
    return [r.getMessage() for r in caplog.records if "SELF-REVIEW] High entropy or low phi" in r.getMessage()]


def test_self_review_reads_the_phi_the_observer_reports(caplog):
    assert _review({"current_phi": 0.55}, caplog) == []


def test_self_review_with_no_reading_flags_nothing(caplog):
    assert _review({"status": "no_ticks_yet"}, caplog) == []


def test_self_review_still_flags_a_low_measured_phi(caplog):
    assert _review({"current_phi": 0.05}, caplog)
