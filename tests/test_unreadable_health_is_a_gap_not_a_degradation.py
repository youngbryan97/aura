"""A service whose health cannot be read is a gap in what is observed, not a degradation of the running process.

LIVE 8 November 2026: "proof integrity degraded: autonomous_resilience: 163 degradations in the last 30m (latest:
RuntimeError: BicameralAdvisory: get_status exposes no recognised health field ...)", seventeen minutes into a process.
Each service the resilience probe could not read was recorded once, at debug, so none reached her log, and together
they kept proof integrity marked degraded.
"""
from __future__ import annotations

import pytest

import core.adaptation.autonomous_resilience as resilience
from core.adaptation.autonomous_resilience import IntegrationAuditor


class _Opaque:
    def get_status(self):
        return {"status": "active"}


@pytest.fixture(autouse=True)
def _fresh_gaps(monkeypatch):
    monkeypatch.setattr(resilience, "_reported_uninstrumented", {})


def test_a_service_with_no_health_of_its_own_is_kept_as_a_gap_and_not_recorded_as_a_degradation(monkeypatch):
    def _recorded(*_args, **_kwargs):
        raise AssertionError("a gap in observation is not a degradation")

    monkeypatch.setattr(resilience, "record_degradation", _recorded)
    probe = IntegrationAuditor._health_probe_for(_Opaque())
    assert probe() == resilience._UNKNOWN_HEALTH_SCORE
    assert resilience.uninstrumented_services() == {"_Opaque": "get_status exposes no recognised health field"}


def test_a_status_that_says_whether_it_is_healthy_is_read():
    class _Doctor:
        def __init__(self, healthy):
            self.healthy = healthy

        def get_status(self):
            return {"status": "healthy" if self.healthy else "unhealthy", "healthy": self.healthy}

    assert IntegrationAuditor._health_probe_for(_Doctor(True))() == 1.0
    assert IntegrationAuditor._health_probe_for(_Doctor(False))() == 0.35
    assert resilience.uninstrumented_services() == {}


def test_the_bicameral_advisory_reads_its_own_health_from_its_bookkeeping():
    from core.brain.bicameral_advisory import BicameralAdvisory

    advisory = BicameralAdvisory()
    probe = IntegrationAuditor._health_probe_for(advisory)
    assert probe() == 1.0
    assert advisory.get_status()["health_score"] == 1.0
    perspective = next(iter(advisory._reliability))
    advisory._reliability[perspective] = float("nan")
    assert probe() == 0.75
    assert resilience.uninstrumented_services() == {}
