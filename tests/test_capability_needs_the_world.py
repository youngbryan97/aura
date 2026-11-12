"""Having a faculty and being able to use it are different facts.

Bryan: "A human is capable of searching too but likely won't get confused if
they try to use google with no internet. They would just know 'it won't
work.'... she should logically be able to infer 'Hey, I can't use web search
right now' and stick to that because she can logically separate tool
capability from the external factors needed to use that tool."

Aura had one axis. The registry knows which skills are registered and
enabled; nothing in it knows whether the world outside the process can
currently support them. A missing network therefore looked exactly like a
missing skill, so she would either claim a capability that could not possibly
work or deny one she has.

The second axis is preconditions, probed from the world. The verdict is
COMPOSED — capability AND preconditions — which is what makes it reasoning
rather than phrasing: unplug the network and the conclusion changes by
itself, with nothing anywhere mentioning networks.

The distinction these tests protect is the one a person makes without
effort: the faculty stays intact while the world is broken.
"""

import threading
import time
import types

import pytest

import core.conversation.capability_preconditions as preconditions
from core.conversation.capability_condition import (
    CapabilityStanding,
    capability_condition_evidence,
    condition_for,
)

pytestmark = pytest.mark.unit


class _Engine:
    def iter_tool_catalog(self, *, include_inactive: bool = True):
        return [{"name": "web_search", "available": True}]


def _await_probe(name: str, *, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    pending_fact = f"{name.replace('_', ' ')} state is being measured"
    while time.monotonic() < deadline:
        state = preconditions.precondition_state(name)
        if state.fact != pending_fact:
            return state
        time.sleep(0.001)
    raise AssertionError(f"precondition probe {name!r} did not finish")


@pytest.fixture
def offline(monkeypatch):
    preconditions.reset_precondition_cache()
    monkeypatch.setitem(
        preconditions._PROBES,
        "network",
        lambda: preconditions.PreconditionState(
            "network", False, "there is no network connection right now"
        ),
    )
    _await_probe("network")
    yield
    preconditions.reset_precondition_cache()


@pytest.fixture
def online(monkeypatch):
    preconditions.reset_precondition_cache()
    monkeypatch.setitem(
        preconditions._PROBES,
        "network",
        lambda: preconditions.PreconditionState("network", True, "there is a network"),
    )
    _await_probe("network")
    yield
    preconditions.reset_precondition_cache()


class TestTheTwoAxesCompose:
    def test_with_the_world_intact_it_is_simply_ready(self, online):
        assert condition_for("web_search", capability_engine=_Engine()).standing is (
            CapabilityStanding.READY
        )

    def test_without_the_network_it_is_blocked_not_absent(self, offline):
        condition = condition_for("web_search", capability_engine=_Engine())
        assert condition.standing is CapabilityStanding.BLOCKED_BY_PRECONDITION
        assert condition.standing is not CapabilityStanding.ABSENT

    def test_the_faculty_survives_the_outage(self, offline):
        """She does not forget how to search when the wifi drops."""
        condition = condition_for("web_search", capability_engine=_Engine())
        assert condition.faculty_intact
        assert condition.is_transient
        assert "network" in condition.missing_preconditions

    def test_the_conclusion_changes_with_the_world(self, monkeypatch):
        """The point of composing rather than asserting."""
        engine = _Engine()
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", True, "there is a network"),
        )
        _await_probe("network")
        assert condition_for("web_search", capability_engine=engine).standing is (
            CapabilityStanding.READY
        )
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", False, "no network"),
        )
        _await_probe("network")
        assert condition_for("web_search", capability_engine=engine).standing is (
            CapabilityStanding.BLOCKED_BY_PRECONDITION
        )


class TestWhatSheIsToldAboutIt:
    def test_the_evidence_separates_faculty_from_world(self, offline):
        block = capability_condition_evidence(
            "can you look up the weather", capability_engine=_Engine()
        )
        assert "held, but it cannot work right now" in block
        assert "intact" in block
        assert "no network connection" in block

    def test_it_is_still_facts_not_a_script(self, offline):
        block = capability_condition_evidence(
            "can you look up the weather", capability_engine=_Engine()
        )
        assert "Say this" not in block and "Do not" not in block  # facts only (F06)
        for canned in ("I can't access", "I'm sorry", "Unfortunately"):
            assert canned not in block


class TestAnUnknownWorldIsNotABrokenOne:
    def test_a_probe_that_cannot_answer_is_not_a_failure(self, monkeypatch):
        """Reporting "there's no network" because a socket raised something
        unexpected is the same confident lie as reporting a missing skill
        because a registry read failed."""
        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", False, "unknown", unknown=True),
        )
        state = _await_probe("network")
        assert state.unknown
        assert preconditions.failing_preconditions("web_search") == ()
        assert condition_for("web_search", capability_engine=_Engine()).standing is (
            CapabilityStanding.READY
        )
        preconditions.reset_precondition_cache()


class TestProbesNeverBlockTheConversation:
    def test_a_slow_accessibility_probe_runs_outside_the_chat_caller(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()

        def slow_probe():
            entered.set()
            assert release.wait(timeout=1.0)
            return preconditions.PreconditionState(
                "accessibility_permission", True, "accessibility is granted"
            )

        preconditions.reset_precondition_cache()
        monkeypatch.setitem(preconditions._PROBES, "accessibility_permission", slow_probe)

        started = time.monotonic()
        state = preconditions.precondition_state("accessibility_permission")
        elapsed = time.monotonic() - started

        assert elapsed < 0.05
        assert state.unknown
        assert entered.wait(timeout=0.5)
        assert preconditions.request_precondition_refresh("accessibility_permission") is False

        release.set()
        measured = _await_probe("accessibility_permission")
        assert measured.satisfied
        assert not measured.unknown
        preconditions.reset_precondition_cache()

    def test_reset_rejects_a_late_result_from_an_obsolete_generation(self, monkeypatch):
        entered = threading.Event()
        release = threading.Event()

        def obsolete_probe():
            entered.set()
            assert release.wait(timeout=1.0)
            return preconditions.PreconditionState("network", False, "obsolete offline")

        preconditions.reset_precondition_cache()
        monkeypatch.setitem(preconditions._PROBES, "network", obsolete_probe)
        assert preconditions.precondition_state("network").unknown
        assert entered.wait(timeout=0.5)

        preconditions.reset_precondition_cache()
        monkeypatch.setitem(
            preconditions._PROBES,
            "network",
            lambda: preconditions.PreconditionState("network", True, "fresh online"),
        )
        release.set()

        measured = _await_probe("network")
        assert measured.satisfied
        assert measured.fact == "fresh online"
        preconditions.reset_precondition_cache()


class TestAccessibilityUsesThePassiveNativeStatusAPI:
    def test_accessibility_is_read_without_osascript(self, monkeypatch):
        monkeypatch.setattr(preconditions.sys, "platform", "darwin")

        application_services = types.SimpleNamespace(**{"AXIsProcessTrusted": lambda: True})
        monkeypatch.setitem(preconditions.sys.modules, "ApplicationServices", application_services)

        state = preconditions._probe_accessibility_permission()

        assert state.satisfied
        assert not state.unknown


class TestNetworkUsesIndependentWebSignals:
    def test_https_probe_uses_the_canonical_read_only_network_owner(self, monkeypatch):
        import core.runtime.network_gateway as network_gateway

        calls: list[tuple[tuple, dict]] = []

        class _Gateway:
            def request(self, *args, **kwargs):
                calls.append((args, kwargs))
                return {"status_code": 503, "ok": False}

        monkeypatch.setattr(network_gateway, "get_network_gateway", lambda: _Gateway())

        assert preconditions._https_endpoint_reachable("https://example.com/probe")
        assert calls == [
            (
                ("HEAD", "https://example.com/probe"),
                {
                    "headers": {"User-Agent": "Aura-Connectivity-Probe/1"},
                    "timeout": preconditions._PROBE_TIMEOUT_SECONDS,
                    "source": "capability_preconditions.network_probe",
                    "read_only": True,
                    "suppress_degradation": True,
                },
            )
        ]

    def test_https_capable_network_is_online_without_public_dns_port_53(self, monkeypatch):
        seen_tcp_ports: list[int] = []
        monkeypatch.setattr(
            preconditions,
            "_https_endpoint_reachable",
            lambda url: url.endswith("generate_204"),
        )

        def tcp_probe(host: str, port: int) -> bool:
            seen_tcp_ports.append(port)
            return False

        monkeypatch.setattr(preconditions, "_tcp_https_endpoint_reachable", tcp_probe)

        state = preconditions._probe_network()

        assert state.satisfied
        assert not state.unknown
        assert 53 not in seen_tcp_ports
        assert all(port == 443 for _, port in preconditions._TCP_HTTPS_TARGETS)

    def test_tcp_443_is_an_independent_route_signal_when_https_providers_fail(self, monkeypatch):
        monkeypatch.setattr(preconditions, "_https_endpoint_reachable", lambda _url: False)
        monkeypatch.setattr(
            preconditions,
            "_tcp_https_endpoint_reachable",
            lambda host, _port: host == "1.1.1.1",
        )

        state = preconditions._probe_network()

        assert state.satisfied
        assert not state.unknown

    def test_offline_requires_every_expected_signal_to_fail(self, monkeypatch):
        monkeypatch.setattr(preconditions, "_https_endpoint_reachable", lambda _url: False)
        monkeypatch.setattr(
            preconditions, "_tcp_https_endpoint_reachable", lambda _host, _port: False
        )

        state = preconditions._probe_network()

        assert not state.satisfied
        assert not state.unknown

    def test_unexpected_probe_fault_is_unknown_not_offline(self, monkeypatch):
        def broken_https(_url: str) -> bool:
            raise RuntimeError("probe implementation failed")

        monkeypatch.setattr(preconditions, "_https_endpoint_reachable", broken_https)
        monkeypatch.setattr(
            preconditions, "_tcp_https_endpoint_reachable", lambda _host, _port: False
        )

        state = preconditions._probe_network()

        assert state.satisfied
        assert state.unknown


class TestPreconditionsAreDeclared:
    def test_a_capability_declares_what_the_world_must_provide(self):
        assert "network" in preconditions.declared_preconditions("web_search")
        assert "desktop_session" in preconditions.declared_preconditions("computer_use")
        assert preconditions.declared_preconditions("some_unknown_skill") == ()
