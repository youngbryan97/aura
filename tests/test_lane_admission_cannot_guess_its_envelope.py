"""An envelope check against a guessed envelope is not a check.

CP126 on core/brain/lane_admission.py. The controller decides whether a
model lane may commit memory on this host — and when it could not observe
the host, it assumed 64GB, sized a ~46GB budget from that, recorded
"observation unavailable", and then bound the decision as a normal fit.

On the developer machine 64GB is the right number, which is exactly why it
survived. Everywhere else it was a fail-open admission control.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from core.brain.lane_admission import (
    ActiveLane,
    LaneAdmissionController,
    QoSClass,
    classify_lane,
)


class _Memory:
    def __init__(self, total_bytes: int, available: bool) -> None:
        self.total_bytes = total_bytes
        self.available = available


class _Observer:
    """A resource observer whose memory reading we control."""

    def __init__(self, total_gb: float | None) -> None:
        self._total_gb = total_gb
        self.provenance = SimpleNamespace(
            source=SimpleNamespace(value="test"), scenario_id="test"
        )

    def memory(self):
        if self._total_gb is None:
            return _Memory(0, False)
        return _Memory(int(self._total_gb * 1024**3), True)


def _controller(total_gb: float | None) -> LaneAdmissionController:
    return LaneAdmissionController(observer=_Observer(total_gb))


# ------------------------------------------------- the fail-open envelope


def test_an_unobservable_host_never_admits_on_a_guessed_budget():
    """The 64GB assumption is gone."""
    decision = _controller(None).admit(
        model_path="/models/trainer-7b", request_gb=30.0, active=()
    )
    assert not decision.admitted
    assert decision.reason.startswith("memory_unobservable")
    assert decision.budget_gb == 0.0
    assert not decision.resource_observation_available


def test_the_cortex_may_still_come_up_but_is_marked_unverified():
    """An observation fault must not become a total outage.

    GUARANTEED is admitted — and the decision says plainly that no envelope
    was established, so nothing downstream can read it as a checked fit.
    """
    decision = _controller(None).admit(
        model_path="/models/resident-renamed",
        request_gb=20.0,
        active=(),
        role="cortex",
    )
    assert decision.admitted
    assert decision.reason == "admitted_without_envelope:memory_unobservable"
    assert not decision.resource_observation_available


def test_an_operator_supplied_absolute_budget_is_honoured(monkeypatch):
    """A declared quantity is not a guess."""
    monkeypatch.setenv("AURA_LANE_BUDGET_GB", "40")
    decision = _controller(None).admit(
        model_path="/models/trainer-7b", request_gb=10.0, active=()
    )
    assert decision.admitted
    assert decision.budget_gb == pytest.approx(40.0)


def test_a_small_host_refuses_what_the_old_default_would_have_admitted():
    """16GB host, 30GB request: the guess used to allow this."""
    decision = _controller(16.0).admit(
        model_path="/models/trainer-7b", request_gb=30.0, active=()
    )
    assert not decision.admitted
    assert "lane_budget_exceeded" in decision.reason


def test_a_real_host_still_admits_a_real_lane():
    """The control: admission must remain reachable."""
    decision = _controller(64.0).admit(
        model_path="/models/qwen-32b-cortex", request_gb=20.0, active=()
    )
    assert decision.admitted
    assert decision.reason == "fits"
    assert decision.resource_observation_available


# ------------------------------------------------------------ numeric safety


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, "twenty"])
def test_a_malformed_request_is_refused_as_malformed(bad):
    """NaN slid past the fit check into the eviction path."""
    decision = _controller(64.0).admit(
        model_path="/models/x", request_gb=bad, active=()
    )
    assert not decision.admitted
    assert decision.reason.startswith("malformed_request_gb")
    assert not decision.evict_first, (
        "a request that means nothing must not produce an eviction advisory"
    )


def test_an_unreadable_active_footprint_refuses_rather_than_undercounting():
    """`> 0.0` is False for NaN, so the lane was silently dropped."""
    decision = _controller(64.0).admit(
        model_path="/models/qwen-32b-cortex",
        request_gb=20.0,
        active=[
            ActiveLane(
                lane="trainer",
                qos=QoSClass.BEST_EFFORT,
                footprint_gb=math.nan,
                model_path="/models/t",
            )
        ],
    )
    assert not decision.admitted
    assert decision.reason.startswith("unreadable_active_footprint")


# ---------------------------------------------------------- the QoS floor


def _cortex_lane() -> ActiveLane:
    return ActiveLane(
        lane="cortex",
        qos=QoSClass.GUARANTEED,
        footprint_gb=20.0,
        model_path="/models/qwen-32b-cortex",
    )


def test_disruptive_eviction_is_an_authority_assertion_verified_upstream():
    """CP126 8de2c849, resolved where the state actually is.

    The flag lets a caller name a GUARANTEED lane — an explicit operator
    handoff has to be able to replace the resident cortex, and this module
    is arithmetic over a snapshot the caller supplies, so it cannot tell an
    authorised handoff from an untrusted trainer.

    The check lives in model_lane_control, against committed state. This
    pins that it is still there, so the authority cannot quietly become
    unchecked.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "runtime" / "model_lane_control.py"
    ).read_text("utf-8")
    ast.parse(source)  # it must at least still be the module we think it is
    for guard in (
        "required_eviction_not_preemptible",
        "required_eviction_already_fenced",
    ):
        assert guard in source, (
            f"model_lane_control no longer refuses on {guard}; lane_admission "
            "relies on it to verify a disruptive handoff's authority"
        )


def test_disruptive_eviction_still_relaxes_the_user_facing_shield():
    """The control: the flag must retain its legitimate effect."""
    warm = ActiveLane(
        lane="trainer",
        qos=QoSClass.BEST_EFFORT,
        footprint_gb=25.0,
        model_path="/models/trainer-7b",
        last_user_facing_age_s=1.0,
    )
    shielded = _controller(32.0).admit(
        model_path="/models/brainstem-14b", request_gb=20.0, active=[warm]
    )
    disruptive = _controller(32.0).admit(
        model_path="/models/brainstem-14b",
        request_gb=20.0,
        active=[warm],
        allow_disruptive_eviction=True,
    )
    assert "/models/trainer-7b" not in shielded.evict_first
    assert "/models/trainer-7b" in disruptive.evict_first


# ------------------------------------------------------ declared vs guessed


def test_a_renamed_model_can_declare_its_role_instead_of_being_guessed():
    lane, qos = classify_lane("/models/aura-primary-v4", purpose="serve")
    assert (lane, qos) == ("auxiliary", QoSClass.BEST_EFFORT)

    declared_lane, declared_qos = classify_lane(
        "/models/aura-primary-v4", purpose="serve", role="cortex"
    )
    assert (declared_lane, declared_qos) == ("cortex", QoSClass.GUARANTEED)


def test_a_declared_class_reaches_the_decision():
    decision = _controller(64.0).admit(
        model_path="/models/aura-primary-v4",
        request_gb=20.0,
        active=(),
        role="cortex",
    )
    assert decision.lane == "cortex"
    assert decision.qos is QoSClass.GUARANTEED


def test_unregistered_size_tokens_do_not_receive_guaranteed_qos():
    decision = _controller(64.0).admit(
        model_path="/models/qwen-32b-cortex", request_gb=20.0, active=()
    )
    assert decision.lane == "auxiliary"
    assert decision.qos is QoSClass.BEST_EFFORT


# ------------------------------------------------------------ the surface


def test_readiness_reflects_whether_an_envelope_can_be_bound():
    """It returned True unconditionally, so nothing could ever have said otherwise."""
    assert _controller(64.0).is_ready()
    assert not _controller(None).is_ready(), (
        "a controller refusing every non-GUARANTEED lane reported ready"
    )


def test_the_snapshot_says_whether_it_can_see_the_host():
    blind = _controller(None).snapshot()
    assert blind["resource_observation_available"] is False
    assert blind["envelope_established"] is False
    assert blind["ready"] is False

    sighted = _controller(64.0).snapshot()
    assert sighted["envelope_established"] is True
    assert sighted["ready"] is True


def test_the_snapshot_counts_decisions_made_without_an_envelope():
    controller = _controller(None)
    for _ in range(3):
        controller.admit(model_path="/models/qwen-32b-cortex", request_gb=20.0, active=())
    assert controller.snapshot()["decisions_without_envelope"] == 3


def test_alive_and_ready_are_not_the_same_question():
    controller = _controller(None)
    assert controller.is_alive()
    assert not controller.is_ready()
