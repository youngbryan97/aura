"""Five things her per-turn code reached for that were never built or never fed.

The governor's rate paced nothing, morphic insights went to a queue nobody
built, theory of mind's watching had no caller, the learning engine's only way
in was a hook nothing triggers, and the opacity monitor was never made. Each is
now built where its reader looks, and each reading moves something.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import core.runtime.background_policy as policy
from core.consciousness import substrate_governor, theory_of_mind
from core.consciousness.structural_opacity import get_structural_opacity_monitor

pytestmark = pytest.mark.unit


@pytest.fixture
def quiet_host(monkeypatch):
    """A host with nothing else capping the loops, so the volition cap is the one read."""
    monkeypatch.setattr(policy, "_VOLITION_RATE", 1.0)
    monkeypatch.setattr(policy, "foreground_only_runtime", lambda: False)
    monkeypatch.setattr(policy, "background_cognition_disabled_reason", lambda **_: "")
    monkeypatch.setattr(policy, "_foreground_activity_reason", lambda: "")
    monkeypatch.setattr(policy, "_read_compute_pressure_reason", lambda: "")
    monkeypatch.setattr(
        policy,
        "_read_memory_pressure_snapshot",
        lambda: SimpleNamespace(pressure_pct=10.0, reason="", refuse_heavy_local_generation=False),
    )
    monkeypatch.setattr(policy, "get_unified_failure_state", lambda: {"pressure": 0.0})
    monkeypatch.setattr("core.runtime.proof_policy.proof_run_active", lambda *a, **k: False)
    monkeypatch.delenv("AURA_CONSTITUTIVE_MAX_HZ", raising=False)
    monkeypatch.setattr(substrate_governor, "_governor", None)


def test_her_volition_sets_the_rate_of_her_background_loops(quiet_host) -> None:
    governor = substrate_governor.get_substrate_governor()
    governor.apply_volition_profile(3)
    assert policy.constitutive_compute_budget("liquid_substrate", 20.0).effective_hz == pytest.approx(20.0)
    governor.apply_volition_profile(1)
    budget = policy.constitutive_compute_budget("liquid_substrate", 20.0)
    assert budget.effective_hz == pytest.approx(10.0)
    assert "volition" in budget.reason


def test_the_kernel_and_the_supervisor_reach_the_same_governor(quiet_host) -> None:
    from core.actuators.process_supervisor import _the_governor

    assert _the_governor() is substrate_governor.get_substrate_governor()


def test_a_morphic_insight_becomes_one_of_her_impulses(monkeypatch) -> None:
    import core.initiative_synthesis as synthesis
    from core.kernel.upgrades_10x import hand_insight_to_her_initiative

    monkeypatch.setattr(synthesis, "_synth_instance", None)
    assert hand_insight_to_her_initiative(SimpleNamespace(content="Tides follow the moon."), curiosity=0.95)
    (impulse,) = synthesis.get_initiative_synthesizer()._impulse_queue
    assert (impulse.source, impulse.drive, impulse.urgency) == ("morphic_clone", "curiosity", pytest.approx(0.95))
    assert not hand_insight_to_her_initiative("", curiosity=0.95)


def test_watching_a_person_is_what_she_then_knows_about_them() -> None:
    who = "watch-test-person"
    for _ in range(3):
        theory_of_mind.watch_the_turn(who, their_act="question", her_act="statement", topic="tides")
    known = theory_of_mind.what_she_knows_about(who, facing="statement")
    assert "tides" in known["stuck_on"]
    assert known["will_probably"] == "question"


def test_a_profile_she_may_not_recall_is_empty(monkeypatch) -> None:
    engine = theory_of_mind.TheoryOfMindEngine.__new__(theory_of_mind.TheoryOfMindEngine)
    monkeypatch.setattr(engine, "_load_user", lambda *a, **k: None, raising=False)
    assert engine.get_profile("someone") == {}


def test_the_opacity_monitor_the_audit_reads_measures_the_substrate() -> None:
    monitor = get_structural_opacity_monitor()
    assert get_structural_opacity_monitor() is monitor
    rng = np.random.default_rng(3)
    signature = monitor.measure(rng.uniform(-1, 1, 64), rng.normal(0, 0.2, (64, 64)))
    assert 0.0 <= signature.opacity_index <= 1.0
