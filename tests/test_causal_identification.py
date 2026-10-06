"""Independent experiments establish identity; stale or ambiguous evidence does not."""
import pytest

from core.agency.causal_identification import CausalIdentification, CausalWitness, within_observed_reach

pytestmark = pytest.mark.unit


def _witness(identity, *, epoch=0, trials=(("a", 3), ("b", 3)), statistic=100.0, effect=80.0):
    return CausalWitness(identity, epoch, trials, statistic, effect)


def test_repeated_independent_probes_identify_a_responder_and_retain_the_evidence():
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    result = ledger.choose([_witness("slider")], visible={"slider"}, established=None, excluded=set())
    assert result == "slider"
    assert ledger.receipts[-1]["independent_trials"] == {"a": 3, "b": 3}
    assert ledger.receipts[-1]["reason"] == "independent repeated trials"


@pytest.mark.parametrize("bad", [
    _witness("slider", trials=(("a", 1), ("b", 20))),
    _witness("slider", trials=(("a", 3), ("a", 3))),
    _witness("slider", statistic=float("nan")),
    _witness("slider", effect=0),
    _witness("slider", epoch=1),
])
def test_insufficient_or_invalid_evidence_remains_unknown(bad):
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    assert ledger.choose([bad], visible={"slider"}, established=None, excluded=set()) is None


def test_an_old_experiment_cannot_identify_after_a_reset():
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    old = _witness("slider")
    ledger.reset("observed contradiction")
    assert ledger.choose([old], visible={"slider"}, established=None, excluded=set()) is None
    assert ledger.choose([_witness("slider", epoch=1)], visible={"slider"}, established=None, excluded=set()) == "slider"


def test_equal_responder_evidence_is_ambiguous():
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    assert ledger.choose([_witness("a"), _witness("b")], visible={"a", "b"}, established=None, excluded=set()) is None


def test_a_competing_historical_score_cannot_revoke_an_established_visible_identity():
    ledger = CausalIdentification(statistic_over=20, effect_over=20)
    assert ledger.choose([_witness("rival", statistic=1e6)], visible={"mine", "rival"},
                         established="mine", excluded=set()) == "mine"
    assert ledger.choose([_witness("rival")], visible={"rival"}, established="mine", excluded={"rival"}) is None


def test_visible_continuity_uses_reachable_motion_and_rejects_teleportation():
    observation = {"previous": (30.0, 50.0), "extent": (4.0, 20.0), "velocity": (0.0, 120.0), "gap": 0.03}
    assert within_observed_reach(position=(32.0, 54.0), **observation)
    assert not within_observed_reach(position=(200.0, 54.0), **observation)
    assert not within_observed_reach(position=(32.0, 54.0), **(observation | {"gap": 3.0}))


def test_attribution_receipts_use_visible_geometry_and_are_bounded():
    from types import SimpleNamespace

    from core.agency.playing_as_it_happens import _Run, _record_control_attribution
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    run = _Run(["a"], 0.0)
    mine = SimpleNamespace(number=1, x=20, y=50)
    moves = SimpleNamespace(shape=(100, 200), things={1: mine})
    hers.number, hers.kind = 1, 0
    for _ in range(4):
        hers._hers.add("a", 80, 0)
    _record_control_attribution(run, moves, hers, 1.0)
    assert run.control_attribution[-1] == {"seconds": 1.0, "thing": 1, "source": "keys", "keys": ["a"], "position": [0.1, 0.5]}
    _record_control_attribution(run, moves, hers, 1.1)
    assert run.attribution_changes == 1
    for n in range(150):
        hers.number = 1 if n % 2 else None
        _record_control_attribution(run, moves, hers, 2.0 + n)
    assert len(run.control_attribution) == 128


def test_identity_claim_runs_the_registered_measurement():
    from core.agency.causal_identification import _causal_identity_invariant
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _causal_identity_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "identity_requires_current_distinguishing_evidence")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
