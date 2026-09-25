"""The discovery step OntologyGenesis spent CP126 admitting it did not have.

CP126 c0a3c26e: "autonomous formation of cognitive laws" was a loop that
logged, slept sixty seconds, and reached a comment saying the real logic went
there — while reporting `active: true`.

The property that makes this a discovery step rather than a second version of
that defect is that it can come back empty, and does. The noise control below
is the load-bearing test in this file: twenty runs where the outcome is
independent of every feature must produce twenty refusals. A loop that always
finds something has not measured anything.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from core.brain import ontology_genesis as genesis
from core.brain.ontology_discovery import (
    CandidateLaw,
    Observation,
    OntologyDiscovery,
    Predicate,
    base_rate,
    candidate_predicates,
    score_split,
)

pytestmark = pytest.mark.unit


def _planted(seed: int, n: int = 400) -> list[Observation]:
    """Episodes where `psi >= 0.7 and queue >= 12` really does raise the rate."""
    rng = random.Random(seed)
    episodes = []
    for i in range(n):
        psi = rng.uniform(0.0, 1.0)
        queue = rng.randint(0, 20)
        real = psi > 0.7 and queue > 12
        episodes.append(
            Observation(
                features={
                    "psi": psi,
                    "queue": queue,
                    "subsystem": rng.choice(["mlx", "memory", "voice"]),
                },
                outcome=rng.random() < (0.85 if real else 0.10),
                at=float(i),
            )
        )
    return episodes


def _noise(seed: int, n: int = 400) -> list[Observation]:
    """Same features, outcome independent of all of them."""
    rng = random.Random(seed)
    return [
        Observation(
            features={
                "psi": rng.uniform(0.0, 1.0),
                "queue": rng.randint(0, 20),
                "subsystem": rng.choice(["mlx", "memory", "voice"]),
                "temp": rng.uniform(0.0, 100.0),
                "depth": rng.randint(0, 9),
            },
            outcome=rng.random() < 0.3,
            at=float(i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The control that decides whether any of this means anything
# ---------------------------------------------------------------------------


def test_pure_noise_yields_no_law_across_twenty_runs():
    engine = OntologyDiscovery(outcome_name="stall")
    found = [seed for seed in range(20) if engine.discover(_noise(1000 + seed)).found]
    assert found == [], f"induced a law from noise on seed(s) {found}"


def test_a_noise_run_says_why_it_found_nothing():
    outcome = OntologyDiscovery(outcome_name="stall").discover(_noise(7))
    assert not outcome.found
    assert outcome.refusal
    assert outcome.candidates_considered > 0, "it did not even look"


# ---------------------------------------------------------------------------
# ...and that it can still find a real one
# ---------------------------------------------------------------------------


def test_a_planted_regularity_is_recovered():
    outcome = OntologyDiscovery(outcome_name="stall").discover(_planted(7))
    assert outcome.found
    law = outcome.discovered
    features = {p.feature for p in law.law.predicates}
    assert features == {"psi", "queue"}
    assert law.evidence.heldout_lift > 1.25
    assert law.evidence.p_value <= 0.01
    assert law.evidence.transfer_lift > 1.0


def test_every_surviving_conjunct_is_load_bearing():
    """Pruning runs first, so the recorded ablation has no free-riders."""
    law = OntologyDiscovery(outcome_name="stall").discover(_planted(7)).discovered
    assert law is not None
    assert law.evidence.ablation
    assert all(delta > 0 for delta in law.evidence.ablation.values()), law.evidence.ablation


def test_the_law_carries_a_provenance_hash():
    law = OntologyDiscovery(outcome_name="stall").discover(_planted(7)).discovered
    assert law is not None
    assert law.provenance().startswith("sha256:")
    assert law.to_dict()["rule"] == law.law.describe()


# ---------------------------------------------------------------------------
# Each refusal is reachable
# ---------------------------------------------------------------------------


def test_too_few_episodes_refuses():
    outcome = OntologyDiscovery().discover(_planted(1, n=10))
    assert not outcome.found and "below" in outcome.refusal


def test_an_outcome_that_never_happens_refuses():
    episodes = [
        Observation({"psi": float(i % 7)}, outcome=False, at=float(i)) for i in range(200)
    ]
    outcome = OntologyDiscovery().discover(episodes)
    assert not outcome.found and "no episode had the outcome" in outcome.refusal


def test_an_outcome_that_always_happens_refuses():
    episodes = [
        Observation({"psi": float(i % 7)}, outcome=True, at=float(i)) for i in range(200)
    ]
    outcome = OntologyDiscovery().discover(episodes)
    assert not outcome.found and "nothing to explain" in outcome.refusal


def test_constant_features_yield_no_predicates():
    episodes = [
        Observation({"psi": 1.0, "mode": "a"}, outcome=bool(i % 2), at=float(i))
        for i in range(200)
    ]
    outcome = OntologyDiscovery().discover(episodes)
    assert not outcome.found
    assert "no feature varied" in outcome.refusal


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


def test_splits_are_by_time_not_at_random():
    """A random split leaks the same minute onto both sides."""
    episodes = _planted(3, n=200)
    train, heldout, transfer = OntologyDiscovery().split(episodes)
    assert max(o.at for o in train) <= min(o.at for o in heldout)
    assert max(o.at for o in heldout) <= min(o.at for o in transfer)


def test_partitioned_discovery_rejects_correlated_rows_and_shared_sources():
    rows = tuple(Observation({"signal": bool(index % 2)}, bool(index % 2),
                             source_id=f"source-{index}") for index in range(40))
    engine = OntologyDiscovery(outcome_name="transfer", min_support=4)
    with pytest.raises(ValueError, match="shares source"):
        engine.discover_partitioned(rows, rows, rows)
    with pytest.raises(ValueError, match="one independent row"):
        engine.discover_partitioned(rows, (*rows[:1], *rows[:1], *rows[1:]), rows)


def test_partitioned_discovery_freezes_fit_before_validation():
    groups = tuple(tuple(Observation(
        {"signal": index % 2 == 0, "noise": index % 3 == 0},
        index % 2 == 0, source_id=f"group-{group}-{index}", at=group * 100 + index
    ) for index in range(40)) for group in range(3))
    result = OntologyDiscovery(outcome_name="transfer", min_support=4).discover_partitioned(
        *groups)
    assert result.found
    assert result.discovered.evidence.validation_method == "exact_conditional_bonferroni"
    assert result.discovered.evidence.permutations == 0
    assert result.discovered.evidence.familywise_candidates >= 1
    assert result.discovered.evidence.transfer_p_value <= 0.01


def test_p_value_never_reaches_zero():
    """The observed arrangement is one of the arrangements under the null."""
    engine = OntologyDiscovery(permutations=99)
    law = CandidateLaw((Predicate("psi", ">=", 0.7),), "stall")
    p = engine.permutation_p_value(law, _planted(11))
    assert p >= 1 / 100


def test_a_predicate_on_a_missing_feature_does_not_hold():
    assert not Predicate("absent", ">=", 1.0).holds({"psi": 5.0})


def test_score_and_base_rate_agree_with_a_hand_count():
    episodes = [
        Observation({"x": 1.0}, outcome=True),
        Observation({"x": 1.0}, outcome=False),
        Observation({"x": 0.0}, outcome=True),
    ]
    law = CandidateLaw((Predicate("x", ">=", 1.0),), "y")
    score = score_split(law, episodes)
    assert (score.support, score.hits) == (2, 1)
    assert base_rate(episodes) == pytest.approx(2 / 3)


def test_candidate_predicates_skip_constant_and_single_valued_features():
    episodes = [
        Observation({"c": 1.0, "v": float(i), "s": "only"}, outcome=bool(i % 2))
        for i in range(20)
    ]
    features = {p.feature for p in candidate_predicates(episodes)}
    assert features == {"v"}


# ---------------------------------------------------------------------------
# The engine: real observations, and integration into a pool with readers
# ---------------------------------------------------------------------------


def test_degradation_records_become_episodes_without_leaking_the_answer():
    records = [
        {
            "subsystem": "mlx",
            "severity": "critical",
            "error_type": "TimeoutError",
            "action": "a",
            "at": 100.0,
        },
        {
            "subsystem": "mlx",
            "severity": "warning",
            "error_type": "ValueError",
            "action": "b",
            "at": 130.0,
        },
    ]
    episodes = genesis.degradation_observations(records)
    assert [o.outcome for o in episodes] == [True, False]
    # Severity is the outcome, so it must not also be a feature; a rule that
    # reads the answer off its own input validates perfectly and predicts
    # nothing.
    for episode in episodes:
        assert "severity" not in episode.features
    assert episodes[1].features["repeat_of_previous"] is True
    assert episodes[1].features["seconds_since_previous"] == pytest.approx(30.0)


def test_an_empty_ring_is_a_refusal_not_a_crash(monkeypatch):
    engine = genesis.OntologyGenesisEngine()
    outcome = asyncio.run(engine.run_discovery_cycle(observations=[]))
    assert not outcome.found
    assert engine.get_status()["last_refusal"]


def test_a_discovered_law_reaches_the_shared_heuristic_pool(monkeypatch):
    """The integration point has readers: curiosity_explorer, dreamer_v2, dream_skill.

    A writer with no reader is indistinguishable from no writer, which is the
    defect class this engine was an instance of.
    """
    ingested: list[tuple[str, str, str]] = []

    class _Pool:
        def ingest_external_heuristic(self, rule, domain="external", source="external"):
            ingested.append((rule, domain, source))
            return True

    import core.adaptation.heuristic_synthesizer as hs

    monkeypatch.setattr(hs, "get_heuristic_synthesizer", lambda: _Pool())

    engine = genesis.OntologyGenesisEngine()
    outcome = asyncio.run(engine.run_discovery_cycle(observations=_planted(7)))

    assert outcome.found
    assert len(ingested) == 1
    rule, domain, source = ingested[0]
    assert domain == "runtime_ontology" and source == "ontology_genesis"
    # The evidence travels with the rule; a heuristic whose support nobody can
    # see is a slogan.
    assert "held-out lift" in rule and "p=" in rule and "transfer lift" in rule

    status = engine.get_status()
    assert status["discoveries"] == 1
    assert status["integrated"] == 1
    assert status["cycles"] == 1
    assert status["last_law"]["evidence"]["p_value"] <= 0.01


def test_a_failing_pool_is_recorded_not_swallowed(monkeypatch):
    class _Pool:
        def ingest_external_heuristic(self, rule, domain="external", source="external"):
            raise RuntimeError("pool unavailable")

    import core.adaptation.heuristic_synthesizer as hs

    monkeypatch.setattr(hs, "get_heuristic_synthesizer", lambda: _Pool())

    engine = genesis.OntologyGenesisEngine()
    outcome = asyncio.run(engine.run_discovery_cycle(observations=_planted(7)))

    assert outcome.found
    # The law is still discovered and logged; only integration failed, and
    # the counter says so rather than the discovery count covering for it.
    assert engine.get_status()["discoveries"] == 1
    assert engine.get_status()["integrated"] == 0


def test_the_discovery_log_is_bounded():
    engine = genesis.OntologyGenesisEngine()
    engine._discovery_log = [{"n": i} for i in range(genesis.MAX_DISCOVERY_LOG + 20)]
    asyncio.run(engine.run_discovery_cycle(observations=[]))
    assert len(engine._discovery_log) <= genesis.MAX_DISCOVERY_LOG + 20
