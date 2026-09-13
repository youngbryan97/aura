"""Tests for the Frontier Discovery Engine and its live, causal, governed wiring.

The engine's whole claim to soundness is that it never asserts the unverified. These
tests pin that contract: exact proofs are PROVEN, false laws are REFUTED with a real
counterexample, unformalizable claims are CONJECTURE (and the rendered text never
asserts them), survivors are committed into the belief substrate, governance can veto,
and the safe parser cannot be tricked into running code.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from core.discovery.frontier_discovery_engine import (
    EpistemicStatus,
    FrontierDiscoveryEngine,
    _compile_integer_poly,
    _PolyParseError,
    get_frontier_discovery_engine,
)


@pytest.fixture
def engine(tmp_path):
    """A hermetic engine: temp KB, commit disabled (no belief-substrate side effects)."""
    return FrontierDiscoveryEngine(
        db_path=str(tmp_path / "fde.db"), commit_enabled=False, default_max_time_s=4.0
    )


# ── epistemic discipline: the assertion gate ────────────────────────────────
def test_true_modular_law_is_proven(engine):
    r = engine.assess_claim("Is n^5 - n divisible by 30 for all integers n?")
    v = r["verdict"]
    assert v["status"] == EpistemicStatus.PROVEN.value
    assert v["exhaustive"] is True
    assert "Proven" in r["rendered"]


def test_false_law_is_refuted_with_counterexample(engine):
    r = engine.assess_claim("is n^5 - n divisible by 7")
    v = r["verdict"]
    assert v["status"] == EpistemicStatus.REFUTED.value
    # n=2: 2^5 - 2 = 30, 30 % 7 == 2 != 0 — a real counterexample.
    assert v["counterexample"] == 2
    assert "Refuted" in r["rendered"]


def test_congruence_phrasing_is_parsed_and_proven(engine):
    r = engine.assess_claim("n^3 ≡ n (mod 6)")
    assert r["verdict"]["status"] == EpistemicStatus.PROVEN.value


def test_unformalizable_claim_is_conjecture_and_never_asserted(engine):
    r = engine.assess_claim("dark matter is made of axions")
    v = r["verdict"]
    assert v["status"] == EpistemicStatus.CONJECTURE.value
    # The cardinal rule: an unverified claim is NEVER phrased as fact.
    rendered = r["rendered"].lower()
    assert "not asserting" in rendered
    assert "conjecture" in rendered


def test_render_distinguishes_supported_from_proven():
    proven = FrontierDiscoveryEngine.render(
        _mk(EpistemicStatus.PROVEN, "n^3 ≡ n (mod 6)", exhaustive=True, trials=6)
    )
    supported = FrontierDiscoveryEngine.render(
        _mk(EpistemicStatus.SUPPORTED, "a_n = n*n", trials=10)
    )
    assert "Proven" in proven
    assert "NOT a proof" in supported and "supported" in supported.lower()


# ── safe parser cannot be tricked into running code ─────────────────────────
@pytest.mark.parametrize(
    "payload",
    ["__import__('os').system('echo x')", "n.__class__", "open('x')", "n + foo", "eval('1')"],
)
def test_safe_poly_parser_rejects_code(payload):
    with pytest.raises(_PolyParseError):
        _compile_integer_poly(payload)


def test_safe_poly_parser_accepts_real_polynomial():
    f = _compile_integer_poly("n**5 - n")
    assert f(2) == 30 and f(3) == 240


# ── the closed discovery loop ────────────────────────────────────────────────
def test_discovery_cycle_proves_real_modular_identities(engine):
    report = engine.run_discovery_cycle(max_time_s=4.0).to_dict()
    statements = {c["statement"] for c in report["proven"]}
    # These are genuine theorems the engine rediscovers by exhaustive residue checking.
    assert "For every integer n, n^3 ≡ n (mod 6)." in statements
    assert "For every integer n, n^5 ≡ n (mod 30)." in statements
    assert report["refuted"] > 0  # it actually kills false candidates, not just confirms
    assert all(c["exhaustive"] for c in report["proven"])


def test_discovery_cycle_is_bounded(engine):
    import time

    start = time.monotonic()
    engine.run_discovery_cycle(max_time_s=1.0)
    assert time.monotonic() - start < 8.0  # generous CI headroom; must not run away


def test_discovery_dedups_across_cycles(engine):
    first = engine.run_discovery_cycle(max_time_s=4.0)
    n_first = len(first.proven) + len(first.supported)
    assert n_first > 0
    # Everything found is now in the KB; a second cycle finds no *new* survivors.
    second = engine.run_discovery_cycle(max_time_s=4.0)
    assert len(second.proven) + len(second.supported) == 0
    assert {d["statement"] for d in engine.knowledge()} >= {c.statement for c in first.proven}


# ── integer-sequence law discovery (SUPPORTED only — finite data is no proof) ─
def test_sequence_law_finds_squares(engine):
    c = engine.discover_sequence_law([i * i for i in range(10)])
    assert c is not None
    assert c.status is EpistemicStatus.SUPPORTED  # never PROVEN from finite terms
    assert "n" in c.formal_form  # closed form is expressed in n, not SafeExpression's 'a'


def test_sequence_law_returns_none_when_no_closed_form(engine):
    # A deliberately irregular sequence the bounded GA should not perfectly fit.
    c = engine.discover_sequence_law([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
    assert c is None or c.status is EpistemicStatus.SUPPORTED


# ── causal commit into the mind (ScientificEngine / world_state) ─────────────
class _FakeSci:
    """Captures the hypothesis→experiment→observe calls the engine makes."""

    def __init__(self):
        self.formed = []
        self.experiments = []
        self.observed = []

    def form_hypothesis(self, claim, *, predicted_observable, expected, prior_confidence):
        self.formed.append((claim, predicted_observable, expected, prior_confidence))
        return f"hyp-{len(self.formed)}"

    def run_experiment(self, hyp_id, **kw):
        self.experiments.append(hyp_id)
        return f"rcpt-{hyp_id}"

    def observe(self, hyp_id, observed, note=""):
        self.observed.append((hyp_id, observed, note))


def test_proven_law_commits_into_belief_substrate(tmp_path):
    sci = _FakeSci()
    eng = FrontierDiscoveryEngine(
        db_path=str(tmp_path / "fde.db"), commit_enabled=True, scientific_engine=sci
    )
    r = eng.assess_claim("Is n^5 - n divisible by 30 for all integers n?")
    assert r["verdict"]["status"] == EpistemicStatus.PROVEN.value
    assert r["verdict"]["committed"] is True
    # The causal chain actually fired into the (fake) ScientificEngine.
    assert len(sci.formed) == 1
    assert sci.experiments == ["hyp-1"]
    assert sci.observed and sci.observed[0][1] == 1.0  # proof => observed pass-rate 1.0


def test_governance_veto_blocks_commit(tmp_path):
    sci = _FakeSci()
    eng = FrontierDiscoveryEngine(
        db_path=str(tmp_path / "fde.db"),
        commit_enabled=True,
        scientific_engine=sci,
        will_decide_fn=lambda payload: {"approved": False},
    )
    r = eng.assess_claim("Is n^5 - n divisible by 30 for all integers n?")
    assert r["verdict"]["status"] == EpistemicStatus.PROVEN.value
    assert r["verdict"]["committed"] is False
    assert sci.formed == []  # veto means nothing was written to belief


def test_refuted_law_never_commits(tmp_path):
    sci = _FakeSci()
    eng = FrontierDiscoveryEngine(
        db_path=str(tmp_path / "fde.db"), commit_enabled=True, scientific_engine=sci
    )
    eng.assess_claim("is n^2 - n divisible by 5")  # false (n=3: 6, 6%5=1)
    assert sci.formed == []


# ── singleton + container registration ───────────────────────────────────────
def test_singleton_and_container_registration():
    eng = get_frontier_discovery_engine()
    assert get_frontier_discovery_engine() is eng
    from core.container import ServiceContainer

    assert ServiceContainer.has(FrontierDiscoveryEngine.SERVICE_NAME)


def test_a_cleared_container_gets_the_engine_back():
    """Registration used to happen once, beside construction.

    So a container cleared afterwards lost the service name for the rest of
    the process while the engine went on existing, and every lookup failed on
    an object that was right there. This test failed inside a chunk and passed
    alone, which is what that shape looks like from outside.
    """
    from core.container import ServiceContainer

    engine = get_frontier_discovery_engine()
    ServiceContainer.clear()
    assert not ServiceContainer.has(FrontierDiscoveryEngine.SERVICE_NAME)

    again = get_frontier_discovery_engine()
    assert again is engine, "clearing the container must not build a second engine"
    assert ServiceContainer.has(FrontierDiscoveryEngine.SERVICE_NAME)


# ── response-lane wiring (live, causal) ──────────────────────────────────────
async def test_response_lane_routes_discovery_question():
    from core.brain.reasoning_strategies import ReasoningStrategies, StrategyResult

    async def _gen(*_a, **_k):
        return ""

    rs = ReasoningStrategies(_gen)
    assert rs._should_discover("is n^5 - n divisible by 30 for all n") is True
    assert rs._should_discover("what's the weather today") is False

    result = await rs._try_discovery("is n^5 - n divisible by 30 for all n")
    assert isinstance(result, StrategyResult)
    assert result.metadata.get("frontier_discovery") is True
    assert result.metadata.get("epistemic_status") == EpistemicStatus.PROVEN.value
    assert "Proven" in result.content


async def test_response_lane_ignores_non_discovery_question():
    from core.brain.reasoning_strategies import ReasoningStrategies

    async def _gen(*_a, **_k):
        return ""

    rs = ReasoningStrategies(_gen)
    assert await rs._try_discovery("tell me a joke") is None


# ── autonomous-loop wiring (live runtime driver) ─────────────────────────────
async def test_autonomous_loop_runs_discovery_tick(tmp_path, monkeypatch):
    import core.discovery.frontier_discovery_engine as fde_mod
    from core.autonomy.autonomous_initiative_loop import AutonomousInitiativeLoop

    # Methods exist and are coroutines; the idle lane has an executable runtime path.
    assert asyncio.iscoroutinefunction(AutonomousInitiativeLoop._discovery_loop)
    assert asyncio.iscoroutinefunction(AutonomousInitiativeLoop._run_discovery_cycle_once)

    fresh = FrontierDiscoveryEngine(db_path=str(tmp_path / "loop.db"), commit_enabled=False)
    monkeypatch.setattr(fde_mod, "get_frontier_discovery_engine", lambda **kw: fresh)

    loop = AutonomousInitiativeLoop(orchestrator=None)
    sentinel = object()
    loop._discovery_task = sentinel
    assert sentinel in loop._core_tasks()  # _core_tasks actually returns the discovery slot

    feed: list = []
    monkeypatch.setattr(loop, "_emit_feed", lambda *a, **k: feed.append((a, k)), raising=False)
    monkeypatch.setattr(loop, "_queue_visible_update", lambda *a, **k: True, raising=False)

    await loop._run_discovery_cycle_once()
    assert any(k.get("category") == "Discovery" for _a, k in feed), feed


# ── helpers ──────────────────────────────────────────────────────────────────
def _mk(status, statement, *, exhaustive=False, trials=0):
    from core.discovery.frontier_discovery_engine import Conjecture

    return Conjecture(
        statement=statement, domain="modular_identity", formal_form="", status=status,
        confidence=0.9, trials=trials, exhaustive=exhaustive,
        falsification_plan="checking all residues mod 6",
    )
