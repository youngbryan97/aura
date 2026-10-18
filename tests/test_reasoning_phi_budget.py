"""Tests for Φ-gated test-time compute allocation in the reasoning amplifier.

The contract under test: cognitive *demand* (free energy, uncertainty, stuck
valence, frustration) buys reasoning depth; Φ-*capacity* (how integrated the mind
is right now) gates how much of that depth it is allowed to spend.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.reasoning_amplifier_v2 import (
    PHI_DELIBERATE,
    PHI_DORMANT,
    ReasoningAmplifierV2 as R,
    ReasoningMode as M,
    _MAX_SAMPLES,
    _MODE_BUDGET,
)


# ── pure budget resolver ────────────────────────────────────────────────────
def test_neutral_signals_keep_base_mode():
    mode, samples, tmult = R._resolve_compute_budget(M.NORMAL, {})
    assert mode is M.NORMAL
    assert samples == _MODE_BUDGET[M.NORMAL]
    assert tmult == 1.0


def test_high_demand_high_phi_deepens_and_spends_more():
    mode, samples, tmult = R._resolve_compute_budget(
        M.NORMAL, {"phi": 0.8, "free_energy": 0.9, "uncertainty": 0.8}
    )
    assert mode in (M.DEEP, M.EXTREME)          # demand bought depth
    assert samples > _MODE_BUDGET[M.NORMAL]       # more candidates
    assert tmult > 1.0                            # more wall-clock


def test_low_phi_spends_no_extra_depth_and_keeps_the_tasks_own():
    # Low capacity damps what demand can buy; it does not take away the
    # depth the task's type starts at.
    mode, samples, _ = R._resolve_compute_budget(
        M.NORMAL, {"phi": 0.02, "free_energy": 1.0, "uncertainty": 1.0}
    )
    assert mode in (M.NORMAL, M.DEEP)
    assert samples >= _MODE_BUDGET[M.NORMAL]


def test_a_deep_task_stays_deep_at_any_phi():
    for phi in (0.0, 0.02, 0.10, 0.5):
        mode, samples, _ = R._resolve_compute_budget(
            M.DEEP, {"phi": phi, "free_energy": 1.0, "uncertainty": 1.0}
        )
        assert mode in (M.DEEP, M.EXTREME)
        assert samples >= _MODE_BUDGET[M.DEEP]


def test_proof_mode_never_downgraded_by_low_phi():
    mode, _, _ = R._resolve_compute_budget(M.PROOF, {"phi": 0.0})
    assert mode is M.PROOF


def test_stuck_mind_deepens_even_without_free_energy():
    # Legacy hard trigger preserved: negative valence / high frustration deepens once.
    mode, _, _ = R._resolve_compute_budget(M.NORMAL, {"phi": 0.4, "valence": -0.6, "frustration": 0.7})
    assert mode is M.DEEP


def test_sample_budget_never_exceeds_max():
    mode, samples, _ = R._resolve_compute_budget(
        M.EXTREME, {"phi": 1.0, "free_energy": 1.0, "uncertainty": 1.0, "frustration": 1.0}
    )
    assert samples <= _MAX_SAMPLES


# ── component functions ─────────────────────────────────────────────────────
def test_phi_capacity_is_monotonic():
    lo = R._phi_capacity(PHI_DORMANT)
    mid = R._phi_capacity((PHI_DORMANT + PHI_DELIBERATE) / 2)
    hi = R._phi_capacity(PHI_DELIBERATE)
    assert lo < mid < hi
    assert hi == 1.0


def test_cognitive_demand_responds_to_each_signal():
    assert R._cognitive_demand({}) == 0.0
    assert R._cognitive_demand({"free_energy": 1.0}) > 0.0
    assert R._cognitive_demand({"uncertainty": 1.0}) > 0.0
    assert R._cognitive_demand({"valence": -1.0}) > 0.0
    assert R._cognitive_demand({"frustration": 1.0}) > 0.0
    # clamped to [0,1] even when every signal is maxed
    assert R._cognitive_demand(
        {"free_energy": 1.0, "uncertainty": 1.0, "valence": -1.0, "frustration": 1.0}
    ) == 1.0


# ── amplify() cache-hit short circuit ───────────────────────────────────────
def test_amplify_cache_hit_bypasses_generation(tmp_path, monkeypatch):
    from core.brain import reasoning_solved_cache as rsc
    from core.brain.reasoning_amplifier_v2 import AmplificationRequest

    # Temp-backed cache, pre-populated with a verifier-clean math answer.
    cache = rsc.ReasoningSolvedCache(tmp_path / "c.json")
    cache.put("what is 6 times 7", "math", answer="42", confidence=0.97, mode="deep", verified=True)
    monkeypatch.setattr(rsc, "get_reasoning_solved_cache", lambda: cache)

    calls = {"n": 0}

    async def _generate(prompt, temperature):
        calls["n"] += 1
        return "should never be called"

    amp = R(_generate)
    req = AmplificationRequest(
        objective="what is 6 times 7",
        task_type="math",
        context={"skip_evidence": True},
    )
    result = asyncio.run(amp.amplify(req))
    assert result.answer == "42"
    assert result.verified is True
    assert result.receipt.strategy_used == "solved_cache"
    assert "solved_cache_hit" in result.receipt.fallbacks_used
    assert calls["n"] == 0  # generation was fully bypassed


def _run_amplify_with_verifier(tmp_path, monkeypatch, *, ok: bool, checked: bool):
    """Run amplify() with an injected verifier verdict; return (result, temp cache)."""
    from core.brain import reasoning_amplifier_v2 as rav
    from core.brain import reasoning_solved_cache as rsc
    from core.brain.verifiers.base import VerificationResult

    cache = rsc.ReasoningSolvedCache(tmp_path / "c.json")
    monkeypatch.setattr(rsc, "get_reasoning_solved_cache", lambda: cache)

    class _FakeVerifier:
        async def verify(self, candidate, *, task_type=None, context=None):
            return VerificationResult(domain="math", ok=ok, checked=checked, score=0.95, engine="fake")

    async def _gen(prompt, temperature):
        return "the answer is 4"

    amp = rav.ReasoningAmplifierV2(_gen, verifier=_FakeVerifier())
    req = rav.AmplificationRequest(
        objective="what is two plus two", task_type="math",
        mode=rav.ReasoningMode.FAST, context={"skip_evidence": True},
    )
    return asyncio.run(amp.amplify(req)), cache


def test_vacuous_pass_is_not_cached(tmp_path, monkeypatch):
    # ok=True but checked=False (verifier had nothing concrete to evaluate) MUST NOT
    # be cached — this is the poisoning bug the hard bench caught.
    _result, cache = _run_amplify_with_verifier(tmp_path, monkeypatch, ok=True, checked=False)
    assert cache.get("what is two plus two", "math") is None


def test_genuinely_checked_pass_is_cached(tmp_path, monkeypatch):
    # ok=True AND checked=True (a real evaluation happened) IS cached.
    _result, cache = _run_amplify_with_verifier(tmp_path, monkeypatch, ok=True, checked=True)
    hit = cache.get("what is two plus two", "math")
    assert hit is not None and "4" in hit.answer
