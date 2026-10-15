"""Tests for verifier-filtered self-consistency (inference-time amplification)."""
from __future__ import annotations

import pytest

from core.brain.reasoning_amplifier import (
    DeliberationEngine,
    amplify,
    default_extract_answer,
)


async def _mark_badmath_invalid(text: str):
    if "BADMATH" in text:
        return False, ["arithmetic: BADMATH"]
    return True, []


async def _all_valid(text: str):
    return True, []


def test_extract_answer():
    assert default_extract_answer("reasoning...\nAnswer: 42") == "42"
    assert default_extract_answer("Therefore: the sky is blue") == "the sky is blue"
    assert default_extract_answer("just one line") == "just one line"


@pytest.mark.asyncio
async def test_clean_majority_wins_with_high_confidence():
    cands = ["Answer: blue", "Answer: blue", "Answer: blue", "Answer: green"]
    r = await amplify(cands, verify=_all_valid)
    assert default_extract_answer(r.answer) == "blue"
    assert r.agreement == pytest.approx(0.75)
    assert r.confidence > 0.6


@pytest.mark.asyncio
async def test_verifier_filter_changes_the_winner():
    # Raw majority is "A" (3 vs 2), but all three A-paths contain a provable error.
    # Verifier-filtered self-consistency must pick the verifier-clean answer "B".
    cands = [
        "BADMATH reasoning... Answer: A",
        "BADMATH reasoning... Answer: A",
        "BADMATH reasoning... Answer: A",
        "clean reasoning... Answer: B",
        "clean reasoning... Answer: B",
    ]
    r = await amplify(cands, verify=_mark_badmath_invalid)
    assert default_extract_answer(r.answer) == "B"     # clean minority beat wrong majority
    assert r.valid_n == 2
    assert r.verified is True


@pytest.mark.asyncio
async def test_all_invalid_falls_back_to_voting_over_all():
    cands = ["BADMATH Answer: X", "BADMATH Answer: X", "BADMATH Answer: Y"]
    r = await amplify(cands, verify=_mark_badmath_invalid)
    assert default_extract_answer(r.answer) == "X"     # still votes, but not 'verified'
    assert r.valid_n == 0 and r.verified is False
    assert r.confidence < 0.9                          # confidence damped (no clean winner)


@pytest.mark.asyncio
async def test_empty_candidates():
    r = await amplify([], verify=_all_valid)
    assert r.answer == "" and r.n == 0 and r.confidence == 0.0


@pytest.mark.asyncio
async def test_real_verifier_discards_arithmetic_error_path():
    # End-to-end with the real deduction engine (SymbolicBridge).
    cands = [
        "We have 3 + 4 = 8, so Answer: eight",      # provable arithmetic error → discarded
        "3 + 4 = 7. Answer: seven",
        "It is 7. Answer: seven",
    ]
    r = await amplify(cands)                            # default real verifier
    assert "seven" in default_extract_answer(r.answer).lower()
    assert r.valid_n == 2


@pytest.mark.asyncio
async def test_deliberation_engine_samples_then_amplifies():
    seq = iter(["Answer: blue", "Answer: blue", "Answer: green"])

    async def _gen(question, temp):
        return next(seq)

    eng = DeliberationEngine(n_samples=3)
    r = await eng.deliberate("what color?", _gen, verify=_all_valid)
    assert default_extract_answer(r.answer) == "blue" and r.n == 3


# ── adaptive escalation + decompose-then-verify ───────────────────────────

@pytest.mark.asyncio
async def test_adaptive_stops_early_on_strong_clean_consensus():
    calls = {"n": 0}

    async def _gen(q, t):
        calls["n"] += 1
        return "Answer: blue"          # unanimous + clean → should stop at min_samples

    eng = DeliberationEngine()
    r = await eng.adaptive_deliberate("color?", _gen, min_samples=3, max_samples=9, verify=_all_valid)
    assert r.answer.endswith("blue") and r.n == 3 and calls["n"] == 3   # didn't escalate


@pytest.mark.asyncio
async def test_adaptive_escalates_when_uncertain():
    # split answers → never reaches target_agreement → escalates to max_samples
    seq = iter(["Answer: A", "Answer: B"] * 20)

    async def _gen(q, t):
        return next(seq)

    eng = DeliberationEngine()
    r = await eng.adaptive_deliberate("x?", _gen, min_samples=3, max_samples=7, batch=2,
                                      target_agreement=0.9, verify=_all_valid)
    assert r.n >= 7        # spent the budget because it stayed uncertain


@pytest.mark.asyncio
async def test_decompose_then_verify_solves_parts():
    async def _decompose(q):
        return ["sub1", "sub2"]

    async def _gen(q, t):
        if q == "sub1":
            return "Answer: 10"
        if q == "sub2":
            return "Answer: 5"
        return "Answer: 15"            # recombine prompt

    eng = DeliberationEngine()
    r = await eng.decompose_and_solve("10+5?", _gen, _decompose, verify=_all_valid)
    assert r.n == 2 and r.answer.endswith("15")


@pytest.mark.asyncio
async def test_decompose_falls_back_when_no_subquestions():
    async def _decompose(q):
        return []

    async def _gen(q, t):
        return "Answer: direct"

    eng = DeliberationEngine()
    r = await eng.decompose_and_solve("simple?", _gen, _decompose, verify=_all_valid)
    assert r.answer.endswith("direct")


def test_an_answer_line_is_found_though_a_remark_follows_it():
    from core.brain.reasoning_amplifier import default_extract_answer

    assert default_extract_answer("Let me work it out.\nAnswer: 42\nWait, let me double check: yes, that holds.") == "42"
    assert default_extract_answer("6 times 7.\nFinal answer: 42") == "42"
    assert default_extract_answer("no conclusion marker here\n41 + 1 = 42") == "41 + 1 = 42"
