"""Tests for the hardened math verifier — derive the exact answer from the question."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.verifiers.math_engine import MathTruthEngine, _derive_exact_answer


def _verify(candidate: str, objective: str):
    return asyncio.run(MathTruthEngine().verify(candidate, context={"objective": objective}))


# ── derivation ──────────────────────────────────────────────────────────────
def test_derive_mod():
    assert _derive_exact_answer("What is 123456 mod 7?") == ("123456 mod 7", "4")


def test_derive_power_words_and_caret():
    assert _derive_exact_answer("17 to the power of 4")[1] == "83521"
    assert _derive_exact_answer("2^10")[1] == "1024"


def test_derive_gcd():
    assert _derive_exact_answer("GCD of 1071 and 462")[1] == "21"


def test_derive_times():
    assert _derive_exact_answer("what is 137 times 248")[1] == str(137 * 248)


def test_derive_caps_huge_power():
    # exponent beyond the bound is not derived (keeps it exact + cheap)
    assert _derive_exact_answer("2 to the power of 100000") is None


def test_derive_none_for_non_arithmetic():
    assert _derive_exact_answer("explain why the sky is blue") is None


# ── verification soundness ──────────────────────────────────────────────────
def test_wrong_mod_is_hard_fail():
    r = _verify("123456 mod 7 is 5", "What is 123456 mod 7?")
    assert r.ok is False and r.checked is True


def test_right_mod_passes():
    r = _verify("The remainder is 4.", "What is 123456 mod 7?")
    assert r.ok is True and r.checked is True


def test_wrong_power_is_hard_fail():
    r = _verify("17^4 = 81000", "What is 17 to the power of 4?")
    assert r.ok is False and r.checked is True


def test_vacuous_prose_rejected_for_derivable_question():
    # The previous poison: prose with no answer used to pass. Now it fails the derived check.
    r = _verify("I am not sure; here is a guess.", "What is 123456 mod 7?")
    assert r.ok is False and r.checked is True


def test_non_derivable_prose_stays_vacuous():
    # No derivable expression and no '=' claim -> nothing to check (unchanged behavior).
    r = _verify("It depends on the context.", "Tell me about prime numbers.")
    assert r.checked is False


def test_correct_answer_among_reasoning_passes():
    r = _verify("First 17^2 = 289, then 289^2 = 83521. Answer: 83521.", "What is 17 to the power of 4?")
    assert r.ok is True and r.checked is True


def test_correct_intermediate_cannot_hide_wrong_final_answer():
    r = _verify(
        "I computed 17^4 as 83521, but my final answer is 81000.",
        "What is 17 to the power of 4?",
    )
    assert r.ok is False and r.checked is True


def test_last_answer_envelope_is_authoritative():
    r = _verify(
        "<answer>83521</answer>\nCorrection: <answer>81000</answer>",
        "What is 17 to the power of 4?",
    )
    assert r.ok is False and r.checked is True


# A draft cut by its token cap before any answer, as an AIME draft was in the F04 typed-lane proof on
# 12 November 2026: working in LaTeX, no final answer, a question no exact value is derived from.
CUT_DRAFT = (
    "Let the box have edges $2x$, $2y$, $2z$, so $4x^2 + 4y^2 = 41$ and $V_{box} = (2x)(2y)(2z) = 8xyz$.\n"
    "The tetrahedron is formed by cutting off 4 corner pyramids from the box. Each corner pyramid has"
)
TETRAHEDRON = "Let ABCD be a tetrahedron such that AB=CD=sqrt(41), AC=BD=sqrt(80), BC=AD=sqrt(89). Find m+n+p."


def test_working_with_no_answer_is_not_a_checked_answer():
    r = _verify(CUT_DRAFT, TETRAHEDRON)
    assert r.ok is True and r.checked is False and r.verdict == "UNCHECKED"


def test_a_wrong_step_still_refutes_without_an_answer_to_compare():
    r = _verify("First 12 * 12 = 154, and then the area follows.", TETRAHEDRON)
    assert r.ok is False and r.checked is True


def test_the_amplifier_searches_past_a_draft_that_reached_no_answer():
    """The incumbent is kept without a search only when a check reached its answer."""
    from core.brain.reasoning_amplifier_v2 import amplify_turn

    asked: list[float] = []

    async def generate(prompt: str, temperature: float) -> str:
        asked.append(temperature)
        return CUT_DRAFT

    result = asyncio.run(amplify_turn(
        TETRAHEDRON, generate, task_type="math", time_budget_s=30.0, sample_budget=2,
        extra_context={"seed_candidates": [CUT_DRAFT], "sealed_evaluation": True, "read_only_evaluation": True,
                       "skip_cache": True}))
    receipt = result.receipt.to_dict()
    assert "incumbent_verified" not in receipt["fallbacks_used"]
    assert receipt["promotion_authority"] != "preserve_incumbent"
    assert asked, "no candidate was asked for"
