"""Trying a different design where a mistake costs nothing.

Aura can change her own code. What she cannot do is try a different design and
find out whether it is better, because finding out means running it and the
only place it runs is where a mistake costs something.

A shadow receives every input the live design receives, produces its own
answer, and that answer goes nowhere. The disagreements are the finding.
"""

from __future__ import annotations

import pytest

from core.promotion.shadow_architecture import (
    MAX_DISAGREEMENT_RATE,
    MIN_DISAGREEMENTS,
    MIN_TRIALS,
    Shadow,
    Standing,
)

CASES = [(n, n * n) for n in range(30)]


def live(n):
    """Right except on multiples of seven."""
    return n * n if n % 7 else n * n + 1


def grade(answer, expected):
    return 1.0 if answer == expected else 0.0


def run(candidate, *, agree=None, grader=grade):
    shadow = Shadow("s", live=live, candidate=candidate, agree=agree, grade=grader)
    for value, expected in CASES:
        shadow.run(value, key=str(value), expected=expected)
    return shadow


# ── the shadow cannot reach the caller ───────────────────────────────────


def test_the_caller_gets_the_live_answer():
    shadow = Shadow("s", live=live, candidate=lambda n: -999)
    assert [shadow.run(n) for n in range(5)] == [live(n) for n in range(5)]


def test_there_is_no_argument_that_returns_the_shadow_answer():
    """A shadow reachable by a flag is a feature flag, and those get flipped."""
    import inspect

    signature = inspect.signature(Shadow.run)
    assert set(signature.parameters) == {"self", "value", "key", "expected"}


@pytest.mark.parametrize(
    "failure",
    [
        lambda n: 1 / 0,
        lambda n: n.nope,
        lambda n: (_ for _ in ()).throw(SystemError("boom")),
        lambda n: [][5],
    ],
)
def test_a_shadow_that_raises_cannot_take_the_live_answer_down(failure):
    """The first version listed five exception types and a ZeroDivisionError
    from the candidate reached the caller."""
    shadow = Shadow("s", live=live, candidate=failure)
    assert [shadow.run(n) for n in range(5)] == [live(n) for n in range(5)]
    assert shadow.errors == 5


def test_a_repeatedly_failing_shadow_does_not_open_an_incident_per_turn():
    import inspect

    source = inspect.getsource(Shadow.run)
    assert "self._errors == 1" in source


# ── the standings ────────────────────────────────────────────────────────


def test_a_strictly_better_design_is_ready():
    proof = run(lambda n: n * n).proof()
    assert proof.standing is Standing.READY
    assert proof.may_migrate is True
    assert proof.mean_gain > 0.0


def test_the_same_design_under_another_name_buys_nothing():
    proof = run(lambda n: n * n if n % 7 else n * n + 1).proof()
    assert proof.standing is Standing.IDENTICAL
    assert proof.disagreements < MIN_DISAGREEMENTS


def test_a_design_that_differs_everywhere_is_not_a_variant():
    proof = run(lambda n: -n).proof()
    assert proof.standing is Standing.DIVERGENT
    assert proof.disagreements / proof.trials > MAX_DISAGREEMENT_RATE


def test_a_design_that_is_no_better_where_it_differs_is_refused():
    proof = run(lambda n: n * n + 2 if n % 5 == 0 else live(n)).proof()
    assert proof.standing in {Standing.NO_BETTER, Standing.REGRESSES}
    assert proof.may_migrate is False


def test_too_few_trials_cannot_settle_it():
    shadow = Shadow("s", live=live, candidate=lambda n: n * n, grade=grade)
    for value, expected in CASES[: MIN_TRIALS - 1]:
        shadow.run(value, key=str(value), expected=expected)
    assert shadow.proof().standing is Standing.UNTRIED


# ── the condition that is usually skipped ────────────────────────────────


def test_a_good_average_does_not_excuse_breaking_something_that_worked():
    """Nine improvements and one destruction has a good mean."""

    def better_but_regresses(n):
        return -1 if n == 3 else n * n

    proof = run(better_but_regresses).proof()
    assert proof.mean_gain > 0.0, "it is better on average"
    assert proof.standing is Standing.REGRESSES
    assert proof.regressions == ("3",)
    assert proof.may_migrate is False


def test_a_regression_needs_the_live_design_to_have_been_right():
    """Both being wrong on a case is not a regression."""

    def wrong_where_live_was_wrong(n):
        return 999 if n == 14 else n * n

    proof = run(wrong_where_live_was_wrong).proof()
    assert "14" not in proof.regressions


def test_failing_the_behavioural_contracts_refuses_migration():
    assert run(lambda n: n * n).proof(contracts_held=False).standing is (
        Standing.REGRESSES
    )


def test_a_design_nobody_can_score_is_one_nobody_should_switch_to():
    """No grader means every disagreement is a tie, and READY is unreachable."""
    proof = run(lambda n: n * n, grader=None).proof()
    assert proof.standing is Standing.NO_BETTER
    assert proof.may_migrate is False


# ── the record ───────────────────────────────────────────────────────────


def test_the_proof_names_what_it_broke():
    proof = run(lambda n: -1 if n == 3 else n * n).proof()
    assert proof.regressions and "broke" in proof.because


def test_a_custom_agreement_function_is_used():
    """Two designs can differ in form and agree in what matters."""
    shadow = run(lambda n: float(live(n)), agree=lambda a, b: abs(a - b) < 1e-9)
    assert shadow.proof().standing is Standing.IDENTICAL


def test_the_snapshot_counts_errors_separately_from_disagreements():
    shadow = Shadow("s", live=live, candidate=lambda n: 1 / 0)
    for n in range(5):
        shadow.run(n)
    snapshot = shadow.snapshot()
    assert snapshot["errors"] == 5 and snapshot["trials"] == 0
