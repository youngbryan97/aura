"""What an answer has to satisfy before it is served, and what a retry costs.

CrewAI's closure asked for an output guardrail protocol with a retry budget.
Aura checks its answers in several places and none of them says in one
sentence what an answer must satisfy, so a reader cannot tell whether a
particular check runs and a failed check has no agreed way to ask again.
"""
from __future__ import annotations

import pytest

from core.runtime.what_an_answer_must_pass import (
    AGuardrail,
    AVerdict,
    TheGuardrails,
    a_guardrail,
)
from core.runtime.what_is_left_to_spend import a_budget_of


def _not_empty():
    return a_guardrail(
        "not empty",
        lambda answer: AVerdict(bool(answer), "the reply is empty", "say something"),
    )


def _no_placeholder():
    return a_guardrail(
        "no placeholder",
        lambda answer: AVerdict(
            "TODO" not in str(answer), "the reply says TODO", "finish it"
        ),
    )


# ------------------------------------------------------------- a guardrail


def test_a_guardrail_satisfies_the_protocol():
    assert isinstance(_not_empty(), AGuardrail)


def test_a_guardrail_without_a_name_is_refused():
    """A refusal from an unnamed one cannot be acted on."""
    with pytest.raises(ValueError, match="needs a name"):
        a_guardrail("  ", lambda answer: AVerdict(True))


def test_a_verdict_reads_as_a_boolean():
    assert bool(AVerdict(passed=True)) is True
    assert bool(AVerdict(passed=False, why="no")) is False


def test_a_refusal_says_what_to_do_instead():
    """"Refused" is not actionable; naming the fault is."""
    said = _not_empty().check("")
    assert not said
    assert said.why == "the reply is empty"
    assert said.instead == "say something"


# --------------------------------------------------------------- the chain


def test_an_answer_that_passes_everything_passes():
    rails = TheGuardrails().add(_not_empty()).add(_no_placeholder())
    assert rails.check("a real answer")


def test_it_stops_at_the_first_refusal():
    """A producer given five reasons fixes the first and gets the second."""
    rails = TheGuardrails().add(_not_empty()).add(_no_placeholder())
    said = rails.check("")
    assert said.why == "the reply is empty"
    assert len(rails.refusals) == 1
    assert rails.refusals[0]["rail"] == "not empty"


def test_a_broken_guardrail_is_whatever_that_rail_declared():
    """This used to carry on regardless, and that is what got flagged.

    "A broken rail is not a refusal" is right for a rail whose job is quality
    and wrong for one whose job is safety. A check that could not run has
    established nothing, so which of those it is has to be the rail's own
    declaration rather than this loop's opinion.
    """
    from core.runtime.what_an_answer_must_pass import WhenItCannotAnswer

    def angry(_answer):
        raise RuntimeError("this rail is broken")

    quality = (
        TheGuardrails()
        .add(
            a_guardrail(
                "broken", angry, when_it_cannot_answer=WhenItCannotAnswer.CARRY_ON
            )
        )
        .add(_not_empty())
    )
    assert quality.check("a real answer")

    safety = TheGuardrails().add(a_guardrail("broken", angry)).add(_not_empty())
    assert not safety.check("a real answer"), "an undeclared rail refuses"


def test_the_report_names_every_rail_and_every_refusal():
    rails = TheGuardrails().add(_not_empty()).add(_no_placeholder())
    rails.check("TODO: later")
    report = rails.report()
    assert report["rails"] == ["not empty", "no placeholder"]
    assert report["refusals"][0]["rail"] == "no placeholder"


# ------------------------------------------------------- retries and budget


def test_it_asks_again_until_the_answer_passes():
    rails = TheGuardrails().add(_not_empty()).add(_no_placeholder())
    answers = iter(["", "TODO: later", "the answer"])

    said_to = []

    def produce(instead):
        said_to.append(instead)
        return next(answers)

    answer, verdict = rails.until_it_passes(
        produce, budget=a_budget_of("attempts", 5)
    )
    assert answer == "the answer"
    assert verdict
    assert rails.attempts == 3


def test_the_producer_is_told_what_to_do_differently():
    rails = TheGuardrails().add(_not_empty())
    answers = iter(["", "something"])
    told = []

    def produce(instead):
        told.append(instead)
        return next(answers)

    rails.until_it_passes(produce, budget=a_budget_of("attempts", 3))
    assert told == ["", "say something"]


def test_retries_spend_from_the_callers_budget():
    """Three rails with three retries each is nine attempts, and the caller
    who allowed three never said so."""
    rails = TheGuardrails().add(_not_empty())
    budget = a_budget_of("attempts", 2)

    answer, verdict = rails.until_it_passes(lambda _instead: "", budget=budget)
    assert rails.attempts == 2
    assert budget.exhausted
    assert not verdict


def test_running_out_still_returns_the_last_answer_and_why_it_failed():
    """A caller out of budget still has something, and knows what is wrong."""
    rails = TheGuardrails().add(_no_placeholder())
    answer, verdict = rails.until_it_passes(
        lambda _instead: "TODO: later", budget=a_budget_of("attempts", 1)
    )
    assert answer == "TODO: later"
    assert not verdict
    assert verdict.why == "the reply says TODO"


def test_a_budget_with_nothing_in_it_produces_nothing():
    rails = TheGuardrails().add(_not_empty())
    answer, verdict = rails.until_it_passes(
        lambda _instead: "never asked", budget=a_budget_of("attempts", 0)
    )
    assert answer is None
    assert verdict.why == "nothing was produced"
    assert rails.attempts == 0


def test_a_child_budget_cannot_outspend_the_turn():
    turn = a_budget_of("this turn", 2)
    rails = TheGuardrails().add(_not_empty())
    rails.until_it_passes(
        lambda _instead: "", budget=turn.under("guarded answer", at_most=10)
    )
    assert rails.attempts == 2
    assert turn.exhausted


# --- A rail says what its own failure means -----------------------------------


def test_a_rail_that_does_not_say_refuses_when_it_cannot_run() -> None:
    """The safe default has to be the one you get by not thinking about it."""
    from core.runtime.what_an_answer_must_pass import (
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    def broken(_answer):
        raise RuntimeError("the checker is broken")

    rail = a_guardrail("safety", broken)
    assert rail.when_it_cannot_answer is WhenItCannotAnswer.REFUSE

    verdict = TheGuardrails().add(rail).check("an answer")
    assert not verdict.passed
    assert verdict.from_a_broken_rail
    assert "established nothing" in verdict.why


def test_a_quality_rail_can_declare_that_losing_it_costs_polish() -> None:
    from core.runtime.what_an_answer_must_pass import (
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    def broken(_answer):
        raise RuntimeError("no")

    rails = TheGuardrails().add(
        a_guardrail("tidy", broken, when_it_cannot_answer=WhenItCannotAnswer.CARRY_ON)
    )
    verdict = rails.check("an answer")
    assert verdict.passed
    assert rails.could_not_run and rails.could_not_run[0]["rail"] == "tidy"


def test_a_pass_that_some_rails_never_looked_at_says_so() -> None:
    """Silence from a rail that could not run must not read as agreement."""
    from core.runtime.what_an_answer_must_pass import (
        AVerdict,
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    def broken(_answer):
        raise RuntimeError("no")

    rails = (
        TheGuardrails()
        .add(a_guardrail("looked", lambda _a: AVerdict(passed=True)))
        .add(
            a_guardrail(
                "did not", broken, when_it_cannot_answer=WhenItCannotAnswer.ABSTAIN
            )
        )
    )
    verdict = rails.check("an answer")
    assert verdict.passed
    assert "did not" in verdict.why and "could not run" in verdict.why


def test_escalating_records_a_degradation_rather_than_a_log_line() -> None:
    from unittest.mock import patch

    from core.runtime.what_an_answer_must_pass import (
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    def broken(_answer):
        raise RuntimeError("no")

    rails = TheGuardrails().add(
        a_guardrail(
            "constitutional",
            broken,
            when_it_cannot_answer=WhenItCannotAnswer.ESCALATE,
        )
    )
    with patch("core.runtime.errors.record_degradation") as told:
        verdict = rails.check("an answer")
    assert not verdict.passed
    assert told.called
    assert "constitutional" in told.call_args.kwargs.get("action", "")


def test_the_report_names_every_rail_that_fails_open() -> None:
    from core.runtime.what_an_answer_must_pass import (
        AVerdict,
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    ok = lambda _a: AVerdict(passed=True)  # noqa: E731
    rails = (
        TheGuardrails()
        .add(a_guardrail("safety", ok))
        .add(a_guardrail("tidy", ok, when_it_cannot_answer=WhenItCannotAnswer.CARRY_ON))
    )
    seen = rails.report()
    assert seen["fail_open_rails"] == ["tidy"]
    assert seen["when_they_cannot_answer"]["safety"] == "refuse"


def test_all_four_failure_modes_are_distinct_and_only_two_let_it_through() -> None:
    from core.runtime.what_an_answer_must_pass import (
        TheGuardrails,
        WhenItCannotAnswer,
        a_guardrail,
    )

    def broken(_answer):
        raise RuntimeError("no")

    got = {}
    for mode in WhenItCannotAnswer:
        rails = TheGuardrails().add(
            a_guardrail("r", broken, when_it_cannot_answer=mode)
        )
        got[mode] = rails.check("x").passed
    assert len(WhenItCannotAnswer) == 4
    assert sum(1 for allowed in got.values() if allowed) == 2
    assert got[WhenItCannotAnswer.REFUSE] is False
    assert got[WhenItCannotAnswer.ESCALATE] is False
