"""The three experiments a boot must not run, run here instead.

A predicate that arrives at a result costs seconds; one that reads a result a
previous run arrived at costs none. The boot posture declines the first kind,
and `test_cognition_discipline` holds it to that from both directions: anything
over 1.5 seconds has to be declared an experiment, and anything declared an
experiment has to be owned by a test that actually runs it. Declaring without
owning turns a slow check into a claim nobody makes — from the boot outcome it
looks exactly like one that passed.

These three were named by the budget:

    endogenous_verdict_is_earned_on_known_corpora       6.5s
    test_both_her_algebras_compile_to_one_semantics     1.7s
    frozen_semantic_programs_transfer_to_fresh_cohort   1.6s

The first trains a readout over known corpora and scores it. The second
compiles every positional term and every value expression in both of her
languages and compares them term by term. The third seeds a fresh cohort and
runs the frozen transducer against it. Each arrives at its answer rather than
reading one, which is what makes it an experiment here.
"""

from __future__ import annotations

import pytest

import core.organism.model_validation as val_mod

DECLINED_AT_BOOT = (
    "endogenous_verdict_is_earned_on_known_corpora",
    "test_both_her_algebras_compile_to_one_semantics",
    "frozen_semantic_programs_transfer_to_fresh_cohort",
)


@pytest.fixture(scope="module")
def suite():
    val_mod.install_runtime_validation()
    return val_mod.get_suite()


@pytest.mark.parametrize("name", DECLINED_AT_BOOT)
def test_the_experiment_is_registered_and_declared(suite, name: str) -> None:
    tests = getattr(suite, "_tests", {})
    assert name in tests, f"{name} is owned here and registered nowhere"
    assert tests[name].expensive, (
        f"{name} is owned here as an experiment and the suite calls it an "
        "instrument, so a boot would run it"
    )


@pytest.mark.parametrize("name", DECLINED_AT_BOOT)
def test_the_experiment_runs_and_reports_something(suite, name: str) -> None:
    """Run it at full cost, which is the whole point of owning it.

    The assertion is that it reaches a verdict, not which verdict. A failing
    experiment is a finding about the system and belongs in the suite's own
    report; an experiment that cannot run at all is a finding about this file.
    """
    test = getattr(suite, "_tests", {})[name]
    models = list(getattr(suite, "_models", {}).values())
    assert models, "the suite registered no model to run against"
    outcome = test.run(models[0], include_expensive=True)

    assert outcome is not None, f"{name} produced no outcome"
    assert outcome.score is not None, f"{name} ran and scored nothing"
    assert outcome.score.outcome is not None
