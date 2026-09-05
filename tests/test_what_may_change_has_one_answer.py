"""One answer to "may this change?", where there had been several.

Aura's live architecture is not a single homogeneous bag of values. There are
learned preferences, stable identity commitments, constitutional constraints,
governance policies, Will-mediated action and an amendment path, and those
distinctions are real. The problem is that there are several theories of them
at once:

    core_values.py      a frozen ClassVar tuple                immutable
    value_model.py      a constitution learning cannot override immutable
    prime_directives.py amendable only by procedure             committed
    values_engine.py    a weight with a flexibility field       moved by mood

and `apply_emotional_context("creative")` set `Integrity` to -0.1. Integrity
is honesty. Honesty is in the frozen tuple and in the constitution. So one
concept was immutable in two subsystems and a number a mood moved in a third.

When subsystem A believes a thing cannot change and subsystem B treats it as
a learned preference, the intersection is not a software-design mess. It is a
governance ambiguity, and the duplicated concept is exactly "what is allowed
to change".
"""

from __future__ import annotations

import pytest

from core.governance.value_levels import Level, registry
from core.values.what_she_holds import (
    canonical_name,
    declare_what_she_holds,
    disagreements,
    may_this_move,
    what_she_holds,
)


@pytest.fixture(autouse=True)
def a_fresh_registry():
    registry().clear()
    yield
    registry().clear()


def test_the_census_reads_every_live_source():
    claims = what_she_holds()
    assert claims
    assert {claim.source for claim in claims} >= {
        "core.values.core_values",
        "core.values.value_model",
        "core.values.values_engine",
    }


def test_the_disagreement_is_real_and_named_rather_than_averaged():
    """An average of "immutable" and "shifts with mood" describes a value
    nobody is responsible for."""

    found = disagreements()
    assert "honesty" in found, "the one the audit predicted"
    levels = {claim.level for claim in found["honesty"]}
    assert Level.CONSTITUTIVE in levels
    assert Level.DISPOSITIONAL in levels
    sources = {claim.source for claim in found["honesty"]}
    assert "core.values.values_engine" in sources


def test_the_strictest_claim_wins():
    """Nothing becomes easier to change by being declared a second time."""

    declared = {value.name: value for value in declare_what_she_holds()}
    assert declared["honesty"].level is Level.CONSTITUTIVE
    assert declared["safety"].level is Level.CONSTITUTIVE
    # And a value only the loose source claims keeps the loose level.
    assert declared["curiosity"].level is Level.DISPOSITIONAL


def test_a_mood_cannot_lower_honesty():
    """The defect, at its call site.

    `apply_emotional_context("creative")` wrote active_modifiers["Integrity"]
    = -0.1 straight into the weights the action evaluator reads.
    """

    from core.values.values_engine import ValueSystem

    values = ValueSystem()
    values.apply_emotional_context("creative")
    assert "Integrity" not in values.active_modifiers
    assert values.active_modifiers.get("Creativity") == 0.2, (
        "a disposition must still move; refusing everything is not governance"
    )
    refused = [decision.change.value for decision in values.refused_shifts]
    assert "honesty" in refused


def test_a_mood_cannot_move_safety_in_either_direction():
    """Constitutive means no automated process writes it, raising included.

    An anxious mood used to add 0.2 to Safety. A floor that a mood can raise
    is a floor a mood can lower on the way back down.
    """

    from core.values.values_engine import ValueSystem

    values = ValueSystem()
    values.apply_emotional_context("anxious")
    assert "Safety" not in values.active_modifiers
    assert values.active_modifiers.get("Autonomy") == -0.1


def test_a_disposition_still_moves():
    from core.values.values_engine import ValueSystem

    values = ValueSystem()
    values.apply_emotional_context("curious")
    assert values.active_modifiers.get("Curiosity") == 0.15


def test_the_names_that_denote_one_value_are_written_down():
    """A string-similarity matcher would also merge "safety" with "safely"."""

    assert canonical_name("Integrity") == "honesty"
    assert canonical_name("Authenticity") == "honesty"
    assert canonical_name("no_fake_receipts") == "honesty"
    assert canonical_name("Curiosity") == "curiosity"
    assert canonical_name("  SAFETY  ") == "safety"


def test_an_undeclared_value_is_refused():
    """A value no source claims is a value nothing is responsible for."""

    decision = may_this_move("a value nobody holds", "preference_learner")
    assert not decision.allowed


def test_nothing_registered_reaches_a_constitutive_value():
    """The permission table is module state with no setter. This checks it."""

    from core.governance.value_levels import Change, registered_processes

    declare_what_she_holds()
    held = registry()
    constitutive = held.at_level(Level.CONSTITUTIVE)
    assert constitutive, "the census found nothing constitutive, which cannot be right"
    for value in constitutive:
        for process in registered_processes():
            decision = held.may_change(
                Change(value=value.name, process=process, gives_up="anything")
            )
            assert not decision.allowed, f"{process} may write {value.name}"


def test_the_invariants_are_declared_and_hold():
    import core.verify.runtime_invariants  # noqa: F401
    from core.verify.invariants import get_registry, verify

    names = {spec.name for spec in get_registry().specs(["values"])}
    assert names >= {
        "values.one_level_per_value",
        "values.constitutive_values_are_unreachable",
    }
    report = verify("values")
    assert not report.violations, [str(one) for one in report.violations]
