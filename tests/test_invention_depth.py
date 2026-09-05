"""Inventions that feed later inventions.

Composing human-supplied primitives is not invention however elaborate the
compositions get. The question is whether what was invented becomes material
for the next invention: if generation five can only build from what a person
wrote down, the expressive power never grew, it only got busier.

Both halves are easy to fake, and one of them has already been faked in this
codebase — a macro check that ran against the base primitives rather than
their closure, so new syntax counted as new meaning and the vocabulary grew
while the expressible set stood still.
"""

from __future__ import annotations

import pytest

from core.cognition.invention_depth import (
    MIN_PROBES,
    Verdict,
    Vocabulary,
)

PROBES = [0, 1, 2, 3, 5, 8, 13, -4, 7, 11]


@pytest.fixture
def vocabulary():
    words = Vocabulary(probes=PROBES)
    words.supply("inc", lambda x: x + 1)
    words.supply("neg", lambda x: -x)
    words.supply("dbl", lambda x: x * 2)
    return words


# ── the new thing has to be new ──────────────────────────────────────────


def test_a_composition_of_two_primitives_is_a_macro(vocabulary):
    proposal = vocabulary.judge("add_two", lambda x: x + 2)
    assert proposal.verdict is Verdict.MACRO
    assert proposal.equivalent_to == ("inc", "inc")


def test_a_composition_three_deep_is_still_a_macro(vocabulary):
    """The check runs against the closure, not the base set."""
    proposal = vocabulary.judge("dec", lambda x: x - 1)
    assert proposal.verdict is Verdict.MACRO
    assert len(proposal.equivalent_to) == 3


def test_something_no_composition_reaches_is_invented(vocabulary):
    assert vocabulary.judge("square", lambda x: x * x).verdict is Verdict.INVENTED


def test_a_second_name_for_an_existing_primitive_is_a_duplicate(vocabulary):
    proposal = vocabulary.judge("increment", lambda x: x + 1)
    assert proposal.verdict is Verdict.DUPLICATE
    assert proposal.equivalent_to == ("inc",)


def test_a_macro_is_not_added_to_the_vocabulary(vocabulary):
    before = vocabulary.names
    vocabulary.invent("add_two", lambda x: x + 2)
    assert vocabulary.names == before


def test_once_something_is_invented_what_it_reaches_becomes_a_macro(vocabulary):
    """Which is correct, and is what stops the count inflating."""
    assert vocabulary.invent("square", lambda x: x * x).accepted
    quartic = vocabulary.judge("quartic", lambda x: (x * x) * (x * x))
    assert quartic.verdict is Verdict.MACRO
    assert quartic.equivalent_to == ("square", "square")


def test_novelty_is_judged_on_what_it_does_not_how_it_is_written(vocabulary):
    written_differently = vocabulary.judge("add_two_v2", lambda x: (x + 3) - 1)
    assert written_differently.verdict is Verdict.MACRO


def test_too_few_probes_cannot_tell_anything_apart():
    thin = Vocabulary(probes=[1, 2])
    thin.supply("inc", lambda x: x + 1)
    proposal = thin.judge("square", lambda x: x * x)
    assert proposal.verdict is Verdict.UNDECIDABLE
    assert str(MIN_PROBES) in proposal.because


def test_a_primitive_that_raises_is_undefined_there_not_equal_to_another(vocabulary):
    """Two primitives that both fail on a probe are not thereby the same."""
    vocabulary.supply("recip", lambda x: 1 / x)
    proposal = vocabulary.judge("recip_squared", lambda x: 1 / (x * x))
    assert proposal.verdict is not Verdict.DUPLICATE


# ── invention has to feed invention ──────────────────────────────────────


def test_a_chain_where_each_stands_on_the_last_grows_the_depth(vocabulary):
    assert vocabulary.invent("square", lambda x: x * x).generation == 1
    assert vocabulary.invent(
        "sq_plus_self", lambda x: x * x + x, depends_on=["square"]
    ).generation == 2
    assert vocabulary.invent(
        "triangular", lambda x: (x * x + x) // 2, depends_on=["sq_plus_self"]
    ).generation == 3
    assert vocabulary.depth == 3


def test_a_wide_vocabulary_is_not_a_deep_one(vocabulary):
    """Every invention built only on what a person supplied is depth one."""
    vocabulary.invent("square", lambda x: x * x)
    vocabulary.invent("mod3", lambda x: x % 3)
    vocabulary.invent("is_prime_ish", lambda x: x in (2, 3, 5, 7, 11, 13))
    assert vocabulary.snapshot()["invented"] == 3
    assert vocabulary.depth == 1


def test_the_lineage_is_the_chain_it_stands_on(vocabulary):
    vocabulary.invent("square", lambda x: x * x)
    vocabulary.invent("sq_plus_self", lambda x: x * x + x, depends_on=["square"])
    vocabulary.invent(
        "triangular", lambda x: (x * x + x) // 2, depends_on=["sq_plus_self"]
    )
    assert vocabulary.lineage("triangular") == ("sq_plus_self", "square")


def test_a_dependency_that_does_not_exist_is_reported_not_dropped(vocabulary):
    """An invention standing on something refused is not standing on anything."""
    proposal = vocabulary.invent("cube", lambda x: x * x * x, depends_on=["nope"])
    assert proposal.accepted
    assert proposal.unknown_dependencies == ("nope",)
    assert proposal.generation == 1


def test_a_refused_proposal_still_reports_its_missing_dependencies(vocabulary):
    proposal = vocabulary.invent("add_two", lambda x: x + 2, depends_on=["nope"])
    assert not proposal.accepted
    assert proposal.unknown_dependencies == ("nope",)


def test_the_probe_set_is_fixed_before_any_proposal(vocabulary):
    """One chosen alongside a proposal can be chosen to make it look new."""
    assert vocabulary.probes == tuple(PROBES)
    assert not hasattr(vocabulary, "set_probes")
    assert not hasattr(vocabulary, "add_probe")


def test_the_snapshot_separates_supplied_from_invented(vocabulary):
    vocabulary.invent("square", lambda x: x * x)
    snapshot = vocabulary.snapshot()
    assert snapshot["supplied"] == 3
    assert snapshot["invented"] == 1
    assert snapshot["by_generation"]["0"] == 3
