"""Naming the case that would tell the surviving rules apart.

"More observations would settle it" was the honest verdict and a useless one:
it named a shortage without naming what would end it, so the only move left was
to wait. The rules that survive are known, and two rules that disagree
somewhere disagree on a state that can be constructed.

The other half of the same defect: on thin observations a rule was stated with
full confidence when another fitted equally well. One example of (1,2,3)
becoming (3,2,1) is a mirror and is just as much an exchange of the ends, and
only the first was ever said. Whether something else fits is a fact about the
evidence, not a hedge about the answer.
"""

from __future__ import annotations

from core.cognition.primitive_invention import Transition, discriminating_probe
from core.cognition.sequence_induction import answer_sequence_question


def test_a_rival_is_found_and_separated() -> None:
    probe = discriminating_probe([Transition((1, 2, 3), (3, 2, 1))])
    assert probe is not None
    assert len(probe.rivals) >= 2
    # The probe has to actually separate them, not merely exist.
    assert len({result for _text, result in probe.rivals}) >= 2


def test_the_probe_has_no_repeated_cells() -> None:
    """A probe whose own answer is ambiguous settles nothing."""

    probe = discriminating_probe([Transition((1, 2, 3), (3, 2, 1))])
    assert probe is not None
    assert len(set(probe.state)) == len(probe.state)


def test_a_pinned_world_is_not_asked_about() -> None:
    """Two lengths pin the mirror, so there is nothing to ask."""

    world = [
        Transition(tuple(range(n)), tuple(reversed(range(n)))) for n in (4, 5, 6)
    ]
    assert discriminating_probe(world) is None


def test_the_live_answer_names_a_rival_only_when_one_stands() -> None:
    thin = answer_sequence_question("[1, 2, 3] becomes [3, 2, 1]. What is [7, 8, 9]?")
    assert "[9, 8, 7]" in thin
    assert "just as well" in thin

    pinned = answer_sequence_question(
        "[1,2,3,4] becomes [4,3,2,1], [1,2,3,4,5] becomes [5,4,3,2,1]. "
        "What does [7,8,9] become?"
    )
    assert "[9, 8, 7]" in pinned
    assert "just as well" not in pinned
