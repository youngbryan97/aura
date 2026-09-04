"""Reject(m) leaves the state exactly as it was, for every m.

The review that prompted this named one operator: letting go of a part
removed it, found the removal paid nothing, logged "she kept it after all",
and left the part gone. The record and the registry said opposite things.

Its sibling — naming what two parts share — did put its head back. That is
the real finding. The rule held because one author remembered and the other
did not, so what these tests hold is the rule at the place actions are
admitted, not at the two places it was got right or wrong by hand.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from core.cognition import what_she_does_about_herself as W
from core.cognition.an_invented_kind import WHERE_FROM
from core.cognition.what_she_can_take_back import (
    WHAT_A_CHANGE_CAN_REACH,
    as_it_stands,
    only_if_it_pays,
    put_it_back,
)
from core.cognition.what_she_could_do_next import (
    WHAT_SHE_COULD_DO,
    what_she_could_do,
)
from core.cognition.what_she_is_made_of import APart


@pytest.fixture(autouse=True)
def _restore_every_registry():
    """These tests write to live registries. Put them all back afterwards."""
    was = as_it_stands()
    actions = dict(WHAT_SHE_COULD_DO)
    yield
    put_it_back(was)
    WHAT_SHE_COULD_DO.clear()
    WHAT_SHE_COULD_DO.update(actions)


def _a_part() -> APart:
    WHERE_FROM["a_word_of_hers"] = ("stood in for the real thing",)
    return APart(
        at="word a_word_of_hers",
        kind="word",
        name="a_word_of_hers",
        term=None,
        used=0,
        idle=999.0,
        holds_up=False,
    )


def _letting_go(verdict: tuple[bool, str]) -> tuple[Any, bool]:
    part = _a_part()
    with patch.object(W, "_probe", return_value=[("fam", ())]), patch.object(
        W, "the_one_she_should_let_go", return_value=part
    ), patch.object(W, "worth_keeping", return_value=verdict):
        W.offer_what_she_can_do_about_what_she_is_made_of()
        action = next(
            one for one in WHAT_SHE_COULD_DO.values() if one.kind == "letting go"
        )
        said = action.do_it(None)
    return said, "a_word_of_hers" in WHERE_FROM


def test_a_removal_that_pays_nothing_puts_the_part_back():
    """The defect, exactly: it returned None over an emptied registry."""
    said, still_there = _letting_go((False, "it paid nothing"))
    assert said is None
    assert still_there, "a rejected removal left the part gone"


def test_a_removal_that_pays_still_removes():
    """The rollback must not be a way of never changing anything."""
    said, still_there = _letting_go((True, "it paid"))
    assert said == "let go of word a_word_of_hers"
    assert not still_there


def test_an_action_that_says_it_did_nothing_leaves_nothing_behind():
    """The invariant is enforced where actions are admitted, not per author."""

    def mutates_then_declines(situation: Any = None) -> str | None:
        WHERE_FROM["written_by_a_forgetful_author"] = lambda a, b: a
        return None

    action = what_she_could_do(
        "an action whose author forgot to undo",
        over="the words",
        kind="a test",
        do_it=mutates_then_declines,
        needs_a_case=False,
    )
    assert action.do_it(None) is None
    assert "written_by_a_forgetful_author" not in WHERE_FROM


def test_an_action_that_says_it_acted_keeps_what_it_did():
    def mutates_and_says_so(situation: Any = None) -> str | None:
        WHERE_FROM["kept_on_purpose"] = lambda a, b: a
        return "wrote a word"

    action = what_she_could_do(
        "an action that says what it did",
        over="the words",
        kind="a test",
        do_it=mutates_and_says_so,
        needs_a_case=False,
    )
    assert action.do_it(None) == "wrote a word"
    assert "kept_on_purpose" in WHERE_FROM


def test_a_change_that_raises_halfway_leaves_nothing_behind():
    """The case where leaving the state alone matters most."""
    was = dict(WHERE_FROM)
    with pytest.raises(RuntimeError):
        with only_if_it_pays("one that breaks"):
            WHERE_FROM["half_written"] = lambda a, b: a
            raise RuntimeError("halfway")
    assert WHERE_FROM == was


def test_the_trial_must_be_told_to_keep():
    """Forgetting is a change that did nothing, which is the safe way to be wrong."""
    with only_if_it_pays("one that forgets"):
        WHERE_FROM["forgotten"] = lambda a, b: a
    assert "forgotten" not in WHERE_FROM


def test_put_it_back_says_what_would_not_come_back():
    """It reports what it restored, rather than claiming a rollback it did not do."""
    was = as_it_stands()
    WHERE_FROM["added"] = lambda a, b: a
    assert put_it_back(was) == ()
    assert "added" not in WHERE_FROM


def test_every_named_registry_is_reachable():
    """A registry named here and missing there is a change nobody can take back."""
    from importlib import import_module

    for module_name, attr in WHAT_A_CHANGE_CAN_REACH:
        registry = getattr(import_module(module_name), attr)
        assert isinstance(registry, dict), f"{module_name}.{attr}"
