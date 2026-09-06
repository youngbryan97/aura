"""A null substitute derived from the contract, not written by hand.

Voyager's closure asked for NullOrgan substitutes satisfying the same
contract, so every major cognitive organ can be lesioned at boot. Aura has the
measurement half — a lesion registry, an influence ledger, treatment against
null — and six lesionable channels out of sixty-nine declared services.
Nothing is measured, and the reason is that almost nothing can be.

Writing a neutral by hand per organ means writing a small fiction per organ,
and a fiction that drifts from the contract stops being a control. This one is
read off the protocol.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest

from core.verify.an_organ_that_does_nothing import (
    an_organ_that_does_nothing,
    the_emptiest,
    what_a_null_organ_cannot_answer,
)


@runtime_checkable
class AFaculty(Protocol):
    def appraise(self, thing: str) -> dict: ...
    def how_strong(self) -> float: ...
    def what_it_saw(self) -> list[str]: ...
    def ready(self) -> bool: ...
    def name_of(self) -> str: ...
    def anything(self) -> Any: ...


@pytest.fixture
def null():
    return an_organ_that_does_nothing(AFaculty, called="interiority")


# ------------------------------------------------------------ the contract


def test_it_satisfies_the_protocol_it_was_made_from(null):
    """A substitute that does not is a second failure mode, not a control."""
    assert isinstance(null, AFaculty)
    assert what_a_null_organ_cannot_answer(AFaculty, null) == []


def test_every_answer_is_the_emptiest_of_its_declared_type(null):
    assert null.appraise("x") == {}
    assert null.how_strong() == 0.0
    assert null.what_it_saw() == []
    assert null.ready() is False
    assert null.name_of() == ""


def test_an_undeclared_return_is_nothing(null):
    assert null.anything() is None


def test_a_mutable_answer_is_fresh_each_time(null):
    """One shared list means a caller that appends changes the next answer."""
    first = null.what_it_saw()
    first.append("something")
    assert null.what_it_saw() == []


def test_it_remembers_being_asked(null):
    """A lesion that changes nothing because nothing called it is a different
    result from one that changes nothing because the organ does not matter."""
    assert null.asked == []
    null.ready()
    null.ready()
    null.how_strong()
    assert null.asked == ["ready", "ready", "how_strong"]


def test_it_says_what_it_stands_in_for(null):
    assert "interiority" in repr(null)


# --------------------------------------------------------- the emptiest


@pytest.mark.parametrize(
    "declared,expected",
    [
        (bool, False),
        (int, 0),
        (float, 0.0),
        (str, ""),
        (list, []),
        (dict, {}),
        (tuple, ()),
        (None, None),
        (Any, None),
    ],
)
def test_the_emptiest_of_each_shape(declared, expected):
    assert the_emptiest(declared) == expected


def test_the_emptiest_of_an_optional_is_nothing():
    assert the_emptiest(str | None) is None
    assert the_emptiest(dict[str, int] | None) is None


def test_the_emptiest_of_a_parameterised_container():
    assert the_emptiest(list[str]) == []
    assert the_emptiest(dict[str, int]) == {}


def test_a_protocol_with_no_methods_makes_an_organ_with_none():
    @runtime_checkable
    class Empty(Protocol):
        pass

    organ = an_organ_that_does_nothing(Empty)
    assert what_a_null_organ_cannot_answer(Empty, organ) == []
    assert organ.asked == []


def test_a_substitute_missing_a_method_is_named():
    class NotQuite:
        def ready(self) -> bool:
            return False

    missing = what_a_null_organ_cannot_answer(AFaculty, NotQuite())
    assert "appraise" in missing
    assert "ready" not in missing
