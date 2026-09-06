"""A null substitute for an organ, so a claim about it can be lesioned.

Voyager's closure asked for OrganBoundary adapters with NullOrgan substitutes
satisfying the same contract, and for every major cognitive organ to support
boot-time lesion. Aura has the measurement half — a lesion registry, an
influence ledger, treatment against null — and six lesionable channels out of
sixty-nine declared services. Nothing is measured, and the reason is that
almost nothing can be.

Making a faculty lesionable is usually not a redesign. It is one place where
the faculty's answer reaches what it changes, and a neutral to put there
instead. What was missing is a neutral that is obviously neutral: writing one
by hand per organ means writing a small fiction per organ, and a fiction that
drifts from the contract stops being a control.

So the neutral is derived from the contract. An organ that does nothing
answers every method its protocol declares, and answers with the emptiest
value of the declared return type — an empty list for a list, zero for a
number, None where nothing is declared. That is what "did not contribute"
looks like, and it cannot drift from the protocol because it is read off it.
"""
from __future__ import annotations

import logging
import typing
from typing import Any, get_type_hints

logger = logging.getLogger("Aura.AnOrganThatDoesNothing")

__all__ = [
    "an_organ_that_does_nothing",
    "the_emptiest",
    "what_a_null_organ_cannot_answer",
]

#: The emptiest value of each shape. A null organ answers with these, so
#: "did not contribute" is the same thing everywhere rather than whatever
#: each author thought neutral meant.
_EMPTIEST: dict[Any, Any] = {
    bool: False,
    int: 0,
    float: 0.0,
    str: "",
    list: [],
    dict: {},
    set: frozenset(),
    tuple: (),
    type(None): None,
}


def the_emptiest(what: Any) -> Any:
    """The emptiest value of a declared type, or None where none was declared."""
    if what is None or what is Any:
        return None
    origin = typing.get_origin(what)
    if origin is not None:
        if origin in (list, set, frozenset, tuple, dict):
            return _EMPTIEST.get(origin, None)
        # `X | None` and Optional[X]: None is the emptiest by construction.
        args = [one for one in typing.get_args(what) if one is not type(None)]
        if len(args) < len(typing.get_args(what)):
            return None
        return the_emptiest(args[0]) if args else None
    if isinstance(what, type):
        for kind, empty in _EMPTIEST.items():
            if isinstance(kind, type) and issubclass(what, kind):
                return [] if empty == [] else ({} if empty == {} else empty)
    return None


def an_organ_that_does_nothing(protocol: type, *, called: str = "") -> Any:
    """An object satisfying ``protocol`` whose every answer is the emptiest one.

    For the counterfactual arm of a measurement: run the real path with this
    in place of the organ and the difference is what the organ contributed.

    Every call is recorded on the substitute, so a trial can also say whether
    the organ was reached at all — a lesion that changes nothing because
    nothing called it is a different result from one that changes nothing
    because the organ does not matter.
    """
    wanted = [
        name
        for name in dir(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name, None))
    ]
    returns: dict[str, Any] = {}
    for name in wanted:
        method = getattr(protocol, name)
        try:
            hints = get_type_hints(method)
        except Exception:  # noqa: BLE001 — an unresolvable hint is no hint
            hints = {}
        returns[name] = hints.get("return")

    class AnOrganThatDoesNothing:
        """Answers everything, contributes nothing, and remembers being asked."""

        def __init__(self) -> None:
            self.asked: list[str] = []
            self.name = called or getattr(protocol, "__name__", "an organ")

        def __repr__(self) -> str:
            return f"<nothing where {self.name} was, asked {len(self.asked)} times>"

    def _answering(name: str, declared: Any) -> Any:
        empty = the_emptiest(declared)

        def answer(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            self.asked.append(name)
            # A fresh empty each time: handing back one shared list means a
            # caller that appends to it changes what the next caller gets.
            return list(empty) if isinstance(empty, list) else (
                dict(empty) if isinstance(empty, dict) else empty
            )

        answer.__name__ = name
        answer.__doc__ = f"nothing, where {name} would have contributed"
        return answer

    for name, declared in returns.items():
        setattr(AnOrganThatDoesNothing, name, _answering(name, declared))
    return AnOrganThatDoesNothing()


def what_a_null_organ_cannot_answer(protocol: type, organ: Any) -> list[str]:
    """Methods the protocol declares that this substitute does not answer.

    Empty is the point: a substitute that does not satisfy the contract is not
    a control, it is a second failure mode.
    """
    missing: list[str] = []
    for name in dir(protocol):
        if name.startswith("_") or not callable(getattr(protocol, name, None)):
            continue
        if not callable(getattr(organ, name, None)):
            missing.append(name)
    return missing
