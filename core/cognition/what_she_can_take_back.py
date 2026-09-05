"""A developmental change that does not pay leaves nothing behind.

Aura has many ways to change what she is made of and no single account of
what a change *is*. Each action was written to undo itself, and each author
had to remember. One of them did — naming what two parts share pops the head
it added when the trial does not pay. One of them did not: letting go of a
part removed it, found the removal paid nothing, logged "she kept it after
all", and left the part gone. The record and the state said opposite things.

The rule this module holds is the one that cannot be got wrong by forgetting:

    a change that is not kept leaves the state as it found it

It holds by construction rather than by discipline. The trial snapshots every
registry a developmental change can reach, and restores them on the way out
unless the body says to keep what it did. An author who forgets to call
``keep`` gets a change that did nothing, which is the safe way to be wrong.

What it cannot see is a value mutated in place — the snapshot holds the
mapping, not a deep copy of what the mapping points at. ``put_it_back``
therefore checks its own work and says what did not come back, rather than
reporting a rollback it did not perform.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Every registry a developmental change reaches, as ``module: name`` pairs.
#: Named rather than discovered: a change that reaches something not on this
#: list is a change this module cannot take back, and saying so is the point.
WHAT_A_CHANGE_CAN_REACH: tuple[tuple[str, str], ...] = (
    ("core.cognition.an_invented_kind", "WHERE_FROM"),
    ("core.cognition.an_invented_kind", "WHAT_OF_IT"),
    ("core.cognition.an_invented_kind", "WAYS_TO_BUILD"),
    ("core.cognition.an_invented_kind", "KINDS"),
    ("core.cognition.an_invented_kind", "UNSETTLED"),
    ("core.cognition.one_algebra", "DERIVED_HEADS"),
    ("core.cognition.a_rule_with_no_shape", "RULES_WITH_NO_SHAPE"),
    ("core.cognition.a_kind_of_thing_she_named", "KINDS_OF_THING"),
    ("core.cognition.what_rests_on_what", "QUARANTINED"),
    ("core.cognition.sequence_induction", "WHAT_WOULD_SETTLE_IT"),
    ("core.cognition.growing_at_any_level", "REGISTRY"),
    # The actions themselves. An action that writes another action is a change
    # to what she can do, and a trial that does not cover it would leave the
    # new action behind after the change that wrote it was taken back.
    ("core.cognition.what_she_could_do_next", "WHAT_SHE_COULD_DO"),
)


def _reach() -> list[tuple[str, dict]]:
    """The live registries, skipping any that will not import here."""
    from importlib import import_module

    found: list[tuple[str, dict]] = []
    for module_name, attr in WHAT_A_CHANGE_CAN_REACH:
        try:
            registry = getattr(import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            logger.debug("cannot reach %s.%s: %s", module_name, attr, exc)
            continue
        if isinstance(registry, dict):
            found.append((f"{module_name}.{attr}", registry))
    return found


@dataclass(frozen=True)
class HowItStood:
    """Every registry as it was, keyed by where it lives."""

    held: dict[str, dict[Any, Any]]

    def restore(self) -> tuple[str, ...]:
        """Put every registry back. Named so a holder needs no import of us.

        ``how_a_change_is_promoted`` keeps whatever a promotion replaced and
        has to be able to undo it without knowing what kind of thing it is.
        Duck-typing the restore keeps that module free of this one.
        """
        return put_it_back(self)

    def what_changed(self) -> dict[str, tuple[int, int]]:
        """Which registries differ from the snapshot, and by how many keys."""
        moved: dict[str, tuple[int, int]] = {}
        for where, registry in _reach():
            was = self.held.get(where)
            if was is None or registry == was:
                continue
            moved[where] = (len(was), len(registry))
        return moved


def as_it_stands() -> HowItStood:
    """Snapshot the registries. Cheap: one shallow copy per registry."""
    return HowItStood(held={where: dict(reg) for where, reg in _reach()})


def put_it_back(was: HowItStood) -> tuple[str, ...]:
    """Restore every registry, and say what would not come back.

    The return value is the honest part. A key-level change is restored
    exactly; a value mutated in place is not, and this names the registry
    where that happened rather than letting the caller believe otherwise.
    """
    for where, registry in _reach():
        held = was.held.get(where)
        if held is None:
            continue
        registry.clear()
        registry.update(held)
    return tuple(sorted(was.what_changed()))


@dataclass
class ATrial:
    """One developmental change, kept only if the body says so."""

    why: str = ""
    kept: bool = False
    before: HowItStood = field(default_factory=as_it_stands)

    def keep(self, why: str = "") -> None:
        """Say the change paid. Without this call, it is taken back."""
        self.kept = True
        self.why = str(why)


@contextmanager
def only_if_it_pays(what: str = "a change") -> Iterator[ATrial]:
    """Run a developmental change, and take it back unless it is kept.

    An exception takes it back too, and then raises: a change that failed
    halfway is the case where leaving the state alone matters most.
    """
    trial = ATrial()
    try:
        yield trial
    except BaseException:
        stubborn = put_it_back(trial.before)
        if stubborn:
            logger.warning(
                "%s raised and these did not come back: %s",
                what, ", ".join(stubborn),
            )
        raise
    if trial.kept:
        return
    stubborn = put_it_back(trial.before)
    if stubborn:
        logger.warning(
            "%s was not kept and these did not come back: %s",
            what, ", ".join(stubborn),
        )
    else:
        logger.info("%s was not kept, and left nothing behind", what)


__all__ = [
    "ATrial",
    "HowItStood",
    "WHAT_A_CHANGE_CAN_REACH",
    "as_it_stands",
    "only_if_it_pays",
    "put_it_back",
]
