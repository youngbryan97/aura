"""The seven destinations, and whether each is really one.

``WHERE_A_TERM_CAN_GO`` names seven places a term can be installed, and the
comment above it makes a specific claim about why that set is the set:

    Each has an installer, a lesion, and a persistence path, and those three
    are what makes something a destination rather than a wish.

That is the right criterion and nothing checked it. A destination with an
installer and no persistence path is a change that does not survive the
process; one with no lesion is a change nobody can measure the absence of.
Either way the set would be smaller than it says.

So the three are declared here, per destination, and the gate resolves them.
A destination whose installer, lesion or persistence cannot be found is
reported by name — which is the difference between a claim in a comment and a
claim that fails when it stops being true.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.WhereATermCanGo")

__all__ = [
    "THE_DESTINATIONS",
    "what_is_not_really_a_destination",
    "where_a_term_can_go",
]

#: Destination → what installs into it, what neutralises it, what keeps it.
#: Every value is an importable name, so the gate resolves rather than reads.
THE_DESTINATIONS: dict[str, dict[str, str]] = {
    "the words": {
        "installs": "core.cognition.an_invented_kind:WHAT_OF_IT",
        "lesion": "core.cognition.what_she_gave_meaning:forget_everything",
        "keeps": "core.cognition.what_she_gave_meaning:keep",
    },
    "the ways of building words": {
        "installs": "core.cognition.an_invented_kind:WAYS_TO_BUILD",
        "lesion": "core.cognition.what_she_gave_meaning:forget_everything",
        "keeps": "core.cognition.what_she_gave_meaning:keep",
    },
    "the ways of computing": {
        "installs": "core.cognition.the_floor_she_stands_on:every_code",
        "lesion": "core.cognition.what_she_gave_meaning:forget_everything",
        "keeps": "core.cognition.what_she_gave_meaning:keep",
    },
    "the shapes a rule can have": {
        "installs": "core.cognition.an_invented_kind:KINDS",
        "lesion": "core.agency.what_she_invented:forget_everything",
        "keeps": "core.agency.what_she_invented:keep",
    },
    "the order she tries them in": {
        "installs": "core.cognition.what_it_is_worth_doing:the_worth_she_wrote",
        "lesion": "core.cognition.what_it_is_worth_doing:the_worth_she_uses",
        "keeps": "core.cognition.the_record_of_her_own_work:keep_the_record",
    },
    "the proposer": {
        # The first version of this named a function in primitive_invention
        # that does not exist, and the gate said so — a resolved name can do
        # that and a comment cannot. What installs a proposer is
        # the_proposer_she_wrote.
        "installs": "core.cognition.the_proposer_she_can_replace:the_proposer_she_wrote",
        "lesion": "core.cognition.the_proposer_she_can_replace:the_proposer_in_use",
        "keeps": "core.cognition.what_she_gave_meaning:keep",
    },
    "what a change is worth": {
        "installs": "core.cognition.what_it_is_worth_doing:the_worth_she_wrote",
        "lesion": "core.cognition.what_it_is_worth_doing:the_worth_she_uses",
        "keeps": "core.cognition.the_record_of_her_own_work:keep_the_record",
    },
}


def _resolves(address: str) -> bool:
    """Whether this dotted name is actually there."""
    module_name, _, attribute = address.partition(":")
    try:
        module = __import__(module_name, fromlist=[attribute or "_"])
    except Exception as exc:  # noqa: BLE001 — an unimportable name does not resolve
        logger.debug("%s did not import: %s", address, exc)
        return False
    return not attribute or hasattr(module, attribute)


def where_a_term_can_go() -> dict[str, Any]:
    """Every destination, and whether its three parts are all there."""
    from core.cognition.what_she_could_do_next import (
        WHERE_A_TERM_CAN_GO,
        the_actions_she_has,
    )

    actions = the_actions_she_has()
    how_many: dict[str, int] = {}
    for one in actions:
        how_many[one.over] = how_many.get(one.over, 0) + 1

    said: dict[str, dict[str, Any]] = {}
    for name in WHERE_A_TERM_CAN_GO:
        parts = THE_DESTINATIONS.get(name)
        if parts is None:
            said[name] = {"declared": False, "actions": how_many.get(name, 0)}
            continue
        said[name] = {
            "declared": True,
            "actions": how_many.get(name, 0),
            **{key: _resolves(where) for key, where in parts.items()},
        }
    return {
        "destinations": len(WHERE_A_TERM_CAN_GO),
        "actions": len(actions),
        "each": said,
        "not_really_a_destination": what_is_not_really_a_destination(said),
    }


def what_is_not_really_a_destination(said: dict[str, Any] | None = None) -> list[str]:
    """Destinations missing an installer, a lesion, a persistence path, or an action.

    A destination with no action installing into it is a slot nothing fills.
    One with no persistence path is a change that does not survive the
    process. One with no lesion is a change nobody can measure the absence of.
    """
    if said is None:
        said = where_a_term_can_go()["each"]
    wrong: list[str] = []
    for name, parts in said.items():
        if not parts.get("declared"):
            wrong.append(f"{name}: nothing declares its three parts")
            continue
        for key in ("installs", "lesion", "keeps"):
            if not parts.get(key):
                wrong.append(f"{name}: its {key} does not resolve")
        if not parts.get("actions"):
            wrong.append(f"{name}: no action installs into it")
    return wrong
