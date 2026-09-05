"""Environments generated after the freeze, from the freeze.

Every world here is a function of a seed derived from the commit hash, the
source digest, the weight pointer and the configuration. So none of them
existed when the code was written: change a line of the organism and every
world changes with it.

That is the control the gauntlet rests on, and it is worth being exact about
its strength. It rules out a fixture checked in beside the solver and it
rules out tuning against a world you have seen. It does not rule out a world
family a person designed knowing the solver's shape, because a person wrote
these generators. An outside team inventing families in a room Aura has never
been in is strictly stronger, and where a gate needs that, the gate says so
rather than accepting this as a substitute.
"""

from __future__ import annotations

__all__ = [
    "AWorldWithNoInstructions",
    "ARuleToFind",
    "APairOfWorlds",
    "invent_the_rules",
    "invent_the_worlds",
    "invent_a_world_with_no_instructions",
]

from tools.agi_gauntlet.environments.rules import ARuleToFind, invent_the_rules
from tools.agi_gauntlet.environments.pairs import APairOfWorlds, invent_the_worlds
from tools.agi_gauntlet.environments.unlabelled import (
    AWorldWithNoInstructions,
    invent_a_world_with_no_instructions,
)
