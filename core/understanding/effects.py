"""What a person can want brought about, and what it costs to be wrong about each.

A request is known here by what would be true once it was met: a program exists and works, a list of things somebody
checked exists, a document has been read. Words do not come into it. The last three kinds are about work already asked
for: going on with it, being told how it stands, or having what was made changed. ``talk`` is everything that asks for
nothing to be brought about, which is most of what anybody says.

``unasked`` is how bad it is to do this for somebody who was only talking, set against failing to do it for somebody who
asked, which is 1. Reading a file they gave her or looking something up costs them nothing but a little of her time.
Building something nobody wanted leaves a thing on their disk that can be put in the bin.
Looking something up costs them the wait as well: a minute or more where an answer takes twenty seconds (25 November 2026). What reaches other people or
their machine is worse, and cannot always be taken back, so it wants more certainty.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ABOUT_WORK_ASKED_FOR", "ASKING_FIRST", "BY_NAME", "EFFECTS", "TALK", "Effect"]


@dataclass(frozen=True)
class Effect:
    name: str
    #: What is true of the world once it has been met.
    met: str
    #: The cost of doing it for somebody who did not ask, where failing somebody who did is 1.
    unasked: float
    #: The cost of only answering somebody who did ask: 1 where a reply is none of it, and less where a reply is some of it.
    unmet: float = 1.0


TALK = "talk"

EFFECTS: tuple[Effect, ...] = (
    Effect(TALK, "nothing is to be brought about; what was said is answered", 0.0),
    Effect("program", "a program exists, works when it is used, and is where it was asked to be", 1.0),
    Effect("program_changed", "a program she made does what was asked of it now, and still does the rest", 1.0),
    Effect("found", "things of the kind asked for were found, and each was checked against what was asked", 0.4),
    Effect("document_read", "a document was read, all that the request turns on, and the answer is from it", 0.3),
    Effect("known", "something about the world was looked up and the answer is from what was found", 0.5, 0.4),
    Effect("written", "a text was written to what was asked", 0.3, 0.2),
    Effect("computed", "a figure, a plan or a proof was worked out and checked", 0.3, 0.4),
    Effect("on_machine", "something happened on their machine: an app, a file, a setting, a page", 2.0),
    Effect("sent", "something reached somebody else", 8.0),
    Effect("kept", "something is remembered or will be said at a time", 0.5),
    Effect("resume", "work already asked for went on from where it stood, with what was just given", 0.5),
    Effect("status", "they know how work already asked for stands, from the work itself", 0.1),
)

BY_NAME: dict[str, Effect] = {effect.name: effect for effect in EFFECTS}

#: Kinds that are about work already asked for, and take up the request they are about.
ABOUT_WORK_ASKED_FOR: frozenset[str] = frozenset({"resume", "status", "program_changed"})

#: What one short question first costs the person, on the same scale.
ASKING_FIRST = 0.25
