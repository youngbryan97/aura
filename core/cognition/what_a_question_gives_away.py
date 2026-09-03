"""What somebody asking tells her, quite apart from the answer.

The good Cluedo players are not the ones who track the answers. Everybody
tracks the answers. They are the ones who watch what other people ASK, because
a question is not free: somebody asking about the dagger is telling the table
they do not hold the dagger, and somebody who asks about the same room three
times is telling everybody they are stuck on it.

That is information nobody gave them and nobody can decline to give. It
arrives whether or not the question is answered, and it is about the asker
rather than about the thing asked.

She listened only for answers. A question put to her was a thing to respond
to; a question put to somebody else was noise. So half of what a conversation
carries went past her — and the half that went past is the half that says what
the other person does not have, which is exactly what she cannot find out any
other way.

Not about games. A colleague's question says where their model runs out. A
system that keeps retrying one endpoint says what it has not got. A person
asking the same thing a third time is telling you the first two answers did
not land, and that is worth more than the question.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WhatTheirAskingSays"]


@dataclass
class WhatTheirAskingSays:
    """What people have asked about, and what that says about them."""

    #: who -> what they asked about -> how many times.
    asked: dict[str, dict[str, int]] = field(default_factory=dict)
    #: who -> things they have stopped asking about since first asking.
    _last_asked: dict[str, dict[str, int]] = field(default_factory=dict)
    turns: int = 0

    def they_asked(self, who: str, about: Iterable[str]) -> None:
        """One question, and who put it."""
        self.turns += 1
        name = str(who)
        for one in set(about):
            thing = str(one)
            self.asked.setdefault(name, {})[thing] = (
                self.asked.setdefault(name, {}).get(thing, 0) + 1
            )
            self._last_asked.setdefault(name, {})[thing] = self.turns

    def they_have_not_got(self, who: str) -> tuple[str, ...]:
        """What asking says they do not have.

        Somebody asking about a thing is telling everybody they do not hold
        it, because holding it is what asking is for. Weaker than being unable
        to answer — a person can ask about something they hold, to mislead —
        and worth having anyway, because most people are not doing that most
        of the time.
        """
        return tuple(sorted(self.asked.get(str(who)) or {}))

    def what_they_are_stuck_on(self, who: str, *, at_least: int = 2) -> tuple[str, ...]:
        """What they have asked about more than once, most-asked first.

        Asking twice is not the same as asking. It says the first answer did
        not settle it, which is a fact about their situation rather than about
        the thing.
        """
        seen = self.asked.get(str(who)) or {}
        return tuple(
            one
            for one, many in sorted(seen.items(), key=lambda one: (-one[1], one[0]))
            if many >= at_least
        )

    def what_they_have_stopped_asking(self, who: str) -> tuple[str, ...]:
        """Things they asked about and have not asked about since.

        Somebody who stops asking has found out. That is the moment their
        knowledge changed, and it is visible from outside without anybody
        saying anything.
        """
        seen = self._last_asked.get(str(who)) or {}
        if not seen:
            return ()
        latest = max(seen.values())
        return tuple(
            one for one, when in sorted(seen.items()) if when < latest
        )

    def who_is_furthest_along(self) -> str:
        """Whoever is asking about the fewest things.

        A short list of questions is a nearly finished one. This is how a
        table knows somebody is about to win before they say so — and how she
        can know which of several parties is closest to an answer without any
        of them telling her.
        """
        if not self.asked:
            return ""
        return min(
            self.asked,
            key=lambda who: (len(self.asked[who]), who),
        )

    def describe(self, who: str) -> str:
        theirs = self.they_have_not_got(who)
        if not theirs:
            return f"{who} has asked nothing yet"
        stuck = self.what_they_are_stuck_on(who)
        said = f"{who} does not have {len(theirs)} thing(s)"
        return f"{said}, and is stuck on {', '.join(stuck)}" if stuck else said

    def as_memory(self) -> dict[str, Any]:
        return {
            "asked": {k: dict(v) for k, v in self.asked.items()},
            "last_asked": {k: dict(v) for k, v in self._last_asked.items()},
            "turns": self.turns,
        }
