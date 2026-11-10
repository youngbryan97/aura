"""The people she meets while watching: who is in a scene, who she has met, and what she recalls of them.

Lifted whole out of `what_moves_her`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio


class _KeepsThePeopleSheMeets:
    """Lifted whole out of WhatMovesHer; see what_moves_her.py."""

    async def _who_is_in(self, text: str) -> list[str]:
        from .what_moves_her import (
            logger,
        )

        if self.whom is None or not text:
            return []
        try:
            return list(await asyncio.to_thread(self.whom, text))
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            logger.debug("who is in it was not read: %s", exc)
            return []

    def _keys_of(self, named: list[str]) -> list[str]:
        from .what_moves_her import (
            _one_person,
        )

        return [key for key, person in self.people.items() if any(_one_person(name, person.name) for name in named)]

    def _met(self, at: float, named: list[str], liking: float | None) -> None:
        """They were in what she heard or saw at ``at``: counted, and how she liked it is part of how she feels about
        them (the running mean of the last ``FEELS_OVER``)."""
        from .what_moves_her import (
            CARES_AT,
            FEELS_OVER,
            Person,
            _clock,
            _one_person,
            logger,
        )

        met: set[str] = set()
        for name in sorted(named, key=len, reverse=True):  # the fullest name first, so "Kevin" joins Kevin Lee
            key = next((key for key, person in self.people.items() if _one_person(name, person.name)), None)
            if key is None:
                key = name.lower()
                self.people[key] = Person(name=name, first_at=at, last_at=at)
            person = self.people[key]
            if len(name) > len(person.name):
                person.name = name  # "Allen" heard first is Allen Pan once he is named in full
            if key in met:
                continue  # named twice in one stretch is in it once
            met.add(key)
            person.named += 1
            person.last_at = at
            if liking is not None:
                person.felt += 1
                person.feeling += (float(liking) - person.feeling) / min(person.felt, FEELS_OVER)
            if person.name not in self.events[-1].people:
                self.events[-1].people.append(person.name)
            if (not person.shown and person.felt >= 3 and abs(person.feeling) >= CARES_AT
                    and self.show is not None):
                person.shown = True
                way = "like" if person.feeling > 0 else "dislike"
                try:
                    self.show(f"{_clock(at)} she has come to {way} {person.name} ({person.feeling:+.2f})")
                except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
                    logger.debug("how she feels about someone was not shown: %s", exc)

    def people_kept(self) -> dict[str, dict[str, float]]:
        """Who was in it, and how she came to feel about them: to keep with her memory of the watching."""
        return {person.name: {"feeling": round(person.feeling, 3), "named": person.named}
                for person in sorted(self.people.values(), key=lambda person: -person.named)[:20]}

    def _maybe_recall(self, at: float, then: str) -> None:
        """A stretch that moved her far enough calls up memories of her own, in the background: shown nowhere."""
        from .what_moves_her import (
            REMINDED_EVERY_S,
            REMINDS_AT,
        )

        if self.recall is None or (self._recalling is not None and not self._recalling.done()):
            return
        if at - self._recalled_at < REMINDED_EVERY_S:
            return
        strongest = max((felt.evidence for felt in self.felt[-6:] if felt.at == at), default=0.0)
        if strongest < REMINDS_AT:
            return
        self._recalled_at = at
        self._recalling = asyncio.get_running_loop().create_task(self._recall(then), name="what_it_calls_up_of_hers")

    async def _recall(self, then: str) -> None:
        from .what_moves_her import (
            logger,
        )

        try:
            called = await self.recall(then) if self.recall is not None else []
        except (RuntimeError, OSError, TimeoutError, TypeError, ValueError) as exc:
            logger.debug("what it calls up of hers was not recalled: %s", exc)
            return
        for episode in called or []:
            if episode and episode not in self.reminded_of:
                self.reminded_of.append(str(episode))

