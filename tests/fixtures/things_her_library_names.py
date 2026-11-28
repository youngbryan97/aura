"""A stand-in for what an exchange is about (core/conation/what_an_exchange_is_about.py), for tests: it names the
things it is given wherever they are said, and makes the most of what is said most."""
from __future__ import annotations

import re

from core.conation.what_an_exchange_is_about import Subject, shares_of


class NamesThese:
    """What her library would name of an exchange, as given: each name, where a text says it, beyond chance by as
    many times as it is said."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self._last: list[Subject] = []

    def read(self, *texts: str) -> list[Subject]:
        out = []
        for place, name in enumerate(self.names):
            said = re.compile(rf"\b{re.escape(name)}(?:s|'s)?\b", re.IGNORECASE)
            counts = [len(said.findall(str(text or ""))) for text in texts]
            if sum(counts):
                out.append(Subject(name, place + 1, 0.5, float(sum(counts)), 0.0, name,
                                   frozenset(i for i, count in enumerate(counts) if count)))
        return sorted(out, key=lambda subject: -subject.beyond_chance)

    def about(self, *texts: str) -> list[str]:
        self._last = self.read(*texts)
        return [subject.title for subject in self._last]

    def read_already(self, *texts: str) -> list[Subject]:
        return self.read(*texts)

    read_soon = read_already

    def shares(self) -> dict[str, float]:
        return shares_of(self._last)

    def subjects_of(self, *texts: str) -> list[str]:
        return [subject.title for subject in self.read(*texts)]
