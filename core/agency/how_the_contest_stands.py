"""How a contest stands: who is ahead, what winning takes, what is left to lose, and whether anything is happening.

A person playing knows more than what the last screen said. They know they
are up three to one, that the game is to five, that one life is left, that the
clock is nearly out, and that nothing has happened for a minute. An end
screen read wrongly ("You You 5 win! Computer o", LIVE 2026-10-04, taken for a
loss) is checked against that, and a game that never says who won is settled
by it.

Read from two things a game gives:

- its words, once: what wins ("first to 5", "score 10 points to win", "reach
  level 3", "collect all 8 stars"), and what loses ("lose all your lives",
  "before time runs out");
- its counters, as they change (core/agency/what_meeting_things_does.py reads
  them): two sides' scores on one row, hers on her side; a score, lives, a
  clock, a level, by the words beside them.

And gives: the standing, what each side still needs, an outcome the counters
alone settle, whether it has stalled against the game's own pace, and a line
for a person watching. Nothing here knows a game.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import median

__all__ = ["ContestStands", "WhatWins", "what_wins"]

#: Words that name the other side of a contest, and the player's own.
_THEIRS = re.compile(r"\b(computer|cpu|ai|opponent|enemy|rival|them|player\s*2|p2|com|bot|away)\b", re.I)
_MINE = re.compile(r"\b(you|your|player\s*1|p1|player|me|home)\b", re.I)

#: The least time with nothing counted changing that is a stall, whatever the game's pace.
STALL_AFTER_S = 30.0
#: How many of the game's usual gaps between changes, with nothing changing, is a stall.
STALL_GAPS = 4.0


@dataclass
class WhatWins:
    """What a game's words say wins and loses it."""

    to_score: int | None = None
    to_level: int | None = None
    to_collect: int | None = None
    lives_lose: bool = False
    timed: bool = False

    def says(self) -> str:
        said = []
        if self.to_score:
            said.append(f"first to {self.to_score} points wins")
        if self.to_level:
            said.append(f"reaching level {self.to_level} wins")
        if self.to_collect:
            said.append(f"collecting all {self.to_collect} wins")
        if self.lives_lose:
            said.append("losing every life loses")
        if self.timed:
            said.append("it runs against a clock")
        return "; ".join(said)


_TO_SCORE = re.compile(
    r"\bfirst (?:player |one |side )?to (?:reach |get |score )?(\d{1,4})\b|"
    r"\b(?:score|get|reach|earn) (\d{1,4}) (?:points? |goals? |runs? )?(?:to win|wins?|first)\b|"
    r"\b(\d{1,4}) (?:points?|goals?) (?:to win|wins?)\b|"
    r"\bwins? (?:at|with|by reaching) (\d{1,4})\b"
)
_TO_LEVEL = re.compile(r"\b(?:reach|complete|beat|clear|finish) (?:all )?(?:level|stage|wave|world) (\d{1,3})\b|\b(?:all|every) (\d{1,3}) (?:levels|stages|waves)\b")
_TO_COLLECT = re.compile(r"\b(?:collect|find|gather|get) all (\d{1,4})\b|\b(?:collect|find|gather) (?:all )?(\d{1,4}) [a-z]+")
_LIVES = re.compile(r"\b(?:lose|losing|lost|run out of|out of) (?:all )?(?:your |of your )?(?:lives|hearts|health)\b|\b\d+ (?:lives|hearts)\b|\blives?\b", re.I)
_TIMED = re.compile(r"\b(?:time runs out|before time|time limit|timer|\d+ (?:seconds|minutes)|against the clock|countdown)\b")


def what_wins(words: str) -> WhatWins:
    """What winning and losing take, as the game's own words say."""
    text = " ".join(str(words or "").lower().split())

    def first_number(pattern: re.Pattern[str]) -> int | None:
        found = pattern.search(text)
        if not found:
            return None
        return next((int(g) for g in found.groups() if g), None)

    return WhatWins(
        to_score=first_number(_TO_SCORE),
        to_level=first_number(_TO_LEVEL),
        to_collect=first_number(_TO_COLLECT),
        lives_lose=bool(_LIVES.search(text)),
        timed=bool(_TIMED.search(text)),
    )


@dataclass
class ContestStands:
    """The state of one contest, followed from its counters and its words."""

    wins: WhatWins = field(default_factory=WhatWins)
    mine: int | None = None
    theirs: int | None = None
    lives: int | None = None
    level: int | None = None
    clock: int | None = None
    changed_at: list[float] = field(default_factory=list)
    _last: dict[str, int] = field(default_factory=dict)
    _started_at: float | None = None

    def heard(self, words: str) -> None:
        """A game's words: what wins and loses, kept where they say something new."""
        said = what_wins(words)
        for name in ("to_score", "to_level", "to_collect"):
            if getattr(said, name) and not getattr(self.wins, name):
                setattr(self.wins, name, getattr(said, name))
        self.wins.lives_lose |= said.lives_lose
        self.wins.timed |= said.timed

    def counted(self, values: Mapping[str, int], where: Mapping[str, tuple[float, float]], her_x: float | None, at: float) -> None:
        """The counters as they now read, by their label (or their place), and where they are."""
        if self._started_at is None:
            self._started_at = at
        if any(self._last.get(k) != v for k, v in values.items() if k in self._last):
            self.changed_at.append(at)
        self._last = dict(values)
        sides = [k for k in values if k.startswith("number at")]
        rows: dict[float, list[str]] = {}
        for key in sides:
            rows.setdefault(round(where.get(key, (0.0, 0.0))[1], 1), []).append(key)
        for row in rows.values():
            if len(row) >= 2 and her_x is not None:
                nearest = min(row, key=lambda k: abs(where.get(k, (0.5, 0.0))[0] - her_x))
                self.mine = values[nearest]
                others = [values[k] for k in row if k != nearest]
                self.theirs = max(others) if others else None
        for key, value in values.items():
            if key.startswith("number at"):
                continue
            words = key.lower()
            if re.search(r"\b(lives?|hearts?|health|hp|ships?|men|tries)\b", words):
                self.lives = value
            elif re.search(r"\b(level|stage|wave|world|round)\b", words):
                self.level = value
            elif re.search(r"\b(time|timer|clock|sec)\b", words):
                self.clock = value
            elif _THEIRS.search(words):
                self.theirs = value
            elif _MINE.search(words) or re.search(r"\b(score|points?|pts)\b", words):
                self.mine = value

    def standing(self) -> str:
        """Ahead, level or behind, and by how much, where both sides' scores are known."""
        if self.mine is None or self.theirs is None:
            return ""
        lead = self.mine - self.theirs
        return "level" if lead == 0 else (f"ahead by {lead}" if lead > 0 else f"behind by {-lead}")

    def to_go(self) -> tuple[int | None, int | None]:
        """What she still needs to win, and what the other side does, where winning is a score."""
        target = self.wins.to_score
        if not target:
            return None, None
        return (max(0, target - self.mine) if self.mine is not None else None,
                max(0, target - self.theirs) if self.theirs is not None else None)

    def settled(self) -> str:
        """"won" or "lost" where the counters and what wins settle it alone; "" where they do not."""
        mine_left, theirs_left = self.to_go()
        if mine_left == 0 and theirs_left != 0:
            return "won"
        if theirs_left == 0 and mine_left != 0:
            return "lost"
        if self.wins.to_level and self.level is not None and self.level > self.wins.to_level:
            return "won"
        if self.lives == 0 and (self.wins.lives_lose or self.lives is not None):
            return "lost"
        return ""

    def stalled(self, now: float) -> bool:
        """Nothing counted has changed in much longer than this game's own pace between changes."""
        since = self.changed_at[-1] if self.changed_at else self._started_at
        if since is None:
            return False
        gaps = [b - a for a, b in zip(self.changed_at, self.changed_at[1:], strict=False)]
        usual = median(gaps) if len(gaps) >= 2 else 0.0
        return now - since > max(STALL_AFTER_S, STALL_GAPS * usual)

    def says(self) -> str:
        """How it stands, in a line for a person watching."""
        parts = []
        # Two counts a thousand times apart are not two sides of one contest: one of them was misread or is another
        # count (LIVE 2026-10-09 "1–94332035, I'm behind by 94332034").
        if (self.mine is not None and self.theirs is not None
                and max(self.mine, self.theirs) <= 1000 * max(1, min(self.mine, self.theirs))):
            standing = self.standing()
            parts.append(f"{self.mine}–{self.theirs}, " + ("level" if standing == "level" else f"I'm {standing}"))
        elif self.mine is not None:
            parts.append(f"I have {self.mine}")
        mine_left, theirs_left = self.to_go()
        if mine_left:
            parts.append(f"{mine_left} more to win" + (f", they need {theirs_left}" if theirs_left else ""))
        if self.lives is not None:
            parts.append(f"{self.lives} {'life' if self.lives == 1 else 'lives'} left")
        if self.clock is not None and self.wins.timed:
            parts.append(f"{self.clock} on the clock")
        return "; ".join(parts) + ("." if parts else "")
