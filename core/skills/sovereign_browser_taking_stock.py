"""Taking stock between the runs of what a page draws: when to stop and ask, with what, and whether it helped.

The faculty itself knows nothing of pages or games (core/cognition/taking_stock.py).
Here it is given its occasions and its sources by the play of one thing:

- the occasions: she has lost the last runs one after another (FAILING); a
  round ran its slice and gained nothing (STUCK). Both come while the thing is
  waiting for her, between runs, so there is time; play itself is never
  stopped for it.
- the sources: what helped her before in this thing; her own copy of
  Wikipedia and the web, kept to what names the thing; her model, where it can
  answer in the time; her own semantic memory.
- what is done with it: the counsel is read beside the screen's own words when
  she next plays (core/skills/screen_pursuit_as_it_happens.py), so a key or a
  way of playing it names is taken up the way a screen's instructions are.
- and whether it helped: the runs after it are held against the runs before,
  as better runs are judged anywhere (more gained, or lasting longer), and
  what helped is kept, and what did not is not offered again.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Stocktaking", "the_thing"]

logger = logging.getLogger("Skills.SovereignBrowser.TakingStock")

#: The time given to taking stock, in seconds, and the least share of what is left to play that it may use.
TAKING_STOCK_S = 25.0
AT_MOST_OF_WHAT_IS_LEFT = 0.2

#: How many times in one thing, and how many runs its counsel is given before it is judged.
AT_MOST = 3
JUDGED_AFTER = 2


def the_thing(goal: str, title: str = "") -> str:
    """The thing she is in, by its own name: as the goal names it ("It is “X”"), else as its page's title begins."""
    named = re.search(r"[“\"]([^”\"]{3,120})[”\"]", str(goal or ""))
    if named:
        return named.group(1).strip()
    return re.split(r"\s+[:|–—-]\s+", str(title or "").strip())[0][:120]


def _better(after: list[dict[str, Any]], before: list[dict[str, Any]]) -> bool:
    """Whether the runs after counsel went better than those before it: a win, more gained, or lasting longer."""
    if any(r.get("ended") == "won" for r in after):
        return True
    if not before:
        return False
    gained = max(int(r.get("gains") or 0) for r in after) - max(int(r.get("gains") or 0) for r in before)
    lasted = max(float(r.get("took_s") or 0.0) for r in after) / max(1.0, max(float(r.get("took_s") or 0.0) for r in before))
    return gained > 0 or (gained == 0 and lasted > 1.25)


@dataclass
class Stocktaking:
    """Taking stock in one thing: how often, the counsel in force, and the runs it is judged by."""

    thing: str
    taken: int = 0
    counsel: Any = None
    at_run: int = 0
    before: list[dict[str, Any]] = field(default_factory=list)

    def due(self, runs: list[dict[str, Any]]) -> str:
        """FAILING where the last two runs were lost and no counsel is still being tried; else ''."""
        from core.cognition.taking_stock import FAILING

        if self.taken >= AT_MOST or len(runs) < 2 or self.counsel is not None and len(runs) - self.at_run < JUDGED_AFTER + 1:
            return ""
        return FAILING if all(r.get("ended") == "lost" for r in runs[-2:]) else ""

    async def take(self, why: str, goal: str, words: list[str], ended: str, runs: list[dict[str, Any]], keep: dict[str, Any],
                   deadline: float) -> bool:
        """Stop, ask, and put what is worth going by where play will read it; whether anything was found."""
        from core.cognition.taking_stock import (
            Situation,
            WhatHelped,
            from_her_corpus,
            from_her_memory,
            from_her_model,
            from_the_web,
            questions_for,
            take_stock,
        )
        from core.rebuilding.her_model import ask_her_model
        from core.skills.looking_it_up import search_and_read
        from core.skills.sovereign_browser_going import _names_it

        seconds = min(TAKING_STOCK_S, max(0.0, deadline - time.monotonic()) * AT_MOST_OF_WHAT_IS_LEFT)
        if seconds < 8.0 or not self.thing:
            return False
        self.taken += 1
        situation = Situation(self.thing, goal, tuple(words[-6:]), ended, why)
        helped = WhatHelped.of(self.thing)
        _tell(f"{why}, so I'm taking stock: {questions_for(situation)[0]}?")
        sources = {
            "what helped me before": helped.source(),
            "what I remember": from_her_memory(),
            "my own copy of Wikipedia": from_her_corpus(thing=self.thing, names_it=_names_it),
            "the web": from_the_web(search_and_read, thing=self.thing, names_it=_names_it),
            "my model": from_her_model(ask_her_model),
        }
        counsel = helped.without_what_did_not(await take_stock(situation, sources, seconds=seconds))
        _tell(counsel.said())
        if not counsel:
            return False
        self.counsel, self.at_run, self.before = counsel, len(runs), list(runs[-3:])
        # Read with the screen's own words from the next run on; what was found before is kept beside it.
        keep["counsel"] = " ".join(t for t in dict.fromkeys([str(keep.get("counsel") or ""), counsel.told]) if t)
        return True

    def judge(self, runs: list[dict[str, Any]]) -> None:
        """Once the counsel has had its runs, whether it helped: said, and kept for next time."""
        from core.cognition.taking_stock import WhatHelped

        if self.counsel is None or len(runs) - self.at_run < JUDGED_AFTER and not any(r.get("ended") == "won" for r in runs[self.at_run:]):
            return
        after = runs[self.at_run:]
        helped = _better(after, self.before)
        WhatHelped.of(self.thing).came_of(self.thing, self.counsel, helped)
        _tell("That helped; I'll remember it." if helped else "Going by that didn't help, so I won't count on it again.")
        logger.info("counsel for %r %s: %s", self.thing, "helped" if helped else "did not help", self.counsel.told[:200])
        self.counsel = None


def _tell(line: str) -> None:
    from core.skills.screen_pursuit import _tell as said

    said(line)
