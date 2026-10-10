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
AT_MOST = 4
JUDGED_AFTER = 2


def the_thing(goal: str, title: str = "") -> str:
    """The thing she is in, by its own name: as the goal names it ("It is “X”"), else as its page's title begins."""
    named = re.search(r"[“\"]([^”\"]{3,120})[”\"]", str(goal or ""))
    if named:
        return named.group(1).strip()
    # A title's own separators first ("Name : Publisher : Site", "Name | Site"); a dash only where there is none, for a
    # dash is often inside a name ("The Powerpuff Girls - Attack Of The Puppybots : Cartoon Network").
    title = str(title or "").strip()
    for separator in (r"\s+[:|]\s+", r"\s+[–—-]\s+"):
        parts = re.split(separator, title)
        if len(parts) > 1:
            return parts[0][:120]
    return title[:120]


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
    #: Where the thing is: its record there is one of what others wrote of it (core/skills/what_others_wrote.py).
    url: str = ""
    taken: int = 0
    counsel: Any = None
    at_run: int = 0
    before: list[dict[str, Any]] = field(default_factory=list)
    #: What play was going by before the counsel in force was added: put back if that counsel does not help.
    counsel_before: str = ""

    def remembered(self) -> bool:
        """Whether she has gone by counsel here before that helped: then she begins with it, not by asking again."""
        from core.cognition.taking_stock import WhatHelped

        helped = WhatHelped.of(self.thing).helped if self.thing else []
        return bool(helped)

    def due(self, runs: list[dict[str, Any]]) -> str:
        """FAILING where the last two runs were lost and no counsel is still being tried; else ''."""
        from core.cognition.taking_stock import FAILING

        if self.taken >= AT_MOST or len(runs) < 2 or self.counsel is not None and len(runs) - self.at_run < JUDGED_AFTER + 1:
            return ""
        return FAILING if all(r.get("ended") == "lost" for r in runs[-2:]) else ""

    async def take(self, why: str, goal: str, words: list[str], ended: str, runs: list[dict[str, Any]], keep: dict[str, Any],
                   deadline: float) -> bool:
        """Stop, ask, and put what is worth going by where play will read it; whether anything was found."""
        from core.brain.llm.thinking_reserve import seconds_to_decode
        from core.cognition.taking_stock import (
            BEFORE,
            Situation,
            WhatHelped,
            from_her_corpus,
            from_her_memory,
            from_her_model,
            from_the_web,
            from_what_others_wrote,
            questions_for,
            take_stock,
        )
        from core.rebuilding.her_model import ask_her_model
        from core.skills.looking_it_up import search_and_read
        from core.skills.sovereign_browser_going import _named_on_its_page, _names_it
        from core.skills.what_others_wrote import what_others_wrote

        # The thing waits for her while she takes stock: she takes the time her model needs to be asked too, within a
        # share of what is left to play. LIVE 2026-10-10 a fixed 25 s left her model out of every stock she took, it
        # needing 35, and nothing else had anything to say of the game.
        her_model_needs = float(seconds_to_decode(220) or 0.0) * 1.2 + 5.0
        seconds = min(max(TAKING_STOCK_S, her_model_needs), max(0.0, deadline - time.monotonic()) * AT_MOST_OF_WHAT_IS_LEFT)
        if seconds < 8.0 or not self.thing:
            return False
        self.taken += 1
        kind = "game" if re.search(r"\b(?:games?|play)\b", goal, re.I) else ""
        situation = Situation(self.thing, goal, tuple(words[-6:]), ended, why, kind=kind)
        helped = WhatHelped.of(self.thing)
        # Said as a person says it, not as a search is typed.
        _tell(f"Before I begin, I'm looking up how {self.thing} is played." if why == BEFORE
              else f"{why}, so I'm taking stock: {questions_for(situation)[0]}?")
        sources = {
            "what helped me before": helped.source(),
            "what I remember": from_her_memory(),
            "my own copy of Wikipedia": from_her_corpus(thing=self.thing, names_it=_names_it),
            "the web": from_the_web(search_and_read, thing=self.thing, names_it=_names_it, kind=kind,
                                    on_its_page=_named_on_its_page),
            # And where people wrote of it: its record where it is kept, its fans' wiki, players' questions.
            "what others wrote": from_what_others_wrote(lambda seconds: what_others_wrote(
                self.thing, seconds, url=self.url, kind=kind, on_its_page=_named_on_its_page)),
            "my model": from_her_model(ask_her_model),
        }
        counsel = helped.without_what_did_not(await take_stock(situation, sources, seconds=seconds))
        # Before play what was found goes into the guide to the place, which says it whole; after, it is counsel.
        _tell(counsel.gathered() if why == BEFORE else counsel.said())
        if not counsel:
            return False
        self.counsel, self.at_run, self.before = counsel, len(runs), list(runs[-3:])
        self.counsel_before = str(keep.get("counsel") or "")
        # Read with the screen's own words from the next run on; what was found before is kept beside it.
        keep["counsel"] = " ".join(t for t in dict.fromkeys([str(keep.get("counsel") or ""), counsel.told]) if t)
        return True

    def judge(self, runs: list[dict[str, Any]], keep: dict[str, Any] | None = None) -> None:
        """Once the counsel has had its runs, whether it helped: said, and kept for next time. Counsel that did not help
        is no longer read by play: what was in force before it is put back."""
        from core.cognition.taking_stock import WhatHelped

        if self.counsel is None or len(runs) - self.at_run < JUDGED_AFTER and not any(r.get("ended") == "won" for r in runs[self.at_run:]):
            return
        after = runs[self.at_run:]
        if not self.before and not any(r.get("ended") == "won" for r in after):
            # Counsel taken before the first run has nothing to be held against but a win; it is kept in force, unjudged.
            self.counsel = None
            return
        helped = _better(after, self.before)
        WhatHelped.of(self.thing).came_of(self.thing, self.counsel, helped)
        _tell("That helped; I'll remember it." if helped else "Going by that didn't help, so I won't count on it again.")
        if not helped and keep is not None:
            keep["counsel"] = self.counsel_before
        logger.info("counsel for %r %s: %s", self.thing, "helped" if helped else "did not help", self.counsel.told[:200])
        self.counsel = None


def _tell(line: str) -> None:
    from core.skills.screen_pursuit import _tell as said

    said(line)
