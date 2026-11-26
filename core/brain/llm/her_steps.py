"""Which of her tools a tool loop calls next, and when the work is done: hers to decide, not her model's.

Bryan, 24 November 2026: "the 27b is an organ. not the central hub. aura should be driving things. not her model."
Her model's tool loop was handed every tool a turn offered and chose which to call, with what, in what order, and when
to stop: her model drove. Now each pass of the loop offers her model one tool, the one her steps hold next, and her model
writes that call's arguments, as an organ of hers fills in what she has decided to do. What comes back is weighed here,
and her steps say whether to try it again, go on to the next, or answer.

- A procedure (the default): the steps her caller decided, in its order. A turn's are the capabilities her reading of
  the request chose; a goal's are her way of changing a body of work (what went wrong, where, read it, change it, check
  it, offer it). A step that needs another (``needs``) is taken only once that one came back done.
- A choice (``choose_by_meaning``): where a caller holds a set of tools and no order, the next is the one her own
  embedding of what she is doing, and of what came back last, puts nearest (core/brain/requests_by_meaning.py). Where
  that reading is not there, her caller's order stands: never her model's choice.

Each step is tried at most ``TRIES`` times: a call that came back not done is tried once more with what it came back
with in view. A step her model writes no call for is let go: what it takes could not be written from what is known, and
a turn somebody is waiting on is not spent asking again. Nothing here knows a tool.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.HerSteps")

__all__ = ["TRIES", "UNWRITTEN_TRIES", "HerSteps", "came_back_done"]

#: How many times one step is tried: once, and once more with what came back in view.
TRIES = 2
#: How many passes a step her model writes no call for is given.
UNWRITTEN_TRIES = 1


def came_back_done(result: Any) -> bool:
    """Whether a call came back done: its own word for it where it gives one."""
    if isinstance(result, Mapping):
        if "ok" in result:
            return bool(result.get("ok"))
        status = str(result.get("status") or "").lower()
        return bool(status) and status not in {"error", "failed", "refused", "denied", "blocked"}
    return result is not None


@dataclass
class HerSteps:
    """Her steps for one tool loop."""

    tools: list[str]
    objective: str = ""
    choose_by_meaning: bool = False
    #: A step and the step it needs to have come back done before it is taken.
    needs: dict[str, str] = field(default_factory=dict)
    #: What each tool does, for choosing by meaning.
    described: dict[str, str] = field(default_factory=dict)
    tries: dict[str, int] = field(default_factory=dict)
    done: set[str] = field(default_factory=set)
    finished: set[str] = field(default_factory=set)
    taken: list[str] = field(default_factory=list)

    @classmethod
    def for_loop(cls, tools: Mapping[str, Any] | None, objective: str = "", *, order: Iterable[str] | None = None,
                 choose_by_meaning: bool = False, needs: Mapping[str, str] | None = None,
                 already: Iterable[tuple[str, bool]] = (), let_go: Iterable[str] = ()) -> HerSteps | None:
        """Her steps over ``tools`` (None where none are offered), in ``order`` where given, with the calls of this
        piece of work already made (``already``: each tool and whether it came back done) counted, and the steps
        already let go in it (``let_go``) not offered again."""
        offered = dict(tools or {})
        if not offered:
            return None
        named = [name for name in (order or ()) if name in offered]
        named += [name for name in offered if name not in named]
        steps = cls(named, objective=str(objective or ""), choose_by_meaning=choose_by_meaning,
                    needs={k: v for k, v in dict(needs or {}).items() if k in offered and v in offered},
                    described={name: str((offered[name] or {}).get("description") or "")
                               if isinstance(offered[name], Mapping) else "" for name in named})
        for name, ok in already:
            if name in offered:
                steps.called(name, ok)
        steps.finished.update(name for name in let_go if name in offered)
        return steps

    def let_go(self) -> list[str]:
        """The steps let go without coming back done: for a piece of work that goes on across loops."""
        return [name for name in self.tools if name in self.finished and name not in self.done]

    def _open(self) -> list[str]:
        open_steps = []
        for name in self.tools:
            if name in self.finished:
                continue
            needed = self.needs.get(name)
            if needed and needed not in self.done:
                if needed in self.finished:
                    # What it waits on was let go: it is let go too.
                    self.finished.add(name)
                continue
            open_steps.append(name)
        return open_steps

    def next(self, last: str = "") -> str | None:
        """The step to take now; None when her steps are done and she answers."""
        open_steps = self._open()
        if not open_steps:
            return None
        if self.choose_by_meaning and len(open_steps) > 1:
            chosen = self._by_meaning(open_steps, last)
            if chosen:
                self.taken.append(chosen)
                logger.info("her step %d: %s, by her reading of what she is doing", len(self.taken), chosen)
                return chosen
        chosen = open_steps[0]
        self.taken.append(chosen)
        logger.info("her step %d: %s, as her steps go", len(self.taken), chosen)
        return chosen

    def _by_meaning(self, open_steps: list[str], last: str) -> str | None:
        from core.brain.requests_by_meaning import nearest_by_meaning

        situation = " ".join(part for part in (self.objective, str(last or "")[-600:]) if part.strip())
        ranked = nearest_by_meaning(situation, {name: self.described.get(name, "") for name in open_steps})
        return ranked[0] if ranked else None

    def called(self, name: str, ok: bool) -> None:
        """A call of ``name`` came back, done or not."""
        self.tries[name] = self.tries.get(name, 0) + 1
        if ok:
            self.done.add(name)
            self.finished.add(name)
        elif self.tries[name] >= TRIES:
            self.finished.add(name)

    def not_filled(self, name: str) -> None:
        """Her model wrote no call for ``name``: what the step takes could not be written from what is known."""
        self.tries[name] = self.tries.get(name, 0) + 1
        logger.info("her model wrote no call for her step %s; it is let go", name)
        if self.tries[name] >= UNWRITTEN_TRIES:
            self.finished.add(name)
