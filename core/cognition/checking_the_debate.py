"""A second look at what she is about to do, against the guide to where she is, learned from what her acts did.

Her deliberation weighs the acts open to her by what they might find out and
where they have led. It does not know what the place is: that a label is a
score to read and not a button, that a corner icon pauses the game, that the
place is worked with the mouse and a key she is about to press does nothing in
it, that a "Buy" button is not hers to press. The guide to the place
(core/cognition/a_guide_to_a_place.py) does. This checks each act against it:
a handful of plain facts about the act and the place, weighed by a logistic
model whose weights start from what those facts mean and are learned from what
the acts she took then did. What it learns is about acts and guides, not about
any place, and it is kept for every place she goes.

It does not choose. It scales what her deliberation already valued, and where
its view and hers part on the act she favours most, it says so in her log.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = ["DebateCheck", "the_check"]

logger = logging.getLogger("Aura.CheckingTheDebate")

#: What each fact about an act is, in order: the model's inputs.
FACTS = ("always", "a control it names", "a way on", "serves a goal", "information to read", "not part of the task",
         "a key where the mouse works it", "past a boundary", "did nothing lately", "what I've noticed of it",
         "fits what the place is", "makes no sense there", "keeps what was built", "throws away what was built")

#: What the facts mean before anything is learned: the weights she starts from.
STARTING = (0.0, 1.0, 1.0, 0.8, -2.0, -1.5, -1.5, -3.0, -1.0, 1.0, 0.6, -1.2, 0.8, -1.0)

#: How far one outcome moves the weights, and how far learning may take a weight from where it started.
LEARNING_RATE = 0.05
DRIFT = 2.5

#: The least and most an act's value is scaled by.
LEAST, MOST = 0.25, 1.75


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


@dataclass
class DebateCheck:
    """The weights of the check, and the facts of each act it last scored, for learning from what the act did."""

    weights: list[float] = field(default_factory=lambda: list(STARTING))
    scored: dict[str, tuple[float, ...]] = field(default_factory=dict)
    seen: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def facts(self, act: str, guide: Any, can_do: Any = None) -> tuple[float, ...]:
        """The facts about one act in the place the guide describes, as 0 or 1."""
        from core.agency.what_i_can_do_here import THE_PICTURE, what_is_clicked
        from core.language.a_way_on import how_much_it_leads_on

        label = what_is_clicked(act) or ""
        key = "" if label else str(act or "").lower()
        keys = set(guide.keys_for_play()) if guide is not None else set()
        pointer = bool(guide is not None and guide.pointer_named())
        named = bool(key and key in keys) or bool(label and pointer and (label == THE_PICTURE or not label.startswith("the ")))
        way_on = bool(label) and how_much_it_leads_on(label) >= 1.4
        goal_words = {w for g in (guide.goals if guide is not None else []) for w in re.findall(r"[a-z]{4,}", g.lower())}
        serves = bool(label) and bool(goal_words & set(re.findall(r"[a-z]{4,}", label.lower())))
        to_read = bool(label) and guide is not None and guide.is_to_read(label) and not way_on
        in_play = guide is not None and any(guide.in_play(n) for n in ("steering", "shooting", "jumping", "thrust",
                                                                        "timing a press", "clicking things"))
        aside = bool(label) and in_play and guide.is_not_for_play(label)
        mouse_only = bool(key) and pointer and not keys
        boundary = bool(label) and guide is not None and _past_a_boundary(label, guide)
        quiet = act in (getattr(can_do, "quiet_since", ()) or ())
        # What her theories say of it: +1 where she has come to think it pays, -1 where it costs (what_she_notices.py).
        notes = getattr(guide, "notes", None) if guide is not None else None
        noticed = notes.paying(key or ("click" if label else "")) if notes is not None else 0
        # What the place is (core/cognition/what_this_place_is.py): an act that fits it, and one that makes no sense
        # there (steering keys in a program for making music), unless the place's own words name it.
        from core.cognition.what_this_place_is import how_an_act_fits

        fits, senseless = how_an_act_fits(guide, label, key)
        senseless = senseless and not named
        # Where she has built something here (a part carried and kept), an option that keeps it is worth more than one
        # that throws it away: a person who built a trap that failed edits it (core/cognition/reading_the_rules.py).
        from core.cognition.reading_the_rules import keeps_the_work

        # Built in this pass, or before it: what she has put in place is kept on the guide across runs, and the screen
        # after a failed test is the start of the next run (LIVE 2026-10-10 she chose "start over" there seven times).
        built = bool(label or key) and (any(getattr(can_do, "carried_to", {}).values()) or bool(getattr(guide, "built", "")))
        # Her own "start over", where a way to begin again is offered, is an option like a label that says so.
        keeps = keeps_the_work(label or key) if built else None
        return tuple(float(f) for f in (1, named, way_on, serves, to_read, aside, mouse_only, boundary, quiet, noticed,
                                        fits, senseless, keeps is True, keeps is False))

    def weigh(self, valued: Mapping[str, float], guide: Any, can_do: Any = None) -> dict[str, float]:
        """What her deliberation valued, each act scaled by how the check sees it; a parting of views logged."""
        if guide is None or not valued:
            return dict(valued)
        out: dict[str, float] = {}
        views: dict[str, float] = {}
        with self.lock:
            for act, value in valued.items():
                facts = self.facts(act, guide, can_do)
                self.scored[act] = facts
                p = _sigmoid(sum(w * f for w, f in zip(self.weights, facts, strict=True)))
                views[act] = p
                out[act] = value * (LEAST + (MOST - LEAST) * p)
        hers = max(valued, key=lambda a: valued[a])
        mine = max(out, key=lambda a: out[a])
        if hers != mine and views[hers] < 0.35:
            logger.info("the check parts from her deliberation: %r (%s) over %r (%s)", mine,
                        ", ".join(n for n, f in zip(FACTS, self.scored[mine], strict=True) if f and n != "always"),
                        hers, ", ".join(n for n, f in zip(FACTS, self.scored[hers], strict=True) if f and n != "always"))
        return out

    def learned(self, act: str, answered: bool) -> None:
        """What an act she took did: the weights moved toward what it says (an act that answered was worth taking)."""
        with self.lock:
            facts = self.scored.pop(act, None)
            if facts is None:
                return
            p = _sigmoid(sum(w * f for w, f in zip(self.weights, facts, strict=True)))
            for i, f in enumerate(facts):
                if f:
                    moved = self.weights[i] + LEARNING_RATE * ((1.0 if answered else 0.0) - p) * f
                    self.weights[i] = max(STARTING[i] - DRIFT, min(STARTING[i] + DRIFT, moved))
            self.seen += 1
            if self.seen % 25 == 0:
                self._keep()

    def _keep(self) -> None:
        from core.runtime.what_she_learned import named, remember

        remember(named("what holds everywhere", "checking the debate"), {"weights": self.weights, "seen": self.seen})


def _past_a_boundary(label: str, guide: Any) -> bool:
    """Whether a label is one of the acts that are the person's to do: signing in, a bot check, what cannot be undone."""
    from core.agency.mechanics_she_knows import BOUNDARY, MECHANICS

    return any(m.kind == BOUNDARY and m.said_in(label) for m in MECHANICS)


_CHECKS: dict[str, DebateCheck] = {}
_MADE = checked_lock("checking_the_debate.made")


def the_check() -> DebateCheck:
    """The one check she has, with the weights she has learned anywhere (one per place her learning is kept in)."""
    from core.runtime.what_she_learned import _kept_in, named, recall

    where = str(_kept_in())
    with _MADE:
        if where not in _CHECKS:
            held = recall(named("what holds everywhere", "checking the debate")) or {}
            weights = list(held.get("weights") or STARTING) if isinstance(held, dict) else list(STARTING)
            # Weights learned before a fact was added keep what they learned; the new facts start where they mean.
            weights = weights + list(STARTING[len(weights):]) if len(weights) < len(STARTING) else weights
            _CHECKS[where] = DebateCheck(weights=weights if len(weights) == len(STARTING) else list(STARTING),
                                         seen=int(held.get("seen") or 0) if isinstance(held, dict) else 0)
        return _CHECKS[where]
