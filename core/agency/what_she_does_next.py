"""What she does next when nothing is asked of her: chosen by her from what could be done, and judged by what came of it.

Measured 24 November 2026 over seven hours: of about 220 choices she made by herself, 205 were one string, "Acting on
curiosity: explore_knowledge". A drive had become an option. Nothing named what would be done, nothing was made by it, and
nothing that came of it was learned from, so it won again. The same was measured on 13 October (379 of 500).

Here a drive is never an option. What she chooses among are acts: each names what is done and has something that does it,
and says what it is for and what put it there. What each is worth is learned from what came of the ones before it, as a
linear model of what is known about an act before it is done, held as a distribution and not as a number: an act is chosen
by what a draw from that distribution says it would be worth (Thompson sampling, Thompson 1933; for linear models Agrawal
and Goyal 2013). So what has paid is done more, what has come to nothing is done less, and what she is unsure of is tried
because she is unsure of it. Old outcomes count for less as new ones come, because what is worth doing changes.

Doing nothing is an act too, and worth nothing: an act that is expected to come to less than nothing is not done.

Every choice is kept with its grounds: what there was to choose from, what each was expected to be worth and how unsure
that was, and what about it counted most. So is what came of it. That record is what tells whether she is choosing well.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.WhatSheDoesNext")

__all__ = ["CameOf", "Chosen", "Option", "REST", "WhatSheDoesNext", "get_what_she_does_next", "reset_for_test"]

#: How sure she is of an act's worth before any has been done: worth lies in [-1, 1], and this is the spread of a number
#: that could be anywhere in it. And how far one outcome is from what the act is worth, as far as is known beforehand.
UNSURE, NOISE = 1.0 / 3.0, 1.0 / 3.0
#: How unsure she is at the start of what acts of every sort have in common: a quarter of it, since what one sort of act comes
#: to says less of another sort than of its own. LIVE 25 November 2026, held as unsure of both, one search that came to nothing
#: took as much off every goal, request and telling as off looking things up, and she rested.
SHARED = UNSURE / 4.0
#: How much the outcomes before still count when a new one comes: the last two or three hundred acts are what is learned from.
KEEPS = 0.995
#: How many choices are kept with their grounds.
KEPT = 400
#: How much an outcome still counts once the way such an act is done has changed: it may have been the way that failed, and
#: what is left of it is what does not turn on the way. LIVE 25 November 2026 nine acts that came to nothing, done by what has
#: since been mended, had her expect nothing of anything, and rest.
DONE_ANOTHER_WAY = 0.25

REST = "rest"


@dataclass(frozen=True)
class CameOf:
    """What came of an act."""

    #: How far it got with what it set out to do, 0 to 1, by what can be seen of it and not by what is said of it.
    got: float = 0.0
    #: What it was worth to her or to who it was for, where something measured that: [-1, 1].
    worth: float | None = None
    #: Whether something stopped it that was not its own: a person speaking, the machine short of room.
    put_down: bool = False
    what: str = ""
    seconds: float = 0.0

    def reward(self, cost_s: float) -> float:
        """What it comes to in all: how far it got, what it was worth where that was measured, and what it took."""
        came = 2.0 * self.got - 1.0 if self.worth is None else 0.5 * (2.0 * self.got - 1.0) + 0.5 * self.worth
        took = 0.1 * math.tanh(max(0.0, self.seconds) / max(60.0, 4.0 * cost_s))
        return max(-1.0, min(1.0, came - took))


@dataclass(frozen=True)
class Option:
    """Something she could do."""

    id: str
    #: What sort of act it is, which what is learned of acts is learned by: ``carry_on``, ``find_out``, ``make``, ``tell``, ...
    kind: str
    #: What she would do, in a sentence.
    what: str
    #: What put it there, as it was read.
    why: str = ""
    #: What is known of it before it is done, each 0 to 1: how much it is wanted, how much is waiting on it, how new it is.
    known: Mapping[str, float] = field(default_factory=dict)
    #: The drives it would feed, and how much each is asking now.
    drives: Mapping[str, float] = field(default_factory=dict)
    #: How long it takes, in seconds, as far as is known.
    cost_s: float = 60.0
    does: Callable[[], Awaitable[CameOf]] | None = field(default=None, compare=False, repr=False)
    #: Whether it is owed to somebody and comes before what is chosen.
    owed: bool = False
    #: How such an act is done now, as a mark that is another one when that changes; "" where it is not told.
    way: str = ""

    def features(self) -> dict[str, float]:
        """What of it what it is worth is learned from."""
        out = {"": 1.0, f"kind:{self.kind}": 1.0}
        out.update({f"known:{name}": float(value) for name, value in self.known.items()})
        out.update({f"{self.kind}/{name}": float(value) for name, value in self.known.items()})
        out.update({f"{self.kind}<{name}": float(value) for name, value in self.drives.items()})
        out["takes"] = math.log1p(max(0.0, self.cost_s) / 60.0) / 5.0
        return out


@dataclass
class Chosen:
    """A choice of hers, with its grounds, and what came of it."""

    at: float
    id: str
    kind: str
    what: str
    why: str
    #: What it was expected to be worth, how unsure that was, and what the draw said.
    expected: float
    unsure: float
    drawn: float
    #: What else there was, the likeliest first: what, its expected worth, how unsure, and the draw.
    over: list[dict[str, Any]] = field(default_factory=list)
    #: What about it counted most, and how much.
    counted: list[tuple[str, float]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    came: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    way: str = ""

    def says(self) -> str:
        """The grounds in a sentence, from what was measured."""
        second = self.over[0] if self.over else None
        best = ", ".join(f"{name.replace('kind:', '').replace('known:', '').replace('<', ' for ').replace('/', ' by how ')} {value:+.2f}" for name, value in self.counted[:3] if name)
        return (f"{self.what}: expected {self.expected:+.2f} (give or take {self.unsure:.2f})"
                + (f", over {second['what']} at {second['expected']:+.2f}" if second else "") + (f". What counted: {best}." if best else "."))


def _unsure_of(feature: str) -> float:
    """How unsure she is at the start of what a feature is worth: less of what every sort of act has."""
    return UNSURE if feature.startswith("kind:") or "/" in feature or "<" in feature else SHARED


class WhatSheDoesNext:
    """What she has learned of what acts are worth, and her choices by it. Kept, and the same after she has been closed."""

    def __init__(self, path: Path | None = None, *, seed: int | None = None) -> None:
        self._path = path
        self._lock = checked_lock("agency.what_she_does_next")
        self._names: dict[str, int] = {}
        self._precision = np.zeros((0, 0))
        self._pulled = np.zeros(0)
        self._rng = np.random.default_rng(seed)
        self._choices: deque[Chosen] = deque(maxlen=KEPT)
        #: How each sort of act was done when it was last chosen among.
        self._ways: dict[str, str] = {}
        self._done = 0
        self._loaded = path is not None and not Path(path).is_file()

    # ── what is learned ──────────────────────────────────────────────

    def _vector(self, features: Mapping[str, float], *, grow: bool) -> np.ndarray:
        if grow:
            for name in features:
                if name not in self._names:
                    self._names[name] = len(self._names)
            n = len(self._names)
            if n > len(self._pulled):
                wider = self._at_the_start()
                wider[: len(self._pulled), : len(self._pulled)] = self._precision
                self._precision, self._pulled = wider, np.concatenate([self._pulled, np.zeros(n - len(self._pulled))])
        x = np.zeros(len(self._names))
        for name, value in features.items():
            at = self._names.get(name)
            if at is not None:
                x[at] = value
        return x

    def _at_the_start(self) -> np.ndarray:
        """How sure she is of each feature's worth before anything is done, as a precision."""
        return np.diag([1.0 / _unsure_of(name) for name in self._names])

    def _posterior(self) -> tuple[np.ndarray, np.ndarray]:
        spread = np.linalg.inv(self._precision) if len(self._pulled) else np.zeros((0, 0))
        return spread @ self._pulled, spread

    def expected(self, option: Option) -> tuple[float, float]:
        """What an act is expected to be worth, and how unsure that is."""
        if option.kind == REST:
            return 0.0, 0.0
        with self._lock:
            self._load()
            x = self._vector(option.features(), grow=False)
            mean, spread = self._posterior()
            unknown = sum(v * v * _unsure_of(n) for n, v in option.features().items() if n not in self._names)
            return float(x @ mean) if len(mean) else 0.0, math.sqrt(max(0.0, float(x @ spread @ x) if len(mean) else 0.0) + unknown)

    def learn(self, chosen: Chosen, came: CameOf, *, cost_s: float = 60.0) -> float | None:
        """What came of an act, learned from; what it came to in all. What something else put down says nothing of the act."""
        chosen.came = asdict(came)
        if came.put_down:
            return None
        reward = came.reward(cost_s)
        chosen.reward = reward
        if chosen.kind != REST:
            with self._lock:
                self._load()
                self._weigh(chosen.features, reward, 1.0)
                self._done += 1
        return reward

    def _weigh(self, features: Mapping[str, float], reward: float, counts: float) -> None:
        """One outcome, into what acts are expected to be worth, as much as it counts."""
        x = self._vector(features, grow=True)
        # What was learned before counts for a little less, and never for less than was known at the start.
        self._precision = KEEPS * self._precision + (1.0 - KEEPS) * self._at_the_start() + counts * np.outer(x, x) / NOISE
        self._pulled = KEEPS * self._pulled + counts * x * reward / NOISE

    def _as_they_are_done_now(self, options: Iterable[Option]) -> None:
        """Where the way a sort of act is done has changed, what it is expected to be worth is taken again from the outcomes kept,
        those of another way counting for less."""
        now = {option.kind: option.way for option in options if option.way}
        changed = sorted(kind for kind, way in now.items() if self._ways.get(kind) != way)
        if not changed:
            return
        self._ways.update(now)
        self._names, self._precision, self._pulled = {}, np.zeros((0, 0)), np.zeros(0)
        for chosen in self._choices:
            if chosen.reward is not None and chosen.kind != REST and chosen.features:
                same = self._ways.get(chosen.kind, chosen.way) == chosen.way
                self._weigh(chosen.features, chosen.reward, 1.0 if same else DONE_ANOTHER_WAY)
        logger.info("🧭 Done another way than when she last chose: %s; what came of it before counts for less.", ", ".join(changed))

    # ── choosing ─────────────────────────────────────────────────────

    def choose(self, options: Iterable[Option]) -> tuple[Option, Chosen] | None:
        """Choose among what could be done; what she chose, and the choice with its grounds. None where the best is to rest."""
        held = [option for option in options if option.does is not None and option.kind != REST]
        owed = [option for option in held if option.owed]
        held = owed or held
        if not held:
            return None
        rows: list[tuple[float, float, float, Option]] = []
        with self._lock:
            self._load()
            self._as_they_are_done_now(held)
            for option in held:
                self._vector(option.features(), grow=True)
            mean, spread = self._posterior()
            theta = self._rng.multivariate_normal(mean, spread, method="cholesky") if len(mean) else mean
            for option in held:
                x = self._vector(option.features(), grow=False)
                rows.append((float(x @ theta), float(x @ mean), math.sqrt(max(0.0, float(x @ spread @ x))), option))
            rows.sort(key=lambda row: -row[0])
            drawn, expected, unsure, best = rows[0]
            x = self._vector(best.features(), grow=False)
            counted = sorted(((name, float(x[at] * mean[at])) for name, at in self._names.items() if x[at]), key=lambda one: -abs(one[1]))[:6]
        chosen = Chosen(time.time(), best.id, best.kind, best.what, best.why, expected, unsure, drawn,
                        over=[{"id": o.id, "kind": o.kind, "what": o.what, "expected": round(e, 3), "unsure": round(u, 3), "drawn": round(d, 3)} for d, e, u, o in rows[1:7]],
                        counted=[(name, round(value, 3)) for name, value in counted], features=best.features(), way=best.way)
        # What has never been done is expected to come to nothing in particular, and is found out by doing it.
        if drawn <= 0.0 and expected < 0.0 and not best.owed:
            logger.info("🧭 She rests: the best of %d things to do, %s, would come to %+.2f.", len(held), best.what[:80], drawn)
            return None
        with self._lock:
            self._choices.append(chosen)
        logger.info("🧭 She chose, of %d: %s", len(held), chosen.says()[:400])
        return best, chosen

    # ── told, and kept ───────────────────────────────────────────────

    def choices(self, last: int = 40) -> list[Chosen]:
        with self._lock:
            self._load()
            return list(self._choices)[-last:]

    def how_she_chooses(self) -> dict[str, Any]:
        """What she has learned of what is worth doing, and whether she chooses better than she did: what came of her acts
        lately beside what came of them before, how many different things she does, and what each kind is expected to be worth."""
        with self._lock:
            self._load()
            mean, spread = self._posterior()
            held = [c for c in self._choices if c.reward is not None]
            kinds = {}
            for name, at in self._names.items():
                if name.startswith("kind:"):
                    kind = name[5:]
                    done = [c.reward for c in held if c.kind == kind]
                    kinds[kind] = {"expected": round(float(mean[at]) + (float(mean[self._names[""]]) if "" in self._names else 0.0), 3),
                                   "unsure": round(math.sqrt(max(0.0, float(spread[at, at]))), 3), "done": len(done),
                                   "came_to": round(float(np.mean(done)), 3) if done else None}
        half = len(held) // 2
        rewards = [c.reward for c in held]
        return {"done": self._done, "kinds": kinds, "different": len({c.id for c in held[-100:]}), "of": len(held[-100:]),
                "before": round(float(np.mean(rewards[:half])), 3) if half else None, "lately": round(float(np.mean(rewards[half:])), 3) if rewards[half:] else None}

    def _where(self) -> Path:
        if self._path is None:
            from core.runtime.state_ownership import state_root

            self._path = state_root() / "data" / "agency" / "what_she_does_next.json"
        return Path(self._path)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._where()
        if not path.is_file():
            return
        try:
            kept = json.loads(path.read_text(encoding="utf-8"))
            names = [str(name) for name in kept["names"]]
            precision, pulled = np.asarray(kept["precision"], dtype=float), np.asarray(kept["pulled"], dtype=float)
            if precision.shape != (len(names), len(names)) or pulled.shape != (len(names),) or not np.isfinite(precision).all():
                raise ValueError("what was kept does not fit together")
            self._names, self._precision, self._pulled = {name: i for i, name in enumerate(names)}, precision, pulled
            self._done = int(kept.get("done") or 0)
            self._ways = {str(kind): str(way) for kind, way in dict(kept.get("ways") or {}).items()}
            for row in kept.get("choices") or []:
                row["counted"] = [tuple(one) for one in row.get("counted") or ()]
                self._choices.append(Chosen(**row))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record_degradation("what_she_does_next", exc, severity="warning", action="what she had learned of what is worth doing is not held; it begins again")

    def keep(self) -> None:
        """Write what is held. Not on the event loop."""
        from core.rebuilding.her_build_writes import governed

        with self._lock:
            kept = {"names": list(self._names), "precision": np.round(self._precision, 9).tolist(), "pulled": np.round(self._pulled, 9).tolist(), "done": self._done, "ways": self._ways,
                    "choices": [asdict(c) for c in self._choices]}
        governed("ensure_directory", self._where().parent, source="agency.what_she_does_next")
        governed("write_text", self._where(), json.dumps(kept, ensure_ascii=False), source="agency.what_she_does_next")


_HELD: WhatSheDoesNext | None = None
_HELD_LOCK = checked_lock("agency.what_she_does_next.singleton")


def get_what_she_does_next() -> WhatSheDoesNext:
    global _HELD
    with _HELD_LOCK:
        if _HELD is None:
            _HELD = WhatSheDoesNext()
        return _HELD


def reset_for_test() -> None:
    global _HELD
    with _HELD_LOCK:
        _HELD = None
