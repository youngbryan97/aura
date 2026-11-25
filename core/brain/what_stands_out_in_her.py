"""Which of her own readings stand out from what they usually are.

Her live state went to her model as JSON with its keys in alphabetical order, cut at 2,600 characters. What came first by name
went on every turn and what came after never did: LIVE 25 November 2026 the cut fell inside ``mind_snapshot``, so of her
substrate, her voice and the time her turns take nothing arrived, and about a thousand tokens of it, read in three seconds, were
the same from one turn to the next.

A reading that is what it always is tells her model nothing. How much one tells is measured: for a reading that is not a number,
from how often it has been this; for a number, from how far it is from its own usual, in its own usual spread; for one that has
only ever gone one way, as a clock and a count do, from how far it moved. What tells most goes first, until the room is taken.
A turn about her is given her readings whole, what tells most first.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

__all__ = ["TELLS", "HerUsual", "get_her_usual", "what_stands_out"]

logger = logging.getLogger("Aura.WhatStandsOut")

#: How many turns a usual is taken over: the window the rest of her record of herself is read over.
WINDOW = 256
#: What a reading has to tell to be given, in bits: more than one this likely as not would.
TELLS = 1.5
#: How many values of one reading are told apart.
VALUES = 32

Path_ = tuple[str, ...]


def leaves(reading: Any, at: Path_ = ()) -> Iterator[tuple[Path_, Any]]:
    """Every value in ``reading`` that holds no other, with the way to it. What is empty, and what says only that it was cut, is none."""
    if isinstance(reading, Mapping):
        for key, value in reading.items():
            if not str(key).startswith("_"):
                yield from leaves(value, (*at, str(key)))
    elif isinstance(reading, (list, tuple)):
        held = [item for item in reading if not isinstance(item, (Mapping, list, tuple))]
        if held and len(held) == len(reading):
            yield at, list(reading)
        else:
            for index, item in enumerate(reading):
                yield from leaves(item, (*at, str(index)))
    elif reading is not None and reading != "":
        yield at, reading


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


class HerUsual:
    """What each of her readings usually is, learned from the turns she has had."""

    def __init__(self, keep: Path | None = None) -> None:
        self._keep = keep
        self._lock = checked_lock("what_stands_out_in_her")
        self._held: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        if self._keep is None:
            return {}
        try:
            held = json.loads(self._keep.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            record_degradation("what_stands_out", exc, severity="warning", action="what her readings usually are is learned again")
            return {}
        return {str(k): v for k, v in held.items() if isinstance(v, dict)} if isinstance(held, dict) else {}

    def tells(self, at: Path_, value: Any) -> float:
        """How much this reading tells, in bits, by what it has usually been."""
        with self._lock:
            held = self._held.get(".".join(at))
        if held is None:
            return 1.0
        number = _number(value)
        if number is None or "mean" not in held:
            seen, of = held.get("values") or {}, float(held.get("n", 0))
            return -math.log2((float(seen.get(self._as_key(value), 0.0)) + 1.0) / (of + 2.0))
        if held["n"] < 4:
            return 1.0
        one_way = not (held.get("ups") and held.get("downs")) and (held.get("ups") or held.get("downs"))
        if one_way:
            number, mean, var = number - held["last"], held["step"], held["step_var"]
        else:
            mean, var = held["mean"], held["var"]
        spread = max(math.sqrt(max(var, 0.0)), 1e-3 * abs(mean), 1e-6)
        away = (number - mean) / spread
        return 0.5 * away * away / math.log(2)

    def learn(self, reading: Any) -> None:
        """One more turn's readings, into what they usually are."""
        with self._lock:
            for at, value in leaves(reading):
                self._one(".".join(at), value)

    def _one(self, name: str, value: Any) -> None:
        held = self._held.setdefault(name, {"n": 0})
        held["n"] = n = min(int(held["n"]) + 1, WINDOW)
        rate = 1.0 / n
        number = _number(value)
        if number is None:
            values = held.setdefault("values", {})
            for key in list(values):
                values[key] *= 1.0 - (rate if n >= WINDOW else 0.0)
            key = self._as_key(value)
            values[key] = values.get(key, 0.0) + 1.0
            while len(values) > VALUES:
                values.pop(min(values, key=values.get))
            return
        if "mean" not in held:
            held.update(mean=number, var=0.0, last=number, step=0.0, step_var=0.0, ups=0, downs=0)
            return
        step = number - held["last"]
        held["ups"], held["downs"] = held["ups"] + (step > 0), held["downs"] + (step < 0)
        for mean, var, x in (("mean", "var", number), ("step", "step_var", step)):
            off = x - held[mean]
            held[mean] += rate * off
            held[var] = (1.0 - rate) * (held[var] + rate * off * off)
        held["last"] = number

    @staticmethod
    def _as_key(value: Any) -> str:
        return json.dumps(value, default=str, ensure_ascii=True)[:80]

    def keep(self) -> None:
        """Kept for the next run, off the loop that is answering."""
        if self._keep is None:
            return
        with self._lock:
            held = json.dumps(self._held)
        try:
            asyncio.get_running_loop().run_in_executor(None, self._write, held)
        # not a failure: nobody is being answered on a loop here, so it is written now.
        except RuntimeError:
            self._write(held)

    def _write(self, held: str) -> None:
        try:
            from core.runtime.her_writes import governed

            if not self._keep.parent.is_dir():
                governed("ensure_directory", self._keep.parent, source="what_stands_out")
            governed("write_text", self._keep, held, source="what_stands_out")
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            record_degradation("what_stands_out", exc, severity="warning", action="what her readings usually are is held in memory only")


def _short(value: Any) -> Any:
    """The same reading in three figures."""
    number = _number(value)
    if number is not None and isinstance(value, float):
        return float(f"{number:.3g}")
    if isinstance(value, list):
        return [_short(item) for item in value]
    return value


def _put(into: dict[str, Any], at: Path_, value: Any) -> None:
    for key in at[:-1]:
        into = into.setdefault(key, {})
    into[at[-1]] = value


def _written(reading: dict[str, Any]) -> str:
    return json.dumps(reading, separators=(",", ":"), default=str, ensure_ascii=True)


def what_stands_out(reading: Mapping[str, Any], usual: HerUsual, *, room: int, whole: bool = False) -> str:
    """Her readings that tell most, as data, within ``room`` characters; all that fit, when ``whole``."""
    telling = [(usual.tells(at, value), index, at, value) for index, (at, value) in enumerate(leaves(reading)) if at]
    given: dict[str, Any] = {}
    for bits, _index, at, value in sorted(telling, key=lambda told: (-told[0], told[1])):
        if not whole and bits < TELLS:
            break
        tried = json.loads(_written(given))
        _put(tried, at, _short(value))
        if len(_written(tried)) <= room:
            given = tried
    return _written(given) if given else ""


_HERS: HerUsual | None = None


def get_her_usual() -> HerUsual:
    global _HERS
    if _HERS is None:
        from core.runtime.state_ownership import state_root

        _HERS = HerUsual(state_root() / "data" / "brain" / "her_usual.json")
    return _HERS
