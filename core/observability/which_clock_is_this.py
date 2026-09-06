"""Six clocks, and the arithmetic that must never mix them.

A timestamp on its own does not say what it can be compared against. Wall time
steps when NTP corrects it and jumps backwards across a suspend; monotonic
never goes back but means nothing between processes; and a simulation clock is
whatever the experiment says it is. Subtract one from another and you get a
plausible number that is wrong, which is the worst kind.

Aura reads the clock 5,018 times. Scanning all of it — inside expressions,
across the locals of a function, and per attribute of a class — found no place
where a value from one clock is subtracted from another. So the discipline is
already there; what is missing is that it is nowhere written down, which is
why the scans below stay as gates rather than as a one-off audit.

The six domains, and what each one is for:

* **wall** — when something happened, for a person or another machine. Steps.
* **monotonic** — how long something took, in this process. Never steps, and
  means nothing outside this process.
* **subjective** — her own sense of elapsed time, which runs at a rate
  cognition sets. Compare only against itself.
* **conversation** — turns, not seconds. A gap of an hour and a gap of a
  minute are the same distance here if nothing was said.
* **simulation** — an experiment's own clock, which may run fast, backwards,
  or stop.
* **model_budget** — tokens and inference seconds. The clock that decides
  whether an answer can be finished, and the only one a turn can run out of.

:func:`domains_with_no_reader` names the ones nothing reads. Four of the six
were built here; a domain nobody reads is a definition, not a clock.
"""
from __future__ import annotations

import ast
import functools
import logging
import pathlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.WhichClockIsThis")

__all__ = [
    "ClockDomain",
    "AStamp",
    "now",
    "how_long_between",
    "MixedClocks",
    "set_the_rate_of",
    "advance",
    "note_a_read",
    "domains_with_no_reader",
    "where_clocks_are_mixed",
    "how_the_clocks_stand",
]


class ClockDomain(StrEnum):
    """Which clock a number came from. Two numbers only subtract within one."""

    WALL = "wall"
    MONOTONIC = "monotonic"
    SUBJECTIVE = "subjective"
    CONVERSATION = "conversation"
    SIMULATION = "simulation"
    MODEL_BUDGET = "model_budget"


class MixedClocks(TypeError):
    """Two stamps from different domains were compared."""


@dataclass(frozen=True, slots=True, order=False)
class AStamp:
    """A number that knows which clock it came from."""

    at: float
    domain: ClockDomain

    def __sub__(self, other: "AStamp") -> float:
        if not isinstance(other, AStamp):
            return NotImplemented
        if other.domain is not self.domain:
            raise MixedClocks(
                f"{self.domain} minus {other.domain}: the difference would be a "
                "plausible number and a wrong one"
            )
        return self.at - other.at

    def __lt__(self, other: "AStamp") -> bool:
        if other.domain is not self.domain:
            raise MixedClocks(f"{self.domain} against {other.domain}")
        return self.at < other.at

    def __str__(self) -> str:
        return f"{self.at:.6f}@{self.domain}"


#: Rate 0 means the clock only moves when something advances it. That is the
#: whole point of the conversation clock: an hour of silence is no distance,
#: because nothing was said. The model budget is the same — it is spent, not
#: elapsed.
_RATES: dict[ClockDomain, float] = {
    ClockDomain.SUBJECTIVE: 1.0,
    ClockDomain.SIMULATION: 1.0,
    ClockDomain.CONVERSATION: 0.0,
    ClockDomain.MODEL_BUDGET: 0.0,
}
_OFFSETS: dict[ClockDomain, float] = {
    ClockDomain.SUBJECTIVE: 0.0,
    ClockDomain.CONVERSATION: 0.0,
    ClockDomain.SIMULATION: 0.0,
    ClockDomain.MODEL_BUDGET: 0.0,
}
_ANCHOR: dict[ClockDomain, float] = {}
_READS: dict[ClockDomain, int] = {}
_LOCK = threading.Lock()


def _reader_for(domain: ClockDomain) -> Callable[[], float]:
    if domain is ClockDomain.WALL:
        return time.time
    if domain is ClockDomain.MONOTONIC:
        return time.monotonic
    return lambda: _derived(domain)


def _derived(domain: ClockDomain) -> float:
    """A clock Aura drives itself: an anchor, a rate, and whatever was added."""
    with _LOCK:
        rate = _RATES.get(domain, 1.0)
        offset = _OFFSETS.get(domain, 0.0)
        anchor = _ANCHOR.setdefault(domain, time.monotonic())
    if rate == 0.0:
        return offset
    return offset + (time.monotonic() - anchor) * rate


def now(domain: ClockDomain = ClockDomain.MONOTONIC) -> AStamp:
    """Read a clock, and carry which one it was."""
    note_a_read(domain)
    return AStamp(at=_reader_for(domain)(), domain=domain)


def note_a_read(domain: ClockDomain) -> None:
    with _LOCK:
        _READS[domain] = _READS.get(domain, 0) + 1


def how_long_between(a: AStamp, b: AStamp) -> float:
    """b - a, refusing where the two came from different clocks."""
    return b - a


def set_the_rate_of(domain: ClockDomain, rate: float) -> None:
    """How fast a driven clock runs. 0 stops it; 2.0 runs it at double."""
    if domain in (ClockDomain.WALL, ClockDomain.MONOTONIC):
        raise ValueError(f"{domain} is the machine's; nothing here sets its rate")
    with _LOCK:
        # Freeze what has elapsed so far before the rate changes, or the
        # change rewrites the past as well as the future.
        anchor = _ANCHOR.setdefault(domain, time.monotonic())
        elapsed = (time.monotonic() - anchor) * _RATES.get(domain, 1.0)
        _OFFSETS[domain] = _OFFSETS.get(domain, 0.0) + elapsed
        _ANCHOR[domain] = time.monotonic()
        _RATES[domain] = float(rate)


def advance(domain: ClockDomain, by: float) -> None:
    """Move a driven clock on by hand. A turn ending, a simulation step."""
    if domain in (ClockDomain.WALL, ClockDomain.MONOTONIC):
        raise ValueError(f"{domain} cannot be advanced by hand")
    with _LOCK:
        _ANCHOR.setdefault(domain, time.monotonic())
        _OFFSETS[domain] = _OFFSETS.get(domain, 0.0) + float(by)


def domains_with_no_reader() -> tuple[str, ...]:
    """Domains nothing has read. A clock nobody reads is a definition."""
    with _LOCK:
        read = set(_READS)
    return tuple(sorted(str(d) for d in ClockDomain if d not in read))


# --- The gates ---------------------------------------------------------------

_WALL_CALLS = {"time"}
_MONO_CALLS = {"monotonic", "perf_counter"}
_ROOTS = ("core", "interface", "skills", "llm", "executors")


def _clock_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "time":
            if node.func.attr in _WALL_CALLS:
                return "wall"
            if node.func.attr in _MONO_CALLS:
                return "mono"
    return None


@functools.lru_cache(maxsize=1)
def where_clocks_are_mixed(repo: str = ".") -> dict[str, list[str]]:
    """Three scans, each for a way one clock reaches the other's arithmetic.

    ``in_one_expression`` subtracts the calls directly. ``through_a_local``
    goes via a variable inside one function. ``through_a_field`` is the same
    attribute of one class written from both, which is how a duration comes
    out wrong in a place neither write can see.
    """
    root = pathlib.Path(repo)
    found: dict[str, list[str]] = {
        "in_one_expression": [],
        "through_a_local": [],
        "through_a_field": [],
    }
    for name in _ROOTS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, ValueError, OSError):
                continue
            _scan_expressions(path, tree, found)
            _scan_locals(path, tree, found)
            _scan_fields(path, tree, found)
    return found


def _scan_expressions(path: pathlib.Path, tree: ast.AST, found: dict) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Add)):
            left, right = _clock_of(node.left), _clock_of(node.right)
            if left and right and left != right:
                found["in_one_expression"].append(f"{path}:{node.lineno} {left}/{right}")


def _scan_locals(path: pathlib.Path, tree: ast.AST, found: dict) -> None:
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        held: dict[str, str] = {}
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                clock = _clock_of(node.value)
                if clock:
                    held[node.targets[0].id] = clock
        if not held:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                def side(x: ast.AST) -> str | None:
                    return _clock_of(x) or (
                        held.get(x.id) if isinstance(x, ast.Name) else None
                    )

                left, right = side(node.left), side(node.right)
                if left and right and left != right:
                    found["through_a_local"].append(
                        f"{path}:{node.lineno} in {fn.name}(): {left} - {right}"
                    )


def _scan_fields(path: pathlib.Path, tree: ast.AST, found: dict) -> None:
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        written: dict[str, set[str]] = {}
        for node in ast.walk(cls):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "self"
            ):
                clock = _clock_of(node.value)
                if clock:
                    written.setdefault(node.targets[0].attr, set()).add(clock)
        for attr, clocks in sorted(written.items()):
            if len(clocks) > 1:
                found["through_a_field"].append(
                    f"{path}::{cls.name}.{attr} {sorted(clocks)}"
                )


def how_the_clocks_stand() -> dict[str, Any]:
    """For the health report: what was read, what was mixed, what nobody uses."""
    mixed = where_clocks_are_mixed()
    with _LOCK:
        reads = {str(d): n for d, n in sorted(_READS.items())}
        rates = {str(d): r for d, r in sorted(_RATES.items())}
    return {
        "domains": [str(d) for d in ClockDomain],
        "reads": reads,
        "rates": rates,
        "with_no_reader": list(domains_with_no_reader()),
        "mixed": {k: len(v) for k, v in mixed.items()},
        "mixed_at": {k: v[:10] for k, v in mixed.items() if v},
        "clean": not any(mixed.values()),
    }
