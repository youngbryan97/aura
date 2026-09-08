"""Cutting a cell out and running the workload again.

Everything else in this package measures what varies with what. A rotation null
makes that honest, but honest correlation is still correlation: a weight from
``effective.py`` licenses the sentence "one cell's past helps predict another's
future" and nothing stronger. The object the docs keep naming,

    w_ij^eff(c) = Effect[do(i), j | c]

wants the ``do``. In a brain that means a laser and an opsin. Here it means
replacing a function with one that does nothing and running the same work
again, which is the one advantage this substrate has over the tissue ones: the
intervention is exact, it is reversible, and every cell is reachable.

What makes the result mean something is not the lesion, it is the control. Cut
any cell out of a running system and something downstream will move; the number
is only evidence if a comparable cell cut out of the same system does not move
the same thing. So every intervention here runs three arms — untouched, the
target silenced, and a degree-matched cell silenced — and reports the target's
effect against the control's rather than against zero.

The lesion never reaches the live runtime. It patches an attribute on an
imported module inside a context manager, restores it in a ``finally``, and
refuses outright unless the process is a test or the caller says in as many
words that it is not one.
"""

from __future__ import annotations

import importlib
import logging
import os
import random
import statistics
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.connectome.types import ConnectomeSnapshot

__all__ = [
    "ArmResult",
    "InterventionReport",
    "LesionRefusedError",
    "degree_matched_control",
    "run_intervention",
    "silence",
    "silenced_calls",
]

logger = logging.getLogger("Aura.Connectome.Intervene")


class LesionRefusedError(RuntimeError):
    """The cut was not made, and the reason is the message."""


#: How many times each silenced cell was called while it was silenced. Read
#: after the context manager exits. A lesion on a cell that was never called is
#: not a lesion, and without this it looks exactly like one that did nothing.
_SUPPRESSED: dict[str, int] = {}


def silenced_calls(uid: str) -> int:
    """Calls the silenced cell absorbed. Zero means the lesion never bit."""
    return _SUPPRESSED.get(uid, 0)


def _guard() -> None:
    if os.environ.get("AURA_TESTING") == "1":
        return
    if os.environ.get("AURA_ALLOW_LESION") == "1":
        return
    raise LesionRefusedError(
        "a lesion patches a live module attribute; set AURA_TESTING=1 for a test "
        "process, or AURA_ALLOW_LESION=1 to say deliberately that this one is not"
    )


def _resolve(uid: str) -> tuple[Any, str, Any]:
    """Find the owner object, the attribute name, and what is there now."""
    module_name, _, qualname = uid.partition(":")
    if not module_name or not qualname:
        raise LesionRefusedError(f"{uid!r} is not module:qualname")
    try:
        owner: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise LesionRefusedError(f"{module_name} will not import: {exc}") from exc
    parts = qualname.split(".")
    for part in parts[:-1]:
        if part == "<locals>":
            raise LesionRefusedError(f"{uid} is defined inside another function")
        owner = getattr(owner, part, None)
        if owner is None:
            raise LesionRefusedError(f"{uid} has no {part}")
    name = parts[-1]
    current = getattr(owner, name, None)
    if current is None:
        raise LesionRefusedError(f"{uid} is not there to cut")
    if not callable(current):
        raise LesionRefusedError(f"{uid} is not callable")
    return owner, name, current


@contextmanager
def silence(uid: str, *, returns: Any = None) -> Iterator[None]:
    """Replace one cell with one that does nothing, and put it back after.

    ``returns`` is what the silenced cell hands back. None is the honest default
    — a cell that has been cut out produces nothing — but a caller who knows the
    consumer cannot survive a None may pass the type's empty value instead, and
    the difference between those two runs is itself a measurement.

    An async cell is replaced by an async stub, because handing a coroutine's
    caller a plain value raises where the lesion should have been silent, and a
    lesion that crashes the workload measures the crash.
    """
    import inspect

    _guard()
    owner, name, original = _resolve(uid)
    _SUPPRESSED[uid] = 0

    if inspect.iscoroutinefunction(original):

        async def stub(*_args: Any, **_kwargs: Any) -> Any:
            _SUPPRESSED[uid] = _SUPPRESSED.get(uid, 0) + 1
            return returns

    else:

        def stub(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[misc]
            _SUPPRESSED[uid] = _SUPPRESSED.get(uid, 0) + 1
            return returns

    stub.__name__ = getattr(original, "__name__", name)
    stub.__qualname__ = getattr(original, "__qualname__", name)
    stub.__doc__ = f"silenced by core.connectome.intervene for {uid}"
    try:
        setattr(owner, name, stub)
    except (AttributeError, TypeError) as exc:
        raise LesionRefusedError(f"{uid} will not take a patch: {exc}") from exc
    try:
        yield
    finally:
        setattr(owner, name, original)


def degree_matched_control(
    snapshot: ConnectomeSnapshot,
    target: str,
    *,
    exclude: Sequence[str] = (),
    seed: int = 0,
) -> str:
    """A cell as connected as the target and not on the path being tested.

    Matched on in-degree and out-degree together, because a cell with the same
    number of callers and a tenth of the callees is not a comparable cut. Ties
    are broken by a seeded shuffle rather than by name, so the control is not
    always the alphabetically first cell in some crowded module.
    """
    degrees: dict[str, list[int]] = {}
    for (pre, post, _kind), connection in snapshot.connections.items():
        degrees.setdefault(pre, [0, 0])[1] += 1
        degrees.setdefault(post, [0, 0])[0] += 1
        del connection
    want = degrees.get(target)
    if want is None:
        raise LesionRefusedError(f"{target} is not in the snapshot")
    banned = {target, *exclude}
    candidates = [
        (abs(value[0] - want[0]) + abs(value[1] - want[1]), uid)
        for uid, value in degrees.items()
        if uid not in banned
    ]
    if not candidates:
        raise LesionRefusedError("no cell left to use as a control")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One arm of an intervention: what the readout said, over repeats."""

    arm: str
    target: str
    readout: dict[str, float]
    spread: dict[str, float]
    repeats: int
    seconds: float
    calls_suppressed: int = 0
    failures: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "target": self.target,
            "readout": {k: round(v, 6) for k, v in self.readout.items()},
            "spread": {k: round(v, 6) for k, v in self.spread.items()},
            "repeats": self.repeats,
            "seconds": round(self.seconds, 2),
            "calls_suppressed": self.calls_suppressed,
            "failures": self.failures,
        }


@dataclass(frozen=True, slots=True)
class InterventionReport:
    """What cutting the target did, next to what cutting a like cell did."""

    target: str
    control: str
    condition: str
    baseline: ArmResult
    lesioned: ArmResult
    control_arm: ArmResult
    predicted: tuple[str, ...] = ()
    spared: tuple[str, ...] = ()
    notes: str = ""
    effects: dict[str, float] = field(default_factory=dict)
    control_effects: dict[str, float] = field(default_factory=dict)

    @property
    def bit(self) -> bool:
        """Did the lesion touch anything at all?"""
        return self.lesioned.calls_suppressed > 0

    def verdict(self) -> str:
        if not self.bit:
            return (
                f"{self.target} was never called under this workload, so nothing was "
                "cut and no conclusion is available"
            )
        moved = [
            key
            for key, value in self.effects.items()
            if abs(value) > abs(self.control_effects.get(key, 0.0))
            and abs(value) > 1e-9
        ]
        if not moved:
            return (
                f"cutting {self.target} moved no readout further than cutting a "
                "degree-matched cell did"
            )
        hit = [key for key in self.predicted if key in moved]
        broke = [key for key in self.spared if key in moved]
        parts = [f"cutting {self.target} moved {', '.join(sorted(moved))}"]
        if self.predicted:
            parts.append(
                f"{len(hit)} of {len(self.predicted)} predicted readouts moved"
            )
        if broke:
            parts.append(f"but it also moved what it was supposed to spare: {broke}")
        return "; ".join(parts)

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "control": self.control,
            "condition": self.condition,
            "bit": self.bit,
            "baseline": self.baseline.summary(),
            "lesioned": self.lesioned.summary(),
            "control_arm": self.control_arm.summary(),
            "effects": {k: round(v, 6) for k, v in self.effects.items()},
            "control_effects": {k: round(v, 6) for k, v in self.control_effects.items()},
            "predicted": list(self.predicted),
            "spared": list(self.spared),
            "notes": self.notes,
            "verdict": self.verdict(),
        }


def _run_arm(
    arm: str,
    target: str,
    workload: Callable[[], Mapping[str, float]],
    *,
    repeats: int,
    lesion: str | None,
    returns: Any,
) -> ArmResult:
    started = time.monotonic()
    gathered: dict[str, list[float]] = {}
    failures = 0

    def once() -> None:
        nonlocal failures
        try:
            reading = workload()
        except Exception as exc:  # noqa: BLE001 - a workload that dies is a reading
            failures += 1
            logger.debug("workload failed under %s: %s", arm, exc)
            return
        for key, value in reading.items():
            gathered.setdefault(key, []).append(float(value))

    if lesion is None:
        for _ in range(repeats):
            once()
        suppressed = 0
    else:
        with silence(lesion, returns=returns):
            for _ in range(repeats):
                once()
        suppressed = silenced_calls(lesion)

    readout = {k: statistics.fmean(v) for k, v in gathered.items() if v}
    spread = {
        k: (statistics.pstdev(v) if len(v) > 1 else 0.0) for k, v in gathered.items() if v
    }
    return ArmResult(
        arm=arm,
        target=target,
        readout=readout,
        spread=spread,
        repeats=repeats,
        seconds=time.monotonic() - started,
        calls_suppressed=suppressed,
        failures=failures,
    )


def run_intervention(
    workload: Callable[[], Mapping[str, float]],
    target: str,
    *,
    snapshot: ConnectomeSnapshot | None = None,
    control: str | None = None,
    condition: str = "baseline",
    repeats: int = 3,
    returns: Any = None,
    predicted: Sequence[str] = (),
    spared: Sequence[str] = (),
    notes: str = "",
    seed: int = 0,
) -> InterventionReport:
    """Run the workload untouched, with the target cut, and with a like cell cut.

    ``predicted`` names the readouts the cut is expected to move and ``spared``
    the ones it is expected to leave alone. Both are recorded in the report
    whether or not they hold, because a lesion that moves everything is as
    uninformative as one that moves nothing, and only a prediction written
    before the run can tell those apart afterwards.
    """
    if control is None:
        if snapshot is None:
            raise LesionRefusedError("give a control cell or a snapshot to pick one from")
        control = degree_matched_control(snapshot, target, seed=seed)

    baseline = _run_arm("baseline", target, workload, repeats=repeats, lesion=None, returns=returns)
    lesioned = _run_arm("lesioned", target, workload, repeats=repeats, lesion=target, returns=returns)
    control_arm = _run_arm(
        "control", control, workload, repeats=repeats, lesion=control, returns=returns
    )

    keys = sorted(set(baseline.readout) | set(lesioned.readout) | set(control_arm.readout))
    effects = {
        key: lesioned.readout.get(key, 0.0) - baseline.readout.get(key, 0.0) for key in keys
    }
    control_effects = {
        key: control_arm.readout.get(key, 0.0) - baseline.readout.get(key, 0.0)
        for key in keys
    }
    return InterventionReport(
        target=target,
        control=control,
        condition=condition,
        baseline=baseline,
        lesioned=lesioned,
        control_arm=control_arm,
        predicted=tuple(predicted),
        spared=tuple(spared),
        notes=notes,
        effects=effects,
        control_effects=control_effects,
    )
