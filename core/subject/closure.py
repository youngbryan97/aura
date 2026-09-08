"""Is the declared core actually the core, or is something outside it steering?

Ten domains were chosen. Choosing them is a claim, and the claim is testable:

    P(K_{t+1} | X_t, E_t) ~= P(K_{t+1} | K_t, E_t)

where X is the rest of the machine. If some variable outside K predicts K's
future better than K does, then K is not causally closed and that variable
belongs inside it — or, worse, it is the broker every architecture diagram
draws as a box in the middle, and the whole graph over the ten domains is an
artefact of what it relays.

That last case is why this test exists rather than the obvious one. A hidden
broker cannot be found by asking whether any of the ten domains is a cut
vertex: the ten are all still strongly connected through it, all still on
cycles, and vertex connectivity over a graph that does not contain the broker
reports a healthy two or three. The synthetic star null in `core.subject.nulls`
does exactly this and passes every graph criterion. What it fails is here.

The periphery is read by walking the live phase objects and taking every
number they carry. No list is written by hand, because a hand-written list of
suspects is a list of the ones already thought of, and the point is to find
the one that was not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.recording import Recording

__all__ = ["ClosureReport", "closure_gain", "read_periphery"]

#: How many periphery numbers to keep. A cap, because one organ holding a large
#: array would otherwise supply more columns than the whole core.
MAX_PERIPHERY: int = 400

#: How deep to walk into an object's attributes. Two levels reaches the state a
#: phase keeps inside a helper it owns, which is where the interesting hidden
#: variables live; deeper than that and the walk starts collecting the runtime's
#: furniture.
MAX_DEPTH: int = 2


def _numbers(obj: Any, prefix: str, out: dict[str, float], depth: int = 0) -> None:
    if len(out) >= MAX_PERIPHERY or depth > MAX_DEPTH:
        return
    for name in sorted(vars(obj)) if hasattr(obj, "__dict__") else ():
        if name.startswith("__") or len(out) >= MAX_PERIPHERY:
            continue
        value = getattr(obj, name, None)
        if isinstance(value, bool):
            out[f"{prefix}.{name}"] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                out[f"{prefix}.{name}"] = number
        elif isinstance(value, (list, tuple, dict, set)):
            out[f"{prefix}.{name}#"] = float(len(value))
        elif hasattr(value, "__dict__") and not callable(value) and depth < MAX_DEPTH:
            _numbers(value, f"{prefix}.{name}", out, depth + 1)


def read_periphery(kernel: Any) -> dict[str, float]:
    """Every number the machine is carrying that is not part of K.

    The phases, the kernel itself, and every organ already instantiated beside
    them: counters, cursors, cached scores, whatever was kept between calls.
    Most of it is genuinely irrelevant. Any of it that predicts the core's next
    state was not, and the report names it rather than reporting an amount.

    Nothing here is a hand-written list of suspects. A list of suspects is a
    list of the ones already thought of, and the point is the one that was not.
    """
    out: dict[str, float] = {}
    _numbers(kernel, "kernel", out, depth=1)
    for phase in getattr(kernel, "_phases", []):
        _numbers(phase, phase.__class__.__name__, out)
    organs = getattr(kernel, "organs", None)
    if isinstance(organs, dict):
        for name, organ in organs.items():
            _numbers(organ, f"organ.{name}", out, depth=1)
    return out


@dataclass(frozen=True, slots=True)
class ClosureReport:
    loss_core: float
    loss_core_and_periphery: float
    loss_shuffled_periphery: float
    leak: float
    shuffled_leak: float
    leak_over_shuffle: float
    periphery_width: int
    top_leaks: tuple[tuple[str, float], ...] = ()

    @property
    def closed(self) -> bool:
        """Closed when the periphery does not improve on K.

        Two ways to be closed and both are here, because the first version used
        only the second and called a core open when the outside variables had
        made prediction strictly worse. A periphery that does not beat K has
        told us K is enough. A periphery that beats K by no more than its own
        shuffled copy has told us the gain was the extra columns, not what was
        in them.
        """
        return self.leak <= 0.0 or self.leak <= self.shuffled_leak

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_core_only": round(self.loss_core, 6),
            "loss_core_and_periphery": round(self.loss_core_and_periphery, 6),
            "loss_shuffled_periphery": round(self.loss_shuffled_periphery, 6),
            "leak": round(self.leak, 6),
            "shuffled_leak": round(self.shuffled_leak, 6),
            "leak_over_shuffled": round(self.leak_over_shuffle, 6),
            "periphery_width": self.periphery_width,
            "closed": self.closed,
            "largest_leaks": [
                {"variable": name, "gain": round(value, 5)} for name, value in self.top_leaks
            ],
        }


def closure_gain(
    recording: Recording,
    periphery: np.ndarray,
    names: tuple[str, ...],
    *,
    seed: int = 0,
    rank: int = 6,
) -> ClosureReport:
    """How much of K's future the rest of the machine explains that K does not.

    The shuffled arm keeps every periphery column and destroys only its
    alignment in time, so a gain that survives it is information rather than
    the extra degrees of freedom a wider model always brings.
    """
    live = recording.live_columns()
    now = recording.x[:-1][:, live]
    nxt = recording.x[1:][:, live]
    outside = periphery[:-1]
    if outside.size == 0 or now.shape[0] < 60:
        return ClosureReport(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, int(outside.shape[1] if outside.size else 0))

    spread = outside.std(axis=0)
    keep = spread > 1e-9
    outside = outside[:, keep]
    kept_names = tuple(name for name, flag in zip(names, keep, strict=True) if flag)
    if outside.shape[1] == 0:
        return ClosureReport(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0)

    train, validate, test = split_rows(now.shape[0])
    core = fit_predict(now, nxt, train=train, validate=validate, test=test)
    both = fit_predict(
        np.hstack([now, outside]),
        nxt,
        train=train,
        validate=validate,
        test=test,
        own_width=now.shape[1],
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(outside.shape[0])
    shuffled = fit_predict(
        np.hstack([now, outside[order]]),
        nxt,
        train=train,
        validate=validate,
        test=test,
        own_width=now.shape[1],
    )

    base = core.loss if core.loss > 1e-12 else 1.0
    leak = (core.loss - both.loss) / base
    shuffled_leak = (core.loss - shuffled.loss) / base
    over = (shuffled.loss - both.loss) / (shuffled.loss if shuffled.loss > 1e-12 else 1.0)

    # Which outside variables carry it, one at a time, so the answer names a
    # thing rather than reporting an amount.
    ranked: list[tuple[str, float]] = []
    if leak > 0.0:
        for index, name in enumerate(kept_names):
            single = fit_predict(
                np.hstack([now, outside[:, index : index + 1]]),
                nxt,
                train=train,
                validate=validate,
                test=test,
                own_width=now.shape[1],
            )
            gain = (core.loss - single.loss) / base
            if gain > 0.0:
                ranked.append((name, float(gain)))
        ranked.sort(key=lambda pair: -pair[1])

    return ClosureReport(
        loss_core=core.loss,
        loss_core_and_periphery=both.loss,
        loss_shuffled_periphery=shuffled.loss,
        leak=float(leak),
        shuffled_leak=float(shuffled_leak),
        leak_over_shuffle=float(over),
        periphery_width=int(outside.shape[1]),
        top_leaks=tuple(ranked[:rank]),
    )
