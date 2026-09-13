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

The periphery is read by walking the live phase objects, the kernel, the organs
and every service the container has already built, taking every number they
carry. No list is written by hand, because a hand-written list of suspects is a
list of the ones already thought of, and the point is to find the one that was
not.

Clocks are dropped before the comparison. A refresh timestamp outside K
predicts K's future for the same reason any monotone series predicts any other
— they are both going one way — and the first version of this test returned
"not closed" on the strength of two of them. The question is whether a hidden
*state* explains the core's future, and elapsed time is not a hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.recording import Recording

__all__ = ["ClosureReport", "closure_gain", "coverage", "read_periphery"]

#: What the last periphery walk saw, and what it could not reach. Read through
#: `coverage()`; a closure result without it is a claim about everything
#: outside the core made by a walk that stopped somewhere.
_COVERAGE: dict[str, Any] = {}

#: How many periphery numbers to keep. A cap, because one organ holding a large
#: array would otherwise supply more columns than the whole core.
MAX_PERIPHERY: int = 400

#: How deep to walk into an object's attributes. Two levels reaches the state a
#: phase keeps inside a helper it owns, which is where the interesting hidden
#: variables live; deeper than that and the walk starts collecting the runtime's
#: furniture.
MAX_DEPTH: int = 2


#: Anything at least this large is a wall-clock instant, not a quantity. Epoch
#: seconds passed a billion in 2001; a counter, a rate, a score or a cached
#: distance never reaches it.
EPOCH_FLOOR: float = 1e9


def _is_clock(value: float) -> bool:
    """A stored instant is the run's time index, not hidden state.

    The statistical filter below catches a counter that ticks every frame. It
    does not catch a timestamp that is written twice in a whole recording,
    because two moves are not enough to establish a direction — and two moves
    are all a summary refresh makes. Both of the largest leaks in the first
    campaign were exactly that: `_last_summary_refresh_at` and its completion
    stamp, predicting the core because everything in a run drifts with time.
    """
    return abs(value) >= EPOCH_FLOOR


#: How many projections an array is summarised into, beyond its four moments.
#: Fixed, so a reservoir of a thousand units and a vector of three cost the
#: same, and drawn from a frozen seed so two runs sketch the same directions.
SKETCH_WIDTH: int = 4
SKETCH_SEED: int = 20250911


def _is_array(value: Any) -> bool:
    """A numpy array, or something that will become one without side effects."""
    if isinstance(value, np.ndarray):
        return True
    # Torch and MLX tensors both answer to these and neither imports cleanly
    # here; going through the array protocol keeps this from knowing which.
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and not callable(value)
        and not isinstance(value, type)
    )


def _sketch(value: Any, name: str, out: dict[str, float]) -> None:
    """An array as a fixed number of columns: four moments and a projection.

    The projection is a seeded Gaussian, which preserves distances in
    expectation, so two peripheral states that differ differ in the sketch. The
    moments are there because a projection of a constant array is a constant
    and the moments say it was one.
    """
    try:
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        out[f"{name}#"] = 0.0
        return
    out[f"{name}#"] = float(flat.size)
    out[f"{name}.mean"] = float(flat.mean())
    out[f"{name}.sd"] = float(flat.std())
    out[f"{name}.min"] = float(flat.min())
    out[f"{name}.max"] = float(flat.max())
    rng = np.random.default_rng(SKETCH_SEED)
    directions = rng.normal(size=(SKETCH_WIDTH, flat.size)) / math.sqrt(flat.size)
    for index, column in enumerate(directions @ flat):
        out[f"{name}.p{index}"] = float(column)


def _numbers(
    obj: Any,
    prefix: str,
    out: dict[str, float],
    depth: int = 0,
    skip: Any = frozenset(),
) -> None:
    if len(out) >= MAX_PERIPHERY or depth > MAX_DEPTH:
        return
    for name in sorted(vars(obj)) if hasattr(obj, "__dict__") else ():
        if name.startswith("__") or len(out) >= MAX_PERIPHERY:
            continue
        # Not what the core's own schema already reads here. That number is
        # part of K, and K predicting K is not a leak.
        if name in skip or name.lstrip("_") in skip:
            continue
        value = getattr(obj, name, None)
        if isinstance(value, bool):
            out[f"{prefix}.{name}"] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number) and not _is_clock(number):
                out[f"{prefix}.{name}"] = number
        elif isinstance(value, (list, tuple, dict, set)):
            out[f"{prefix}.{name}#"] = float(len(value))
        elif _is_array(value):
            # An array fell through every branch above and through the one
            # below, because a numpy array is not a sequence this walk
            # recognised and has no `__dict__`. So every tensor the machine was
            # carrying was invisible to the closure test: a broker keeping its
            # state in one could not have been found. The sketch is fixed in
            # width, so a large array costs the same as a small one.
            _sketch(value, f"{prefix}.{name}", out)
        elif hasattr(value, "__dict__") and not callable(value) and depth < MAX_DEPTH:
            _numbers(value, f"{prefix}.{name}", out, depth + 1)


#: Name fragments that say a number is a stored instant rather than a quantity.
#: Provenance, not shape: a learning counter, accumulated evidence, a
#: developmental step and a depleting resource are all monotonic over a window
#: and all of them are hidden state the core's future may legitimately depend
#: on. Dropping every one-way column to be rid of the clocks threw those away
#: with them, and a closure test that cannot see a leak cannot report one.
CLOCK_NAMES: tuple[str, ...] = (
    "_at", "_time", "_ts", "timestamp", "last_update", "started", "finished",
    "deadline", "expires", "epoch", "clock",
)


def _named_clock(name: str) -> bool:
    """Whether the name says this is a moment rather than an amount."""
    leaf = name.rsplit(".", 1)[-1].lower().rstrip("#")
    return any(fragment in leaf for fragment in CLOCK_NAMES)


def _one_way(values: np.ndarray) -> np.ndarray:
    """Columns that only ever move one way.

    Kept as a description rather than used as a filter. What it identifies is
    a quantity whose level is mostly the run's own position in time, and the
    answer to that is to enter its increment rather than to drop it: see
    `_as_increments`.
    """
    if values.shape[0] < 8:
        return np.zeros(values.shape[1], dtype=bool)
    steps = np.diff(values, axis=0)
    moving = np.abs(steps) > 1e-12
    counts = moving.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(counts > 0, (steps > 0).sum(axis=0) / np.maximum(counts, 1), 0.0)
        down = np.where(counts > 0, (steps < 0).sum(axis=0) / np.maximum(counts, 1), 0.0)
    return ((up >= 0.99) | (down >= 0.99)) & (counts >= 4)


def _as_increments(values: np.ndarray, rising: np.ndarray) -> np.ndarray:
    """Enter a one-way column as how much it moved, not where it has got to."""
    if not rising.any():
        return values
    out = values.copy()
    steps = np.diff(values[:, rising], axis=0, prepend=values[:1, rising])
    out[:, rising] = steps
    return out


def _core_attributes() -> dict[str, set[str]]:
    """Which attribute of which organ the core's own schema already reads.

    The periphery is the machine minus the core. Without this it was the
    machine including the core: the walk goes through `kernel.organs` and every
    container service, and the workspace, the substrate, the self model and the
    world model *are* the core's organs. Their ignition level is G, their
    valence is C, their beliefs are S. A copy of the core predicting the core is
    not a leak, it is the same number twice, and a closure test run against a
    set that contains what it is testing can only ever say no.
    """
    from core.subject.state import _SCHEMAS

    out: dict[str, set[str]] = {}
    for schema in _SCHEMAS.values():
        for source in schema.sources:
            if not source.startswith("organ:"):
                continue
            path = source[len("organ:") :]
            organ, _, attribute = path.partition(".")
            if organ and attribute:
                out.setdefault(organ, set()).add(attribute.split(".")[0])
    return out


#: Container and organ names that hold one of the core's organs, by the organ
#: key the schema uses for it. The same object reaches the walk under several
#: names, and the core's own readings have to be excluded under all of them.
_ORGAN_ALIASES: dict[str, str] = {
    "workspace": "workspace",
    "global_workspace": "workspace",
    "substrate": "substrate",
    "conscious_substrate": "substrate",
    "liquid_substrate": "substrate",
    "liquid_state": "substrate",
    "self_model": "self_model",
    "world_model": "world_model",
    "unified_world_model": "world_model",
    "free_energy": "free_energy",
    "free_energy_engine": "free_energy",
    "self_prediction": "self_prediction",
    "agency": "agency",
    "comparator": "comparator",
    "ontogeny": "ontogeny",
}


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
    core = _core_attributes()
    _numbers(kernel, "kernel", out, depth=1)
    for phase in getattr(kernel, "_phases", []):
        _numbers(phase, phase.__class__.__name__, out)
    organs = getattr(kernel, "organs", None)
    if isinstance(organs, dict):
        for name, organ in organs.items():
            _numbers(
                organ,
                f"organ.{name}",
                out,
                depth=1,
                skip=core.get(_ORGAN_ALIASES.get(name, name), frozenset()),
            )
    # And every service the container has already built. That is where the
    # hidden state would be if there were any: a phase mostly holds references,
    # a service holds what it has accumulated.
    try:
        from core.container import ServiceContainer

        built = getattr(ServiceContainer, "_services", {}) or {}
        for name in sorted(built):
            if len(out) >= MAX_PERIPHERY:
                break
            _numbers(
                built.get(name),
                f"service.{name}",
                out,
                depth=1,
                skip=core.get(_ORGAN_ALIASES.get(name, name), frozenset()),
            )
    except (AttributeError, ImportError, LookupError, RuntimeError, TypeError, ValueError) as exc:
        # An absent container is an absent periphery. Named rather than bare:
        # every way this can fail is the container not being importable, not
        # holding services yet, or holding something `_numbers` cannot read,
        # and a genuinely unexpected failure while measuring the periphery
        # should reach somebody rather than read as "there is none".
        _COVERAGE["reader_failures"] = _COVERAGE.get("reader_failures", 0) + 1
        _COVERAGE["last_reader_failure"] = f"{type(exc).__name__}: {exc}"[:160]
        return out
    finally:
        _COVERAGE["read"] = len(out)
        _COVERAGE["capped"] = bool(len(out) >= MAX_PERIPHERY)
    return out


def coverage() -> dict[str, Any]:
    """What the periphery walk could and could not see.

    A closure result is a claim about everything outside the core, and a walk
    that stopped at four hundred numbers or two levels down has not seen
    everything outside the core. Reporting the caps is the difference between
    "K is closed" and "K is closed as far as this looked".
    """
    return {
        "numbers_read": int(_COVERAGE.get("read", 0)),
        "cap": MAX_PERIPHERY,
        "hit_the_cap": bool(_COVERAGE.get("capped", False)),
        "max_depth": MAX_DEPTH,
        "reader_failures": int(_COVERAGE.get("reader_failures", 0)),
        "last_reader_failure": _COVERAGE.get("last_reader_failure", ""),
    }


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
    #: How many permutations the floor was read from, and its upper tail.
    shuffled_draws: int = 1
    floor_high: float = 0.0
    #: One-way columns entered as their increment rather than their level, and
    #: columns dropped because their name says they hold a moment.
    differenced: tuple[str, ...] = ()
    dropped_as_clocks: tuple[str, ...] = ()

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
        return self.leak <= 0.0 or self.leak <= max(self.shuffled_leak, self.floor_high)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_core_only": round(self.loss_core, 6),
            "loss_core_and_periphery": round(self.loss_core_and_periphery, 6),
            "loss_shuffled_periphery": round(self.loss_shuffled_periphery, 6),
            "leak": round(self.leak, 6),
            "shuffled_leak": round(self.shuffled_leak, 6),
            "leak_over_shuffled": round(self.leak_over_shuffle, 6),
            "periphery_width": self.periphery_width,
            "shuffled_draws": self.shuffled_draws,
            "shuffled_leak_q95": round(self.floor_high, 6),
            "differenced": list(self.differenced),
            "dropped_as_clocks": list(self.dropped_as_clocks),
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
    draws: int = 16,
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
    clocks = np.array([_named_clock(name) for name in names], dtype=bool)
    keep = (spread > 1e-9) & ~clocks
    outside = outside[:, keep]
    kept_names = tuple(name for name, flag in zip(names, keep, strict=True) if flag)
    # A one-way column enters as its increment. Its level is mostly where the
    # run has got to, and every column in a recording drifts with that; what it
    # carries about the next state is how much it moved.
    rising = _one_way(outside)
    differenced = tuple(
        name for name, flag in zip(kept_names, rising, strict=True) if flag
    )
    outside = _as_increments(outside, rising)
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
    # Several shuffled draws, not one. The shuffled arm is the floor the leak
    # has to clear, and one draw of a permutation is one sample of it — a floor
    # read off a single draw is as likely to be lucky as the thing it is
    # measuring.
    rng = np.random.default_rng(seed)
    base = core.loss if core.loss > 1e-12 else 1.0
    shuffled_losses: list[float] = []
    for _ in range(max(1, draws)):
        order = rng.permutation(outside.shape[0])
        shuffled_losses.append(
            fit_predict(
                np.hstack([now, outside[order]]),
                nxt,
                train=train,
                validate=validate,
                test=test,
                own_width=now.shape[1],
            ).loss
        )
    shuffled_loss = float(np.mean(shuffled_losses))
    shuffled_leaks = [(core.loss - loss) / base for loss in shuffled_losses]
    floor_high = float(np.quantile(shuffled_leaks, 0.95)) if len(shuffled_leaks) > 1 else shuffled_leaks[0]

    leak = (core.loss - both.loss) / base
    shuffled_leak = float(np.mean(shuffled_leaks))
    over = (shuffled_loss - both.loss) / (shuffled_loss if shuffled_loss > 1e-12 else 1.0)

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
        loss_shuffled_periphery=shuffled_loss,
        leak=float(leak),
        shuffled_leak=float(shuffled_leak),
        leak_over_shuffle=float(over),
        periphery_width=int(outside.shape[1]),
        top_leaks=tuple(ranked[:rank]),
        shuffled_draws=int(len(shuffled_leaks)),
        floor_high=float(floor_high),
        differenced=differenced,
        dropped_as_clocks=tuple(
            name for name, flag in zip(names, clocks, strict=True) if flag
        ),
    )
