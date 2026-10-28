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
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from typing import Any

import numpy as np

from core.subject.estimate import fit_predict
from core.subject.recording import Recording

# How an array is summarised, shared with the core's own readers so the two
# describe a recurrent state with the same numbers. See core/subject/sketch.py.
from core.subject.sketch import SKETCH_SEED, SKETCH_WIDTH  # noqa: F401
from core.subject.sketch import is_array as _is_array
from core.subject.sketch import sketch as _sketch_array

__all__ = [
    "PeripheryAccumulator",
    "TURN_CLOSES",
    "periphery_by_turn",
    "periphery_matrix","ClosureReport", "closure_gain", "coverage", "read_periphery"]

#: The frame that closes a turn, whose row the turn series keeps
#: (`Recording.turn_rows`). The periphery is read at it and nowhere else.
TURN_CLOSES: str = "ontogeny"

#: The four readings of the gate (`ClosureReport.verdict`).
OPEN, CLOSED, UNRESOLVED, NOT_MEASURED = "OPEN", "CLOSED_WITHIN_MARGIN", "UNRESOLVED", "NOT_MEASURED"

#: The margin within which a closed reading is a claim, as a leak: the share
#: of what K and its environment cannot predict about K's change that the
#: outside explains, which is the partial R-squared of the outside given K and
#: E. Cohen's convention for a small partial effect is f-squared 0.02 (Cohen,
#: Statistical Power Analysis for the Behavioral Sciences, 2nd ed., 1988),
#: which is a partial R-squared of 0.02 / 1.02. A published convention rather
#: than a number read off her: every value proposed before it (0.025, 0.03,
#: 0.045, 0.059) came after seed 23's leak had been read. Named in the ISC-v6
#: addendum of 25 October, night, beside the interventional test closure now
#: also needs (`core/subject/carrier_reads.py`).
CLOSURE_MARGIN: float | None = 0.02 / 1.02

#: How the margin is drawn when no number is named. The least-domain rule
#: (`least_domain_margin`) was preregistered on 25 October and withdrawn the same
#: day: a part of K the rest needs can carry almost nothing the rest's own past
#: does not, so the least of those leaks sits near zero in a world built with
#: every domain coupled, and on her. No rule is in force, so nothing closes
#: until a margin is named.
LEAST_DOMAIN = "least_domain"
CLOSURE_MARGIN_RULE: str | None = None

#: Draws behind each bootstrap bound, as the partition sweep's design uses
#: (core/subject/isc_v5.py).
BOOTSTRAP_DRAWS: int = 1000

#: What the last periphery walk saw, and what it could not reach. Read through
#: `coverage()`; a closure result without it is a claim about everything
#: outside the core made by a walk that stopped somewhere.
_COVERAGE: dict[str, Any] = {}

#: How many periphery numbers to keep. A cost bound and nothing else: what
#: stops one organ's large array from supplying more columns than the whole
#: core is the per-source ceiling in `_numbers`, not this.
#:
#: It was 400, and the walk is exhausted at 2,758 on this machine — so closure
#: was judged on a seventh of the periphery and reported `hit_the_cap` every
#: time, which is the gate on the carrier decided by where the walk stopped.
#: Set above what this machine carries, with headroom, from the measurement
#: rather than from a round number that looked safe. The full read costs
#: 101 ms a frame against 3.3 ms at the old cap; a gate that is wrong is worse
#: than a gate that is slow. `coverage()["hit_the_cap"]` still says when a
#: larger machine binds it.
MAX_PERIPHERY: int = 16000
#: Raised from 4000 on 25 October. Once the reader took in deques and nested
#: containers the machine carried 8,051 to 8,108 numbers over three rounds of
#: a fresh organism, and the walk, which reads the kernel, phases and organs
#: before the services, spent all 4,000 before reaching any of 289 services:
#: closure was judged without a single service in its periphery. Twice the
#: measured width, because histories keep filling as a run goes on.

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


def _sketch(value: Any, name: str, out: dict[str, float]) -> None:
    """An array as a fixed number of columns: four moments and a projection."""
    try:
        size = int(np.isfinite(np.asarray(value, dtype=np.float64)).sum())
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return
    summary = _sketch_array(value)
    if summary is None:
        out[f"{name}#"] = 0.0
        return
    out[f"{name}#"] = float(size)
    for part, number in summary.items():
        out[f"{name}.{part}"] = number


def _numbers(
    obj: Any,
    prefix: str,
    out: dict[str, float],
    depth: int = 0,
    skip: Any = frozenset(),
    ceiling: int | None = None,
    seen: set[int] | None = None,
    own: Mapping[int, set[str]] | None = None,
) -> None:
    """Collect the numbers this object is carrying, up to `ceiling` entries.

    The ceiling is per source rather than global. Attributes are walked in name
    order, so a global cap is spent by whichever subtree sorts first: the
    kernel alone filled all four hundred slots with configuration constants,
    every one of them the same on every frame, and the closure test then
    compared a core against a periphery of four hundred columns that never
    moved. It read `closed` because nothing had been measured.
    """
    stop = MAX_PERIPHERY if ceiling is None else min(MAX_PERIPHERY, ceiling)
    if len(out) >= stop or depth > MAX_DEPTH:
        return
    # An object is walked once per call. Every phase holds a reference back to
    # the kernel, so the walk re-entered the kernel through each of them and
    # spent the whole budget re-reading the same configuration under a
    # different prefix: four hundred columns, none of which moved, and closure
    # then compared the core against nothing.
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    # One of K's own organs, however the walk arrived at it: through a
    # service's `instance`, or a phase's attribute holding it. The names it
    # was skipped under at the top of a service did not reach it there.
    skip = set(skip) | (own or {}).get(id(obj), set())
    for name in sorted(vars(obj)) if hasattr(obj, "__dict__") else ():
        if name.startswith("__") or len(out) >= stop:
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
        elif isinstance(value, (list, tuple, dict, set, deque)):
            out[f"{prefix}.{name}#"] = float(len(value))
            _contents(value, f"{prefix}.{name}", out, seen=seen, own=own)
        elif _is_array(value):
            # An array fell through every branch above and through the one
            # below, because a numpy array is not a sequence this walk
            # recognised and has no `__dict__`. So every tensor the machine was
            # carrying was invisible to the closure test: a broker keeping its
            # state in one could not have been found. The sketch is fixed in
            # width, so a large array costs the same as a small one.
            _sketch(value, f"{prefix}.{name}", out)
        elif hasattr(value, "__dict__") and not callable(value) and depth < MAX_DEPTH:
            _numbers(value, f"{prefix}.{name}", out, depth + 1, ceiling=stop, seen=seen, own=own)


def _contents(
    value: Any,
    name: str,
    out: dict[str, float],
    depth: int = 0,
    seen: set[int] | None = None,
    own: Mapping[int, set[str]] | None = None,
) -> None:
    """The numbers a container holds, not only how many things it holds.

    Until 25 October a dict or list entered the periphery as its length, so a
    broker that kept its state as a value in a dict of fixed size was invisible
    to the gate: a review planted one changing from 0.5 to -3.7 and the reader
    returned the same number both times. A container of a few numbers enters
    one column per entry, in key order; a wider one enters as the same
    fixed-width sketch an array does, over its values in key order. A set has no
    order, so its numbers are sorted first.
    """
    if isinstance(value, dict):
        items = sorted(((str(key), item) for key, item in value.items()), key=lambda pair: pair[0])
    elif isinstance(value, set):
        try:
            items = [(str(index), item) for index, item in enumerate(sorted(value))]
        # not a failure: a set of things that do not order holds no series to read.
        except TypeError:
            return
    else:
        items = [(str(index), item) for index, item in enumerate(value)]
    # A container of containers, or a history of objects. Until 25 October only
    # numbers sitting directly inside were read, and a deque not at all, so a
    # dict of buffers (affect grounding's `_buf`, eight deques) and a deque of
    # moments (the phenomenal-now engine's `_recent`) entered as nothing: the
    # closure test could not see a leak kept in any history. A nested container
    # is read as its own container, two levels down; a history of records (a
    # dataclass per moment) is read through its most recent record, which is
    # the one its readers use. A list of organs is not a history: the kernel's
    # phases are walked under their own names, with K's attributes kept out.
    if depth < 2:
        for key, item in items:
            if isinstance(item, (list, tuple, dict, set, deque)):
                out[f"{name}[{key}]#"] = float(len(item))
                _contents(item, f"{name}[{key}]", out, depth + 1, seen=seen, own=own)
        if items and not isinstance(value, (dict, set)):
            last = items[-1][1]
            if is_dataclass(last) and not isinstance(last, type):
                _numbers(last, f"{name}[-1]", out, depth=MAX_DEPTH - 1, seen=seen, own=own)
    numbers = [
        (key, float(item))
        for key, item in items
        if isinstance(item, (int, float)) and not isinstance(item, bool)
        and math.isfinite(float(item)) and not _is_clock(float(item))
    ]
    if not numbers:
        return
    if len(numbers) <= SKETCH_WIDTH:
        for key, number in numbers:
            out[f"{name}[{key}]"] = number
        return
    _sketch(np.array([number for _key, number in numbers]), f"{name}[]", out)


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
                read = attribute.split(".")[0]
                out.setdefault(organ, set()).add(read)
                out[organ].update(_VIEWS_OF.get((organ, read), ()))
    return out


#: Attributes that are fixed slices of one the core reads, so reading the one
#: is reading them. The unified field learns its inputs in one matrix and keeps
#: the per-source blocks as its column slices (core/consciousness/unified_field_learning.py).
_VIEWS_OF: dict[tuple[str, str], tuple[str, ...]] = {
    ("field", "_W_input_batched"): ("W_mesh", "W_chem", "W_bind", "W_intero", "W_substrate"),
}


#: Container and organ names that hold one of the core's organs, by the organ
#: key the schema uses for it. The same object reaches the walk under several
#: names, and the core's own readings have to be excluded under all of them.
_ORGAN_ALIASES: dict[str, str] = {
    "what_she_wants": "wants",
    "neural_mesh": "mesh",
    "unified_field": "field",
    "homeostatic_coupling": "homeostasis",
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


def _own_attributes(organs: Any) -> dict[int, set[str]]:
    """K's organs by identity, each with the attributes the core's schema reads of it.

    The skip by name only held at the top of a service. The walk reaches an
    organ through the service's `instance`, or first through a phase holding
    it, and there it read K's own arrays as periphery: the field's state, the
    mesh's weights, the phi core's visits, each under another projection than
    K's, reported as state K did not take in (22 October).
    """
    if organs is None:
        return {}
    out: dict[int, set[str]] = {}
    for key, attributes in _core_attributes().items():
        organ = getattr(organs, key, None)
        if organ is not None:
            out.setdefault(id(organ), set()).update(attributes)
    return out


def read_periphery(kernel: Any, organs: Any = None) -> dict[str, float]:
    """Every number the machine is carrying that is not part of K.

    The phases, the kernel itself, and every organ already instantiated beside
    them: counters, cursors, cached scores, whatever was kept between calls.
    Most of it is genuinely irrelevant. Any of it that predicts the core's next
    state was not, and the report names it rather than reporting an amount.

    Nothing here is a hand-written list of suspects. A list of suspects is a
    list of the ones already thought of, and the point is the one that was not.

    ``organs`` are the live objects K reads (`core.subject.state.Organs`). Given
    them, what K reads of each is left out wherever the walk meets it.
    """
    out: dict[str, float] = {}
    core = _core_attributes()
    own = _own_attributes(organs)
    # What K* holds beyond K's ten domains (core/subject/carrier_reads.py) is
    # read whole under `held.`, and left out of the periphery wherever the walk
    # meets the unit that carries it, so it is on one side of the test only.
    from core.subject.carrier_reads import _units, held_attributes_of, read_held

    from core.subject.carrier_reads import ORGANS_WHOLE, organs_of

    units = _units(kernel, organs)
    for unit, organ in units.items():
        if organ is not None:
            own.setdefault(id(organ), set()).update(held_attributes_of(unit))
    out.update(read_held(kernel, units, organs))
    # K's organs are K*'s whole, so the walk takes none of them, by whatever
    # path it meets them.
    seen: set[int] = {id(organ) for organ in organs_of(organs).values()} if ORGANS_WHOLE else set()
    _numbers(kernel, "kernel", out, depth=1, seen=seen, own=own)
    for phase in getattr(kernel, "_phases", []):
        _numbers(phase, phase.__class__.__name__, out, seen=seen, own=own)
    organs = getattr(kernel, "organs", None)
    if isinstance(organs, dict):
        for name, organ in organs.items():
            _numbers(
                organ,
                f"organ.{name}",
                out,
                depth=1,
                skip=core.get(_ORGAN_ALIASES.get(name, name), frozenset()),
                seen=seen,
                own=own,
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
                seen=seen,
                own=own,
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
    #: How many surrogates the floor was read from, and its upper tail.
    shuffled_draws: int = 1
    floor_high: float = 0.0
    #: One-way columns entered as their increment rather than their level, and
    #: columns dropped because their name says they hold a moment.
    differenced: tuple[str, ...] = ()
    dropped_as_clocks: tuple[str, ...] = ()
    #: What each leak is. Diagnostic: it does not change the verdict.
    audit: dict[str, Any] = field(default_factory=dict)
    #: One entry per forward fold, in time order: what each scored and what it
    #: read, with its own bounds and verdict.
    folds: tuple[dict[str, Any], ...] = ()
    #: How many folds read open on their own.
    leaking_folds: int = 0
    #: The bounds of the pooled leak from a block bootstrap over the scored
    #: rows, one-sided at 95% each. None when they were not computed.
    leak_lower: float | None = None
    leak_upper: float | None = None
    #: The smallest planted leak the same estimator called open on this
    #: recording, read on the same folds and against the same periphery.
    smallest_leak_resolved: float | None = None
    #: The margin a closed reading is claimed within, named before a fresh seed
    #: is scored. None until then, and then the gate can never say closed.
    margin: float | None = None
    #: How many declared environment columns both models were given.
    environment_width: int = 0

    @property
    def measured(self) -> bool:
        """Whether anything outside the core was actually compared against it.

        A walk that returned no usable column leaves `leak` at 0.0, and 0.0 is
        also what a genuinely closed core reads. Both campaigns were reporting
        `closed` on a periphery of width zero: four hundred numbers were read
        and every one of them was the same on every frame, so the comparison
        never happened and its absence looked exactly like a pass.
        """
        return self.periphery_width > 0

    @property
    def resolution(self) -> float | None:
        """The smallest leak the gate could still find on this recording."""
        return self.smallest_leak_resolved

    @property
    def verdict(self) -> str:
        """OPEN, CLOSED_WITHIN_MARGIN, UNRESOLVED or NOT_MEASURED.

        Until 25 October closure was the null hypothesis: the core was called
        closed whenever the periphery failed to beat it, so a gate with no power
        reached "closed" by default, and the rule meant to qualify that,
        `leak >= resolution`, called a leak of exactly zero not decisive. Six
        independent reviews of the briefing found it and it was confirmed in
        this file.

        Open is a finding: the pooled leak clears the surrogate floor's upper
        tail and its lower bound clears the floor's centre, or one fold does so
        at a level divided over the folds, because a leak confined to one
        stretch of a run is still a leak. Closed is a claim of equivalence: a
        margin named in advance, the leak's upper bound below it, the gate shown
        to find a planted leak no larger than it, and no fold open. Everything
        else is unresolved, which is a result in its own right and not a
        failure to be rounded either way.
        """
        if not self.measured:
            return NOT_MEASURED
        if self.leaking_folds > 0 or self._pooled_open():
            return OPEN
        if self.estimator_failed:
            return UNRESOLVED
        if (
            self.margin is not None
            and self.leak_upper is not None
            and self.leak_upper < self.margin
            and self.smallest_leak_resolved is not None
            and math.isfinite(self.smallest_leak_resolved)
            and 0.0 < self.smallest_leak_resolved <= self.margin
        ):
            return CLOSED
        return UNRESOLVED

    @property
    def estimator_failed(self) -> tuple[int, ...]:
        """Folds where the wider model did measurably worse than the one it contains.

        Nested function classes order their optimal population risks, not the
        test errors of fitted predictors. A negative held-out gain can result
        from estimation error or drift. Such a fold cannot certify closure.
        MEASURED 25 October on worlds that develop from inside, as she does: the
        earlier estimator failed a fold in every one, and three real brokers
        read closed through those folds.
        """
        return tuple(
            index for index, fold in enumerate(self.folds)
            if float(fold.get("leak_upper", 0.0)) < 0.0 and not fold.get("dropped")
        )

    @property
    def dropped_folds(self) -> tuple[int, ...]:
        """Failed folds left out of the pooled reading because at least two valid ones remained."""
        return tuple(index for index, fold in enumerate(self.folds) if fold.get("dropped"))

    def _pooled_open(self) -> bool:
        if self.leak_lower is None:
            return False
        return self.leak > self.floor_high and self.leak_lower > self.shuffled_leak

    @property
    def closed(self) -> bool:
        """Closed within the preregistered margin, and nothing less."""
        return self.verdict == CLOSED

    @property
    def decisive(self) -> bool:
        """Whether the gate reached an answer, open or closed."""
        return self.verdict in (OPEN, CLOSED)

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
            "surrogate": "circular_shift",
            "scored": "forward_folds",
            "folds": [
                {key: (value if isinstance(value, (str, bool)) else round(float(value), 6))
                 for key, value in fold.items()}
                for fold in self.folds
            ],
            "leaking_folds": self.leaking_folds,
            "shuffled_leak_q95": round(self.floor_high, 6),
            "differenced": list(self.differenced),
            "dropped_as_clocks": list(self.dropped_as_clocks),
            "verdict": self.verdict,
            "closed": self.closed,
            "measured": self.measured,
            "decisive": self.decisive,
            "leak_lower": None if self.leak_lower is None else round(self.leak_lower, 6),
            "leak_upper": None if self.leak_upper is None else round(self.leak_upper, 6),
            "margin": self.margin,
            "environment_width": self.environment_width,
            "smallest_leak_resolved": self.resolution,
            "estimator_failed_folds": list(self.estimator_failed),
            "dropped_folds": list(self.dropped_folds),
            "periphery_entered_as": "increments",
            "target": "the change in K",
            "why_unresolved": (
                ""
                if self.verdict != UNRESOLVED
                else "the wider model did measurably worse than the narrower on a fold, so the estimator failed there"
                if self.estimator_failed
                else "the planted resolution is not a finite positive effect size"
                if self.smallest_leak_resolved is not None and (
                    not math.isfinite(self.smallest_leak_resolved) or self.smallest_leak_resolved <= 0.0
                )
                else "no margin is named, so closure cannot be claimed"
                if self.margin is None
                else "the leak's upper bound or the gate's resolution is not below the margin"
            ),
            "why_not_measured": (
                ""
                if self.measured
                else "no periphery column varied, so the core was never compared against anything"
            ),
            "largest_leaks": [
                {"variable": name, "gain": round(value, 5)} for name, value in self.top_leaks
            ],
            "audit": self.audit,
        }


def periphery_by_turn(matrix: np.ndarray, recording: Recording) -> np.ndarray:
    """The periphery one row per turn, whether it was read once a turn or once a frame.

    Closure fits K's next turn from its last (`Recording.turn_rows`), so the
    periphery it is compared against is the one at each turn's closing frame.
    Read at every frame, the walk of four thousand numbers was done thirty-three
    times a turn for one row that was used: at 370 rounds, about two and a half
    hours of the run. A matrix read at every frame, as runs before 23 October
    saved it, is taken at the same rows.
    """
    rows = recording.turn_rows(TURN_CLOSES)
    if matrix.shape[0] == len(rows):
        return matrix
    if matrix.shape[0] == recording.x.shape[0]:
        return matrix[rows]
    raise ValueError(
        f"a periphery of {matrix.shape[0]} rows matches neither the {len(rows)} turns "
        f"nor the {recording.x.shape[0]} frames of the recording it is compared with"
    )


class PeripheryAccumulator:
    """Every frame's periphery reading, in one array rather than one dict each.

    `read_periphery` returns a fresh dict of up to four hundred entries whose
    keys it formats fresh every call, so keeping one per frame keeps four
    hundred new strings per frame as well. Measured on the offline organism it
    costs 284 KB a frame against 2.5 KB for everything else the recording
    holds: at sixty rounds that is 4.3 GB and at three hundred it is 21.5 GB,
    which is what took a campaign to thirty-five gigabytes resident and put the
    machine into swap at about three hundred seconds a turn.

    The readings are numbers in a fixed set of slots, so they belong in an
    array. A key that first appears part way through widens the matrix and its
    earlier rows read zero, which is what `periphery_matrix` gives them too.

    Columns come out sorted by name, so the array is the one
    `periphery_matrix` would have built from the same dicts.
    """

    def __init__(self, *, chunk: int = 4096) -> None:
        self._chunk = max(64, int(chunk))
        self._index: dict[str, int] = {}
        self._data: np.ndarray = np.zeros((self._chunk, 0), dtype=np.float64)
        self._rows = 0

    def note(self, reading: Mapping[str, float]) -> None:
        """Take one frame's reading.

        New columns are added in one step. Widening once per key copied the
        whole buffer once per key, and the first frame brings four hundred of
        them at once — which allocated 5.6 GB of transient copies before a
        single row had been written.
        """
        fresh = [name for name in reading if name not in self._index]
        if fresh:
            width = self._data.shape[1]
            for offset, name in enumerate(fresh):
                self._index[name] = width + offset
            self._data = np.hstack(
                [self._data, np.zeros((self._data.shape[0], len(fresh)), dtype=np.float64)]
            )
        if self._rows >= self._data.shape[0]:
            self._data = np.vstack(
                [self._data, np.zeros_like(self._data)] if self._data.shape[0]
                else [np.zeros((self._chunk, self._data.shape[1]), dtype=np.float64)]
            )
        row = self._data[self._rows]
        row[:] = 0.0
        for name, value in reading.items():
            try:
                row[self._index[name]] = float(value)
            except (TypeError, ValueError):
                continue
        self._rows += 1

    def rows(self) -> int:
        return self._rows

    def matrix(self) -> tuple[np.ndarray, tuple[str, ...]]:
        """The readings so far, columns sorted by name."""
        if not self._index:
            return np.zeros((self._rows, 0)), ()
        names = tuple(sorted(self._index))
        order = [self._index[name] for name in names]
        return np.array(self._data[: self._rows][:, order], copy=True), names


def periphery_matrix(rows: Sequence[Mapping[str, float]]) -> tuple[np.ndarray, tuple[str, ...]]:
    """One row per frame, one column per number the machine carried.

    A key missing from a frame reads 0.0 rather than being dropped, because a
    counter that only exists once the thing it counts has happened is a real
    reading of zero before then.
    """
    names = tuple(sorted({key for row in rows for key in row}))
    if not names:
        return np.zeros((len(rows), 0)), ()
    matrix = np.array(
        [[float(row.get(name, 0.0)) for name in names] for row in rows], dtype=np.float64
    )
    return matrix, names


#: How many surrogate peripheries the floor is read from. Sixteen gave a 95th
#: percentile that moved from 0.014 to 0.021 on one recording with nothing but
#: the draws changed (the carrier at 8f548ed79, whose leak was 0.024), so the
#: gate's answer was the luck of the draw. At 64 it moves by a few thousandths.
SURROGATE_DRAWS: int = 64


def _surrogate(outside: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The periphery with its alignment to K broken and its own course kept.

    The whole block is turned around in time by one offset, at least a tenth
    of the run from where it was, so every column keeps its own autocorrelation
    and its relations to the other periphery columns, and loses only its
    relation to K. A row permutation broke both. A slow periphery column then
    became white noise in the floor, while the real one could still follow K's
    own development across the run, and two series that only drift together
    read as a cause: the spurious regression of Granger and Newbold (1974).
    On the carrier at 8f548ed79 the permutation floor put the leak at p 0.03
    and the circular one at p 0.26.
    """
    rows = outside.shape[0]
    margin = max(1, rows // 10)
    if rows <= 2 * margin:
        return outside[rng.permutation(rows)]
    return np.roll(outside, int(rng.integers(margin, rows - margin)), axis=0)


def forward_folds(rows: int, folds: int = 3) -> list[tuple[slice, slice, slice]]:
    """Train on the past and score the future, over a growing window.

    One contiguous 60/20/20 scores the last fifth of the run and nothing else,
    and on a system that develops that one fifth is not representative of it.
    MEASURED on the carrier at 8f548ed79: K predicts its own next turn to a loss
    of 0.0091 on the final fifth — it explains 99.1% of itself there — and to
    0.0617, 0.1100, 0.1785 and 0.2851 on the other four. The leak is a fraction
    of what K cannot predict, so its denominator moved by thirty times depending
    on which rows the gate happened to score, and the gate's answer moved with
    it: the carrier's periphery cleared its floor on one fifth and not on the
    others.

    Scoring every fifth is not the fix. Training on later rows and testing on
    earlier ones extrapolates a drifting periphery backwards, and the first
    fifth returned a leak of -322,469.

    So the window grows forward: each fold fits on rows strictly earlier than
    the rows it is scored on, and the last fold is the split the gate used
    before. Three honest estimates say whether an answer belongs to the carrier
    or to the slice, and one cannot.
    """
    out: list[tuple[slice, slice, slice]] = []
    parts = folds + 2
    for k in range(folds):
        validate_start = int(rows * (k + 1) / parts)
        test_start = int(rows * (k + 2) / parts)
        test_stop = int(rows * (k + 3) / parts)
        out.append(
            (slice(0, validate_start), slice(validate_start, test_start), slice(test_start, test_stop))
        )
    return out


def environment_of_transitions(recording: Recording) -> tuple[np.ndarray, tuple[str, ...]]:
    """What arrived from outside during each step from turn t to t + 1.

    The closure the gate tests is conditional on the environment,
    P(K_next | K, E, X) = P(K_next | K, E), and until 25 October neither model
    was given E: an organ that only echoed the condition schedule read as a
    leak. The input that drives a step is the one recorded on the row it ends
    on, so this is row t + 1's condition, one column per condition, and its
    declared readings. A column that only counts time, or that is the
    condition again under a number, is left out.
    """
    rows = recording.x.shape[0]
    if rows < 2:
        return np.zeros((0, 0)), ()
    labels = tuple(recording.conditions[1:])
    kinds = sorted(set(recording.conditions))
    onehot = np.array([[1.0 if label == kind else 0.0 for kind in kinds] for label in labels])
    env = np.asarray(recording.env, dtype=np.float64)[1:]
    names: list[str] = [f"condition={kind}" for kind in kinds]
    blocks = [onehot]
    if env.size:
        counting = _one_way(env)
        for index, name in enumerate(recording.env_names):
            column = env[:, index]
            if counting[index] or _named_clock(name) or column.std() <= 1e-12:
                continue
            # The condition under another name: constant within each condition.
            if all(np.ptp(column[[i for i, label in enumerate(labels) if label == kind]]) <= 1e-12
                   for kind in kinds if kind in labels):
                continue
            blocks.append(column[:, None])
            names.append(name)
    return np.hstack(blocks), tuple(names)


def _block_length(series: np.ndarray) -> int:
    """How many consecutive rows a bootstrap block keeps.

    The scored rows are a time series, so resampling them one at a time treats
    two thousand dependent turns as two thousand independent replications and
    makes the bounds too narrow. A block is at least n^(1/3) rows, the order
    that minimises the error of a block-bootstrap variance (Hall, Horowitz and
    Jing 1995), and at least as long as the series stays correlated with
    itself beyond the band an uncorrelated series of this length would show.
    """
    n = int(series.shape[0])
    if n < 4:
        return 1
    centred = series - series.mean()
    total = float(centred @ centred)
    band = 2.0 / math.sqrt(n)
    lag = 1
    while lag < n // 4 and total > 0.0:
        if abs(float(centred[:-lag] @ centred[lag:]) / total) < band:
            break
        lag += 1
    return max(lag, int(round(n ** (1.0 / 3.0))))


def _pooled_bounds(
    gains: Sequence[np.ndarray], bases: Sequence[np.ndarray], *, seed: int, draws: int = BOOTSTRAP_DRAWS,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """One-sided bounds on sum(gain) / sum(base), from circular blocks within each fold."""
    rng = np.random.default_rng(seed)
    lengths = [_block_length(g) for g in gains]
    values = np.empty(draws)
    for draw in range(draws):
        top = bottom = 0.0
        for gain, base, length in zip(gains, bases, lengths, strict=True):
            n = gain.shape[0]
            if n == 0:
                continue
            starts = rng.integers(0, n, size=int(math.ceil(n / length)))
            index = (starts[:, None] + np.arange(length)[None, :]).reshape(-1)[:n] % n
            top += float(gain[index].sum())
            bottom += float(base[index].sum())
        values[draw] = top / bottom if bottom > 1e-12 else 0.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _added_variable_scores(
    own: np.ndarray, target: np.ndarray, block: np.ndarray, rows: slice, alpha: float
) -> np.ndarray:
    """Each added column's summed squared correlation with what `own` leaves unexplained in each target, on `rows`."""
    from core.subject.estimate import _ridge, _standardise

    x, _keep = _standardise(own[rows], own[rows])
    y = target[rows]
    spread = y.std(axis=0)
    y = (y - y.mean(axis=0)) / np.where(spread > 1e-12, spread, 1.0)
    residual = y - x @ _ridge(x, y, alpha)
    residual = residual / np.where(residual.std(axis=0) > 1e-12, residual.std(axis=0), 1.0)
    added = block[rows]
    added = (added - added.mean(axis=0)) / np.where(added.std(axis=0) > 1e-12, added.std(axis=0), 1.0)
    correlation = (added.T @ (residual - residual.mean(axis=0))) / max(1, added.shape[0])
    return (correlation**2).sum(axis=1)


def _fit_nested(
    features: np.ndarray, targets: np.ndarray, *, train: slice, validate: slice, test: slice,
    own_width: int | None = None,
) -> Any:
    """A nested fit whose added block enters through the columns that beat every time-shifted copy of themselves.

    One ridge penalty shared by thousands of outside columns drowns a single
    informative one: on seed 23 with her organs whole, a near-perfect copy of
    a target among two thousand columns was found at a leak of 0.197 and missed
    at 0.013, while the margin is 0.0196. Screened on the training rows only,
    an added column enters when its added-variable score exceeds the largest
    score of the same block circularly shifted, which keeps every column's
    course in time and breaks its alignment with K. The surrogate floor and the
    planted cause pass through the same screen, so false entries are matched
    on both sides. On worlds that develop from inside a real broker stayed open
    in 30 of 30; on seed 23 the gate then found a planted leak of 0.010.
    """
    if own_width is None or own_width >= features.shape[1]:
        return fit_predict(features, targets, train=train, validate=validate, test=test, own_width=own_width)
    own, added = features[:, :own_width], features[:, own_width:]
    rows = slice(train.start, train.stop)
    alpha = max(float(fit_predict(own, targets, train=train, validate=validate, test=validate).alpha), 1e-3)
    real = _added_variable_scores(own, targets, added, rows, alpha)
    count = train.stop - train.start
    rng = np.random.default_rng(int(added.shape[1]) + int(train.stop))
    shifted = added.copy()
    shifted[rows] = np.roll(added[rows], int(rng.integers(max(1, count // 10), max(2, count - count // 10))), axis=0)
    threshold = float(_added_variable_scores(own, targets, shifted, rows, alpha).max())
    keep = np.flatnonzero(real > threshold)
    if keep.size == 0:
        return fit_predict(own, targets, train=train, validate=validate, test=test)
    return fit_predict(
        np.hstack([own, added[:, keep]]), targets, train=train, validate=validate, test=test, own_width=own_width,
    )


def _scored(
    now: np.ndarray,
    env: np.ndarray,
    nxt: np.ndarray,
    outside: np.ndarray,
    *,
    seed: int,
    draws: int,
    only: Sequence[int] | None = None,
) -> dict[str, Any]:
    """K + E against K + E + P on the forward folds: pooled leak, bounds, floor, and each fold's reading.

    `only` pools over those folds and no others: a planted cause is read on the
    folds the verdict read, not on a fold its own strength rescued.
    """
    own = np.hstack([now, env]) if env.size else now
    folds = forward_folds(now.shape[0])
    per_fold: list[dict[str, Any]] = []
    gains: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    sses: list[tuple[float, float, np.ndarray]] = []
    for index, (train, validate, test) in enumerate(folds):
        rng = np.random.default_rng(seed + index)
        narrow = fit_predict(own, nxt, train=train, validate=validate, test=test)
        wide = _fit_nested(
            np.hstack([own, outside]), nxt, train=train, validate=validate, test=test, own_width=own.shape[1],
        )
        surrogates = [
            _fit_nested(
                np.hstack([own, _surrogate(outside, rng)]), nxt,
                train=train, validate=validate, test=test, own_width=own.shape[1],
            )
            for _ in range(max(1, draws))
        ]
        base = narrow.sse if narrow.sse > 1e-12 else 1.0
        gain = np.asarray(narrow.row_sse) - np.asarray(wide.row_sse)
        fold_floor = [(narrow.sse - fit.sse) / base for fit in surrogates]
        lower, upper = _pooled_bounds(
            [gain], [np.asarray(narrow.row_sse)], seed=seed + 100 + index, alpha=0.05 / len(folds),
        )
        leak = (narrow.sse - wide.sse) / base
        floor_median = float(np.median(fold_floor))
        floor_high = float(np.quantile(fold_floor, 1.0 - 0.05 / len(folds))) if len(fold_floor) > 1 else fold_floor[0]
        per_fold.append({
            "scored_rows": f"{test.start}-{test.stop}",
            "loss_core": float(narrow.loss),
            "loss_core_and_periphery": float(wide.loss),
            "loss_shuffled_periphery": float(np.median([fit.loss for fit in surrogates])),
            "leak": float(leak),
            "leak_lower": lower,
            "leak_upper": upper,
            "shuffled_leak": floor_median,
            "floor_high": floor_high,
            "open": bool(leak > floor_high and lower > floor_median),
        })
        gains.append(gain)
        bases.append(np.asarray(narrow.row_sse))
        sses.append((float(narrow.sse), float(wide.sse), np.array([fit.sse for fit in surrogates])))
    # A fold where the wider model did measurably worse than the narrower one
    # it contains is a fold where the estimator failed: its upper bound is below
    # zero, which no real effect can produce. With at least two valid folds left
    # it is dropped from the pooled reading and named in the report; with fewer,
    # nothing is pooled away and the reading cannot close (`estimator_failed`).
    # MEASURED 25 October: on worlds that develop from inside, dropping failed
    # folds left a real broker open in 30 of 30 and closed an idle periphery in
    # 27 of 30; on seed 23 the first fold, trained on her first fifth with her
    # organs whole, read -0.21 while the other two read +0.0055 and +0.0053.
    valid = list(only) if only is not None else [i for i, fold in enumerate(per_fold) if fold["leak_upper"] >= 0.0]
    if 2 <= len(valid) < len(per_fold) or (only is not None and len(valid) < len(per_fold)):
        for i, fold in enumerate(per_fold):
            if i not in valid:
                fold["dropped"] = True
    else:
        valid = list(range(len(per_fold)))
    sse_own = sum(sses[i][0] for i in valid)
    sse_both = sum(sses[i][1] for i in valid)
    floor_sse = np.sum([sses[i][2] for i in valid], axis=0)
    pooled = (sse_own - sse_both) / sse_own if sse_own > 1e-12 else 0.0
    floor = (sse_own - floor_sse) / sse_own if sse_own > 1e-12 else np.zeros_like(floor_sse)
    lower, upper = _pooled_bounds([gains[i] for i in valid], [bases[i] for i in valid], seed=seed)
    return {
        "folds": per_fold,
        "leak": float(pooled),
        "leak_lower": lower,
        "leak_upper": upper,
        "shuffled_leak": float(np.median(floor)),
        "floor_high": float(np.quantile(floor, 0.95)) if floor.size > 1 else float(floor[0]),
        "loss_core": float(np.median([fold["loss_core"] for fold in per_fold])),
        "loss_core_and_periphery": float(np.median([fold["loss_core_and_periphery"] for fold in per_fold])),
        "loss_shuffled_periphery": float(np.median([fold["loss_shuffled_periphery"] for fold in per_fold])),
    }


def _screened(
    own: np.ndarray, change: np.ndarray, outside: np.ndarray, train: slice, validate: slice, *, limit: int
) -> list[int]:
    """The periphery columns that could add most to K*'s prediction of K's change, on the fitting rows.

    The added-variable score: a column's gain, added to a model that already
    holds K*, comes through its correlation with what that model leaves
    unexplained in each target, so each column is scored by the sum over
    targets of its squared correlation with the residual.
    """
    from core.subject.estimate import _ridge, _standardise

    rows = slice(train.start, validate.stop)
    fit = fit_predict(own, change, train=train, validate=validate, test=validate)
    x, _keep = _standardise(own[rows], own[rows])
    y = change[rows]
    spread = y.std(axis=0)
    y = (y - y.mean(axis=0)) / np.where(spread > 1e-12, spread, 1.0)
    weights = _ridge(x, y, max(float(fit.alpha), 1e-3))
    residual = y - x @ weights
    residual = residual / np.where(residual.std(axis=0) > 1e-12, residual.std(axis=0), 1.0)
    block = outside[rows]
    block = (block - block.mean(axis=0)) / np.where(block.std(axis=0) > 1e-12, block.std(axis=0), 1.0)
    correlation = (block.T @ (residual - residual.mean(axis=0))) / max(1, block.shape[0])
    scores = (correlation**2).sum(axis=1)
    return [int(i) for i in np.argsort(-scores)[:limit]]


def _opens(scored: Mapping[str, Any]) -> bool:
    if any(fold["open"] and not fold.get("dropped") for fold in scored["folds"]):
        return True
    return scored["leak"] > scored["floor_high"] and scored["leak_lower"] > scored["shuffled_leak"]


def _resolution(
    now: np.ndarray, env: np.ndarray, nxt: np.ndarray, outside: np.ndarray, *, seed: int, draws: int,
    only: Sequence[int] | None = None,
) -> dict[str, Any]:
    """The smallest planted leak the same estimator calls open, among the same periphery.

    A planted cause is the NEXT value of the column K predicts worst (at row t
    the value at t + 1, a leak by construction), spoilt by more and more noise
    and appended to the real periphery, so it has to be found among the same
    four hundred columns and on the same folds the verdict reads. Planted
    alone beside K, as the control did until 25 October, it was found more
    easily than any real leak could be.
    """
    order = np.argsort(-np.abs(np.diff(nxt, axis=0)).std(axis=0))
    ladder: list[dict[str, Any]] = []
    resolved: float | None = None
    carried = nxt[:, order[:1]]

    def plant(columns: np.ndarray, noise: float) -> np.ndarray:
        rng = np.random.default_rng(seed + 1)
        spread = columns.std(axis=0)
        spread = np.where(spread > 1e-12, spread, 1.0)
        return columns + rng.standard_normal(columns.shape) * spread * noise

    # How many of K's hardest columns the plant carries. One column predicted
    # perfectly is one of K's several hundred targets: on seed 23, once a
    # failing fold was dropped, it could not explain more than 0.013 of what K
    # leaves unexplained, under the surrogate floor's 0.0137, so the gate could
    # never be shown to find a leak the size of the margin however good it was.
    # The plant doubles its columns until, noiseless, it reaches the margin.
    target = CLOSURE_MARGIN if CLOSURE_MARGIN is not None else 0.0
    width = 1
    while width < min(len(order), 64):
        trial = _scored(now, env, nxt, np.hstack([outside, nxt[:, order[:width]]]), seed=seed + 7, draws=1, only=only)
        if trial["leak"] >= target:
            break
        width *= 2
    carried = nxt[:, order[: min(width, len(order))]]

    def rung(noise: float, refined: bool = False) -> bool:
        nonlocal resolved
        planted = plant(carried, noise)
        scored = _scored(now, env, nxt, np.hstack([outside, planted]), seed=seed + 7, draws=draws, only=only)
        found = _opens(scored)
        ladder.append({
            "noise_in_spreads": round(noise, 6),
            "leak": round(scored["leak"], 6),
            "leak_lower": round(scored["leak_lower"], 6),
            "shuffled_q95": round(scored["floor_high"], 6),
            "detected": found,
            "refined": refined,
        })
        # A plant read open shows the gate's reach only by a positive pooled
        # leak: one fold can open while the pooled leak is below zero, and a
        # resolution below zero passed `<= margin` whatever the gate could find
        # (the carrier of 26 October recorded -0.0014).
        usable = math.isfinite(scored["leak"]) and scored["leak"] > 0.0
        if found and usable and (resolved is None or scored["leak"] < resolved):
            resolved = scored["leak"]
        return found

    found_at: float | None = None
    missed_at: float | None = None
    for noise in (0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0):
        if not rung(noise):
            # Spoiling it further only makes the plant fainter.
            missed_at = noise
            break
        found_at = noise
    # The rungs are coarse where the planted leak falls fastest: on seed 23 with
    # her organs whole it was found at 0.197 and missed at the next rung,
    # 0.0117, with the margin between them, so the ladder reported the grid's
    # step as the gate's reach. Bisecting between the last rung found and the
    # first missed measures the smallest leak the gate actually finds.
    if found_at is not None and missed_at is not None:
        for _ in range(6):
            middle = (found_at + missed_at) / 2.0
            if rung(middle, refined=True):
                found_at = middle
            else:
                missed_at = middle
    return {
        "carried_next_value_of": f"the {carried.shape[1]} columns K predicts worst, spoilt by noise, among the real periphery",
        "ladder": ladder,
        "detected": resolved is not None,
        "smallest_leak_resolved": resolved,
        "draws": draws,
    }


def least_domain_margin(recording: Recording, *, seed: int = 0, environment: bool = True) -> dict[str, Any]:
    """The smallest leak one of K's own domains shows from outside. Withdrawn as the margin.

    It was preregistered as the margin on 25 October and withdrawn the same day,
    and is kept so the record can be checked. The argument for it was that the
    partition sweep shows every domain is needed by the rest, so each is a broker
    known to matter. That conflates two things: a domain can be needed by the
    rest and still carry almost nothing the rest's own past does not, because
    the rest already holds most of what it would say. In a world built with every
    domain coupled the least of these leaks sits near zero, and on seed 23 four
    of her ten read at or below zero, so as a margin it made closure impossible
    whatever the gate measured. What follows is the original reasoning.


    Each domain in turn is left out of the core and read as its periphery, by
    the same estimator on the same folds with the same environment: the rest of
    K predicting its own next turn, against the rest of K and that domain. The
    partition sweep decides that every domain is needed by the others, so each
    is a broker known to matter, at her run length and her grain. The smallest
    of their leaks is the margin. Closed within it says that everything outside
    K together tells K's future less than the least part of K does, and no
    fraction or constant was chosen to say so.

    The margin is not a bound on its own estimate. A domain that tells the rest
    nothing given E makes the margin zero or less, and then no closed reading
    is possible: if a part of K is as idle as that, "less than any part of K"
    is not a claim the gate can make.
    """
    live = recording.live_columns()
    rows = recording.x.shape[0] - 1
    env = np.zeros((rows, 0))
    if environment:
        found, _names = environment_of_transitions(recording)
        if found.shape[0] == rows:
            env = found
    by_domain: dict[str, float] = {}
    for key in recording.live_domains():
        inside = np.zeros(live.shape[0], dtype=bool)
        inside[recording.slices[key]] = True
        mine, rest = live & inside, live & ~inside
        if not mine.any() or not rest.any() or rows < 60:
            continue
        outside = recording.x[:-1][:, mine]
        outside = _as_increments(outside, np.ones(outside.shape[1], dtype=bool))
        before = recording.x[:-1][:, rest]
        scored = _scored(before, env, recording.x[1:][:, rest] - before, outside, seed=seed, draws=1)
        by_domain[key] = float(scored["leak"])
    if not by_domain:
        return {"rule": LEAST_DOMAIN, "margin": None, "domain": None, "by_domain": {}}
    least = min(by_domain, key=lambda key: by_domain[key])
    return {
        "rule": LEAST_DOMAIN,
        "margin": by_domain[least],
        "domain": least,
        "by_domain": {key: round(value, 6) for key, value in by_domain.items()},
    }


def closure_gain(
    recording: Recording,
    periphery: np.ndarray,
    names: tuple[str, ...],
    *,
    seed: int = 0,
    rank: int = 6,
    draws: int = SURROGATE_DRAWS,
    margin: float | None = None,
    environment: bool = True,
) -> ClosureReport:
    """How much of K's future the rest of the machine explains that K and its environment do not.

    The floor is the same comparison against surrogate peripheries that keep
    every column's course in time and lose its alignment with K (`_surrogate`),
    so a gain that clears it is information rather than the extra degrees of
    freedom a wider model always brings, or a slow column drifting beside a
    core that develops. The leak is pooled over three forward folds as the sum
    of what the periphery saves over the sum of what K and E leave, so a
    stretch of the run where K predicts itself almost perfectly cannot inflate
    a tiny absolute gain into a large ratio. `margin` defaults to the
    preregistered `CLOSURE_MARGIN`.
    """
    margin = CLOSURE_MARGIN if margin is None else margin
    live = recording.live_columns()
    now = recording.x[:-1][:, live]
    nxt = recording.x[1:][:, live]
    # K*: what K's transition reads beyond its ten domains, recorded under
    # `held.` (core/subject/carrier_reads.py), is part of the carrier and never
    # the periphery. It predicts; K's own columns are what is predicted. With
    # her organs whole that is about six thousand columns, and every one of them
    # as a target made each fit some 10^14 operations; whether anything outside
    # moves the held state itself is asked exactly, by the interventional test
    # (ISC-v6 addendum, 25 October, night), which compares every held reading.
    if periphery.size and names:
        held = np.array([name.startswith("held.") for name in names], dtype=bool)
        if held.any():
            block = periphery[:, held]
            moving = (block.std(axis=0) > 1e-9) & ~np.array(
                [_named_clock(name) for name, flag in zip(names, held, strict=True) if flag], dtype=bool
            )
            block = block[:, moving]
            if block.shape[1] and block.shape[0] == recording.x.shape[0]:
                now = np.hstack([now, block[:-1]])
            periphery = periphery[:, ~held]
            names = tuple(name for name, flag in zip(names, held, strict=True) if not flag)
    outside = periphery[:-1]
    if outside.size == 0 or now.shape[0] < 60:
        return ClosureReport(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, int(outside.shape[1] if outside.size else 0),
                             margin=margin)
    env, _env_names = environment_of_transitions(recording) if environment else (np.zeros((now.shape[0], 0)), ())
    if env.shape[0] != now.shape[0]:
        env = np.zeros((now.shape[0], 0))

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
    # Every outside column enters as its increment, not only the one-way ones,
    # and every model predicts K's change rather than its next level. On a
    # system that develops, a drifting level regresses spuriously on a core that
    # drifts beside it: MEASURED 25 October on worlds whose development runs
    # inside K, as hers does, unrelated drifting columns read open in 4 of 15
    # and the estimator failed a fold in all 15; entered this way, 3 of 30 read
    # open, 2 of 30 failed a fold, and a real broker opened in 30 of 30. Ridge
    # then shrinks toward no change, which is persistence, rather than toward
    # the training mean of something that moves.
    outside = _as_increments(outside, np.ones(outside.shape[1], dtype=bool))
    if outside.shape[1] == 0:
        return ClosureReport(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0, margin=margin)
    change = nxt - now[:, : nxt.shape[1]]

    scored = _scored(now, env, change, outside, seed=seed, draws=draws)
    folds = forward_folds(now.shape[0])
    own = np.hstack([now, env]) if env.size else now
    train, validate, test = folds[-1]
    core = fit_predict(own, change, train=train, validate=validate, test=test)
    base = core.loss if core.loss > 1e-12 else 1.0

    # Which outside variables carry it, one at a time, so the answer names a
    # thing rather than reporting an amount.
    ranked: list[tuple[str, float]] = []
    if scored["leak"] > 0.0:
        # Screened first, then fitted exactly. With K* holding her organs whole
        # each exact fit is a model of six and a half thousand columns, and one
        # per periphery column was two hours of a run for a diagnostic. A column
        # can only carry gain through what K* leaves unexplained, so the columns
        # whose increments correlate most with that residual on the training rows
        # are the ones fitted; the rest cannot rank above them by much.
        candidates = _screened(own, change, outside, train, validate, limit=max(4 * rank, 40))
        for index in candidates:
            name = kept_names[index]
            single = _fit_nested(
                np.hstack([own, outside[:, index : index + 1]]),
                change,
                train=train,
                validate=validate,
                test=test,
                own_width=own.shape[1],
            )
            gain = (core.loss - single.loss) / base
            if gain > 0.0:
                ranked.append((name, float(gain)))
        ranked.sort(key=lambda pair: -pair[1])

    audit = audit_leaks(own, change, outside, kept_names, ranked[:rank], differenced, seed=seed, draws=draws)
    resolution: dict[str, Any] = {"skipped": "the gate read open, so its resolution decides nothing"}
    drawn: dict[str, Any] = {"named": margin}
    if not _opens(scored):
        read = [i for i, fold in enumerate(scored["folds"]) if not fold.get("dropped")]
        resolution = _resolution(now, env, change, outside, seed=seed, draws=max(8, draws // 4), only=read)
        if margin is None and CLOSURE_MARGIN_RULE == LEAST_DOMAIN:
            drawn = least_domain_margin(recording, seed=seed, environment=environment)
            margin = drawn["margin"]
    audit["planted_cause"] = resolution
    audit["margin"] = drawn
    shuffled = scored["shuffled_leak"]
    shuffled_loss = scored["loss_shuffled_periphery"]
    return ClosureReport(
        audit=audit,
        loss_core=scored["loss_core"],
        loss_core_and_periphery=scored["loss_core_and_periphery"],
        loss_shuffled_periphery=shuffled_loss,
        leak=float(scored["leak"]),
        shuffled_leak=float(shuffled),
        leak_over_shuffle=float(
            (shuffled_loss - scored["loss_core_and_periphery"]) / (shuffled_loss if shuffled_loss > 1e-12 else 1.0)
        ),
        periphery_width=int(outside.shape[1]),
        top_leaks=tuple(ranked[:rank]),
        shuffled_draws=int(draws),
        floor_high=float(scored["floor_high"]),
        folds=tuple(scored["folds"]),
        leaking_folds=int(sum(1 for fold in scored["folds"] if fold["open"] and not fold.get("dropped"))),
        differenced=differenced,
        dropped_as_clocks=tuple(
            name for name, flag in zip(names, clocks, strict=True) if flag
        ),
        leak_lower=scored["leak_lower"],
        leak_upper=scored["leak_upper"],
        smallest_leak_resolved=resolution.get("smallest_leak_resolved"),
        margin=margin,
        environment_width=int(env.shape[1]),
    )


def _organ_of(name: str) -> str:
    """The organ a periphery column belongs to, under the name K uses for it."""
    head = name.split(".instance.", 1)[0]
    head = head[len("service."):] if head.startswith("service.") else head.split(".", 1)[0]
    return _ORGAN_ALIASES.get(head, head)


def audit_leaks(
    now: np.ndarray,
    nxt: np.ndarray,
    outside: np.ndarray,
    names: tuple[str, ...],
    ranked: Sequence[tuple[str, float]],
    differenced: tuple[str, ...],
    *,
    seed: int = 0,
    draws: int = SURROGATE_DRAWS,
) -> dict[str, Any]:
    """What each leak is, and whether the test can find a cause it was not shown.

    Each leaking column is read three ways. One that only moves one way is
    entered as its increment, and is named a clock. One whose gain is gone
    once K's own previous frame is in the model was carrying the core's own
    past: K is not Markov at this grain, and the column is a memory of K,
    not a cause from outside. One whose organ K already reads is that organ's
    state K does not take in. What is left is an outside cause or unread state.

    The positive control is `_resolution`, run on the verdict's own estimator.
    Its history, kept from when it lived here:

    It plants the periphery with the NEXT value of the column K predicts worst —
    at row t the value at t + 1, which is a leak by construction — and then
    spoils it with more and more noise until the gate loses it. What comes back
    is the smallest leak the gate could still resolve, which the recording's own
    leak is read against.

    Widening the plant instead of spoiling it cannot answer this. One hard column
    carried exactly already leaks 0.0594 on the carrier, against its own leak of
    0.0243, so every rung of a widening ladder sits above the number in question
    and none of them says whether it could have been seen.

    Two earlier controls could not have fired. Until 25 October it planted a
    readout of the whole next frame, and K's present already predicts 99% of
    that frame on the carrier, so the plant carried nothing K lacked. The
    replacement planted one random direction of what K's present does not
    predict, with as much noise again: one direction of a 359-column surprise
    carries about a third of a percent of what the loss scores, and the loss
    standardises every target column and weights them equally. Both reported
    `detected False` on every recording they were ever run on.
    """
    from core.subject.estimate import fit_predict, split_rows

    core_organs = set(_ORGAN_ALIASES.values())
    rows = now.shape[0]
    out: dict[str, Any] = {"leaks": []}
    if rows < 61:
        return out
    past_now, past_prev, past_next = now[1:], now[:-1], nxt[1:]
    train, validate, test = split_rows(rows - 1)
    with_past = np.hstack([past_now, past_prev])
    base_past = fit_predict(with_past, past_next, train=train, validate=validate, test=test)
    index_of = {name: i for i, name in enumerate(names)}
    for name, gain in ranked:
        column = outside[1:, index_of[name] : index_of[name] + 1]
        single = fit_predict(
            np.hstack([with_past, column]), past_next,
            train=train, validate=validate, test=test, own_width=with_past.shape[1],
        )
        scale = base_past.loss if base_past.loss > 1e-12 else 1.0
        gain_given_past = (base_past.loss - single.loss) / scale
        organ = _organ_of(name)
        if name in differenced:
            kind = "a clock, entered as its increment"
        elif gain_given_past <= 0.0:
            kind = "a summary of the core's own past"
        elif organ in core_organs:
            kind = "state of an organ K reads, which K does not take in"
        else:
            kind = "an outside cause or unread state"
        out["leaks"].append({
            "variable": name,
            "gain": round(float(gain), 6),
            "gain_given_the_core_past": round(float(gain_given_past), 6),
            "organ": organ,
            "kind": kind,
        })

    return out
