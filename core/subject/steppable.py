"""Advance the free-running cognitive layers by a count instead of by a clock.

Eleven layers run their own loops on wall-clock timers. A paired intervention
cannot leave them running — the arm that runs while the machine is busy gets
fewer iterations than the arm that runs while it is idle, and the two arms then
differ by the host's schedule as well as by the displacement — so the battery
stopped them, and stopping them meant that whatever those layers contribute to
the coupling between domains was absent from every number it reported.

Both arms take the same number of steps here, because the harness counts them.
Nothing new computes: each entry below calls the same methods the layer's own
loop calls, in the same order and on the same sub-schedule, so there is one
implementation of the cognition and two schedules for it.

Three things this has to get right, and two of them it got wrong.

**The whole loop body, not the first method of it.** A layer declares the
methods one iteration of its loop calls. The first version asked for a single
entry point and took the first name that resolved, so a counted neurochemical
iteration ran `_metabolic_tick` and never `_push_modulation`; interoception
sampled the hardware and never pushed it to the mesh or triggered a
neurochemical event; oscillatory binding stepped its oscillators and never
emitted a binding moment. In every case the method that was dropped is the one
that leaves the layer — the coupling itself — and the battery then reported
the coupling as weak. A measurement that stops executing the coupling cannot
be evidence about it.

**The layer's own rate against the experiment's own clock.** A frame used to
be worth a declared half-second while the clock advanced by whatever
`calibrate_clock` measured, so the two disagreed by construction; and the
frames-between-steps count was clamped at one, which capped every layer at two
hertz. The mesh runs at ten, the field at twenty, the oscillators at a hundred:
the fastest of them was being integrated at a fiftieth of its rate, and the
mesh at a fifth. Here a layer's iterations in a frame are read off its own
declared rate and the frame's own duration, and a rate above one per frame runs
more than once.

**A schedule that is a function of the frame index.** No accumulator, so two
arms handed the same frame numbers run the same layers the same number of
times without carrying any state across the fork.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Subject.Steppable")

__all__ = ["Steps", "frame_seconds", "layers_of", "missing_entry_points", "step_once"]


@dataclass(frozen=True, slots=True)
class Call:
    """One method of a loop body, and how often the body calls it.

    `every` is in iterations of that loop, not frames. Oscillatory binding
    steps its oscillators on every internal tick and emits a binding moment on
    every tenth, which is one loop with two rates in it.
    """

    method: str
    every: int = 1


@dataclass(frozen=True, slots=True)
class Layer:
    """One cognitive layer, its rate in life, and what one iteration calls."""

    name: str
    #: Where the object lives: ("bridge", "neural_mesh") or ("consciousness", …).
    holder: str
    attribute: str
    #: The loop's own iteration rate, in hertz, as its own configuration
    #: declares it. Read from the layer rather than chosen here.
    hz: float
    #: In order, what one iteration of the loop body calls.
    body: tuple[Call, ...]


#: What a frame is worth when no experiment clock is installed. A run installs
#: one and this is not used; it exists so the schedule is defined for a caller
#: that steps the layers without one.
DEFAULT_FRAME_SECONDS: float = 0.05


def frame_seconds() -> float:
    """How much time one frame of the harness is worth.

    The experiment's clock decides, because it is the clock the organism reads.
    A separate constant here would be a second timeline: a layer declared to
    run every five minutes would run every five minutes of one clock while
    everything around it aged on the other.
    """
    try:
        from core.subject.clock import installed_clock

        clock = installed_clock()
    except ImportError:  # pragma: no cover - the clock ships with this package
        clock = None
    if clock is None:
        return DEFAULT_FRAME_SECONDS
    return max(1e-6, float(clock.step))


#: The nine free-running cognitive layers, in the order the bridge brings them
#: up. Each rate and each method name is the one inside that loop's own
#: `while self._running:` body, so nothing here is a second implementation.
LAYERS: tuple[Layer, ...] = (
    Layer("NeuralMesh", "bridge", "neural_mesh", 10.0, (Call("_tick"),)),
    Layer(
        "Neurochemical",
        "bridge",
        "neurochemical",
        2.0,
        (Call("_metabolic_tick"), Call("_push_modulation")),
    ),
    Layer(
        "EmbodiedInteroception",
        "bridge",
        "interoception",
        1.0,
        (
            Call("_sample_hardware"),
            Call("_push_to_mesh"),
            Call("_trigger_neurochemical_events"),
        ),
    ),
    # A hundred hertz internally, to resolve forty-hertz gamma, emitting at
    # ten: the sub-schedule is the loop's own `_internal_tick % output_every`.
    Layer(
        "OscillatoryBinding",
        "bridge",
        "oscillatory_binding",
        100.0,
        (
            Call("_oscillator_step"),
            Call("_compute_synchronization", every=10),
            Call("_emit_binding_moment", every=10),
        ),
    ),
    Layer("UnifiedField", "bridge", "unified_field", 20.0, (Call("_tick"),)),
    # A generation every five minutes.
    Layer("SubstrateEvolution", "bridge", "substrate_evolution", 1.0 / 300.0, (Call("_run_generation"),)),
    Layer("ConsciousnessBridge", "bridge", "", 10.0, (Call("_integration_tick"),)),
    Layer("ClosedCausalLoop", "consciousness", "closed_loop", 1.0 / 2.5, (Call("step"),)),
    Layer("StreamOfBeing", "consciousness", "stream_of_being", 1.0 / 2.5, (Call("step"),)),
)


@dataclass
class Steps:
    """What was advanced, how often, and what could not be."""

    counts: dict[str, int] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    unsteppable: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": dict(sorted(self.counts.items())),
            "calls": dict(sorted(self.calls.items())),
            "unsteppable": dict(sorted(self.unsteppable.items())),
            "failures": dict(sorted(self.failures.items())),
        }


def _target(organism: Any, layer: Layer) -> Any:
    holder = getattr(organism, layer.holder, None)
    if holder is None:
        return None
    if not layer.attribute:
        return holder
    return getattr(holder, layer.attribute, None)


def layers_of(organism: Any) -> dict[str, Any]:
    """Every declared layer that is actually here, by name."""
    found: dict[str, Any] = {}
    for layer in LAYERS:
        target = _target(organism, layer)
        if target is not None:
            found[layer.name] = target
    return found


def iterations_at(layer: Layer, frame: int, seconds: float) -> tuple[int, int]:
    """The layer's first iteration index in this frame, and how many it takes.

    A function of the frame index alone, so nothing has to be carried across a
    fork and two arms handed the same frames run the same iterations. The long
    run rate is exact: the boundaries are floors of the same product, so the
    fractions that do not fit in one frame are not lost, they land in the next.
    """
    rate = max(0.0, float(layer.hz)) * max(0.0, float(seconds))
    if rate <= 0.0:
        return 0, 0
    start = math.floor(frame * rate)
    end = math.floor((frame + 1) * rate)
    return start, max(0, end - start)


async def step_once(
    organism: Any,
    frame: int,
    steps: Steps | None = None,
    *,
    seconds: float | None = None,
) -> Steps:
    """Advance every layer by the iterations its own rate calls for in one frame.

    `frame` is the harness's own count, so which layers move on a given call is
    a function of the count and not of the clock.
    """
    record = steps if steps is not None else Steps()
    span = frame_seconds() if seconds is None else max(1e-6, float(seconds))
    for layer in LAYERS:
        target = _target(organism, layer)
        if target is None:
            continue
        start, count = iterations_at(layer, frame, span)
        if count <= 0:
            continue
        missing = _missing(target, layer)
        if missing:
            record.unsteppable[layer.name] = (
                f"{type(target).__name__} has no {', '.join(missing)}"
            )
            continue
        for index in range(count):
            iteration = start + index
            for call in layer.body:
                if call.every > 1 and iteration % call.every:
                    continue
                method = getattr(target, call.method, None)
                try:
                    outcome = method()
                    if inspect.isawaitable(outcome):
                        await outcome
                    record.calls[f"{layer.name}.{call.method}"] = (
                        record.calls.get(f"{layer.name}.{call.method}", 0) + 1
                    )
                except Exception as exc:  # noqa: BLE001 - a layer that raises has still been asked
                    record.failures[f"{layer.name}.{call.method}"] = (
                        f"{type(exc).__name__}: {exc}"[:160]
                    )
            record.counts[layer.name] = record.counts.get(layer.name, 0) + 1
    return record


def _missing(target: Any, layer: Layer) -> tuple[str, ...]:
    """The declared methods this object does not have. Empty is the bar."""
    return tuple(
        call.method
        for call in layer.body
        if not callable(getattr(target, call.method, None))
    )


def missing_entry_points(organism: Any) -> dict[str, str]:
    """Which live layers cannot be advanced by a count. Empty is the bar.

    A layer counts as unsteppable when any method of its loop body is absent,
    not when all of them are: running part of an iteration and reporting the
    layer stepped is how the coupling came to be missing from the measurement
    while the report said the layer had run.
    """
    out: dict[str, str] = {}
    for layer in LAYERS:
        target = _target(organism, layer)
        if target is None:
            continue
        missing = _missing(target, layer)
        if missing:
            out[layer.name] = f"{type(target).__name__} has no {', '.join(missing)}"
    return out


def names(layers: Sequence[Layer] = LAYERS) -> tuple[str, ...]:
    return tuple(layer.name for layer in layers)
