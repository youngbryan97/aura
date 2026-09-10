"""Advance the free-running cognitive layers by a count instead of by a clock.

Eleven layers run their own loops on wall-clock timers. A paired intervention
cannot leave them running — the arm that runs while the machine is busy gets
fewer iterations than the arm that runs while it is idle, and the two arms then
differ by the host's schedule as well as by the displacement — so the battery
stopped them, and stopping them meant that whatever those layers contribute to
the coupling between domains was absent from every number it reported.

Both arms take the same number of steps here, because the harness counts them.
Nothing new computes: each entry below calls the same method the layer's own
loop calls, so there is one implementation of the cognition and two schedules
for it. A layer that is running and has no entry point is named in the report
rather than passed over, because an unsteppable live loop is the thing this
module exists to make impossible.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.Subject.Steppable")

__all__ = ["Steps", "layers_of", "step_once"]


@dataclass(frozen=True, slots=True)
class Layer:
    """One cognitive layer, and the methods one iteration of its loop calls."""

    name: str
    #: Where the object lives: ("bridge", "neural_mesh") or ("consciousness", …).
    holder: str
    attribute: str
    #: In order, the methods the loop body calls once per iteration.
    methods: tuple[str, ...]
    #: How many of the harness's frames go by between two steps. The layers run
    #: at very different rates in life — the mesh at tens of hertz, evolution
    #: once every five minutes — and stepping evolution as often as the mesh
    #: would be a different organism, not a faithful one.
    every: int = 1


#: How long a frame of the harness is worth, in seconds of the experiment's
#: clock. The layers below step at their own declared rates against it, so a
#: layer that runs at 1 Hz in life runs at 1 Hz here and one that runs every
#: two and a half seconds runs every two and a half seconds. Stepping them all
#: once per frame would be a different organism — faster in some layers, slower
#: in others — rather than the same one on a countable schedule.
FRAME_SECONDS: float = 0.5


def _every(period_s: float) -> int:
    """Frames between two steps of a layer whose own period is `period_s`."""
    return max(1, int(round(period_s / FRAME_SECONDS)))


#: The nine free-running cognitive layers, in the order the bridge brings them
#: up. The method names are the ones inside each loop's own
#: `while self._running:` body, so nothing here is a second implementation.
LAYERS: tuple[Layer, ...] = (
    # 10 Hz in life, and the harness's frame is 0.5 s: once a frame is the
    # closest a counted schedule gets without running it more often than the
    # clock says.
    Layer("NeuralMesh", "bridge", "neural_mesh", ("_tick",)),
    Layer("Neurochemical", "bridge", "neurochemical", ("_metabolic_tick", "_push_modulation")),
    Layer(
        "EmbodiedInteroception",
        "bridge",
        "interoception",
        ("_sample_hardware", "_push_to_mesh", "_trigger_neurochemical_events"),
        every=_every(1.0),
    ),
    Layer(
        "OscillatoryBinding",
        "bridge",
        "oscillatory_binding",
        ("_oscillator_step", "_compute_synchronization", "_emit_binding_moment"),
    ),
    Layer("UnifiedField", "bridge", "unified_field", ("_tick",)),
    # A generation every five minutes.
    Layer(
        "SubstrateEvolution",
        "bridge",
        "substrate_evolution",
        ("_run_generation",),
        every=_every(300.0),
    ),
    Layer("ConsciousnessBridge", "bridge", "", ("_integration_tick",)),
    Layer(
        "ClosedCausalLoop",
        "consciousness",
        "closed_loop",
        ("step",),
        every=_every(2.5),
    ),
    Layer(
        "StreamOfBeing",
        "consciousness",
        "stream_of_being",
        ("step",),
        every=_every(2.5),
    ),
)


@dataclass
class Steps:
    """What was advanced, how often, and what could not be."""

    counts: dict[str, int] = field(default_factory=dict)
    unsteppable: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "steps": dict(sorted(self.counts.items())),
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


async def step_once(organism: Any, frame: int, steps: Steps | None = None) -> Steps:
    """Advance every layer whose turn it is, once.

    `frame` is the harness's own count, so which layers move on a given call is
    a function of the count and not of the clock. Two arms handed the same
    frame numbers therefore run the same layers the same number of times.
    """
    record = steps if steps is not None else Steps()
    for layer in LAYERS:
        target = _target(organism, layer)
        if target is None:
            continue
        if frame % layer.every:
            continue
        method = _entry(target, layer)
        if method is None:
            record.unsteppable[layer.name] = (
                f"none of {', '.join(layer.methods)} on {type(target).__name__}"
            )
            continue
        try:
            outcome = method()
            if inspect.isawaitable(outcome):
                await outcome
            record.counts[layer.name] = record.counts.get(layer.name, 0) + 1
        except Exception as exc:  # noqa: BLE001 - a layer that raises has still been asked
            record.failures[layer.name] = f"{type(exc).__name__}: {exc}"[:160]
    return record


def _entry(target: Any, layer: Layer) -> Any:
    for name in layer.methods:
        method = getattr(target, name, None)
        if callable(method):
            return method
    return None


def missing_entry_points(organism: Any) -> dict[str, str]:
    """Which live layers cannot be advanced by a count. Empty is the bar."""
    out: dict[str, str] = {}
    for layer in LAYERS:
        target = _target(organism, layer)
        if target is None:
            continue
        if _entry(target, layer) is None:
            out[layer.name] = (
                f"{type(target).__name__} has none of {', '.join(layer.methods)}"
            )
    return out


def names(layers: Sequence[Layer] = LAYERS) -> tuple[str, ...]:
    return tuple(layer.name for layer in layers)
