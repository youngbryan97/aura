"""A counted iteration has to be the loop body, not the first line of it.

The battery cannot leave the free-running cognitive layers running — two arms
would differ by the machine's schedule as well as by the displacement — so it
advances them by a count. That is only sound if one counted iteration performs
what one iteration of the production loop performs.

It did not. The layer declaration says, in order, the methods one iteration
calls; the harness asked for a single entry point and took the first name that
resolved. So a counted neurochemical iteration ran `_metabolic_tick` and never
`_push_modulation`, interoception sampled the hardware and never pushed it to
the mesh or triggered a neurochemical event, and oscillatory binding stepped
its oscillators and never emitted a binding moment. Every one of the dropped
methods is the one that leaves the layer. The harness stopped executing the
coupling and the battery then reported the coupling as weak.

The test that existed established that a layer had an entry point, that
`step_once` did not raise, and that enough names appeared in the counts. All
three were true while the defect was there. This compares what a counted
iteration calls against what the loop body calls.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core.subject.steppable import LAYERS, Steps, iterations_at, step_once

ROOT = Path(__file__).resolve().parents[1]


class _Recorder:
    """Stands in for a layer and writes down what was asked of it."""

    def __init__(self, methods):
        self.calls: list[str] = []
        for name in methods:
            setattr(self, name, self._make(name))

    def _make(self, name):
        def call():
            self.calls.append(name)

        return call


class _Holder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _organism_for(layer):
    recorder = _Recorder([call.method for call in layer.body])
    if layer.attribute:
        holder = _Holder(**{layer.attribute: recorder})
    else:
        holder = recorder
    return _Holder(**{layer.holder: holder}), recorder


@pytest.mark.parametrize("layer", LAYERS, ids=lambda item: item.name)
async def test_one_counted_iteration_calls_the_whole_body(layer):
    organism, recorder = _organism_for(layer)
    # A frame long enough that this layer takes at least one iteration.
    seconds = max(1.0 / layer.hz, 1e-6)
    steps = Steps()
    frame = 0
    while not recorder.calls and frame < 4:
        await step_once(organism, frame, steps, seconds=seconds)
        frame += 1
    assert recorder.calls, f"{layer.name} never ran"
    every_iteration = [call.method for call in layer.body if call.every == 1]
    for method in every_iteration:
        assert method in recorder.calls, (
            f"{layer.name} declares {method} in its loop body and the counted "
            f"iteration called only {recorder.calls}"
        )


@pytest.mark.parametrize("layer", LAYERS, ids=lambda item: item.name)
def test_the_declared_body_is_what_the_production_loop_calls(layer):
    """Read the loop out of the source rather than trusting the declaration."""
    target = _loop_source(layer)
    if target is None:
        pytest.skip(f"{layer.name} has no locatable loop body")
    for call in layer.body:
        assert call.method in target, (
            f"{layer.name} declares {call.method}, which does not appear in "
            "its own loop"
        )


def _loop_source(layer) -> str | None:
    """The text of the module that owns this layer's loop."""
    known = {
        "NeuralMesh": "core/consciousness/neural_mesh.py",
        "Neurochemical": "core/consciousness/neurochemical_system.py",
        "EmbodiedInteroception": "core/consciousness/embodied_interoception.py",
        "OscillatoryBinding": "core/consciousness/oscillatory_binding.py",
        "UnifiedField": "core/consciousness/unified_field.py",
        "SubstrateEvolution": "core/consciousness/substrate_evolution.py",
        "ConsciousnessBridge": "core/consciousness/consciousness_bridge.py",
        "ClosedCausalLoop": "core/consciousness/closed_loop.py",
        "StreamOfBeing": "core/consciousness/stream_of_being.py",
    }
    path = known.get(layer.name)
    if path is None:
        return None
    file = ROOT / path
    return file.read_text() if file.exists() else None


async def test_a_sub_schedule_runs_at_its_own_ratio():
    """Oscillatory binding steps at a hundred hertz and emits at ten."""
    layer = next(item for item in LAYERS if item.name == "OscillatoryBinding")
    organism, recorder = _organism_for(layer)
    steps = Steps()
    # One second of frames at a tenth of a second each.
    for frame in range(10):
        await step_once(organism, frame, steps, seconds=0.1)
    stepped = recorder.calls.count("_oscillator_step")
    emitted = recorder.calls.count("_emit_binding_moment")
    assert stepped == 100, stepped
    assert emitted == 10, emitted


def test_the_rate_is_exact_over_the_long_run():
    """Fractions that do not fit in one frame land in the next, not nowhere."""
    layer = next(item for item in LAYERS if item.name == "NeuralMesh")
    seconds = 0.02
    total = sum(iterations_at(layer, frame, seconds)[1] for frame in range(1000))
    # 10 Hz over 1000 frames of 0.02 s is 20 seconds of life.
    assert total == pytest.approx(layer.hz * 1000 * seconds, abs=1)


def test_the_frame_is_worth_what_the_experiment_clock_says():
    """Two timelines is one too many.

    A declared frame length beside a measured clock step is two clocks: a
    layer told to run every five minutes runs every five minutes of one of
    them while everything around it ages on the other.
    """
    from core.subject.clock import ExperimentClock
    from core.subject.steppable import frame_seconds

    clock = ExperimentClock(0.037)
    clock.install()
    try:
        assert frame_seconds() == pytest.approx(0.037)
    finally:
        clock.uninstall()


def test_a_layer_missing_any_declared_method_is_unsteppable():
    """Part of an iteration reported as an iteration is the original defect."""
    from core.subject.steppable import missing_entry_points

    layer = next(item for item in LAYERS if len(item.body) > 1)
    recorder = _Recorder([layer.body[0].method])
    holder = _Holder(**{layer.attribute: recorder}) if layer.attribute else recorder
    organism = _Holder(**{layer.holder: holder})
    missing = missing_entry_points(organism)
    assert layer.name in missing, missing
    assert layer.body[1].method in missing[layer.name]


def test_the_declaration_names_no_method_the_object_lacks():
    """A typo in a declared name would read as an unsteppable layer for ever."""
    for layer in LAYERS:
        for call in layer.body:
            assert call.method.isidentifier(), (layer.name, call.method)


def test_no_layer_is_scheduled_by_a_second_constant():
    """`FRAME_SECONDS` was that second constant; it must not come back."""
    source = (ROOT / "core" / "subject" / "steppable.py").read_text()
    tree = ast.parse(source)
    assigned = {
        node.targets[0].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and node.targets
        and isinstance(node.targets[0], ast.Name)
    }
    assert "FRAME_SECONDS" not in assigned


def test_step_once_is_a_coroutine():
    assert inspect.iscoroutinefunction(step_once)
