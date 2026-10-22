"""A run on one clock trains the world model at its own rate, on the run's time.

On 22 October the pair at 4b1022f25 lost P,M -> W in both arms. One clock
stopped the world model's training lane at bring-up and nothing stepped it, so
the model never learned during a run, and the precision her perception and
memory set together on each of its updates never reached it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.subject.steppable import LAYERS, step_once
from core.world_model.learned_world_model import TRAIN_INTERVAL_S, LearnedWorldModel


def test_the_training_lane_is_a_stepped_layer_at_its_own_rate():
    layer = next(layer for layer in LAYERS if layer.name == "LearnedWorldModel")
    assert layer.hz == 1.0 / TRAIN_INTERVAL_S
    assert [call.method for call in layer.body] == ["train_cycle"]
    assert callable(getattr(LearnedWorldModel, "train_cycle"))


def test_the_harness_runs_a_training_cycle_every_interval_of_run_time():
    cycles: list[int] = []
    organism = SimpleNamespace(world_model=SimpleNamespace(train_cycle=lambda: cycles.append(1)))

    async def run() -> None:
        for frame in range(int(4 * TRAIN_INTERVAL_S)):
            await step_once(organism, frame, seconds=1.0)

    asyncio.run(run())
    assert len(cycles) == 4


def test_a_cycle_learns_from_what_arrived_and_nothing_else():
    model = LearnedWorldModel()
    before = model._pending_since_train
    model.train_cycle()  # nothing arrived: no update, no error
    assert model._pending_since_train == before == 0
