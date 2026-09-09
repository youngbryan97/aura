"""A forward model that is never trained does not have a small error.

`UnifiedWorldModel.observe(..., learn=True)` appends the step to a replay
buffer, and the gradient steps are taken by a background lane that has to be
started. The only caller of `start_training` in the tree was the ontogeny
organ, on its own separate model. The one the cognitive cycle observes into —
the one affect grounding reads as prediction error and the free-energy engine
takes as a signal — collected a replay buffer for the whole of every session
and never took a single step.

Its surprise was not low. It was arbitrary, and two subsystems read it as
evidence about the world.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_the_cycles_model_starts_its_own_training_lane() -> None:
    from core.world_model.learned_world_model import get_learned_world_model
    from core.world_model.unified_world_model import UnifiedWorldModel

    model = get_learned_world_model()
    model.stop_training()
    assert model._trainer_thread is None

    unified = UnifiedWorldModel()
    assert unified.learned is not None
    try:
        assert model._trainer_thread is not None, "the lane the cycle feeds never starts"
        assert model._trainer_thread.is_alive()
    finally:
        model.stop_training()


def test_stopping_the_lane_actually_stops_it() -> None:
    from core.world_model.learned_world_model import get_learned_world_model

    model = get_learned_world_model()
    model.start_training(interval_s=0.05)
    assert model._trainer_thread is not None
    model.stop_training()
    assert model._trainer_thread is None
    # And it can be started again, which a flag left set would prevent.
    model.start_training(interval_s=0.05)
    assert model._trainer_thread is not None
    model.stop_training()


def test_training_moves_the_weights() -> None:
    from core.world_model.learned_world_model import get_learned_world_model

    model = get_learned_world_model()
    model.stop_training()
    rng = np.random.default_rng(11)
    before = np.array(model.W_dec, copy=True)
    for _ in range(64):
        model.observe(rng.random(model.config.observation_dim), learn=True)
    steps = model._train_steps
    model.train_now(passes=4)
    assert model._train_steps > steps
    assert not np.allclose(before, model.W_dec), "a training pass moved nothing"


def test_the_battery_winds_the_thread_down_too() -> None:
    """Cancelling every asyncio task leaves a plain daemon thread running."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "subject" / "organism.py").read_text()
    assert "_stop_threads" in source
    assert "stop_training" in source
