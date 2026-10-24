"""A measurement cannot freeze weights by forgetting a live trainer."""
from __future__ import annotations

import threading

import pytest

from core.world_model.learned_world_model import LearnedWorldModel, WorldModelConfig


def test_timed_out_stop_retains_ownership_and_cannot_spawn_a_second_trainer(monkeypatch):
    model = LearnedWorldModel(WorldModelConfig(observation_dim=3, latent_dim=2, hidden_dim=3, action_dim=2))
    entered, release = threading.Event(), threading.Event()

    def held_cycle():
        entered.set()
        release.wait(2)

    monkeypatch.setattr(model, "train_cycle", held_cycle)
    model.start_training(interval_s=0.001)
    trainer = model._trainer_thread
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="not quiescent"):
            model.stop_training(timeout_s=0.001)
        assert model._trainer_thread is trainer and trainer.is_alive()
        assert model._trainer_stop.is_set()
        model.start_training(interval_s=0.001)
        assert model._trainer_thread is trainer
    finally:
        release.set()
        model.stop_training(timeout_s=2)
    assert model._trainer_thread is None and not model._trainer_stop.is_set()
    model.start_training(interval_s=0.01)
    try:
        assert model._trainer_thread is not trainer and model._trainer_thread.is_alive()
    finally:
        model.stop_training()
