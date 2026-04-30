"""Local embodiment simulator bridge.

Provides a cheap 2D physics world so Aura can form sensorimotor traces without
risking hardware.  The same interface can be implemented by ROS/MuJoCo later,
but this module is already executable and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimObservation:
    position: tuple[float, float]
    velocity: tuple[float, float]
    target: tuple[float, float]
    distance: float

    def to_vector(self) -> np.ndarray:
        return np.array([*self.position, *self.velocity, *self.target, self.distance], dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"position": self.position, "velocity": self.velocity, "target": self.target, "distance": self.distance}


class SimulatorInterface:
    def reset(self, seed: int = 0) -> SimObservation:
        raise RuntimeError("SimulatorInterface.reset must be implemented by a concrete simulator")

    def step(self, action: tuple[float, float]) -> SimObservation:
        raise RuntimeError("SimulatorInterface.step must be implemented by a concrete simulator")


class LocalPhysics2DSimulator(SimulatorInterface):
    def __init__(self, *, dt: float = 0.1, damping: float = 0.92) -> None:
        self.dt = dt
        self.damping = damping
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target = np.array([1.0, 1.0], dtype=np.float32)

    def reset(self, seed: int = 0) -> SimObservation:
        rng = np.random.default_rng(seed)
        self.position = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return self._obs()

    def step(self, action: tuple[float, float]) -> SimObservation:
        force = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)
        self.velocity = (self.velocity + force * self.dt) * self.damping
        self.position = np.clip(self.position + self.velocity * self.dt, -2.0, 2.0)
        return self._obs()

    def proportional_policy(self) -> tuple[float, float]:
        delta = self.target - self.position
        return tuple(np.clip(delta, -1.0, 1.0).astype(float))

    def rollout(self, steps: int = 32) -> list[SimObservation]:
        observations = [self._obs()]
        for _ in range(steps):
            observations.append(self.step(self.proportional_policy()))
        return observations

    def _obs(self) -> SimObservation:
        distance = float(np.linalg.norm(self.target - self.position))
        return SimObservation(
            position=tuple(float(x) for x in self.position),
            velocity=tuple(float(x) for x in self.velocity),
            target=tuple(float(x) for x in self.target),
            distance=distance,
        )


__all__ = ["SimulatorInterface", "SimObservation", "LocalPhysics2DSimulator"]
