"""core/worlds/mujoco_backend.py
──────────────────────────────
MuJoCo-backed high-fidelity simulation for Aura's worlds.

Two integration seams, both honest about what MuJoCo adds (full 6-DoF
rigid-body dynamics with oriented boxes, soft contacts, and a mature
constraint solver) versus the native engine (exact, digest-stable,
dependency-free):

1. ``MujocoAgentSimulator`` implements the existing
   ``SimulatorInterface`` (reset/step toward a target with 2D force
   actions), so every trainer written against the local 2D simulator
   can run on MuJoCo unchanged — the promise simulator_bridge made.
2. ``blueprint_to_mjcf`` compiles a generated WorldBlueprint into MJCF,
   so Aura's procedurally generated (and sculpted) worlds can be
   simulated at high fidelity: oriented boxes, real friction cones.

MuJoCo is optional: ``mujoco_available()`` gates everything, and the
native engine remains the canonical, digest-stable substrate.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.embodiment.simulator_bridge import SimObservation, SimulatorInterface
from core.worlds.generation import WorldBlueprint
from core.worlds.physics import PhysicsError

_MUJOCO_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    # not a failure: the module is optional here, and its absence is the answer.
    except ImportError:
        return False


def blueprint_to_mjcf(blueprint: WorldBlueprint, *, terrain_columns: int = 8) -> str:
    """Compile a WorldBlueprint into MJCF, mirroring the native
    realization (ground plane, terrain columns, props)."""
    size = blueprint.size
    heights = np.asarray(blueprint.heightfield, dtype=np.float64)
    columns = max(1, min(int(terrain_columns), size))
    cell = size / columns
    parts: list[str] = [
        '<mujoco model="aura_world">',
        '  <option timestep="0.004" gravity="0 0 -9.81"/>',
        "  <worldbody>",
        '    <geom name="ground" type="plane" size="0 0 1" friction="0.8 0.005 0.0001"/>',
    ]
    for i in range(columns):
        for j in range(columns):
            sample_x = min(int((i + 0.5) * cell), size - 1)
            sample_y = min(int((j + 0.5) * cell), size - 1)
            height = float(heights[sample_x, sample_y])
            if height < 0.05:
                continue
            parts.append(
                f'    <geom name="terrain_{i}_{j}" type="box" '
                f'pos="{(i + 0.5) * cell - size / 2.0:.4f} '
                f'{(j + 0.5) * cell - size / 2.0:.4f} {height / 2.0:.4f}" '
                f'size="{cell / 2.0:.4f} {cell / 2.0:.4f} {max(height / 2.0, 1e-3):.4f}"/>'
            )
    for entity in blueprint.entities:
        x, y, z = (float(v) for v in entity["position"])
        entity_id = str(entity["entity_id"])
        mass = float(entity.get("mass", 1.0) or 0.0)
        if str(entity.get("shape")) == "sphere":
            radius = float(entity.get("radius", 0.4))
            geom = f'<geom type="sphere" size="{radius:.4f}"/>'
        else:
            hx, hy, hz = (float(v) for v in entity.get("half_extents", (0.4, 0.4, 0.4)))
            geom = f'<geom type="box" size="{hx:.4f} {hy:.4f} {hz:.4f}"/>'
        if mass > 0.0:
            parts.append(
                f'    <body name="{entity_id}" pos="{x:.4f} {y:.4f} {z:.4f}">'
                f'<freejoint/>{geom}</body>'
            )
        else:
            parts.append(
                f'    <body name="{entity_id}" pos="{x:.4f} {y:.4f} {z:.4f}">{geom}</body>'
            )
    parts.extend(["  </worldbody>", "</mujoco>"])
    return "\n".join(parts)


def simulate_blueprint(
    blueprint: WorldBlueprint, *, seconds: float = 2.0
) -> dict[str, Any]:
    """Run a blueprint under MuJoCo and report settled body poses —
    a cross-engine sanity surface for the native world."""
    if not mujoco_available():
        raise PhysicsError("mujoco is not installed")
    import mujoco

    model = mujoco.MjModel.from_xml_string(blueprint_to_mjcf(blueprint))
    data = mujoco.MjData(model)
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    poses: dict[str, list[float]] = {}
    for body_index in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_index)
        if name and name != "world":
            poses[name] = [round(float(v), 5) for v in data.xpos[body_index]]
    return {
        "engine": "mujoco",
        "timestep": float(model.opt.timestep),
        "steps": steps,
        "bodies": poses,
    }


class MujocoAgentSimulator(SimulatorInterface):
    """SimulatorInterface on MuJoCo: a force-actuated ball seeking a
    target on a plane — drop-in high-fidelity replacement for the
    LocalPhysics2DSimulator."""

    _XML = """
    <mujoco model="aura_agent_sim">
      <option timestep="0.01" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 1" friction="0.4 0.005 0.0001"/>
        <body name="agent" pos="0 0 0.2">
          <freejoint/>
          <geom type="sphere" size="0.2" mass="1.0"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def __init__(self, *, force_scale: float = 6.0):
        if not mujoco_available():
            raise PhysicsError("mujoco is not installed")
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(self._XML)
        self.data = mujoco.MjData(self.model)
        self.force_scale = float(force_scale)
        self.target = np.array([1.0, 1.0], dtype=np.float64)

    def reset(self, seed: int = 0) -> SimObservation:
        rng = np.random.default_rng(seed)
        self._mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:2] = rng.uniform(-1.0, 1.0, size=2)
        self.target = rng.uniform(-1.0, 1.0, size=2)
        self._mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: tuple[float, float]) -> SimObservation:
        fx = float(np.clip(action[0], -1.0, 1.0)) * self.force_scale
        fy = float(np.clip(action[1], -1.0, 1.0)) * self.force_scale
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise PhysicsError("action must be finite")
        self.data.xfrc_applied[1, 0] = fx
        self.data.xfrc_applied[1, 1] = fy
        self._mujoco.mj_step(self.model, self.data)
        return self._obs()

    def _obs(self) -> SimObservation:
        position = (float(self.data.qpos[0]), float(self.data.qpos[1]))
        velocity = (float(self.data.qvel[0]), float(self.data.qvel[1]))
        distance = float(math.hypot(
            position[0] - self.target[0], position[1] - self.target[1]))
        return SimObservation(
            position=position,
            velocity=velocity,
            target=(float(self.target[0]), float(self.target[1])),
            distance=distance,
        )
