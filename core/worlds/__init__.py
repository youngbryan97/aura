"""core/worlds — spatial world simulation, generation, and hosting.

Three layers, each independently tested:

- physics: deterministic 3D rigid-body dynamics (translational v1 —
  impulse contacts, restitution, Coulomb friction, sleeping). Fixed
  timestep, fixed iteration order, digest-stable: the same world stepped
  twice produces bit-identical state.
- generation: seeded procedural terrain + entity placement. The same
  seed always regenerates the same world; worlds are reproducible
  artifacts, not transient randomness.
- hosting: persistent named worlds with governed atomic persistence and
  an event journal — Aura's worlds survive restarts and remember what
  happened in them (persistent subjective worlds).

Honest scope: translational rigid-body dynamics with spheres, static
planes, and axis-aligned boxes. No rotational dynamics yet — that is a
declared limitation, not an approximation smuggled in as realism.
"""
from core.worlds.physics import Body, PhysicsWorld, PhysicsError
from core.worlds.generation import WorldBlueprint, generate_world
from core.worlds.embodied import EmbodiedAgent, RayHit
from core.worlds.hosting import WorldHost, get_world_host

__all__ = [
    "Body",
    "PhysicsWorld",
    "PhysicsError",
    "EmbodiedAgent",
    "RayHit",
    "WorldBlueprint",
    "generate_world",
    "WorldHost",
    "get_world_host",
]
