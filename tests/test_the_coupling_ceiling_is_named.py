"""A constant inside a tick is a constant nobody can sweep.

The inter-column matrix is renormalised every ten ticks against a ceiling that
was written as a bare 15.0 in the middle of the tick. Anything measuring how
much coupling the mesh needs had no way to reach it, and worse, no way to find
out that it never fires: the matrix this mesh builds has a norm of about 0.65,
forty times below the ceiling meant to bound it.

Naming it changed no behaviour. These tests hold both halves — that the default
is what the tick used to hardcode, and that it is far enough above the mesh's
own coupling to be inert, so a later reading of it as a cap on recruitment has
something to fail against.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from core.consciousness.neural_mesh import MeshConfig, NeuralMesh


def _quiet_ticks(mesh: NeuralMesh, count: int) -> None:
    width = mesh.cfg.sensory_end * mesh.cfg.neurons_per_column
    for _ in range(count):
        mesh.inject_sensory(np.zeros(width, dtype=np.float32))
        mesh._tick_inner()


def test_the_default_is_what_the_tick_used_to_hardcode() -> None:
    assert MeshConfig().inter_column_weight_norm == 15.0


def test_the_ceiling_does_not_bind_the_mesh_it_guards() -> None:
    """If this fails the mesh's coupling has grown into its own ceiling."""
    mesh = NeuralMesh(MeshConfig())
    norm = float(np.linalg.norm(mesh._inter_W))

    assert norm < mesh.cfg.inter_column_weight_norm / 4.0, (
        f"the inter-column norm is {norm:.2f} against a ceiling of "
        f"{mesh.cfg.inter_column_weight_norm}; the guard now bites and every "
        "coupling measurement has to account for it"
    )


def test_a_lower_ceiling_does_bind() -> None:
    """The knob has to work, or naming it bought nothing."""
    mesh = NeuralMesh(dataclasses.replace(MeshConfig(), inter_column_weight_norm=0.2))
    _quiet_ticks(mesh, 12)

    assert float(np.linalg.norm(mesh._inter_W)) <= 0.2 + 1e-5


def test_scaling_the_coupling_survives_the_guard() -> None:
    """A sweep of the coupling must not be quietly undone ten ticks later."""
    mesh = NeuralMesh(MeshConfig())
    mesh._inter_W = mesh._inter_W * 4.0
    scaled = float(np.linalg.norm(mesh._inter_W))
    _quiet_ticks(mesh, 12)

    assert float(np.linalg.norm(mesh._inter_W)) == scaled
