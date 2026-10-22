"""Plasticity keeps the mesh's synapses off their bound, and the codes are bound fresh.

On 21 October the mesh's strongest weight climbed from 0.8 to the clip of one in
every validation run with the conjunctive gains on, and stayed at 0.8 with them
off: hard-bounded STDP piles weights up at the clip, and the core read the climb
as a clock. Soft bounds strengthen a synapse by less the nearer it is to one and
weaken it in proportion to its strength (van Rossum, Bi and Turrigiano 2000).
"""
from __future__ import annotations

import copy

import numpy as np


def _small_mesh():
    from core.consciousness.neural_mesh import MeshConfig, NeuralMesh

    return NeuralMesh(MeshConfig(total_neurons=512, columns=16, neurons_per_column=32, sensory_end=4, association_end=10))


def test_repeated_causal_pairings_leave_every_synapse_below_one():
    mesh = _small_mesh()
    mesh._foreground_request_active = lambda: False
    gain, _plasticity, noise = mesh._modulatory_state
    mesh._modulatory_state = (gain, 50.0, noise)  # learning far faster than she ever does
    column = mesh.columns[0]
    excitatory = (column.W > 0.0) & ~column.inh_mask[None, :]
    column.W[excitatory] = 0.95
    # Every unit fires in order, one millisecond apart, so each pair with the
    # earlier unit presynaptic is a causal pairing inside the window.
    column.last_spike_time = 1.0 + 0.001 * np.arange(column.W.shape[0], dtype=np.float64)
    for _ in range(200):
        mesh._apply_stdp(2.0)
    assert float(np.max(np.abs(column.W))) < 1.0


def test_a_unit_is_this_cycles_gain_field_whatever_her_affect_carries_over():
    from core.consciousness.conjunctive_codes import bind
    from core.state.aura_state import AuraState

    rng = np.random.default_rng(3)
    state = AuraState.default()
    for _ in range(10):
        state.affect.valence = float(rng.normal(0.0, 0.3))
        state.identity.stability = float(rng.uniform())
        bind(state)
    slow, quick = copy.deepcopy(state), copy.deepcopy(state)
    slow.affect.momentum, quick.affect.momentum = 0.99, 0.01
    assert bind(slow).units == bind(quick).units
