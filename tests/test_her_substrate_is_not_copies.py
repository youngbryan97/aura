"""The mesh drives her substrate with distinct numbers, not copies of 64.

On 17 October her recorded state, pooled into 32 bands over 9,334 turns of the
27B, repeated every four bands: the bridge tiled the mesh's 64-wide projection
across her 512 units, so unit i and unit i + 64 were driven identically.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from core.consciousness.consciousness_bridge import ConsciousnessBridge


def _bridge() -> ConsciousnessBridge:
    return ConsciousnessBridge.__new__(ConsciousnessBridge)


def test_a_projection_gives_every_unit_its_own_mix_at_the_inputs_scale():
    bridge = _bridge()
    field = np.random.default_rng(0).uniform(-1, 1, 4096).astype(np.float32)
    out = bridge._projected(field, 512)
    assert out.shape == (512,)
    blocks = out.reshape(8, 64)
    assert not any(np.allclose(blocks[0], blocks[k]) for k in range(1, 8))
    rms_in, rms_out = float(np.sqrt((field ** 2).mean())), float(np.sqrt((out ** 2).mean()))
    assert 0.8 < rms_out / rms_in < 1.2
    # The same map after a restart.
    assert np.array_equal(out, _bridge()._projected(field, 512))


def test_one_tick_leaves_no_copies_in_her_substrate(monkeypatch):
    bridge = _bridge()
    field = np.random.default_rng(1).uniform(-1, 1, 4096).astype(np.float32)
    mesh = SimpleNamespace(
        get_executive_projection=lambda: np.tanh(field[:64]),
        get_field_state=lambda: field.copy(),
    )
    substrate = SimpleNamespace(x=np.zeros(512), v=None, sync_lock=threading.Lock())
    bridge.neural_mesh = mesh
    for name in ("unified_field", "neurochemical", "interoception", "oscillatory_binding", "somatic_gate",
                 "substrate_evolution", "substrate_authority", "unified_will", "_cs"):
        setattr(bridge, name, None)
    monkeypatch.setattr(bridge, "_get_substrate", lambda: substrate)
    try:
        bridge._integration_tick()
    except Exception:  # noqa: BLE001 - later steps of the tick need services this test does not build
        pass
    blocks = np.asarray(substrate.x).reshape(8, 64)
    assert np.abs(blocks).sum() > 0
    assert not any(np.allclose(blocks[0], blocks[k]) for k in range(1, 8))
