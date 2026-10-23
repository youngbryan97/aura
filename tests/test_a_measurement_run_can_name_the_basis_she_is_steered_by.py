"""A measurement run can name the basis she is steered by.

The reports ground is read from a run with her readings withheld, and the only
route left from her state to her words is the steering channel. The signed
generation's directions shut it (22 October). Directions fitted to her replies
hold the certificate, and a run names their directory in
AURA_STEERING_MEASUREMENT_BASIS. Named, they must exist, be bound to her
checkpoint and carry their fit; unnamed, nothing changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.affective_steering import (
    AFFECTIVE_DIMENSIONS,
    MEASUREMENT_BASIS_ENV,
    SteeringVectorLibrary,
    measurement_basis,
)

HERS = "a" * 64


def _basis(root, descriptor: str = HERS, layer: int = 3, width: int = 8) -> None:
    rng = np.random.default_rng(0)
    for dim in AFFECTIVE_DIMENSIONS:
        v = rng.normal(size=width).astype(np.float32)
        np.savez(root / f"{dim['key']}_layer{layer}.npz", v=v / np.linalg.norm(v),
                 source="fitted_to_her_replies", extracted=True, model_descriptor_sha256=descriptor,
                 reference=0.1, spread=0.2, push=0.01)


def test_unnamed_she_is_steered_by_the_signed_generation(monkeypatch):
    monkeypatch.delenv(MEASUREMENT_BASIS_ENV, raising=False)
    assert measurement_basis() is None


def test_a_named_basis_that_is_not_there_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv(MEASUREMENT_BASIS_ENV, str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="steering_measurement_basis_missing"):
        measurement_basis()


def test_a_named_basis_loads_with_its_fit(monkeypatch, tmp_path):
    _basis(tmp_path)
    monkeypatch.setenv(MEASUREMENT_BASIS_ENV, str(tmp_path))
    named = measurement_basis()
    library = SteeringVectorLibrary(cache_dir=named, source_dirs=[], allow_derivation=False,
                                    expected_model_identity={"descriptor_sha256": HERS})
    layer = library.load_or_derive(None, None, [3], 8)[3]
    assert sorted(layer) == sorted(d["key"] for d in AFFECTIVE_DIMENSIONS)
    assert all((sv.reference, sv.spread, sv.push) == (0.1, 0.2, 0.01) for sv in layer.values())


def test_a_basis_bound_to_another_checkpoint_does_not_load(tmp_path):
    _basis(tmp_path, descriptor="b" * 64)
    library = SteeringVectorLibrary(cache_dir=tmp_path, source_dirs=[], allow_derivation=False,
                                    expected_model_identity={"descriptor_sha256": HERS})
    with pytest.raises(ValueError, match="qualified_steering_vector_unavailable"):
        library.load_or_derive(None, None, [3], 8)
