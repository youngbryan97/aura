"""The random-direction control keeps her steering's lengths and loses its directions, and is read as nothing else.

Steering-off and an untuned model said her steering is what carries her state
to her rating. Whether her directions matter, or any push of that size would
do, is the random-direction control: her run with every direction replaced at
its own length. These pin that the basis it runs under differs from hers only
in where it points, that the control runs under that basis and steered, that
the basis is never read as her run, and that a control nothing steered stops.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.make_random_steering_basis import SOURCE, make_random_basis, random_like
from tools.run_report_grounding import (
    RANDOM_BASIS_SOURCE,
    STEERED_CONTROLS,
    basis_refusal,
    control_environment,
    fusion_refusal,
    steering_refusal,
)

pytestmark = pytest.mark.unit

DIGEST = "52d313c2c435d343cf6acfa2b2ca61bc70334c01b8b17877df79f32ec3c5283c"


def _her_basis(root: Path) -> Path:
    """A small basis shaped like hers: two files, one carrying her felt push everywhere."""
    basis = root / "hers"
    basis.mkdir()
    rng = np.random.default_rng(0)
    common = {"source": "fitted_to_her_replies", "model_descriptor_sha256": DIGEST, "reference": 0.138,
              "spread": 0.142, "push": 0.009, "requested_layer": 43, "selected_layer": 43}
    np.savez(basis / "valence_positive_layer43.npz", v=rng.standard_normal(64).astype(np.float32),
             everywhere=(rng.standard_normal(64) * 9).astype(np.float32), everywhere_slot=np.int64(14), **common)
    np.savez(basis / "arousal_layer43.npz", v=rng.standard_normal(64).astype(np.float32), **common)
    (basis / "basis.json").write_text(json.dumps({"source": "fitted_to_her_replies", "model_descriptor_sha256": DIGEST,
                                                  "operating_alpha": 0.5}))
    return basis


def test_a_random_direction_has_the_length_it_stands_for() -> None:
    rng = np.random.default_rng(1)
    vector = np.arange(1.0, 5121.0, dtype=np.float32)
    drawn = random_like(vector, rng)
    assert np.isclose(np.linalg.norm(drawn), np.linalg.norm(vector), rtol=1e-5)
    assert abs(float(drawn @ vector) / float(np.linalg.norm(drawn) * np.linalg.norm(vector))) < 0.1
    assert not random_like(np.zeros(8), rng).any()


def test_the_random_basis_is_hers_with_only_the_directions_replaced(tmp_path: Path) -> None:
    hers = _her_basis(tmp_path)
    statement = make_random_basis(hers, tmp_path / "random", seed=1)
    assert statement["source"] == SOURCE == RANDOM_BASIS_SOURCE
    assert statement["model_descriptor_sha256"] == DIGEST and statement["everywhere"] == 1
    assert sorted(p.name for p in (tmp_path / "random").glob("*.npz")) == sorted(p.name for p in hers.glob("*.npz"))
    for path in hers.glob("*.npz"):
        before, after = np.load(path), np.load(tmp_path / "random" / path.name)
        assert before.files == after.files
        for key in before.files:
            if key in ("v", "everywhere"):
                assert np.isclose(np.linalg.norm(after[key]), np.linalg.norm(before[key]), rtol=1e-5)
                assert not np.allclose(after[key], before[key])
            else:
                assert np.array_equal(after[key], before[key])


def test_the_random_basis_is_drawn_from_its_seed(tmp_path: Path) -> None:
    hers = _her_basis(tmp_path)
    make_random_basis(hers, tmp_path / "a", seed=1)
    make_random_basis(hers, tmp_path / "b", seed=1)
    make_random_basis(hers, tmp_path / "c", seed=2)
    name = "valence_positive_layer43.npz"
    assert np.array_equal(np.load(tmp_path / "a" / name)["v"], np.load(tmp_path / "b" / name)["v"])
    assert not np.array_equal(np.load(tmp_path / "a" / name)["v"], np.load(tmp_path / "c" / name)["v"])


def test_a_random_basis_is_never_written_over_or_made_from_another(tmp_path: Path) -> None:
    hers = _her_basis(tmp_path)
    make_random_basis(hers, tmp_path / "random", seed=1)
    with pytest.raises(FileExistsError):
        make_random_basis(hers, tmp_path / "random", seed=2)
    with pytest.raises(ValueError):
        make_random_basis(tmp_path / "random", tmp_path / "twice", seed=2)


def test_the_control_keeps_her_steering_and_her_cortex() -> None:
    assert "random-direction" in STEERED_CONTROLS
    assert control_environment(["--whole", "--control", "random-direction"]) == {}
    with pytest.raises(SystemExit):
        control_environment(["--whole", "--control", "random-direction", "--model", "/models/base"])


def test_the_control_must_be_steered() -> None:
    assert steering_refusal(True, "random-direction", {"active": False})
    assert steering_refusal(True, "random-direction", {"active": None})
    assert steering_refusal(True, "random-direction", {"active": True}) is None


RANDOM = {"kind": "measurement", "path": "/b/random", "statement": {"source": "random_directions"}}
HERS = {"kind": "measurement", "path": "/b/hers", "statement": {"source": "fitted_to_her_replies"}}


def test_the_control_runs_under_a_random_basis_and_nothing_else_does() -> None:
    assert basis_refusal("random-direction", RANDOM) is None
    assert basis_refusal("random-direction", HERS)
    assert basis_refusal("random-direction", {"kind": "signed_generation"})
    for control in (None, "steering-off", "untuned"):
        assert basis_refusal(control, RANDOM)
        assert basis_refusal(control, HERS) is None


def _answers(*alphas: float | None) -> dict[str, list[dict]]:
    return {"raised": [{"user_facing": True, "steering_alpha": a} for a in alphas]}


def test_a_steered_control_that_nothing_steered_stops() -> None:
    assert fusion_refusal("random-direction", _answers(0.0, 0.0, None))
    assert fusion_refusal("random-direction", _answers(0.0, 0.5)) is None
    # An anchor her cortex did not answer says nothing about the channel.
    assert fusion_refusal("random-direction", {"raised": [{"user_facing": False, "steering_alpha": 0.0}]}) is None
    assert fusion_refusal("steering-off", _answers(0.0)) is None
    assert fusion_refusal(None, _answers(0.0)) is None
