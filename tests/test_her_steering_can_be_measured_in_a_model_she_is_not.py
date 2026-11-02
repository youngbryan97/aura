"""Her steering can be measured in a model she is not, and only when a run says so by name.

Her steering-off and untuned controls detach her steering, so neither can tell
her account of her reports from the rival that steering any model along her
direction moves its rating. The run that can is an untuned model in her
cortex's place with her steering kept (item 18). Her worker steers only the
model her signed active-cortex record names. These pin the one exception: a run
names the model, beside a named basis, and the worker steers that model, at
exactly that path, under that model's own identity, so only a basis bound to it
loads and only a certificate earned on it opens the channel. Nothing else is
touched: her cortex, a non-cortex lane nobody named, and the desktop, which
never names one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.consciousness import affective_steering
from core.consciousness.affective_steering import (
    MEASUREMENT_BASIS_ENV,
    MEASUREMENT_MODEL_ENV,
    measured_or_cortex,
    measurement_model,
)
from tools.choose_felt_cell import choose
from tools.rebind_steering_basis import rebind_basis
from tools.run_report_grounding import (
    CONTROLS,
    MODEL_CONTROLS,
    STEERED_CONTROLS,
    basis_refusal,
    control_environment,
    steering_refusal,
)

pytestmark = pytest.mark.unit

THEIRS = {"descriptor_sha256": "b" * 64, "canonical_path": "/models/base"}


def _non_cortex(path: Path) -> SimpleNamespace:
    return SimpleNamespace(status="non_cortex_model", model_path=path, descriptor=None, reason="not_active")


@pytest.fixture
def named(monkeypatch, tmp_path):
    """A run that names a model and a basis, with the model's identity built without reading weights."""
    model, basis = tmp_path / "base", tmp_path / "basis"
    model.mkdir()
    basis.mkdir()
    monkeypatch.setenv(MEASUREMENT_MODEL_ENV, str(model))
    monkeypatch.setenv(MEASUREMENT_BASIS_ENV, str(basis))
    import core.brain.llm.model_artifact_profile as profile

    monkeypatch.setattr(profile, "build_model_artifact_descriptor", lambda path: {**THEIRS, "canonical_path": str(path)})
    return model


def test_unnamed_nothing_changes(monkeypatch, tmp_path):
    monkeypatch.delenv(MEASUREMENT_MODEL_ENV, raising=False)
    assert measurement_model() is None
    lane = _non_cortex(tmp_path)
    assert measured_or_cortex(lane) is lane


def test_a_named_model_that_is_not_there_refuses(monkeypatch, tmp_path):
    monkeypatch.setenv(MEASUREMENT_MODEL_ENV, str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="steering_measurement_model_missing"):
        measurement_model()


def test_the_named_model_is_steered_under_its_own_identity(named):
    resolved = measured_or_cortex(_non_cortex(named))
    assert resolved.status == "measurement_model" and not resolved.matched
    assert resolved.descriptor["descriptor_sha256"] == THEIRS["descriptor_sha256"]
    assert resolved.model_path == named.resolve()


def test_her_cortex_and_unnamed_lanes_are_left_alone(named, tmp_path):
    cortex = SimpleNamespace(status="matched", model_path=tmp_path / "hers", descriptor={"descriptor_sha256": "a" * 64})
    assert measured_or_cortex(cortex) is cortex
    other = _non_cortex(tmp_path / "brainstem")
    assert measured_or_cortex(other) is other


def test_a_named_model_without_a_named_basis_refuses(named, monkeypatch):
    monkeypatch.delenv(MEASUREMENT_BASIS_ENV, raising=False)
    with pytest.raises(ValueError, match="steering_measurement_model_without_basis"):
        measured_or_cortex(_non_cortex(named))


def test_the_worker_attaches_to_the_named_model_with_its_identity(named, monkeypatch):
    from core.brain.llm import mlx_worker, model_registry

    seen: dict = {}

    class Engine:
        _alpha = 0.5
        _hooks = [object()]

        def attach(self, _model, _tokenizer, model_path=None, model_identity=None):
            seen.update(model_path=model_path, model_identity=model_identity)

    monkeypatch.setattr(model_registry, "resolve_cortex_bound_artifact", lambda path: _non_cortex(Path(path)))
    monkeypatch.setattr(affective_steering, "get_steering_engine", Engine)
    monkeypatch.setattr(mlx_worker, "_finish_affective_attachment", lambda engine, **_: True)
    monkeypatch.setattr(mlx_worker, "_remember_fusion_identity", lambda descriptor: seen.update(fusion=descriptor))

    result = mlx_worker._attach_affective_steering(
        SimpleNamespace(), SimpleNamespace(), None, None, SimpleNamespace(value=False), model_path=str(named)
    )
    assert result.active is True
    assert seen["model_identity"]["descriptor_sha256"] == THEIRS["descriptor_sha256"]
    assert seen["fusion"]["descriptor_sha256"] == THEIRS["descriptor_sha256"]


def test_the_control_puts_the_model_in_her_place_keeps_her_steering_and_names_it():
    assert {"untuned-steered", "random-direction"} <= STEERED_CONTROLS
    assert MODEL_CONTROLS == {"untuned", "untuned-steered"} and set(CONTROLS) >= MODEL_CONTROLS
    pins = control_environment(["--whole", "--control", "untuned-steered", "--model", "/models/base"])
    assert pins == {
        "AURA_LLM__MLX_MODEL_PATH": "/models/base",
        "AURA_LLM__MLX_DEEP_MODEL_PATH": "/models/base",
        MEASUREMENT_MODEL_ENV: "/models/base",
    }
    assert "AURA_DISABLE_AFFECTIVE_STEERING" not in pins
    with pytest.raises(SystemExit):
        control_environment(["--whole", "--control", "untuned-steered"])
    assert steering_refusal(True, "untuned-steered", {"active": False})
    assert steering_refusal(True, "untuned-steered", {"active": True}) is None


def test_the_control_runs_under_a_named_basis_that_is_not_random():
    named = {"kind": "measurement", "path": "/b/rebound", "statement": {"source": "fitted_to_her_replies"}}
    assert basis_refusal("untuned-steered", named) is None
    assert basis_refusal("untuned-steered", {"kind": "signed_generation"})
    assert basis_refusal("untuned-steered", {**named, "statement": {"source": "random_directions"}})


def _basis(root: Path, digest: str = "a" * 64) -> Path:
    root.mkdir()
    np.savez(root / "valence_positive_layer43.npz", v=np.ones(8, dtype=np.float32),
             everywhere=np.full(8, 2.0, dtype=np.float32), model_descriptor_sha256=np.array(digest),
             reference=0.1, spread=0.2, push=0.01)
    (root / "basis.json").write_text(json.dumps({"source": "fitted_to_her_replies", "model_descriptor_sha256": digest}))
    (root / "metadata.json").write_text("{}")
    return root


def test_rebinding_changes_only_the_checkpoint(tmp_path):
    hers = _basis(tmp_path / "hers")
    made = rebind_basis(hers, tmp_path / "rebound", THEIRS, tmp_path)
    assert made["model_descriptor_sha256"] == THEIRS["descriptor_sha256"]
    assert made["rebound"]["from_model_descriptor_sha256"] == "a" * 64
    before = np.load(hers / "valence_positive_layer43.npz")
    after = np.load(tmp_path / "rebound" / "valence_positive_layer43.npz")
    assert str(after["model_descriptor_sha256"]) == THEIRS["descriptor_sha256"]
    for key in before.files:
        if key != "model_descriptor_sha256":
            assert np.array_equal(before[key], after[key])
    assert not (tmp_path / "rebound" / "metadata.json").exists()
    with pytest.raises(ValueError):
        rebind_basis(tmp_path / "rebound", tmp_path / "twice", THEIRS, tmp_path)


#: Her reading of 24 October (felt-wide-layers-1024.json): swing, conversations up, margin felt, margin random.
HER_CELLS = {
    "35x4": (0.394, 6, 0.1187, 0.0359), "35x8": (0.900, 6, 0.2125, -0.0391),
    "43x4": (0.631, 6, 0.1812, 0.1875), "43x8": (0.981, 6, 0.3031, -0.0328),
    "51x4": (0.688, 6, -0.0125, -0.0609), "51x8": (1.012, 6, -1.0094, -0.1906),
    "59x2": (0.074, 4, 0.0094, 0.0375), "59x4": (0.429, 5, -0.1469, 0.1531),
}


def _reading(cells: dict) -> dict:
    return {"cells": {name: {"swing": s, "up": u, "margin_felt": f, "margin_random": r, "1": [0] * 6, "-1": [0] * 6}
                      for name, (s, u, f, r) in cells.items()}}


def test_the_rule_chooses_her_cell_from_her_reading():
    chosen = choose(_reading(HER_CELLS))
    assert (chosen["layer"], chosen["size"]) == (43, 8.0)
    assert "+0.981, 6 of 6" in chosen["reading"] and "+0.303 against -0.033" in chosen["reading"]


def test_no_cell_no_push():
    assert choose(_reading({"59x4": (0.429, 5, -0.1469, 0.1531)})) is None


# Her steering engine has a second gate, after the worker's: it attaches only where her signed generation resolves,
# and any checkpoint but hers resolves as incompatible. Item 18's first certificates (1 November, 23:46) stopped
# there, "steering did not attach; nothing to certify", before anything was measured. For the named model, the
# named basis stands where the generation stood, and names the layers hooked.


def _incompatible():
    from core.brain.llm.model_bound_steering import CHECKPOINT_INCOMPATIBLE, SteeringGenerationResolution

    return SteeringGenerationResolution(CHECKPOINT_INCOMPATIBLE, reason="active_cortex_descriptor_mismatch")


def test_the_named_basis_stands_where_her_generation_stood_for_the_named_model(named):
    from core.consciousness.steering_geometry import measured_generation

    resolved = measured_generation(_incompatible(), named)
    assert resolved.status == "measurement_model" and not resolved.expected_detachment
    assert resolved.cache_dir == (named.parent / "basis").resolve()


def test_any_other_resolution_or_checkpoint_is_left_as_it_was(named, tmp_path, monkeypatch):
    from core.brain.llm.model_bound_steering import SteeringGenerationResolution
    from core.consciousness.steering_geometry import measured_generation

    hers = SteeringGenerationResolution("qualified", cache_dir=tmp_path)
    assert measured_generation(hers, named) is hers
    other = tmp_path / "other"
    other.mkdir()
    mismatch = _incompatible()
    assert measured_generation(mismatch, other) is mismatch
    monkeypatch.delenv(MEASUREMENT_MODEL_ENV)
    assert measured_generation(mismatch, named) is mismatch


def test_the_named_model_without_a_named_basis_refuses_at_the_engine(named, monkeypatch):
    from core.consciousness.steering_geometry import measured_generation

    monkeypatch.delenv(MEASUREMENT_BASIS_ENV)
    with pytest.raises(ValueError, match="steering_measurement_model_without_basis"):
        measured_generation(_incompatible(), named)


def test_the_layers_hooked_on_the_named_model_are_the_basis_files_layers(named, monkeypatch):
    import core.brain.llm.model_bound_steering as bound

    basis = named.parent / "basis"
    for layer in (3, 43, 63):
        (basis / f"valence_layer{layer}.npz").write_bytes(b"")
    monkeypatch.setattr(bound, "resolve_active_generation", lambda **_: _incompatible())
    engine = affective_steering.AffectiveSteeringEngine.__new__(affective_steering.AffectiveSteeringEngine)
    engine._model_info = {"model_descriptor_sha256": "b" * 64, "n_layers": 64, "d_model": 5120, "model_path": str(named)}
    monkeypatch.setattr(type(engine), "_runtime_vector_cache_dir", staticmethod(lambda **_: named.parent / "cache"))
    assert engine._qualified_target_layers(64) == [3, 43, 63]
    engine._model_info["model_path"] = str(named.parent / "elsewhere")
    assert engine._qualified_target_layers(64) == []
