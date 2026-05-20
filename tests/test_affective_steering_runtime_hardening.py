from types import SimpleNamespace

import numpy as np

from core.consciousness import affective_steering
from core.consciousness.affective_steering import (
    AFFECTIVE_DIMENSIONS,
    SteeringCalibrator,
    SteeringVectorLibrary,
)
from core.consciousness.caa.vector_registry import (
    RegisteredVector,
    VectorProvenance,
    VectorRegistry,
)


def test_derivation_failure_uses_neutral_disabled_vector(tmp_path, monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        affective_steering,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    library = SteeringVectorLibrary(cache_dir=tmp_path)

    def fail_derivation(**_kwargs):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(library, "_derive_caa", fail_derivation)

    vector = library._derive_or_fallback(
        model=object(),
        tokenizer=object(),
        dim_spec=AFFECTIVE_DIMENSIONS[0],
        target_layer=1,
        d_model=8,
    )

    assert vector.source == "disabled_neutral"
    assert vector.selection_reason == "disabled_after_derivation_failure"
    assert np.allclose(vector.v, np.zeros(8, dtype=np.float32))
    assert recorded == [("affective_steering", "RuntimeError")]


def test_invalid_cached_vector_is_rejected_before_derivation(tmp_path, monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        affective_steering,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )
    np.savez(
        tmp_path / "valence_positive_layer1.npz",
        v=np.zeros(8, dtype=np.float32),
        source="cached_caa",
        selected_layer=1,
    )
    library = SteeringVectorLibrary(cache_dir=tmp_path)

    def fail_derivation(**_kwargs):
        raise RuntimeError("fresh derivation unavailable")

    monkeypatch.setattr(library, "_derive_caa", fail_derivation)

    vectors = library.load_or_derive(
        model=object(),
        tokenizer=object(),
        target_layers=[1],
        d_model=8,
    )

    assert vectors[1]["valence_positive"].source == "disabled_neutral"
    assert ("affective_steering", "ValueError") in recorded
    assert ("affective_steering", "RuntimeError") in recorded


def test_vector_registry_reports_disabled_neutral_vectors():
    registry = VectorRegistry()
    registry.register(
        RegisteredVector(
            key="curiosity",
            layer_idx=1,
            d_model=4,
            v=np.zeros(4, dtype=np.float32),
            substrate_idx=4,
            substrate_fn="tanh",
            provenance=VectorProvenance(source="disabled_neutral"),
        )
    )

    status = registry.status(expected_layers=[1], expected_keys=["curiosity"])

    assert status["disabled_neutral_count"] == 1
    assert status["fallback_random_count"] == 0
    assert status["sources"] == {"disabled_neutral": 1}


def test_calibration_import_failure_returns_structured_unavailable(monkeypatch):
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        affective_steering,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    original_import = __import__

    def blocking_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("mlx missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocking_import)
    engine = SimpleNamespace(_alpha=2.0, set_alpha=lambda alpha: setattr(engine, "_alpha", alpha))
    calibrator = SteeringCalibrator(engine, model=object(), tokenizer=object())

    result = calibrator.run_calibration([0.0])

    assert result["ok"] is False
    assert "MLX unavailable" in result["error"]
    assert engine._alpha == 2.0
    assert recorded == [("affective_steering", "ImportError")]
