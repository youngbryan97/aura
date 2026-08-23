from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _write_vector(
    path: Path,
    values: list[float],
    *,
    source: str = "extracted_caa",
    extracted: bool = True,
    model_descriptor_sha256: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        v=np.asarray(values, dtype=np.float32),
        source=source,
        extracted=extracted,
        derived_at=123.0,
        model_descriptor_sha256=model_descriptor_sha256,
    )


def test_steering_vector_library_resolves_exact_and_nearest_layers(tmp_path):
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS, SteeringVectorLibrary

    cache_dir = tmp_path / "training" / "vectors"
    for dim in AFFECTIVE_DIMENSIONS:
        _write_vector(cache_dir / f"{dim['key']}_layer25.npz", [1.0, 0.0, 0.0, 0.0])
        _write_vector(cache_dir / f"{dim['key']}_layer32.npz", [0.0, 1.0, 0.0, 0.0])

    library = SteeringVectorLibrary(cache_dir=cache_dir)
    resolved = library.load_or_derive(
        model=object(),
        tokenizer=object(),
        target_layers=[25, 30],
        d_model=4,
        force_rederive=False,
    )

    assert set(resolved) == {25, 30}
    assert all(vector.exact_layer_match for vector in resolved[25].values())
    assert all(vector.selected_layer == 32 for vector in resolved[30].values())
    assert all(vector.selection_reason.startswith("nearest_layer") for vector in resolved[30].values())

    status = library.registry.status(expected_layers=[25, 30], expected_keys=[dim["key"] for dim in AFFECTIVE_DIMENSIONS])
    assert status["coverage_ratio"] == 1.0
    assert status["exact_match_count"] == len(AFFECTIVE_DIMENSIONS)
    assert status["nearest_match_count"] == len(AFFECTIVE_DIMENSIONS)


def test_steering_vector_library_rejects_wrong_d_model_cache(tmp_path):
    from core.consciousness.affective_steering import SteeringVectorLibrary

    cache_dir = tmp_path / "vectors"
    _write_vector(cache_dir / "valence_positive_layer25.npz", [1.0, 0.0, 0.0])

    library = SteeringVectorLibrary(cache_dir=cache_dir)

    assert library._resolve_cached_path("valence_positive", 25, d_model=4) is None


def test_steering_vector_library_rejects_same_width_from_another_model(tmp_path):
    from core.consciousness.affective_steering import SteeringVectorLibrary

    cache_dir = tmp_path / "vectors"
    _write_vector(
        cache_dir / "valence_positive_layer25.npz",
        [1.0, 0.0, 0.0, 0.0],
        model_descriptor_sha256="a" * 64,
    )
    library = SteeringVectorLibrary(
        cache_dir=cache_dir,
        expected_model_identity={"descriptor_sha256": "b" * 64},
    )

    assert library._resolve_cached_path("valence_positive", 25, d_model=4) is None


def test_steering_vector_library_accepts_only_exact_model_basis(tmp_path):
    from core.consciousness.affective_steering import SteeringVectorLibrary

    cache_dir = tmp_path / "vectors"
    identity = {"descriptor_sha256": "c" * 64}
    _write_vector(
        cache_dir / "valence_positive_layer25.npz",
        [1.0, 0.0, 0.0, 0.0],
        model_descriptor_sha256=identity["descriptor_sha256"],
    )
    library = SteeringVectorLibrary(
        cache_dir=cache_dir,
        expected_model_identity=identity,
    )

    resolved = library._resolve_cached_path("valence_positive", 25, d_model=4)

    assert resolved is not None
    assert resolved[0] == 25


def test_steering_vector_library_discovers_sources_with_explicit_runtime_cache(
    tmp_path,
    monkeypatch,
):
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS, SteeringVectorLibrary

    runtime_cache = tmp_path / "runtime_cache"
    source_dir = tmp_path / "packaged_vectors"
    for dim in AFFECTIVE_DIMENSIONS:
        _write_vector(
            source_dir / f"{dim['key']}_layer25.npz",
            [1.0, 0.0, 0.0, 0.0],
            source="packaged_caa",
        )

    monkeypatch.setenv("AURA_STEERING_DIR", str(source_dir))
    library = SteeringVectorLibrary(cache_dir=runtime_cache)

    resolved = library.load_or_derive(
        model=object(),
        tokenizer=object(),
        target_layers=[25],
        d_model=4,
        force_rederive=False,
    )

    assert all(vector.source == "packaged_caa" for vector in resolved[25].values())
    assert all(Path(vector.file_path).parent == source_dir for vector in resolved[25].values())


def test_affective_steering_runtime_cache_is_partitioned_by_geometry():
    from core.consciousness.affective_steering import AffectiveSteeringEngine

    small = AffectiveSteeringEngine._runtime_vector_cache_dir(n_layers=28, d_model=2368)
    large = AffectiveSteeringEngine._runtime_vector_cache_dir(n_layers=64, d_model=4096)

    assert small != large
    assert "dmodel_2368_layers_28" in str(small)
    assert "dmodel_4096_layers_64" in str(large)


def test_affective_steering_runtime_cache_is_partitioned_by_model_identity():
    from core.consciousness.affective_steering import AffectiveSteeringEngine

    first = AffectiveSteeringEngine._runtime_vector_cache_dir(
        n_layers=64,
        d_model=5120,
        model_identity={"descriptor_sha256": "1" * 64},
    )
    second = AffectiveSteeringEngine._runtime_vector_cache_dir(
        n_layers=64,
        d_model=5120,
        model_identity={"descriptor_sha256": "2" * 64},
    )

    assert first != second
    assert "model_1111111111111111" in str(first)
    assert "model_2222222222222222" in str(second)


def test_steering_vector_weight_prefers_direct_substrate_index():
    from core.consciousness.affective_steering import SteeringVector

    vector = SteeringVector(
        key="curiosity",
        layer_idx=1,
        d_model=4,
        v=np.ones(4, dtype=np.float32),
        substrate_idx=4,
        substrate_fn="tanh",
    )
    substrate = np.zeros(64, dtype=np.float32)
    substrate[4] = 0.75

    assert vector.compute_weight({"motivation": -1.0}) < 0.0
    assert vector.compute_weight_from_state(substrate) > 0.6


def test_substrate_sync_prefers_shared_state_vector_over_mood_projection():
    from core.consciousness.affective_steering import SubstrateSyncThread

    shared = {"state_vector": np.array([0.1, 0.2, 0.0, 0.3, 0.4], dtype=np.float32)}
    sync = SubstrateSyncThread(hooks=[], engine=object(), shared_state=shared)

    vector, source = sync._read_substrate_vector()

    assert source == "shared_state"
    assert vector is not None
    assert np.allclose(vector, np.array([0.1, 0.2, 0.0, 0.3, 0.4], dtype=np.float32))


def test_substrate_sync_reads_registered_liquid_substrate_vector(monkeypatch):
    from core.consciousness.affective_steering import SubstrateSyncThread
    from core.container import ServiceContainer

    ServiceContainer.clear()
    substrate = SimpleNamespace(get_state_vector=lambda: np.array([0.5, -0.25], dtype=np.float32))
    ServiceContainer.register_instance("liquid_substrate", substrate)
    try:
        sync = SubstrateSyncThread(hooks=[], engine=object())

        vector, source = sync._read_substrate_vector()

        assert source == "liquid_substrate"
        assert vector is not None
        assert np.allclose(vector, np.array([0.5, -0.25], dtype=np.float32))
    finally:
        ServiceContainer.clear()


def test_affective_steering_geometry_falls_back_to_packaged_vector_dim(
    tmp_path,
    monkeypatch,
):
    from core.consciousness.affective_steering import AFFECTIVE_DIMENSIONS, AffectiveSteeringEngine

    source_dir = tmp_path / "packaged_vectors"
    for dim in AFFECTIVE_DIMENSIONS:
        _write_vector(
            source_dir / f"{dim['key']}_layer25.npz",
            [1.0] + [0.0] * 512,
            source="packaged_caa",
        )
        _write_vector(
            source_dir / f"{dim['key']}_layer30.npz",
            [1.0] + [0.0] * 512,
            source="packaged_caa",
        )
        _write_vector(
            source_dir / f"{dim['key']}_layer35.npz",
            [1.0] + [0.0] * 512,
            source="packaged_caa",
        )
    monkeypatch.setenv("AURA_STEERING_DIR", str(source_dir))

    model = SimpleNamespace(layers=[object() for _ in range(64)])
    engine = AffectiveSteeringEngine()

    assert engine._discover_model_geometry(model) == (64, 513)


def test_surface_alpha_override_survives_adaptive_alpha_update():
    from core.consciousness.affective_steering import AffectiveSteeringEngine

    hooks = [SimpleNamespace(_alpha=5.0), SimpleNamespace(_alpha=5.0)]
    engine = AffectiveSteeringEngine()
    engine._hooks = hooks

    engine.set_surface_alpha_override(0.25)
    engine.set_alpha(4.7)

    assert engine._alpha == 4.7
    assert [hook._alpha for hook in hooks] == [0.25, 0.25]

    engine.set_surface_alpha_override(None)
    engine.set_alpha(4.7)

    assert [hook._alpha for hook in hooks] == [4.7, 4.7]


def test_surface_alpha_override_can_neutralize_residual_injection():
    from core.consciousness.affective_steering import AffectiveSteeringEngine

    hooks = [SimpleNamespace(_alpha=5.0), SimpleNamespace(_alpha=5.0)]
    engine = AffectiveSteeringEngine()
    engine._hooks = hooks

    engine.set_surface_alpha_override(0.0)
    engine.set_alpha(4.7)

    assert engine._alpha == 4.7
    assert [hook._alpha for hook in hooks] == [0.0, 0.0]


def test_production_caa_adapts_alpha_and_detects_collapse(tmp_path):
    from core.consciousness.caa import (
        ProductionCAA,
        RegisteredVector,
        VectorProvenance,
        VectorRegistry,
    )

    cache_dir = tmp_path / "vectors"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for key in ("valence_positive", "arousal"):
        _write_vector(cache_dir / f"{key}_layer25.npz", [1.0, 0.0, 0.0, 0.0])
    registry = VectorRegistry()
    for key in ("valence_positive", "arousal"):
        registry.register(
            RegisteredVector(
                key=key,
                layer_idx=25,
                d_model=4,
                v=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                substrate_idx=0,
                substrate_fn="tanh",
                provenance=VectorProvenance(
                    source="extracted_caa",
                    file_path=str(cache_dir / f"{key}_layer25.npz"),
                    requested_layer=25,
                    selected_layer=25,
                    selection_reason="exact",
                    extracted=True,
                    exact_layer_match=True,
                ),
            )
        )

    production = ProductionCAA(base_alpha=5.0, vectors_dir=cache_dir)
    status = production.ingest_registry(
        registry,
        expected_layers=[25],
        expected_keys=["valence_positive", "arousal"],
        model_path="",
    )
    assert status["readiness"]["level"] == "validated"
    assert status["alpha_state"]["current_alpha"] > 5.0

    collapse = production.observe_generation("the drift the drift the drift the drift the drift the drift")
    assert collapse["collapse"]["severity"] in {"warning", "critical"}
    assert collapse["alpha_state"]["current_alpha"] <= status["alpha_state"]["current_alpha"]


def test_finetune_pipe_persists_processing_metadata(tmp_path):
    from core.adaptation.finetune_pipe import FinetunePipe

    pipe = FinetunePipe(data_dir=str(tmp_path))
    asyncio.run(
        pipe.register_success(
            task_description="experiential_moment",
            context="Context",
            reasoning="Reasoning",
            final_action="Action",
            quality_score=0.9,
            metadata={"steering": {"readiness_level": "production", "adaptive_alpha": 7.5}},
        )
    )
    asyncio.run(pipe.flush())

    lines = pipe.dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"_meta"' in lines[0]
    assert '"readiness_level": "production"' in lines[0]


def test_crsm_lora_bridge_quality_bonus_tracks_processing_context(monkeypatch, tmp_path):
    import core.consciousness.crsm_lora_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "PERSIST_PATH", tmp_path / "crsm_buffer.jsonl")
    bridge = bridge_mod.CRSMLoraBridge()
    monkeypatch.setattr(
        bridge,
        "_capture_processing_context",
        lambda: {"steering": {"readiness_level": "production", "adaptive_alpha": 7.5}, "mood": {"valence": 0.7}},
    )

    bridge.pre_inference_capture(
        context_text="context",
        surprise_magnitude=0.35,
        hedonic_score=0.3,
        crsm_hidden_norm=0.8,
    )
    bridge.post_inference_capture("response", hedonic_after=0.45)

    status = bridge.get_status()
    assert status["buffer_size"] == 1
    assert status["last_processing_context"]["steering"]["readiness_level"] == "production"
    assert bridge._buffer[-1].quality_score >= 0.88
