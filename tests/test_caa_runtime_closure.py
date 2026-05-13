from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np


def _write_vector(path: Path, values: list[float], *, source: str = "extracted_caa", extracted: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        v=np.asarray(values, dtype=np.float32),
        source=source,
        extracted=extracted,
        derived_at=123.0,
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


def test_production_caa_adapts_alpha_and_detects_collapse(tmp_path):
    from core.consciousness.caa import ProductionCAA, RegisteredVector, VectorProvenance, VectorRegistry

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
