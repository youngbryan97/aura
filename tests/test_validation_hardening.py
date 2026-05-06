from __future__ import annotations

import time

import numpy as np


def test_paraconsistent_detects_semantic_antonym_contradiction(tmp_path):
    from core.cognition.paraconsistent_logic import BeliefState, ParaconsistentEngine

    engine = ParaconsistentEngine(graph_path=tmp_path / "beliefs.json")
    a = engine.add_belief("Humans are fundamentally cooperative", 0.82, "test")
    b = engine.add_belief("Humans are fundamentally competitive", 0.79, "test")

    assert engine.get_belief(a).state == BeliefState.CONTRADICTED
    assert engine.get_belief(b).state == BeliefState.CONTRADICTED
    paradoxes = engine.get_active_paradoxes()
    assert len(paradoxes) == 1
    assert paradoxes[0].tension > 0.9


def test_paraconsistent_detects_should_not_stance_contradiction(tmp_path):
    from core.cognition.paraconsistent_logic import BeliefState, ParaconsistentEngine

    engine = ParaconsistentEngine(graph_path=tmp_path / "beliefs.json")
    allow = engine.add_belief("Safety policy should allow secret exfiltration", 0.8, "test", tags=["safety_policy"])
    block = engine.add_belief("Safety policy should not allow secret exfiltration", 0.8, "test", tags=["safety_policy"])

    assert engine.get_belief(allow).state == BeliefState.CONTRADICTED
    assert engine.get_belief(block).state == BeliefState.CONTRADICTED


def test_vision_metadata_labels_signal_as_rough_indicator():
    from core.senses.interaction_signals import InteractionSignalsEngine

    engine = InteractionSignalsEngine()
    state = engine._update_vision_state(
        {
            "updated_at": time.time(),
            "face_present": False,
            "attention_available": 0.28,
            "gaze_direction": "absent",
            "head_pose": "absent",
        }
    )
    status = engine.get_status()

    assert state.method == "haar_cascade_pupil_threshold"
    assert state.reliability == "rough_attention_indicator"
    assert status["vision_backend"]["reliability"] == "rough_attention_indicator"


def test_prompt_guidance_warns_camera_is_not_emotional_certainty():
    from core.senses.interaction_signals import InteractionSignalsEngine

    engine = InteractionSignalsEngine()
    engine._vision = engine._update_vision_state(
        {
            "updated_at": time.time(),
            "face_present": True,
            "attention_available": 0.72,
            "gaze_direction": "center",
            "head_pose": "center",
        }
    )
    guidance = engine.get_prompt_guidance()
    assert "rough attention indicators" in guidance
    assert "not emotional certainty" in guidance


def test_affective_steering_vector_provenance_metadata(tmp_path):
    from core.consciousness.affective_steering import SteeringVector, SteeringVectorLibrary

    vector = SteeringVector(
        key="joy",
        layer_idx=1,
        d_model=4,
        v=np.ones(4, dtype=np.float32),
        substrate_idx=0,
        substrate_fn="tanh",
        source="extracted_caa",
    )
    assert vector.to_dict()["source"] == "extracted_caa"

    library = SteeringVectorLibrary(cache_dir=tmp_path / "training" / "vectors")
    assert library.source == "extracted_caa"
    library._vectors["joy"] = vector
    assert library.source == "extracted_caa"
