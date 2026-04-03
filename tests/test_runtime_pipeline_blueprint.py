from core.runtime.pipeline_blueprint import kernel_phase_attribute_order, legacy_runtime_phase_specs


def test_legacy_runtime_phase_specs_match_cognitive_engine_pipeline():
    names = [spec.name for spec in legacy_runtime_phase_specs(include_executive_closure=False)]

    assert names == [
        "proprioceptive_loop",
        "social_context",
        "sensory_ingestion",
        "memory_retrieval",
        "affect_update",
        "cognitive_routing",
        "response_generation",
        "memory_consolidation",
        "identity_reflection",
        "initiative_generation",
        "consciousness",
    ]


def test_legacy_runtime_phase_specs_match_mind_tick_pipeline():
    names = [spec.name for spec in legacy_runtime_phase_specs(include_executive_closure=True)]

    assert names == [
        "proprioceptive_loop",
        "social_context",
        "sensory_ingestion",
        "memory_retrieval",
        "affect_update",
        "executive_closure",
        "cognitive_routing",
        "response_generation",
        "memory_consolidation",
        "identity_reflection",
        "initiative_generation",
        "consciousness",
    ]


def test_kernel_phase_attribute_order_matches_shared_runtime_pipeline():
    assert kernel_phase_attribute_order() == (
        "proprioceptive_phase",
        "social_context_phase",
        "sensory_ingestion_phase",
        "multimodal",
        "eternal",
        "memory_retrieval_phase",
        "perfect_emotion",
        "affect_phase",
        "phi_phase",
        "motivation_phase",
        "executive_closure_phase",
        "evolution_guard",
        "growth",
        "evolution",
        "inference_phase",
        "conversational_dynamics_phase",
        "bonding_phase",
        "routing_phase",
        "godmode_tools",
        "response_phase",
        "repair_phase",
        "memory_consolidation_phase",
        "identity_reflection_phase",
        "initiative_generation_phase",
        "consciousness_phase",
        "self_review_phase",
        "learning_phase",
        "legacy_bridge",
    )
