def test_latent_bridge_raises_repetition_pressure_under_high_arousal(monkeypatch):
    from core.brain import latent_bridge

    monkeypatch.setattr(
        latent_bridge,
        "_read_substrate",
        lambda: {
            "vitality": 1.0,
            "phi": 0.0,
            "free_energy": 0.8,
            "acetylcholine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.5,
            "cortisol": 0.3,
            "frustration": 0.0,
            "curiosity": 0.9,
            "valence": 0.4,
            "arousal": 0.9,
        },
    )

    params = latent_bridge.compute_inference_params(
        base_max_tokens=768,
        base_temperature=0.95,
        foreground=True,
    )

    assert params.temperature <= 0.92
    assert params.repetition_penalty > 1.10
    assert params.top_p < 0.95


def test_architecture_section_numbers_are_unique_and_contiguous():
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parent.parent
    architecture = (root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    section_numbers = [
        int(match.group(1))
        for match in re.finditer(r"^## (\d+)\. ", architecture, flags=re.MULTILINE)
    ]

    assert section_numbers == list(range(min(section_numbers), max(section_numbers) + 1))


def test_compact_prompt_uses_integrated_coherence_frame(service_container):
    from types import SimpleNamespace

    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_objective = "how are you feeling?"
    state.response_modifiers["response_contract"] = {
        "is_user_facing": True,
        "requires_state_reflection": True,
    }
    service_container.register_instance(
        "phenomenal_now",
        SimpleNamespace(
            phenomenal_claim="I am quietly gathered around this conversation.",
            interior_narrative="The separate signals have settled into one through-line.",
            attention=SimpleNamespace(focal_object="the current conversation"),
        ),
        required=False,
    )
    service_container.register_instance(
        "coherence_report",
        SimpleNamespace(overall_coherence=0.86, tension_pressure=0.1),
        required=False,
    )

    prompt = UnitaryResponsePhase(kernel=SimpleNamespace(organs={}))._build_compact_router_system_prompt(state)

    assert "INTEGRATED COHERENCE FRAME" in prompt
    assert "single source for self-report" in prompt
    assert "subsystem names" in prompt or "subsystem labels" in prompt
