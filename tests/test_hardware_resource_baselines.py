from core.autonomic.resource_stakes import ResourceStakesLedger, ViabilityState
from core.evaluation.baselines import (
    ControllerObservation,
    LinearStateParameterMimic,
    compare_to_mimic,
)
from core.evaluation.hardware_reality import (
    HardwareRealityAuditor,
    ModelMemoryProfile,
    WorkloadProfile,
    bryan_m5_64gb_profile,
    legacy_low_memory_profile,
)


def test_bryan_m5_64gb_treats_32b_as_cortex_not_heartbeat():
    auditor = HardwareRealityAuditor(
        bryan_m5_64gb_profile(),
        workload=WorkloadProfile(os_reserved_gib=4.0, safety_margin_gib=2.0),
    )
    verdict = auditor.evaluate(
        ModelMemoryProfile(
            name="32B-4bit",
            parameters_b=32,
            quantization_bits=4,
            hidden_size=5120,
            layers=64,
        )
    )

    assert verdict.feasible is True
    assert verdict.classification == "batch_or_high_level_cortex"
    assert verdict.realtime_heartbeat_feasible is False
    assert verdict.recommended_tier == "7B-4bit"


def test_legacy_low_memory_profile_rejects_32b_realtime_claim():
    auditor = HardwareRealityAuditor(
        legacy_low_memory_profile(),
        workload=WorkloadProfile(os_reserved_gib=4.0, safety_margin_gib=2.0),
    )
    verdict = auditor.evaluate(
        ModelMemoryProfile(
            name="32B-4bit",
            parameters_b=32,
            quantization_bits=4,
            hidden_size=5120,
            layers=64,
        )
    )

    assert verdict.classification in {"not_feasible", "batch_or_high_level_cortex"}
    assert verdict.realtime_heartbeat_feasible is False
    assert any("32B" in warning for warning in verdict.warnings)
    assert verdict.recommended_tier in {"7B-4bit", "1.5B-4bit"}


def test_resource_stakes_persist_and_constrain_actions(tmp_path):
    db = tmp_path / "stakes.sqlite3"
    ledger = ResourceStakesLedger(
        db,
        initial=ViabilityState(
            energy=0.45,
            tool_budget=0.35,
            memory_budget=0.30,
            storage_budget=0.30,
            integrity=0.40,
        ),
    )

    ledger.consume("expensive_generation", energy=0.30, tool_budget=0.20, memory_budget=0.15)
    envelope = ledger.action_envelope("high")

    assert envelope.allowed is True
    assert envelope.effort == "low"
    assert "large_model_cortex" in envelope.disabled_capabilities

    reloaded = ResourceStakesLedger(db)
    assert reloaded.state().integrity < 0.40
    assert reloaded.state().degradation_events >= 1


def test_resource_stakes_can_deny_outward_action(tmp_path):
    ledger = ResourceStakesLedger(
        tmp_path / "stakes.sqlite3",
        initial=ViabilityState(
            energy=0.08,
            tool_budget=0.08,
            memory_budget=0.08,
            storage_budget=0.08,
            integrity=0.08,
        ),
    )

    envelope = ledger.action_envelope("high")

    assert envelope.allowed is False
    assert envelope.max_tokens == 0
    assert "llm_generation" in envelope.disabled_capabilities


def test_organism_status_exposes_resource_stakes(tmp_path, service_container):
    from core.runtime.organism_status import get_organism_status

    ledger = ResourceStakesLedger(
        tmp_path / "stakes.sqlite3",
        initial=ViabilityState(
            energy=0.25,
            tool_budget=0.25,
            memory_budget=0.25,
            storage_budget=0.25,
            integrity=0.25,
        ),
    )
    service_container.register_instance("resource_stakes", ledger)

    status = get_organism_status()

    assert status["resource_stakes"]["viability"] == ledger.state().viability
    assert status["resource_stakes"]["action_envelope"]["effort"] == "low"


def test_strong_mimic_baseline_is_endpoint_not_process_evidence():
    observations = [
        ControllerObservation(
            state={"valence": 0.1 * i, "arousal": 1.0 - 0.05 * i},
            params={"temperature": 0.6 + 0.02 * i, "tokens": 512 + 12 * i},
            coherence=0.82,
            process_trace=(0.15 + 0.01 * i, 0.40 + 0.02 * i, 0.75 + 0.01 * i),
        )
        for i in range(12)
    ]
    mimic = LinearStateParameterMimic(["valence", "arousal"], ["temperature", "tokens"])
    mimic.fit(observations)
    predictions = [mimic.predict(row.state) for row in observations]

    comparison = compare_to_mimic(
        observations,
        predictions,
        param_keys=["temperature", "tokens"],
        min_trace_distance=0.05,
        min_coherence_delta=0.05,
    )

    assert comparison.endpoint_mae < 1e-6
    assert comparison.trace_distance > 0.05
    assert comparison.process_advantage is True
