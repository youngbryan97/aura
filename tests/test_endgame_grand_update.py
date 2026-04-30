from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_mutation_tiers_seal_core_immune_math_and_bus_paths():
    from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path

    sealed = [
        "core/consciousness/phi_core.py",
        "core/consciousness/hierarchical_phi.py",
        "core/memory/scar_formation.py",
        "core/bus/actor_bus.py",
        "core/runtime/autonomy_conductor.py",
        "core/self_modification/fault_pipeline.py",
    ]
    for path in sealed:
        assert classify_mutation_path(path).tier is MutationTier.SEALED

    assert classify_mutation_path("core/brain/inference_gate.py").tier is MutationTier.PROPOSE_ONLY
    assert classify_mutation_path("tests/test_generated.py").tier is MutationTier.FREE_AUTO_FIX
    assert classify_mutation_path("core/consciousness/endogenous_fitness.py").tier is MutationTier.SHADOW_VALIDATED_AUTO_FIX


def test_repair_approval_allows_obvious_low_risk_bugfixes_without_prior_calibration():
    from core.self_modification.repair_approval import RepairApprovalPolicy

    decision = RepairApprovalPolicy().decide(
        target_file="core/consciousness/endogenous_fitness.py",
        candidate_changed=True,
        deterministic=True,
        candidate_confidence=0.91,
        calibration_probability=0.55,
        calibration_attempts=0,
    )
    assert decision.approved
    assert decision.stage == "auto_apply_after_shadow"
    assert decision.observation_mode


def test_fault_pipeline_builds_precise_bug_packet_and_eligible_nameerror_patch(tmp_path):
    source_path = tmp_path / "core" / "consciousness" / "sample_bug.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def broken():\n    return get_mlx_client()\n\nbroken()\n", encoding="utf-8")

    try:
        exec(compile(source_path.read_text(), str(source_path), "exec"), {})
    except Exception as exc:
        from core.self_modification.fault_pipeline import FaultToPatchPipeline

        result = FaultToPatchPipeline(tmp_path).diagnose(exc)
    else:  # pragma: no cover
        raise AssertionError("sample bug did not fail")

    assert result.packet.error_type == "NameError"
    assert result.packet.file == "core/consciousness/sample_bug.py"
    assert result.candidate is not None
    assert "from core.brain.llm.mlx_client import get_mlx_client" in result.candidate.after_source
    assert result.promotion_allowed


def test_scar_court_prevents_single_event_from_high_influence_scar():
    from core.memory.scar_formation import ScarDomain, ScarFormationSystem

    system = ScarFormationSystem()
    system._scars.clear()
    scar = system.form_scar(
        ScarDomain.SECURITY_BREACH,
        "Accessibility permission timeout during screen inspection",
        "accessibility_timeout_test",
        severity=0.9,
        event_id="evt1",
        source_id="screen_probe",
        confidence=0.6,
        verified_threat=False,
    )
    assert scar.maturity_status in {"provisional", "reduced"}
    assert scar.effective_severity() <= 0.15
    assert not system.eligible_for_lora_consolidation("accessibility_timeout_test")

    scar.reinforce(event_id="evt2", source_id="audit_log", confidence=0.9, verified_threat=True)
    scar.reinforce(event_id="evt3", source_id="will_receipt", confidence=0.9, verified_threat=True)
    assert scar.evidence_count >= 3
    assert scar.source_diversity >= 2


def test_stdp_external_validation_external_signal_beats_controls():
    from core.consciousness.stdp_external_validation import STDPExternalValidator

    report = STDPExternalValidator(seed=7).run(steps=96)
    assert report.passed
    margins = report.to_dict()["margins"]
    assert all(value > 0 for value in margins.values())


def test_substrate_policy_head_outputs_decision_weights_and_ablation_delta():
    from core.consciousness.substrate_policy_head import POLICY_KEYS, SubstratePolicyHead, SubstratePolicyInput

    head = SubstratePolicyHead()
    inputs = SubstratePolicyInput(
        state64=[0.1] * 64,
        phi=1.2,
        valence=0.3,
        arousal=0.7,
        dominance=0.2,
        prediction_error=0.4,
        scar_pressure=0.1,
        resource_headroom=0.8,
        continuity=0.9,
    )
    policy = head.compute(inputs)
    assert set(policy.weights) == set(POLICY_KEYS)
    assert all(0.0 <= value <= 1.0 for value in policy.weights.values())
    assert head.ablation_report(inputs)["full_vs_prompt_mean_abs_delta"] > 0


def test_metabolic_scheduler_improves_when_stable_and_repairs_when_unstable():
    from core.autonomy.metabolic_budget import MetabolicBudgetScheduler, MetabolicState

    scheduler = MetabolicBudgetScheduler()
    stable = scheduler.allocate(MetabolicState(stability=0.95, resource_headroom=0.9, novelty_budget=0.8, benchmark_gap=0.5))
    broken = scheduler.allocate(MetabolicState(stability=0.4, tests_passing=False))
    assert stable.improvement > 0.04
    assert broken.mode == "repair"
    assert broken.repair > stable.repair


def test_behavioral_contracts_and_canary_runtime_gate_regressions():
    from core.promotion.canary_runtime import CanaryRuntime, ReplayExample

    examples = [ReplayExample("ex1", "hello", "the answer is stable and useful")]
    report = CanaryRuntime().compare(
        examples,
        lambda ex: "the answer is stable and useful",
        metrics={
            "phi": 0.5,
            "governance_receipt_coverage": 1.0,
            "scar_false_positive_rate": 0.0,
            "event_loop_lag_p95_s": 0.01,
            "tool_success_rate": 0.9,
            "memory_retrieval_f1": 0.8,
        },
    )
    assert report.passed


def test_keep_awake_uses_screen_saver_friendly_caffeinate_flags():
    from core.runtime.keep_awake import MacKeepAwakeController

    cmd = MacKeepAwakeController().build_command()
    assert cmd == ("caffeinate", "-i", "-m", "-s")
    assert "-d" not in cmd


def test_output_gate_routes_background_self_talk_to_secondary():
    from core.utils.output_gate import AutonomousOutputGate

    gate = AutonomousOutputGate()
    target, metadata = gate._foreground_policy(
        "Self-Initiated: Brief Curiosity Scan",
        "system",
        "primary",
        {},
    )
    assert target == "secondary"
    assert metadata["voice"] is False
    assert metadata["suppress_bus"] is True


def test_governance_primitives_fail_closed_when_runtime_active(tmp_path):
    from core.container import ServiceContainer
    from core.governance_context import governed_scope_sync
    from core.runtime.consequential_primitives import guarded_write_text

    old_locked = getattr(ServiceContainer, "_registration_locked", False)
    ServiceContainer._registration_locked = True
    try:
        with pytest.raises(Exception):
            guarded_write_text(tmp_path / "blocked.txt", "no receipt")

        class Decision:
            receipt_id = "WR-test"
            domain = "file_write"
            source = "test"

        with governed_scope_sync(Decision()):
            guarded_write_text(tmp_path / "allowed.txt", "ok")
        assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "ok"
    finally:
        ServiceContainer._registration_locked = old_locked


def test_memory_benchmark_graph_selective_reduces_tokens():
    from core.memory.memory_benchmarking import GraphMemoryIndex, MemoryBenchmarkCase, MemoryBenchmarkRunner, MemoryScope, ScopedMemoryRecord

    index = GraphMemoryIndex()
    index.add(ScopedMemoryRecord("a", "python repair traceback import mlx client", MemoryScope.APPLICATION, "bryan", "coder", links=("b",)))
    index.add(ScopedMemoryRecord("b", "pytest validates deterministic bug packet", MemoryScope.APPLICATION, "bryan", "coder"))
    for idx in range(20):
        index.add(ScopedMemoryRecord(f"noise{idx}", "unrelated memory about dinner plans", MemoryScope.APPLICATION, "bryan", "coder"))
    result = MemoryBenchmarkRunner(index).run([MemoryBenchmarkCase("mlx import traceback repair", ("a", "b"))])
    assert result["graph_selective"].mean_tokens < result["full_context"].mean_tokens


def test_toolweaver_synthetic_flywheel_and_simulator_are_operational(tmp_path):
    from core.embodiment.simulator_bridge import LocalPhysics2DSimulator
    from core.learning.synthetic_data_flywheel import SyntheticDataFlywheel
    from core.tools.toolweaver import ToolSpec, ToolWeaverIndex

    index = ToolWeaverIndex()
    index.fit([
        ToolSpec("pytest", "run python tests", ("test", "code")),
        ToolSpec("web_read", "read web pages", ("research", "browser")),
    ])
    assert index.retrieve("run code test")[0].name == "pytest"

    traces = SyntheticDataFlywheel(tmp_path).generate_from_success(
        {"id": "s1", "task": "fix import", "output": "import added", "score": 0.95, "task_type": "repair"},
        variants=4,
    )
    assert len(traces) == 4
    assert SyntheticDataFlywheel(tmp_path).write_jsonl(traces).exists()

    sim = LocalPhysics2DSimulator()
    start = sim.reset(seed=1).distance
    end = sim.rollout(steps=20)[-1].distance
    assert end < start


def test_activation_auditor_reconciles_safe_custom_loop():
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec

    state = {"started": False}

    async def starter(_orch):
        state["started"] = True
        return {"ok": True}

    auditor = ActivationAuditor((ActivationSpec("custom", required=True, auto_start=True, starter=starter),))
    report = asyncio.run(auditor.audit(reconcile=True))
    assert state["started"]
    assert report.statuses[0].reconciled


def test_caa_validator_reads_existing_vector_artifacts():
    from training.caa_32b_validation import CAA32BValidator

    report = CAA32BValidator(vectors_dir=Path("training/vectors")).run()
    assert report["vector_count"] > 0
    assert "activation_vectors_present" in report["pass_conditions"]
