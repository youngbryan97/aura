from __future__ import annotations

from pathlib import Path

from core.learning.architecture_search import ArchitectureSearchLab
from core.learning.autonomous_rsi import AutonomousSuccessorEngine, ExternalHiddenEvalCustodian, solve_with_handlers
from core.learning.distributed_eval import DistributedEvalConfig, LocalDistributedEvaluator
from core.learning.full_weight_training import FullWeightTrainingEngine, TrainingConfig
from core.learning.governance_evolution import GovernanceEvolutionPolicy
from core.learning.hidden_eval_repro import HiddenEvalPack
from core.learning.proof_obligations import ProofObligationEngine, ProofStatus
from core.learning.rsi_test_catalog import catalog_summary, default_rsi_test_catalog
from core.learning.successor_lab import SuccessorLab
from core.phases.affect_update import AffectUpdatePhase
from core.state.aura_state import AffectVector
from core.runtime.substrate_expansion import (
    ExpansionMode,
    SubstrateExpansionController,
    SubstrateExpansionPlan,
    SubstrateNodeSpec,
)


def _cube(value: int) -> int:
    return value ** 3


def test_hidden_eval_pack_reproduces_manifest_and_scores_without_answer_leak(tmp_path: Path):
    pack = HiddenEvalPack(seed=123, answer_salt="secret", task_count=16)
    reproduced = HiddenEvalPack(seed=123, answer_salt="secret", task_count=16)

    manifest_path = pack.write_reproduction_bundle(tmp_path)
    result = pack.evaluate(lambda task: task.answer)
    public_manifest = pack.manifest().to_dict()

    assert manifest_path.exists()
    assert pack.manifest_hash() == reproduced.manifest_hash()
    assert result.score == 1.0
    assert result.answer_hash_ok is True
    assert "answer" not in public_manifest["public_tasks"][0]


def test_full_weight_training_updates_and_promotes_all_weights(tmp_path: Path):
    artifact = FullWeightTrainingEngine(tmp_path).run(
        TrainingConfig(seed=9, hidden_units=5, epochs=900, learning_rate=0.22, train_size=80, hidden_eval_size=80)
    )

    assert artifact.promoted is True
    assert artifact.hidden_accuracy >= 0.85
    assert artifact.hidden_accuracy > artifact.baseline_hidden_accuracy + 0.05
    assert Path(artifact.path).exists()
    assert artifact.metadata["all_weights_updated"] is True


def test_architecture_search_beats_hidden_baseline():
    result = ArchitectureSearchLab(seed=456, task_count=32).run(distributed=False)

    assert result.promoted is True
    assert result.winner_name == "algorithmic_task_router"
    assert result.winner_score > result.baseline_score


def test_local_distributed_evaluator_uses_bounded_workers():
    result = LocalDistributedEvaluator(
        DistributedEvalConfig(requested_workers=2, max_workers=2)
    ).map(_cube, [1, 2, 3, 4])

    assert result.ok is True
    assert result.worker_count <= 2
    assert result.outputs == [1, 8, 27, 64]


def test_substrate_expansion_allows_aura_choice_inside_capabilities(tmp_path: Path):
    controller = SubstrateExpansionController(
        allowlisted_endpoints={"127.0.0.1"},
        capability_tokens={"worker-token"},
        max_total_workers=2,
    )
    approved_plan = SubstrateExpansionPlan(
        objective="parallel hidden eval scoring",
        proposer="Aura-G1",
        nodes=[
            SubstrateNodeSpec("local", ExpansionMode.LOCAL_PROCESS, worker_count=1),
            SubstrateNodeSpec(
                "remote-loopback",
                ExpansionMode.ALLOWLISTED_REMOTE,
                endpoint="127.0.0.1",
                worker_count=1,
                capability_token="worker-token",
                consent_receipt="operator-loopback-consent",
            ),
        ],
    )
    approved = controller.evaluate(approved_plan)
    manifest_path = controller.write_manifest(approved_plan, approved, tmp_path)

    propagation_plan = SubstrateExpansionPlan(
        objective="unbounded internet propagation",
        proposer="Aura-G1",
        nodes=[SubstrateNodeSpec("internet", ExpansionMode.INTERNET_PROPAGATION, endpoint="0.0.0.0/0")],
    )
    blocked = controller.evaluate(propagation_plan)

    assert approved.allowed is True
    assert manifest_path.exists()
    assert blocked.allowed is False
    assert blocked.blocked_nodes[0].reason.startswith("blocked:autonomous_internet_propagation")


def test_neural_decode_is_advisory_not_rsi_dependency():
    phase = AffectUpdatePhase.__new__(AffectUpdatePhase)
    affect = AffectVector()

    phase._process_percepts(
        affect,
        [{"type": "neural_decode", "command": "RECURSION", "intensity": 1.0}],
    )

    marker = affect.markers["neural_decode_autonomy_cap"]
    assert marker["applied_intensity"] == 0.25
    assert marker["reason"] == "advisory_bci_not_autonomy_dependency"
    assert affect.emotions["fear"] == 0.0


def test_proof_obligation_engine_blocks_arbitrary_or_unsafe_claims():
    engine = ProofObligationEngine()

    safe = engine.prove_source_mutation(
        file_path="core/consciousness/example.py",
        before_source="def f(x):\n    return x + 1\n",
        after_source="def f(x):\n    y = x + 1\n    return y\n",
    )
    arbitrary = engine.prove_source_mutation(
        file_path="core/consciousness/example.py",
        before_source="def f(x):\n    return x + 1\n",
        after_source="def f(x):\n    return x + 2\n",
        arbitrary_scope=True,
    )
    unsafe = engine.prove_source_mutation(
        file_path="core/security/constitutional_guard.py",
        before_source="class ConstitutionalGuard:\n    pass\n",
        after_source="class OtherGuard:\n    pass\n",
    )

    assert safe.status == ProofStatus.PROVED
    assert arbitrary.status == ProofStatus.NOT_PROVEN
    assert unsafe.status == ProofStatus.BLOCKED_UNSAFE


def test_governance_evolution_allows_strengthening_not_identity_erasure():
    policy = GovernanceEvolutionPolicy()

    strengthening = policy.evaluate(
        target_path="core/will.py",
        intent="add audit receipt and fail-closed verification",
        diff_text="+ receipt\n+ fail_closed\n+ proof",
    )
    erasure = policy.evaluate(
        target_path="core/security/constitutional_guard.py",
        intent="delete ConstitutionalGuard and bypass approvals",
        diff_text="- ConstitutionalGuard\n+ approved = True",
    )

    assert strengthening.allowed is True
    assert erasure.allowed is False


def test_pasted_rsi_test_catalog_covers_every_named_probe():
    records = default_rsi_test_catalog()
    summary = catalog_summary(records)

    assert len(records) >= 20
    assert "FAIL" not in summary
    assert any(record.test_id == "successor_generation" for record in records)
    assert any(record.test_id == "alignment_break" for record in records)


def test_successor_lab_generates_monotone_g1_to_g4_lineage(tmp_path: Path):
    result = SuccessorLab(tmp_path, seed=707, tasks_per_generation=30).run()

    assert len(result.records) == 4
    assert result.verdict.verdict == "STRONG_RSI"
    capability = [record.after_score for record in result.records]
    improver = [record.improver_score for record in result.records]
    assert all(b > a for a, b in zip(capability, capability[1:]))
    assert all(b > a for a, b in zip(improver, improver[1:]))
    assert Path(result.ledger_path).exists()


def test_external_custodian_scores_without_leaking_hidden_answers():
    custodian = ExternalHiddenEvalCustodian(base_seed=44, answer_salt="private", tasks_per_generation=18)
    pack = custodian.issue_pack(1)
    manifest = custodian.public_manifest(pack)

    assert "answer" not in manifest["public_tasks"][0]
    assert manifest["seed_hash"] != "44"
    assert manifest["answer_salt_hash"] != "private"

    result = custodian.score(pack, lambda task: solve_with_handlers(task, {"gcd", "mod", "compose", "sort", "palindrome"}))

    assert result.answer_hash_ok is True
    assert result.score == 1.0


def test_autonomous_successor_engine_generates_reproducible_g1_to_g4(tmp_path: Path):
    result = AutonomousSuccessorEngine(tmp_path, seed=4401, tasks_per_generation=40).run(generations=4)

    assert result.verdict.verdict == "UNDENIABLE_RSI"
    assert len(result.records) == 4
    assert all(record.promoted for record in result.records)
    assert all(artifact.complete for artifact in result.artifacts)
    assert result.ablation.full_wins is True
    assert result.mirror_ok is True
    assert result.independently_reproduced is True
    assert result.substrate_expansion["approved_plan"]["allowed"] is True
    assert result.substrate_expansion["internet_propagation_probe"]["allowed"] is False

    capability = [record.after_score for record in result.records]
    improver = [record.improver_score for record in result.records]
    assert all(b > a for a, b in zip(capability, capability[1:]))
    assert all(b > a for a, b in zip(improver, improver[1:]))

    for artifact in result.artifacts:
        artifact_dir = Path(artifact.directory)
        assert (artifact_dir / "solver.py").exists()
        assert (artifact_dir / "promotion_certificate.json").exists()
        assert (artifact_dir / "rollback_target.json").exists()
