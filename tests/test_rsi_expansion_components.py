from __future__ import annotations

from pathlib import Path

from core.learning.architecture_search import ArchitectureSearchLab
from core.learning.distributed_eval import DistributedEvalConfig, LocalDistributedEvaluator
from core.learning.full_weight_training import FullWeightTrainingEngine, TrainingConfig
from core.learning.governance_evolution import GovernanceEvolutionPolicy
from core.learning.hidden_eval_repro import HiddenEvalPack
from core.learning.proof_obligations import ProofObligationEngine, ProofStatus
from core.learning.rsi_test_catalog import catalog_summary, default_rsi_test_catalog
from core.learning.successor_lab import SuccessorLab


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
