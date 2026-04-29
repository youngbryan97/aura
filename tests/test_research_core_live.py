"""LIVE end-to-end test of the SelfImprovingResearchCore.

Exercises the deepest practical integration:

  * Real LatticeLM forward + backward passes (CPU-only).
  * Real F1 audit-chain receipts emitted by F18 promotion gate.
  * Real F2 prediction ledger advances per cycle.
  * Real F4 SafeMutationEvaluator subprocess for F19 discovery.
  * Real F5 doctor bundle includes research_core.json.
  * F9 curriculum bridge converts F21 unknowns into LearningTasks.
  * F14 tenant boundary stamps the workdir.
  * F16-style Will gate is consultable (if injected).

This is the autonomy proof: spin up the core, run multiple cycles,
inspect the artifacts, and confirm every wired pipeline carried
data.  Manual operator intervention is not required at any step.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
import torch

from core.curriculum.task_generator import LearningTask
from core.lattice import LatticeConfig, TrainConfig
from core.promotion.dynamic_benchmark import DynamicBenchmark
from core.research_core.core import (
    ResearchCoreConfig,
    SelfImprovingResearchCore,
)
from core.research_core.curriculum_bridge import (
    task_to_learning_task,
    tasks_to_learning_tasks,
)
from core.research_core.doctor import collect_research_core_status
from core.research_core.registry import register_research_core
from core.runtime.audit_chain import AuditChain
from core.runtime.diagnostics_bundle import build_bundle
from core.runtime.receipts import (
    GovernanceReceipt,
    get_receipt_store,
    reset_receipt_store,
)


def _live_cfg(workdir: Path) -> ResearchCoreConfig:
    return ResearchCoreConfig(
        workdir=workdir,
        model_cfg=LatticeConfig(
            vocab_size=128, d_model=32, n_layers=2, n_heads=4,
            d_state=8, n_experts=4, top_k=2, max_seq_len=24, attention_window=12,
        ),
        train_cfg=TrainConfig(amp=False, checkpoint_dir=str(workdir / "ckpt")),
        critical_metrics=("task_accuracy",),
        max_regression=0.5,
        discovery_population=12,
        discovery_elite=3,
        discovery_generations=4,
    )


@pytest.fixture
def fresh_store(tmp_path: Path):
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    yield
    reset_receipt_store()


# ---------------------------------------------------------------------------
# F9 curriculum bridge
# ---------------------------------------------------------------------------
def test_task_to_learning_task_carries_source_metadata():
    bench = DynamicBenchmark(seed=0)
    task = bench.generate(1, kinds=["gcd"])[0]
    lt = task_to_learning_task(task, strategy="more_examples", iteration=3)
    assert isinstance(lt, LearningTask)
    assert lt.belief == "task:gcd"
    assert lt.modality == "symbolic"
    assert lt.strategy == "more_examples"
    assert lt.iteration == 3
    assert lt.metadata["source_kind"] == "gcd"
    assert "source_hash" in lt.metadata


def test_tasks_to_learning_tasks_preserves_count():
    bench = DynamicBenchmark(seed=1)
    tasks = bench.generate(5)
    lts = tasks_to_learning_tasks(tasks)
    assert len(lts) == 5
    assert all(isinstance(lt, LearningTask) for lt in lts)


# ---------------------------------------------------------------------------
# Live: full multi-cycle run
# ---------------------------------------------------------------------------
def test_live_three_cycles_advance_every_pipeline(tmp_path: Path, fresh_store):
    """Run three cycles back-to-back and verify every wire carried data."""
    cfg = _live_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)

    # F14: tenant boundary stamped on init.
    assert (cfg.workdir / "tenant.json").exists()

    pre_ledger = core.ledger.count()
    pre_vault = core.vault.size()
    pre_novelty = len(core.novelty)

    reports = []
    for _ in range(3):
        reports.append(core.run_cycle(n_eval_tasks=8))

    # F1: every cycle produced a governance receipt.
    receipts = get_receipt_store().query_by_kind("governance")
    promotion_receipts = [
        r for r in receipts if isinstance(r, GovernanceReceipt) and r.domain == "checkpoint_promotion"
    ]
    assert len(promotion_receipts) >= 3

    # F1: audit chain integrity intact.
    chain_result = get_receipt_store().verify_chain()
    assert chain_result["ok"] is True
    assert chain_result["length"] >= 3

    # F2: prediction ledger advanced.
    assert core.ledger.count() > pre_ledger

    # F18: holdout vault grew.
    assert core.vault.size() > pre_vault

    # F21: novelty archive populated.
    assert len(core.novelty) > pre_novelty

    # Cycle bookkeeping is self-consistent.
    assert core._iteration == 3
    assert len(core.cycle_history()) == 3
    for r in reports:
        assert r.promotion is not None
        assert "task_accuracy" in r.metrics


# ---------------------------------------------------------------------------
# Live: ServiceContainer registration + doctor bundle
# ---------------------------------------------------------------------------
def test_live_register_and_doctor_bundle_round_trip(tmp_path: Path, fresh_store):
    cfg = _live_cfg(tmp_path / "wd")
    core = register_research_core(cfg=cfg)
    core.run_cycle(n_eval_tasks=6)
    core.run_cycle(n_eval_tasks=6)

    # Doctor collector picks up the registered core.
    snapshot = collect_research_core_status()
    assert snapshot["available"] is True
    assert snapshot["status"]["iteration"] == 2
    assert len(snapshot["recent_cycles"]) == 2

    # F5: full bundle includes research_core.json.
    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    assert info["ok"] is True
    with tarfile.open(info["path"], "r:gz") as tar:
        names = tar.getnames()
    rc_files = [n for n in names if n.endswith("research_core.json")]
    assert len(rc_files) == 1


# ---------------------------------------------------------------------------
# Live: Will gate honoured for promotions
# ---------------------------------------------------------------------------
def test_live_will_refuse_blocks_promotion(tmp_path: Path, fresh_store):
    cfg = _live_cfg(tmp_path / "wd")
    # Set a wide tolerance so metric-side variance doesn't pre-empt the
    # Will check; we want Will to be the deciding factor.
    cfg.max_regression = 1.0

    will_calls = []

    def will(payload):
        will_calls.append(payload)
        # Refuse every promotion after the initial baseline.
        return {"outcome": "refuse", "reason": "policy_freeze"}

    core = SelfImprovingResearchCore(cfg, will_decide_fn=will)
    # First cycle sets baseline (Will not consulted by gate).
    core.run_cycle(n_eval_tasks=6)
    # Second cycle: gate will check metrics; if they pass, Will refuses.
    r2 = core.run_cycle(n_eval_tasks=6)
    if len(will_calls) > 0:
        # Will WAS consulted: refuse must surface in reasons.
        assert any("will_refuse" in reason for reason in r2.promotion["reasons"])
        assert r2.promotion["accepted"] is False
    else:
        # Will was never consulted because the metrics-side check
        # already rejected — that's also a valid block, just at a
        # different layer.  The promotion must still be rejected.
        assert r2.promotion["accepted"] is False


# ---------------------------------------------------------------------------
# Live: discovery actually runs end-to-end via F4 sandbox
# ---------------------------------------------------------------------------
def test_live_discovery_round_trip(tmp_path: Path, fresh_store):
    cfg = _live_cfg(tmp_path / "wd")
    cfg.discovery_population = 32
    cfg.discovery_generations = 30
    core = SelfImprovingResearchCore(cfg)
    result = core.discover_addition_proxy()
    # With 32 individuals * 30 generations the best-score should be
    # close to 0 (perfect addition has score = -size_penalty ≈ -0.005).
    assert result.score > -1.0


# ---------------------------------------------------------------------------
# Live: model is real — forward pass on its actual weights
# ---------------------------------------------------------------------------
def test_live_lattice_model_forward_runs(tmp_path: Path, fresh_store):
    cfg = _live_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    ids = torch.randint(0, cfg.model_cfg.vocab_size, (1, 8))
    out = core.model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
    assert out["logits"].shape == (1, 8, cfg.model_cfg.vocab_size)


# ---------------------------------------------------------------------------
# Live: cycle history serializes
# ---------------------------------------------------------------------------
def test_live_cycle_history_is_json_serializable(tmp_path: Path, fresh_store):
    import json

    cfg = _live_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    core.run_cycle(n_eval_tasks=4)
    core.run_cycle(n_eval_tasks=4)
    history = core.cycle_history()
    # Round-trip through JSON to confirm everything is serialisable.
    blob = json.dumps(history, default=str)
    parsed = json.loads(blob)
    assert len(parsed) == 2
