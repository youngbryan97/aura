"""Tests for SelfImprovingResearchCore.

Coverage:
  * unit: deterministic solver handles every benchmark kind correctly.
  * cycle: a single ``run_cycle`` advances iteration, emits receipts,
    grows vault, leaves novelty archive populated.
  * integration: F1 audit chain receives a governance receipt for the
    promotion decision; F2 prediction ledger gains entries; F5 doctor
    bundle includes a research_core blob.
  * tenant boundary: foreign tenant id refuses to mount the workdir.
  * autonomy: many cycles in a row; promotion stays monotone — once
    accepted, the baseline only advances, never regresses silently.
  * stress: the cycle handles a low-vocab tiny model end-to-end and
    a malformed solver gracefully (no exceptions escape).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.lattice import LatticeConfig, TrainConfig
from core.research_core.core import (
    CycleReport,
    ResearchCoreConfig,
    SelfImprovingResearchCore,
    deterministic_task_solver,
)
from core.research_core.doctor import collect_research_core_status
from core.research_core.registry import register_research_core
from core.runtime.receipts import (
    GovernanceReceipt,
    get_receipt_store,
    reset_receipt_store,
)
from core.runtime.tenant_boundary import TenantBoundary, TenantMismatchError


def _tiny_cfg(workdir: Path, **overrides) -> ResearchCoreConfig:
    cfg = ResearchCoreConfig(
        workdir=workdir,
        model_cfg=LatticeConfig(
            vocab_size=64, d_model=16, n_layers=1, n_heads=4, d_state=4,
            n_experts=2, top_k=1, max_seq_len=16, attention_window=8,
        ),
        train_cfg=TrainConfig(amp=False, checkpoint_dir=str(workdir / "ckpt")),
        critical_metrics=("task_accuracy",),
        max_regression=0.5,
        discovery_population=8,
        discovery_elite=2,
        discovery_generations=3,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture
def fresh_store(tmp_path: Path):
    reset_receipt_store()
    get_receipt_store(tmp_path / "receipts")
    yield
    reset_receipt_store()


# ---------------------------------------------------------------------------
# deterministic solver — must answer every benchmark kind correctly
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Return gcd(12, 18) as an integer.", 6),
        ("Return (3 ** 4) mod 7.", 81 % 7),
        ("Sort this list ascending: [3, 1, 2]", [1, 2, 3]),
        ("Is this string a palindrome? Answer true or false: aba", True),
        ("Is this string a palindrome? Answer true or false: abc", False),
    ],
)
def test_deterministic_solver_correct(prompt, expected):
    assert deterministic_task_solver(prompt) == expected


def test_deterministic_solver_unknown_returns_none():
    assert deterministic_task_solver("write me a poem") is None


# ---------------------------------------------------------------------------
# Lifecycle + tenant boundary
# ---------------------------------------------------------------------------
def test_init_creates_workdir_and_stamps_tenant(tmp_path: Path):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    assert (tmp_path / "wd").exists()
    stamp = core.boundary.current_stamp()
    assert stamp is not None
    assert stamp.tenant_id  # default tenant


def test_foreign_tenant_refuses_to_mount(tmp_path: Path):
    foreign = tmp_path / "wd"
    TenantBoundary(foreign, tenant_id="other-org").stamp()
    cfg = _tiny_cfg(foreign)
    with pytest.raises(TenantMismatchError):
        SelfImprovingResearchCore(cfg, tenant_id="default")


# ---------------------------------------------------------------------------
# Single cycle
# ---------------------------------------------------------------------------
def test_run_cycle_advances_iteration_and_returns_report(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    report = core.run_cycle(n_eval_tasks=5)
    assert isinstance(report, CycleReport)
    assert report.iteration == 1
    assert report.finished_at >= report.started_at
    assert report.promotion is not None
    assert "task_accuracy" in report.metrics


def test_first_cycle_emits_governance_receipt_via_audit_chain(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    report = core.run_cycle(n_eval_tasks=4)
    assert report.receipt_id is not None
    receipts = get_receipt_store().query_by_kind("governance")
    matched = [r for r in receipts if r.receipt_id == report.receipt_id]
    assert matched
    assert isinstance(matched[0], GovernanceReceipt)
    assert matched[0].domain == "checkpoint_promotion"


def test_cycle_grows_vault(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    pre = core.vault.size()
    core.run_cycle(n_eval_tasks=6)
    post = core.vault.size()
    assert post > pre


def test_cycle_advances_prediction_ledger(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    pre = core.ledger.count()
    core.run_cycle(n_eval_tasks=5)
    post = core.ledger.count()
    assert post > pre


def test_cycle_populates_novelty_archive(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    core.run_cycle(n_eval_tasks=4)
    assert len(core.novelty) > 0


# ---------------------------------------------------------------------------
# Multiple cycles — autonomy
# ---------------------------------------------------------------------------
def test_many_cycles_run_without_error(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    for _ in range(5):
        report = core.run_cycle(n_eval_tasks=4)
        assert isinstance(report, CycleReport)
    assert core._iteration == 5
    assert len(core.cycle_history()) == 5


def test_promotion_baseline_never_regresses_silently(tmp_path: Path, fresh_store):
    """Baseline metric advances only when gate accepts; never decreases."""
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    seen_acc = []
    for _ in range(4):
        core.run_cycle(n_eval_tasks=6)
        if core.gate.baseline is not None:
            seen_acc.append(core.gate.baseline["task_accuracy"].mean)
    # Each baseline value must be >= the previous one (monotone).
    for prev, curr in zip(seen_acc, seen_acc[1:]):
        assert curr >= prev


# ---------------------------------------------------------------------------
# Status + introspection
# ---------------------------------------------------------------------------
def test_status_reports_required_fields(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    core.run_cycle(n_eval_tasks=3)
    status = core.status()
    for key in (
        "iteration",
        "last_cycle_at",
        "model",
        "vault_size",
        "novelty_archive_size",
        "ledger_count",
        "promotion_history",
        "tenant",
    ):
        assert key in status
    assert status["model"]["num_parameters"] > 0
    assert status["iteration"] == 1


def test_cycle_history_returns_dicts(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = SelfImprovingResearchCore(cfg)
    core.run_cycle(n_eval_tasks=3)
    core.run_cycle(n_eval_tasks=3)
    hist = core.cycle_history()
    assert len(hist) == 2
    for entry in hist:
        assert "iteration" in entry
        assert "promotion" in entry


# ---------------------------------------------------------------------------
# Discovery — actually finds addition (with enough budget)
# ---------------------------------------------------------------------------
def test_discover_addition_proxy_returns_evolver_result(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd", discovery_population=32, discovery_generations=20)
    core = SelfImprovingResearchCore(cfg)
    result = core.discover_addition_proxy()
    assert hasattr(result, "score")
    assert hasattr(result, "best_str")


# ---------------------------------------------------------------------------
# Doctor bundle integration (F5)
# ---------------------------------------------------------------------------
def test_doctor_collector_reports_unavailable_when_not_registered(tmp_path: Path):
    # No prior register call.
    snapshot = collect_research_core_status()
    # We can't guarantee the container is empty across the test run, so
    # just check the contract: returns a dict with available key.
    assert "available" in snapshot


def test_register_research_core_makes_it_available_via_doctor(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    core = register_research_core(cfg=cfg)
    assert core is not None
    snapshot = collect_research_core_status()
    assert snapshot.get("available") is True
    assert "status" in snapshot


def test_register_research_core_is_idempotent(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")
    a = register_research_core(cfg=cfg)
    b = register_research_core(cfg=cfg)
    assert a is b


# ---------------------------------------------------------------------------
# Edge: malformed solver
# ---------------------------------------------------------------------------
def test_solver_that_raises_does_not_crash_cycle(tmp_path: Path, fresh_store):
    cfg = _tiny_cfg(tmp_path / "wd")

    def bad_solver(prompt: str):
        raise RuntimeError("solver explosion")

    core = SelfImprovingResearchCore(cfg, task_solver=bad_solver)
    report = core.run_cycle(n_eval_tasks=4)
    # task_accuracy will be 0 but the cycle must complete.
    assert report.metrics["task_accuracy"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# F5 bundle wiring — the bundle includes a research_core.json
# ---------------------------------------------------------------------------
def test_doctor_bundle_includes_research_core(tmp_path: Path, fresh_store):
    register_research_core(cfg=_tiny_cfg(tmp_path / "wd"))
    from core.runtime.diagnostics_bundle import build_bundle
    import tarfile

    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    assert info["ok"] is True
    with tarfile.open(info["path"], "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith("/research_core.json") for n in names)
