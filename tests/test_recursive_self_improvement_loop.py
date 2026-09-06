from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.learning.recursive_self_improvement import (
    ImprovementPlan,
    ImprovementScorecard,
    RecursiveSelfImprovementLoop,
)
from core.self_modification.structural_improver import StructuralImprover


class FakeLearner:
    def __init__(self, train_results):
        self.train_results = list(train_results)
        self.force_train_calls = 0
        self.rollback_calls = 0

    def get_learning_stats(self):
        return {
            "buffer_size": 12,
            "session_avg_quality": 0.4,
            "training_policy": {
                "fine_tune_type": "full",
                "full_weights_unlocked": True,
            },
        }

    async def force_train(self):
        self.force_train_calls += 1
        return self.train_results.pop(0) if self.train_results else True

    def rollback_adapter(self):
        self.rollback_calls += 1
        return True


@pytest.mark.asyncio
async def test_recursive_loop_reenters_after_verified_gain(tmp_path: Path):
    learner = FakeLearner([True, True])
    scores = iter([0.2, 0.35, 0.35, 0.5])

    loop = RecursiveSelfImprovementLoop(
        live_learner=learner,
        evaluator=lambda: ImprovementScorecard(score=next(scores)),
        ledger_path=tmp_path / "rsi.jsonl",
        max_depth=2,
        min_score_delta=0.05,
        require_will_authorization=False,
    )
    loop.record_signal("test", "training_data_ready", severity=0.7)

    result = await loop.run_cycle("improve reasoning", force=True)

    assert result.promoted is True
    assert len(result.child_results) == 1
    assert learner.force_train_calls == 2
    assert learner.rollback_calls == 0
    assert (tmp_path / "rsi.jsonl").exists()


@pytest.mark.asyncio
async def test_recursive_loop_rolls_back_weight_update_without_gain(tmp_path: Path):
    learner = FakeLearner([True])
    scores = iter([0.5, 0.49])
    loop = RecursiveSelfImprovementLoop(
        live_learner=learner,
        evaluator=lambda: ImprovementScorecard(score=next(scores)),
        ledger_path=tmp_path / "rsi.jsonl",
        max_depth=2,
        min_score_delta=0.01,
        require_will_authorization=False,
    )
    loop.record_signal("test", "training_data_ready", severity=0.7)

    result = await loop.run_cycle("avoid regression", force=False)

    assert result.promoted is False
    assert result.rollback_performed is True
    assert learner.rollback_calls == 1


@pytest.mark.asyncio
async def test_recursive_loop_uses_native_system2_to_rank_rsi_actions(monkeypatch, tmp_path: Path):
    class Receipt:
        commitment_reason = "System 2 prefers safer structural repair first"
        will_receipt_id = "will-system2-rsi"

        def to_dict(self):
            return {
                "commitment_reason": self.commitment_reason,
                "will_receipt_id": self.will_receipt_id,
            }

    class FakeSystem2:
        def __init__(self):
            self.calls = []

        async def rank_actions(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                search_id="s2-rsi-test",
                confidence=0.82,
                committed_action=SimpleNamespace(name="code_refinement", metadata={}),
                receipt=Receipt(),
            )

    fake = FakeSystem2()
    monkeypatch.setattr(ServiceContainer, "_services", dict(ServiceContainer._services))
    monkeypatch.setattr(ServiceContainer, "_aliases", dict(ServiceContainer._aliases))
    ServiceContainer.register_instance("native_system2", fake)

    loop = RecursiveSelfImprovementLoop(
        ledger_path=tmp_path / "rsi.jsonl",
        require_will_authorization=False,
    )
    plan = ImprovementPlan(
        objective="improve RSI planning",
        actions=["weight_update", "code_refinement"],
        rationale=["both actions are available"],
        depth=0,
    )

    refined = await loop._refine_plan_with_native_system2(
        plan,
        ImprovementScorecard(score=0.4, metrics={"quality": 0.4}),
    )

    assert fake.calls
    assert refined.actions[0] == "code_refinement"
    assert refined.system2_search_id == "s2-rsi-test"
    assert refined.system2_selected_action == "code_refinement"
    assert refined.system2_confidence == 0.82
    assert refined.system2_receipt["will_receipt_id"] == "will-system2-rsi"


@pytest.mark.asyncio
async def test_authorized_rsi_execution_resolves_the_system2_outcome(monkeypatch, tmp_path: Path):
    """A ranked plan teaches only after it is authorized, run, and evaluated."""

    class Receipt:
        commitment_reason = "measured preference"
        will_receipt_id = "will-rsi-outcome"

        def to_dict(self):
            return {"commitment_reason": self.commitment_reason}

    class FakeSystem2:
        def __init__(self):
            self.opened: list[tuple[str, str]] = []
            self.resolved: list[tuple[str, float, str]] = []

        async def rank_actions(self, **_kwargs):
            return SimpleNamespace(
                search_id="s2-rsi-outcome",
                confidence=0.8,
                committed_action=SimpleNamespace(name="weight_update", metadata={}),
                receipt=Receipt(),
            )

        def open_outcome_receipt(self, search_id, *, category, horizon_s):
            self.opened.append((search_id, category))
            assert horizon_s == 7200.0
            return "outcome-rsi"

        def resolve_outcome_receipt(self, receipt_id, observed, *, note):
            self.resolved.append((receipt_id, observed, note))
            return True

    fake = FakeSystem2()
    monkeypatch.setattr(ServiceContainer, "_services", dict(ServiceContainer._services))
    monkeypatch.setattr(ServiceContainer, "_aliases", dict(ServiceContainer._aliases))
    ServiceContainer.register_instance("native_system2", fake)

    learner = FakeLearner([True])
    scores = iter([0.2, 0.4])
    loop = RecursiveSelfImprovementLoop(
        live_learner=learner,
        evaluator=lambda: ImprovementScorecard(score=next(scores)),
        ledger_path=tmp_path / "rsi.jsonl",
        min_score_delta=0.05,
        auto_recurse=False,
        require_will_authorization=False,
    )
    loop.record_signal("test", "training_data_ready", severity=0.8)

    result = await loop.run_cycle("improve reasoning")

    assert result.promoted is True
    assert result.plan.system2_outcome_receipt_id == "outcome-rsi"
    assert fake.opened == [("s2-rsi-outcome", "recursive_self_improvement")]
    assert fake.resolved and fake.resolved[0][:2] == ("outcome-rsi", 1.0)


@pytest.mark.asyncio
async def test_code_refinement_falls_through_to_self_modifier_after_structural_error(tmp_path: Path):
    class BrokenStructuralImprover:
        def __init__(self):
            self.calls = 0

        def find_and_fix(self, max_repairs: int):
            self.calls += 1
            raise OSError("scan unavailable")

    class SelfModifier:
        def run_refinement_cycle(self):
            return {"success": True, "source": "safe_self_modifier"}

    structural_improver = BrokenStructuralImprover()
    loop = RecursiveSelfImprovementLoop(
        structural_improver=structural_improver,
        self_modifier=SelfModifier(),
        ledger_path=tmp_path / "rsi.jsonl",
        require_will_authorization=False,
    )

    result = await loop._run_code_refinement()

    assert result["ok"] is True
    assert structural_improver.calls == 1
    assert result["result"]["source"] == "safe_self_modifier"
    assert result["deterministic"]["ok"] is False
    assert result["deterministic"]["reason"].startswith("structural_improver:OSError")


def _somewhere_repairs_may_be_applied(root: Path) -> Path:
    """A path the mutation constitution puts in the automatic tier.

    The tiers are matched on the path, and everything outside the declared
    low-risk surface — tests, docs, generated skills, the proposal workspace —
    is propose-only or sealed. A repair written into a bare temporary
    directory is therefore correctly refused, so a test that wants the applying
    half has to ask about a path the constitution allows.
    """

    where = root / "patches" / "proposals"
    where.mkdir(parents=True, exist_ok=True)
    return where


def test_structural_improver_finds_and_repairs_missing_os_import(tmp_path: Path):
    at = _somewhere_repairs_may_be_applied(tmp_path)
    source = at / "mod.py"
    source.write_text(
        "def enabled():\n"
        "    return os.environ.get('AURA_FLAG') == '1'\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")

    issues = improver.scan()
    result = improver.apply_known_repair(issues[0])

    assert result.success is True, result.message
    assert "import os" in source.read_text(encoding="utf-8")


def test_a_repair_outside_the_allow_surface_is_refused_rather_than_applied(tmp_path: Path):
    """The deterministic path consults the same constitution the model path does.

    This asserted that the repair was applied, which was true and was the
    defect: the seal over the control plane held on the model-driven repair
    path and not on this one, while both are reachable from the same RSI
    action. "Sealed from every self-modification path" was true of one path
    and said of all of them.
    """

    source = tmp_path / "mod.py"
    source.write_text(
        "def enabled():\n"
        "    return os.environ.get('AURA_FLAG') == '1'\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")

    issues = improver.scan()
    assert issues, "the improver stopped finding the defect"
    result = improver.apply_known_repair(issues[0])

    assert result.success is False
    assert "not applied automatically" in result.message
    assert "import os" not in source.read_text(encoding="utf-8"), (
        "a refused repair was written anyway"
    )


def test_structural_improver_repairs_generated_gateway_mkdir(tmp_path: Path):
    source = _somewhere_repairs_may_be_applied(tmp_path) / "mod.py"
    source.write_text(
        "from pathlib import Path\n\n"
        "def make(root):\n"
        "    target = root / 'x'\n"
        "    get_task_tracker().create_task(get_storage_gateway().create_dir(target, cause='test'))\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")

    issue = next(i for i in improver.scan() if i.kind == "unsafe_async_gateway_mkdir")
    result = improver.apply_known_repair(issue)

    text = source.read_text(encoding="utf-8")
    assert result.success is True, result.message
    assert "Path(target).mkdir(parents=True, exist_ok=True)" in text
    assert "get_storage_gateway" not in text


def test_structural_improver_reports_rollback_failure(tmp_path: Path, monkeypatch):
    from core.self_modification import structural_improver as module

    source = _somewhere_repairs_may_be_applied(tmp_path) / "mod.py"
    source.write_text(
        "def enabled():\n"
        "    return os.environ.get('AURA_FLAG') == '1'\n",
        encoding="utf-8",
    )
    improver = StructuralImprover(tmp_path, ledger_path=tmp_path / "ledger.jsonl")
    issue = improver.scan()[0]
    writes = []

    # Patched at the gateway rather than at atomic_write_text, because the
    # gateway is now the write path: these two calls modify Aura's own source
    # and used to reach the primitive directly, skipping the governance check
    # and the ownership record. Stubbing here asserts the routing as well as
    # the rollback-failure report.
    class _FlakyGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown", **_):
            writes.append((str(path), source))
            if len(writes) == 2:
                raise OSError("rollback target locked")
            Path(path).write_text(text, encoding=encoding)

    monkeypatch.setattr(module, "get_file_write_gateway", _FlakyGateway)
    monkeypatch.setattr(improver, "_validate_files", lambda _paths: {"ok": False})

    result = improver.apply_known_repair(issue)

    assert result.success is False
    assert result.message == "validation failed; rollback failed"
    assert result.validation["rollback_error"] == "OSError: rollback target locked"
    # Both writes went through the gateway, each naming itself.
    assert [source for _path, source in writes] == [
        "self_modification.structural_improver.repair:missing_import_os",
        "self_modification.structural_improver.rollback",
    ]
