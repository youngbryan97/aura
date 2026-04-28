"""Tests for the capability-delta benchmark harness."""
from __future__ import annotations

import pytest

from aura_bench.capability_delta import (
    ABLATION_PROFILES,
    AblationProfile,
    DeltaReport,
    DeltaResult,
    profile_by_name,
    run_capability_delta,
)
from aura_bench.capability_delta.adapters.arithmetic_smoke import (
    ArithmeticSmokeAdapter,
)
from aura_bench.capability_delta.adapters.external_stubs import ALL_STUBS
from aura_bench.capability_delta.profiles import KNOWN_SUBSYSTEMS
from aura_bench.capability_delta.stub_llm import make_stub_llm


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def test_full_profile_enables_every_known_subsystem():
    full = profile_by_name("full")
    assert full.enabled_subsystems == KNOWN_SUBSYSTEMS


def test_base_llm_only_disables_everything():
    base = profile_by_name("base_llm_only")
    assert base.enabled_subsystems == frozenset()


def test_each_single_ablation_disables_exactly_one_subsystem():
    full = profile_by_name("full").enabled_subsystems
    for profile in ABLATION_PROFILES:
        if profile.name in {"full", "base_llm_only"}:
            continue
        diff = full - profile.enabled_subsystems
        assert len(diff) == 1, f"{profile.name} disables {len(diff)} subsystems"


def test_profile_by_name_unknown_raises():
    with pytest.raises(KeyError):
        profile_by_name("not_real")


def test_profile_set_is_closed():
    expected = {
        "full",
        "no_memory",
        "no_homeostasis",
        "no_global_workspace",
        "no_will",
        "no_affect",
        "base_llm_only",
    }
    assert {p.name for p in ABLATION_PROFILES} == expected


# ---------------------------------------------------------------------------
# stub LLM behaviour
# ---------------------------------------------------------------------------
def test_stub_llm_full_profile_is_perfect_on_arithmetic():
    llm = make_stub_llm()
    assert llm("What is 2 + 3?", "full") == "5"
    assert llm("What is 10 - 7?", "full") == "3"
    assert llm("What is 6 * 7?", "full") == "42"


def test_stub_llm_base_only_is_meaningfully_worse():
    llm = make_stub_llm(base_accuracy=0.2)
    correct = 0
    for a in range(20):
        prompt = f"What is {a} + {a + 1}?"
        truth = str(a + (a + 1))
        if llm(prompt, "base_llm_only") == truth:
            correct += 1
    # With base_accuracy=0.2 we expect ~4/20 correct, allow generous
    # tolerance because the hash-based degradation is deterministic.
    assert correct < 12


# ---------------------------------------------------------------------------
# end-to-end smoke run
# ---------------------------------------------------------------------------
def test_arithmetic_smoke_run_full_vs_base_shows_capability_delta():
    adapter = ArithmeticSmokeAdapter()
    llm = make_stub_llm(base_accuracy=0.25)
    report = run_capability_delta(adapter, llm=llm)

    assert isinstance(report, DeltaReport)
    assert "full" in report.by_profile
    assert "base_llm_only" in report.by_profile
    full = report.by_profile["full"]
    base = report.by_profile["base_llm_only"]
    assert full.mean_score == pytest.approx(1.0)  # stub gets full right
    assert base.mean_score < 0.6  # base is meaningfully degraded
    assert report.capability_delta > 0.4  # full beats base by a wide margin


def test_smoke_report_has_outcome_per_task_per_profile():
    adapter = ArithmeticSmokeAdapter()
    llm = make_stub_llm()
    report = run_capability_delta(adapter, llm=llm, max_tasks=5)
    for profile in ABLATION_PROFILES:
        result = report.by_profile[profile.name]
        assert isinstance(result, DeltaResult)
        assert len(result.outcomes) == 5
        for outcome in result.outcomes:
            assert outcome.profile_name == profile.name


def test_report_serialises_to_dict():
    adapter = ArithmeticSmokeAdapter()
    llm = make_stub_llm()
    report = run_capability_delta(adapter, llm=llm, max_tasks=3)
    payload = report.to_dict()
    assert payload["adapter_name"] == adapter.name
    assert "capability_delta" in payload
    assert "by_profile" in payload
    assert set(payload["by_profile"].keys()) == {p.name for p in ABLATION_PROFILES}


# ---------------------------------------------------------------------------
# stub adapters all conform to the contract
# ---------------------------------------------------------------------------
def test_all_external_stubs_implement_adapter_contract():
    llm = make_stub_llm()
    for stub in ALL_STUBS:
        assert isinstance(stub.name, str) and stub.name
        tasks = list(stub.tasks())
        assert tasks, f"{stub.name} produced no tasks"
        outcome = stub.run(tasks[0], "full", llm)
        assert outcome.task_id == tasks[0].task_id
        assert outcome.profile_name == "full"
        assert outcome.metadata.get("synthetic") is True


def test_smoke_adapter_max_tasks_caps_run():
    adapter = ArithmeticSmokeAdapter()
    llm = make_stub_llm()
    report = run_capability_delta(adapter, llm=llm, max_tasks=2)
    for result in report.by_profile.values():
        assert result.n_tasks == 2


# ---------------------------------------------------------------------------
# delta semantics
# ---------------------------------------------------------------------------
def test_capability_delta_is_zero_when_full_and_base_match():
    """A degenerate adapter where every profile yields 1.0 should give delta=0."""

    class AlwaysWinAdapter:
        name = "always_win"

        def tasks(self):
            return [BenchTaskFromName(f"task-{i}") for i in range(3)]

        def run(self, task, profile_name, llm):
            from aura_bench.capability_delta.adapter import TaskOutcome
            return TaskOutcome(
                task_id=task.task_id,
                profile_name=profile_name,
                score=1.0,
                runtime_seconds=0.0,
                success=True,
            )

    from dataclasses import dataclass

    @dataclass
    class BenchTaskFromName:
        task_id: str
        prompt: str = ""
        metadata: dict = None

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

    report = run_capability_delta(
        AlwaysWinAdapter(), llm=make_stub_llm(), max_tasks=3
    )
    assert report.capability_delta == pytest.approx(0.0)
