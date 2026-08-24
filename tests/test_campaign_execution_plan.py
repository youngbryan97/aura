"""The campaign's cost is counted, and no optimization buys speed with science.

Every number here is read from a retained receipt. An optimization whose
benefit cannot be counted is reported unmeasured rather than claimed, and one
that would change any arm's measured compute is rejected outright — including
one that makes a control cheaper, because equal-compute cuts both ways.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan_27b_campaign_execution as planner


@pytest.fixture(scope="module")
def plan():
    built = planner.build()
    if built.get("blocked"):
        pytest.skip(built["blocked"])
    return built


# ── The counts come from the receipt ────────────────────────────────────


def test_the_decode_count_includes_retries(plan):
    measured = plan["measured_reference"]
    # 300 arm rows, 364 actual calls. Counting rows would undercount the work
    # by every retry, which is where two of the arms spend their time.
    assert measured["decode_calls"] > measured["arm_rows"]
    assert measured["decode_calls"] - measured["arm_rows"] == measured["retry_decodes"]


def test_retries_concentrate_in_the_arms_designed_to_fail(plan):
    retries = plan["measured_reference"]["retries_by_arm"]
    assert retries["coefficient_lesion"] > retries["treatment"]
    assert retries["matched_wrong_state"] > retries["treatment"]
    # The two clean controls never retry: their responses are well-formed.
    assert retries["ordinary_base"] == 0
    assert retries["matched_wire_base"] == 0


def test_the_arm_seconds_reconstruct_the_elapsed_time(plan):
    measured = plan["measured_reference"]
    total = sum(measured["arm_seconds"].values())
    assert abs(total - measured["elapsed_seconds"]) / measured["elapsed_seconds"] < 0.01


def test_the_dominant_cost_is_named(plan):
    measured = plan["measured_reference"]
    assert measured["dominant_arm"] == "ordinary_base"
    # Half the campaign is the ordinary control, which generates several times
    # the tokens the typed arms do. It is also the floor the claim is measured
    # against, so its length is not available as a saving.
    assert measured["arm_seconds"]["ordinary_base"] > measured["elapsed_seconds"] * 0.4


# ── Nothing already-existing is claimed ─────────────────────────────────


def test_prefix_reuse_is_reported_as_existing_not_as_a_saving(plan):
    entry = next(
        e for e in plan["optimizations"] if e["name"] == "prompt_prefix_reuse"
    )
    assert entry["status"] == "already_implemented"
    assert entry["counted_saving"].startswith("0")
    assert plan["measured_reference"]["rows_whose_prefix_was_already_cached"] > 0


# ── Nothing load-bearing is optimized away ──────────────────────────────


def test_dropping_retries_is_rejected(plan):
    entry = next(
        e for e in plan["optimizations"] if e["name"] == "drop_serialization_retries"
    )
    assert entry["status"] == "rejected"
    assert entry["affects_measured_compute"] is True


def test_cross_arm_batching_is_rejected_for_counterbalancing(plan):
    entry = next(
        e
        for e in plan["optimizations"]
        if e["name"] == "batch_greedy_decodes_across_arms"
    )
    assert entry["status"] == "rejected"
    assert "counterbalanc" in entry["reason"]


def test_sharing_post_treatment_state_is_rejected(plan):
    entry = next(
        e
        for e in plan["optimizations"]
        if e["name"] == "share_post_treatment_state_between_arms"
    )
    assert entry["status"] == "rejected"


def test_every_rejected_optimization_names_measured_compute(plan):
    for entry in plan["optimizations"]:
        if entry["status"] == "rejected":
            assert entry["affects_measured_compute"] is True, entry["name"]


def test_every_adopted_optimization_leaves_measured_compute_alone(plan):
    for entry in plan["optimizations"]:
        if entry["status"] == "adopt":
            assert entry["affects_measured_compute"] is False, entry["name"]


# ── The scientific workload is unchanged ────────────────────────────────


def test_the_decode_call_count_is_identical_before_and_after(plan):
    counted = plan["counted_changes"]
    assert counted["decode_calls_before"] == counted["decode_calls_after"]


def test_the_saving_is_in_loads_and_tokenizations_not_seconds(plan):
    counted = plan["counted_changes"]
    assert counted["model_loads_before"] > counted["model_loads_after"]
    assert counted["task_tokenizations_before"] > counted["task_tokenizations_after"]
    assert "none" in plan["wall_clock_claim"].lower()


def test_no_wall_clock_speedup_is_asserted_anywhere(plan):
    for entry in plan["optimizations"]:
        if entry["status"] == "adopt":
            assert entry["wall_clock"].startswith("unmeasured")


def test_the_arm_equality_rule_covers_cheaper_controls(plan):
    # An optimization making a control cheaper breaks equal-compute in the
    # other direction, and reads as a saving rather than as damage.
    assert "control cheaper" in plan["arm_equality_rule"]


def test_tokenization_counts_only_the_immutable_half(plan):
    tokenization = plan["tokenization"]
    assert tokenization["task_prompt_tokenizations_required"] == (
        plan["measured_reference"]["task_count"]
    )
    assert "context" in tokenization["note"]


def test_the_plan_is_readable_without_the_model(plan):
    # Everything here is a receipt read; nothing loads weights.
    assert plan["measured_reference"]["source"].endswith("result.json")


def test_a_missing_reference_blocks_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(planner, "_reference", lambda: None)
    built = planner.build()
    assert built.get("blocked")
    assert built.get("measured") is None
