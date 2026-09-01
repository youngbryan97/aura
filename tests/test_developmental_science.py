"""Did living through the earlier tasks help, and what did each organ earn.

Cards 026, 042, 052, 054, 068, 071, 072, 086, 093, 121, 123, 129, 184, 185,
189, 200, 201, 202, 205, 206, A6.4, A6.6.
"""
from __future__ import annotations

import random

import pytest

from core.science.continual_metrics import (
    DEFAULT_FORGETTING_BUDGET,
    Artifact,
    ArtifactEcology,
    ContinualLedger,
)
from core.science.developmental_campaign import Arm, DevelopmentalCampaign
from core.science.organ_accounting import (
    ArmKind,
    OrganMeasurement,
    hochberg,
    reset_organ_accounting_for_test,
    synergy,
)


def _campaign(*, grown_slope, lesion_slope=0.0, grown_context=1000, blocks=8, seed=3, repeat_answers=False):
    rng = random.Random(seed)
    campaign = DevelopmentalCampaign(seed=seed)
    for block in range(blocks):
        for task in range(10):
            key = f"a{task}" if repeat_answers else f"a{block}_{task}"
            campaign.record(block, f"t{block}_{task}", Arm.GROWN,
                            0.5 + grown_slope * block + rng.gauss(0, 0.02),
                            answer_key=key, context_tokens=grown_context)
            campaign.record(block, f"t{block}_{task}", Arm.RESET,
                            0.5 + rng.gauss(0, 0.02), answer_key=key, context_tokens=1000)
            campaign.record(block, f"t{block}_{task}", Arm.GROWN_LESIONED,
                            0.5 + lesion_slope * block + rng.gauss(0, 0.02),
                            answer_key=key, context_tokens=1000)
    return campaign.verdict()


# ── the decisive experiment ───────────────────────────────────────────────

def test_a_growing_gap_that_lesioning_erases_is_the_claim():
    verdict = _campaign(grown_slope=0.03, lesion_slope=0.0)
    assert verdict.effect == "compounding"
    assert verdict.lesion_restores_baseline
    assert "lesioning the acquired artifacts erases it" in verdict.statement


def test_a_growing_gap_that_survives_lesioning_is_something_else():
    verdict = _campaign(grown_slope=0.03, lesion_slope=0.03)
    assert verdict.effect == "compounding"
    assert verdict.lesion_restores_baseline is False
    assert "something else is causing it" in verdict.statement


def test_repeating_answers_across_blocks_voids_the_campaign():
    verdict = _campaign(grown_slope=0.03, repeat_answers=True)
    assert verdict.void
    assert "growth that is recall is recall" in verdict.statement


def test_giving_the_grown_arm_more_context_voids_the_campaign():
    verdict = _campaign(grown_slope=0.03, grown_context=4000)
    assert verdict.void
    assert "wearing development's clothes" in verdict.statement


def test_too_few_blocks_voids_the_campaign():
    verdict = _campaign(grown_slope=0.03, blocks=3)
    assert verdict.void
    assert "a slope needs at least" in verdict.statement


def test_a_true_null_reports_no_effect_at_about_the_nominal_rate():
    positives = sum(
        1 for seed in range(40)
        if _campaign(grown_slope=0.0, seed=seed).effect != "no effect"
    )
    assert positives <= 6, f"{positives}/40 false positives is more than a 95% interval allows"


def test_a_campaign_with_no_lesion_arm_says_it_has_no_causal_claim():
    rng = random.Random(5)
    campaign = DevelopmentalCampaign(seed=5)
    for block in range(8):
        for task in range(10):
            campaign.record(block, f"t{block}_{task}", Arm.GROWN,
                            0.5 + 0.03 * block + rng.gauss(0, 0.02),
                            answer_key=f"a{block}_{task}", context_tokens=1000)
            campaign.record(block, f"t{block}_{task}", Arm.RESET,
                            0.5 + rng.gauss(0, 0.02),
                            answer_key=f"a{block}_{task}", context_tokens=1000)
    verdict = campaign.verdict()
    assert verdict.lesion_restores_baseline is None
    assert "no lesion arm was run" in verdict.statement


def test_the_three_arms_running_one_task_is_not_contamination():
    verdict = _campaign(grown_slope=0.03)
    assert verdict.contaminated_fraction == 0.0


# ── organ accounting ──────────────────────────────────────────────────────

def test_an_organ_whose_effect_is_its_compute_is_named_as_such():
    accounting = reset_organ_accounting_for_test()
    accounting.measure(OrganMeasurement(
        "theatre", {"full": 0.80, "lesioned": 0.60, "compute_matched": 0.798}, n=50, p_value=0.002
    ))
    assert accounting.verdicts()[0].classification == "compute_not_computation"


def test_an_organ_whose_content_does_not_matter_is_named_as_such():
    accounting = reset_organ_accounting_for_test()
    accounting.measure(OrganMeasurement(
        "seat_at_the_table",
        {"full": 0.80, "lesioned": 0.60, "compute_matched": 0.70, "noise_control": 0.799},
        n=50, p_value=0.002,
    ))
    assert accounting.verdicts()[0].classification == "presence_not_content"


def test_an_organ_that_survives_every_control_is_load_bearing():
    accounting = reset_organ_accounting_for_test()
    accounting.measure(OrganMeasurement(
        "workspace",
        {"full": 0.80, "lesioned": 0.60, "compute_matched": 0.62, "noise_control": 0.63},
        n=50, p_value=0.001,
    ))
    assert accounting.verdicts()[0].classification == "load_bearing"


def test_an_organ_with_no_lesion_arm_is_unmeasured_not_assumed_useful():
    accounting = reset_organ_accounting_for_test()
    accounting.measure(OrganMeasurement("never_tested", {"full": 0.8}, n=50))
    assert accounting.verdicts()[0].classification == "unmeasured"


def test_a_lesion_effect_with_no_matched_arm_is_causal_but_unpriced():
    accounting = reset_organ_accounting_for_test()
    accounting.measure(OrganMeasurement(
        "expensive", {"full": 0.8, "lesioned": 0.5}, n=50, p_value=0.01
    ))
    assert accounting.verdicts()[0].classification == "causal_unpriced"


def test_one_marginal_result_among_twenty_nine_nulls_does_not_survive():
    """The case that matters: a p of 0.04 found by looking at thirty organs."""
    assert not any(hochberg([0.04] + [0.9] * 29))


def test_thirty_organs_all_at_p_equals_point_zero_four_do_all_survive():
    """Hochberg is a step-up test and this is the correct behaviour, not a bug.

    When every organ is marginal the joint result is unlikely, and the step-up
    rule rejects the whole family. Bonferroni would keep none of them. The
    difference is why the correction is named in the report rather than
    applied silently.
    """
    assert all(hochberg([0.04] * 30))


def test_one_strong_effect_survives_among_many_weak_ones():
    assert hochberg([0.0001] + [0.6] * 29)[0]


def test_synergy_is_negative_when_two_organs_interfere():
    assert synergy(0.6, 0.7, 0.65) < 0
    assert synergy(0.9, 0.7, 0.6) > 0


def test_interfering_pairs_are_listed_rather_than_averaged_away():
    accounting = reset_organ_accounting_for_test()
    accounting.record_synergy("workspace", "planner", pair=0.6, a_alone=0.7, b_alone=0.65)
    accounting.record_synergy("workspace", "memory", pair=0.9, a_alone=0.7, b_alone=0.6)
    assert accounting.report()["interfering_pairs"] == ["planner+workspace"]


# ── continual metrics ─────────────────────────────────────────────────────

def test_learning_a_new_task_that_breaks_an_old_one_is_measured():
    ledger = ContinualLedger()
    for block, task in enumerate(("a", "b", "c")):
        ledger.trained(task, block)
        ledger.record(task, block, 0.9)
    ledger.record("a", 2, 0.55)
    ledger.record("b", 2, 0.88)
    backward = ledger.backward_transfer(2)
    assert backward["forgetting"]
    assert backward["worst"][0] == "a"


def test_a_block_that_costs_too_much_backward_transfer_does_not_promote():
    ledger = ContinualLedger()
    for block, task in enumerate(("a", "b")):
        ledger.trained(task, block)
        ledger.record(task, block, 0.9)
    ledger.record("a", 1, 0.5)
    verdict = ledger.promotion_verdict(1, own_task_gain=0.4)
    assert not verdict["promote"]
    assert verdict["budget"] == DEFAULT_FORGETTING_BUDGET


def test_a_block_that_gains_without_forgetting_promotes():
    ledger = ContinualLedger()
    for block, task in enumerate(("a", "b")):
        ledger.trained(task, block)
        ledger.record(task, block, 0.9)
    ledger.record("a", 1, 0.9)
    assert ledger.promotion_verdict(1, own_task_gain=0.3)["promote"]


def test_forward_transfer_needs_a_naive_comparison_to_mean_anything():
    ledger = ContinualLedger()
    ledger.trained("b", 1)
    ledger.record("b", 0, 0.6)
    assert not ledger.forward_transfer("b")["measurable"]
    ledger.record("b", 0, 0.6, naive_score=0.3)
    assert ledger.forward_transfer("b")["fwt"] == pytest.approx(0.3)


# ── the ecology ───────────────────────────────────────────────────────────

def test_an_artifact_that_costs_more_than_it_earns_is_retired():
    ecology = ArtifactEcology()
    ecology.add(Artifact("good", "procedure", benefit_per_use=1.0, match_cost=0.05))
    ecology.add(Artifact("bad", "procedure", benefit_per_use=0.01, match_cost=0.5, interference=2.0))
    for _ in range(5):
        ecology.use("good")
        ecology.use("bad")
    assert [a.artifact_id for a in ecology.retire_what_does_not_pay()] == ["bad"]


def test_retiring_is_not_deleting_so_she_can_say_what_she_lost():
    ecology = ArtifactEcology()
    ecology.add(Artifact("bad", "rule", benefit_per_use=0.0, storage_cost=5.0))
    for _ in range(3):
        ecology.use("bad")
    ecology.retire_what_does_not_pay()
    lost = ecology.report()["what_she_stopped_being_able_to_do"]
    assert lost and lost[0]["artifact_id"] == "bad" and lost[0]["why"]


def test_a_barely_used_artifact_is_not_retired_on_one_bad_run():
    ecology = ArtifactEcology()
    ecology.add(Artifact("new", "procedure", benefit_per_use=0.0, storage_cost=1.0))
    ecology.use("new")
    assert ecology.retire_what_does_not_pay() == []


def test_interference_is_counted_against_an_artifact_not_just_its_own_cost():
    quiet = Artifact("quiet", "rule", uses=5, benefit_per_use=0.5)
    noisy = Artifact("noisy", "rule", uses=5, benefit_per_use=0.5, interference=10.0)
    assert quiet.retention_value > noisy.retention_value
