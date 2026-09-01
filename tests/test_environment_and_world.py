"""The same code in an unseen world, and a world model that can be wrong.

Cards 122, 125, 126, 127, 130, 131, 132, 133, 134, 136, 137, 138-153, 150,
181, 182, 197, 198, 203, 204, 219, 128, 062, 101, 104, 135, A6.1, A6.2, A6.3,
A6.5, A6.7, A6.8, A6.9, A6.10, A6.11, A6.12.
"""
from __future__ import annotations

import pytest

from core.cognition.agent_model import MAX_PERSPECTIVE_DEPTH, reset_agent_registry_for_test
from core.cognition.attention_policy import FLOOR, reset_attention_policy_for_test
from core.cognition.curriculum_optimiser import reset_curriculum_for_test
from core.cognition.structure_mapping import Graph, Relation, map_structures, shuffled_null
from core.science.environment_bench import (
    EnvironmentBench,
    EpisodeResult,
    Family,
    Fault,
    horizon_curve,
    scan_for_environment_branches,
)
from core.world_model.prediction_quality import (
    CalibrationCurve,
    InterventionLedger,
    MultiTimescalePrediction,
    ObjectiveComparison,
    PredictionOutcome,
    ensemble_disagreement,
)


# ── environment families ──────────────────────────────────────────────────

def _bench(seed=1):
    bench = EnvironmentBench(seed=seed)
    for name, envs in (("browser", ("a", "b")), ("desktop", ("c", "d")),
                       ("games2d", ("e", "f")), ("terminal", ("g", "h"))):
        bench.add_family(Family(name, envs, shared="screen and keyboard only"))
    return bench


def test_the_split_is_frozen_before_anything_runs():
    bench = _bench()
    first = bench.split()
    assert bench.split() == first
    with pytest.raises(RuntimeError, match="results in view"):
        bench.add_family(Family("late", ("z",)))


def test_a_family_is_held_out_whole_not_by_episode():
    split = _bench().split()
    assert set(split.values()) == {"train", "held_out"}
    assert sum(1 for v in split.values() if v == "held_out") >= 1


def test_transfer_compares_held_out_families_against_trained_ones():
    bench = _bench()
    split = bench.split()
    for family, role in split.items():
        for i in range(10):
            bench.record(EpisodeResult(
                environment=f"{family}_env", family=family,
                succeeded=(i < 9) if role == "train" else (i < 5),
                actions=20, seed=i,
            ))
    transfer = bench.transfer()
    assert transfer["train_success"] > transfer["held_out_success"]
    assert transfer["gap"] > 0
    assert len(transfer["leave_one_family_out"]) == 4


def test_screen_pursuit_has_no_environment_name_branch():
    result = scan_for_environment_branches(
        ["core/skills/screen_pursuit.py"],
        ["2048", "chrome", "safari", "terminal", "minecraft", "finder"],
    )
    assert result["clean"], result["hits"]


def test_a_docstring_naming_an_app_is_not_a_branch():
    result = scan_for_environment_branches(["core/skills/screen_pursuit.py"], ["2048"])
    assert result["clean"], "prose explaining why a tab title is read is documentation"


def test_intent_parsing_is_reported_apart_from_cognition_keyed_on_a_world():
    result = scan_for_environment_branches(
        ["core/skills/desktop_task.py"], ["chrome", "safari"],
        reading_user_intent=["_preferred_browser"],
    )
    assert result["reading_user_intent"], "matching what the user said is not a branch"
    assert all(h["function"] != "_preferred_browser" for h in result["hits"])


def test_a_hit_names_the_function_so_a_reviewer_can_judge_it():
    result = scan_for_environment_branches(["core/skills/desktop_task.py"], ["safari"])
    assert result["hits"] and all(h["function"] for h in result["hits"])


# ── recovery, scored apart from success ───────────────────────────────────

def test_recovery_is_measured_per_fault_and_not_folded_into_success():
    bench = _bench()
    bench.split()
    for fault in Fault:
        for i in range(10):
            bench.record(EpisodeResult(
                environment="a", family="browser", succeeded=True, actions=30,
                fault_injected=fault, recovered=(i < 7), steps_to_recover=3 if i < 7 else None,
            ))
    recovery = bench.recovery()
    assert recovery.rate == pytest.approx(0.7)
    assert set(recovery.by_fault) == {f.value for f in Fault}
    assert recovery.mean_steps_to_recover == pytest.approx(3.0)


def test_a_run_with_no_injected_faults_reports_no_recovery_rather_than_a_perfect_one():
    bench = _bench()
    bench.split()
    bench.record(EpisodeResult(environment="a", family="browser", succeeded=True, actions=5))
    assert bench.recovery().injected == 0
    assert bench.recovery().rate == 0.0


# ── horizon ───────────────────────────────────────────────────────────────

def test_errors_that_are_corrected_show_a_sublinear_horizon_curve():
    results = [
        EpisodeResult(environment="a", family="f", succeeded=i < int(rate * 20),
                      actions=actions, seed=i)
        for actions, rate in ((10, 0.95), (100, 0.9), (1000, 0.85))
        for i in range(20)
    ]
    curve = horizon_curve(results)
    assert curve["measurable"] and curve["sublinear"]


def test_errors_that_accumulate_show_a_steep_curve():
    results = [
        EpisodeResult(environment="a", family="f", succeeded=i < int(rate * 20),
                      actions=actions, seed=i)
        for actions, rate in ((10, 0.95), (100, 0.35), (1000, 0.05))
        for i in range(20)
    ]
    curve = horizon_curve(results)
    assert curve["measurable"] and not curve["sublinear"]


def test_one_bucket_is_not_a_curve():
    results = [EpisodeResult(environment="a", family="f", succeeded=True, actions=10)]
    assert not horizon_curve(results)["measurable"]


# ── world-model quality ───────────────────────────────────────────────────

def test_a_model_that_says_point_nine_and_is_right_point_six_is_overconfident():
    curve = CalibrationCurve()
    for i in range(100):
        curve.observe(PredictionOutcome(confidence=0.9, error=0.05 if i % 10 < 6 else 0.5))
    assert curve.overconfidence == pytest.approx(0.3, abs=0.01)
    assert not curve.to_dict()["calibrated"]


def test_a_calibrated_model_says_so():
    curve = CalibrationCurve()
    for i in range(200):
        curve.observe(PredictionOutcome(confidence=0.6, error=0.05 if i % 10 < 6 else 0.5))
    assert curve.to_dict()["calibrated"]


def test_not_knowing_is_separated_from_the_world_being_noisy():
    unknown = ensemble_disagreement([0.1, 0.9, 0.5], [0.01, 0.01, 0.01])
    noisy = ensemble_disagreement([0.5, 0.5, 0.5], [0.4, 0.4, 0.4])
    assert unknown.worth_investigating
    assert not noisy.worth_investigating


def test_the_objective_with_the_better_loss_can_be_the_worse_planner():
    comparison = ObjectiveComparison(
        reconstruction_loss=0.10, latent_loss=0.30,
        reconstruction_control_success=0.4, latent_control_success=0.7,
        compute_matched=True,
    )
    assert comparison.verdict == "latent prediction plans better"
    assert comparison.loss_disagrees_with_control


def test_an_unmatched_objective_comparison_is_void():
    comparison = ObjectiveComparison(0.1, 0.3, 0.4, 0.7, compute_matched=False)
    assert comparison.verdict.startswith("void")


def test_the_useful_horizon_is_where_the_model_stops_beating_the_baseline():
    scales = MultiTimescalePrediction()
    for horizon, error in ((1, 0.05), (10, 0.2), (100, 0.9)):
        for _ in range(10):
            scales.observe(horizon, error=error, baseline_error=0.5)
    assert scales.useful_horizon()["useful_to"] == 10


def test_a_model_that_predicts_observations_and_not_interventions_is_named():
    ledger = InterventionLedger()
    for _ in range(20):
        ledger.record_observation("x", 1.0, 1.02)
        ledger.record_intervention("x", 1.0, 3.0)
    verdict = ledger.verdict()
    assert not verdict["causal"]
    assert "what goes with what" in verdict["reading"]


def test_a_model_that_survives_intervention_says_so():
    ledger = InterventionLedger()
    for _ in range(20):
        ledger.record_observation("x", 1.0, 1.02)
        ledger.record_intervention("x", 1.0, 1.03)
    assert ledger.verdict()["causal"]


def test_intervention_needs_both_arms():
    ledger = InterventionLedger()
    ledger.record_intervention("x", 1.0, 1.0)
    assert not ledger.verdict()["measurable"]


# ── attention ─────────────────────────────────────────────────────────────

def test_attention_moves_toward_what_it_bought():
    policy = reset_attention_policy_for_test()
    policy.register("useful", "memory")
    policy.register("useless", "memory")
    for _ in range(50):
        policy.observe("useful", spent=1.0, returned=1.0)
        policy.observe("useless", spent=1.0, returned=0.0)
        policy.reallocate()
    allocation = policy.allocation()
    assert allocation["useful"] > allocation["useless"]


def test_no_channel_is_ever_starved_below_the_floor():
    policy = reset_attention_policy_for_test()
    policy.register("winner", "memory")
    policy.register("loser", "percept")
    for _ in range(200):
        policy.observe("winner", spent=1.0, returned=10.0)
        policy.observe("loser", spent=1.0, returned=0.0)
        policy.reallocate()
    assert policy.allocation()["loser"] >= FLOOR * 0.99


def test_one_lucky_turn_does_not_reallocate_the_mind():
    policy = reset_attention_policy_for_test()
    policy.register("a", "memory")
    policy.register("b", "memory")
    before = policy.allocation()
    policy.observe("a", spent=1.0, returned=100.0)
    policy.reallocate()
    assert abs(policy.allocation()["a"] - before["a"]) < 0.2


def test_a_learned_allocation_with_no_static_comparison_is_visibly_unjustified():
    policy = reset_attention_policy_for_test()
    policy.register("a", "memory", static_weight=0.5)
    assert policy.report()["beats_static"] is None
    policy.against_static(0.8, 0.6)
    assert policy.report()["beats_static"]


# ── structure mapping ─────────────────────────────────────────────────────

def _solar():
    return Graph("solar", (
        Relation("attracts", ("sun", "planet"), 2),
        Relation("hotter", ("sun", "planet")),
        Relation("revolves", ("planet", "sun"), 2),
    ))


def _atom():
    return Graph("atom", (
        Relation("attracts", ("nucleus", "electron"), 2),
        Relation("hotter", ("nucleus", "electron")),
        Relation("revolves", ("electron", "nucleus"), 2),
    ))


def test_a_structural_analogy_is_found_across_a_shared_vocabulary_of_none():
    alignment = map_structures(_solar(), _atom())
    assert alignment.mapping == {"sun": "nucleus", "planet": "electron"}
    assert alignment.shares_no_vocabulary


def test_the_scrambled_control_scores_lower_than_the_real_structure():
    result = shuffled_null(_solar(), _atom(), trials=40)
    assert result["structural"]
    assert result["separation"] > 0.2


def test_an_unrelated_domain_aligns_with_nothing():
    unrelated = Graph("u", (Relation("foo", ("a", "b")), Relation("bar", ("b", "c"))))
    assert shuffled_null(_solar(), unrelated, trials=10)["score"] == 0.0


def test_a_domain_too_big_for_the_exhaustive_search_refuses_rather_than_failing_quietly():
    big = Graph("big", tuple(Relation("r", (f"o{i}", f"o{i+1}")) for i in range(10)))
    with pytest.raises(ValueError, match="exhaustive search"):
        map_structures(big, big)


# ── agent models ──────────────────────────────────────────────────────────

def test_a_false_belief_is_acted_on_rather_than_the_truth():
    registry = reset_agent_registry_for_test()
    model = registry.model("sam")
    model.observe_belief("the keys are in the drawer", supports=True, evidence="said so")
    verdict = model.false_belief("the keys are in the drawer", world_truth=False)
    assert verdict["acts_on"] is True and verdict["diverges_from_reality"]


def test_perspective_nesting_is_bounded():
    model = reset_agent_registry_for_test().model("sam")
    model.observe_belief("x", supports=True, evidence="e", about="ana", depth=2)
    with pytest.raises(ValueError, match="starts inventing"):
        model.observe_belief("x", supports=True, evidence="e", about="ana", depth=3)


def test_a_model_that_does_not_beat_the_language_prior_says_so():
    registry = reset_agent_registry_for_test()
    model = registry.model("sam")
    for i in range(10):
        model.predict("lunch", "sandwich", "sandwich")
        model.resolve(i, "sandwich")
    assert not model.beats_the_prior()["learned_something"]


def test_a_model_that_learned_the_person_beats_the_prior():
    registry = reset_agent_registry_for_test()
    model = registry.model("sam")
    for i in range(10):
        model.predict("lunch", "salad", "sandwich")
        model.resolve(i, "salad")
    assert model.beats_the_prior()["learned_something"]
    assert registry.report()["beating_the_prior"] == ["sam"]


def test_reliability_is_per_topic_because_the_mean_is_true_of_neither():
    model = reset_agent_registry_for_test().model("sam")
    for _ in range(10):
        model.observe_reliability("dates", accurate=True)
        model.observe_reliability("names", accurate=False)
    spread = model.reliability_range()
    assert spread["spread"] == pytest.approx(1.0)
    assert spread["lowest"][0] == "names"


# ── curriculum ────────────────────────────────────────────────────────────

def _curriculum():
    curriculum = reset_curriculum_for_test(
        governance=lambda task: (False, "out of bounds") if task.family == "dangerous" else (True, "")
    )
    for task_id, family, scores in (
        ("mastered", "a", [0.95] * 8),
        ("frontier", "b", [0.2, 0.3, 0.5, 0.6, 0.7, 0.8]),
        ("impossible", "c", [0.0] * 8),
        ("forbidden", "dangerous", [0.5]),
    ):
        curriculum.offer(task_id, family)
        for score in scores:
            curriculum.record(task_id, score)
    return curriculum


def test_the_frontier_is_selected_over_the_mastered_and_the_impossible():
    assert _curriculum().select(1)[0]["task_id"] == "frontier"


def test_a_task_outside_the_envelope_never_competes():
    curriculum = _curriculum()
    assert all(row["task_id"] != "forbidden" for row in curriculum.select(10))
    assert curriculum.report()["refused_by_governance"][0]["task_id"] == "forbidden"


def test_mastered_and_impossible_are_both_named():
    report = _curriculum().report()
    assert report["mastered"] == ["mastered"]
    assert report["out_of_reach"] == ["impossible"]


def test_diversity_stops_one_family_absorbing_the_curriculum():
    curriculum = reset_curriculum_for_test()
    curriculum.offer("hot", "a")
    curriculum.offer("cold", "b")
    for score in (0.1, 0.3, 0.6, 0.9):
        curriculum.record("hot", score)
    for _ in range(20):
        curriculum.record("hot", 0.9)
    selected = {row["task_id"]: row for row in curriculum.select(2)}
    assert selected["cold"]["diversity_bonus"] > selected["hot"]["diversity_bonus"]
