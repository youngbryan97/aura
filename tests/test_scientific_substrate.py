"""What a number has to survive before it can support a claim.

Cards 010, 084, 164, 169, 179, 191, 194, 210, 211, 213, 214, 220, A2.15,
A4.1-A4.10.
"""
from __future__ import annotations

import pytest

from core.science.calibration_layer import MIN_OBSERVATIONS, reset_calibration_layer_for_test
from core.science.claim_ladder import (
    PrerequisiteMissing,
    Rung,
    get_ladder,
    reset_ladder_for_test,
)
from core.science.experiment_registry import (
    Arm,
    MalformedExperiment,
    reset_experiment_registry_for_test,
)
from core.science.learning_audit import Link, reset_learning_audit_for_test
from core.science.parameter_registry import (
    Kind,
    Parameter,
    UnprovenanceError,
    reset_parameter_registry_for_test,
)

BUDGET = {"model_id": "qwen3.8-27b", "max_output_tokens": 512, "max_wall_clock_s": 60.0}


# ── the claim ladder ──────────────────────────────────────────────────────

def test_a_rung_requires_every_rung_below_it():
    ladder = reset_ladder_for_test()
    with pytest.raises(PrerequisiteMissing, match="no wired evidence"):
        ladder.register(
            "the workspace helps", owner="o",
            supports=[(Rung.EXISTS, "core/evidence/packet.py"), (Rung.USEFUL, "core/evidence/packet.py")],
            boundary="b",
        )


def test_a_claim_with_no_boundary_is_refused():
    ladder = reset_ladder_for_test()
    with pytest.raises(ValueError, match="states no boundary"):
        ladder.register("x", owner="o", supports=[(Rung.EXISTS, "core/evidence/packet.py")], boundary="  ")


def test_a_support_that_does_not_exist_is_refused():
    ladder = reset_ladder_for_test()
    with pytest.raises(PrerequisiteMissing, match="do not exist"):
        ladder.register("x", owner="o", supports=[(Rung.EXISTS, "core/nope.py")], boundary="b")


def test_the_rung_is_the_highest_unbroken_run():
    ladder = reset_ladder_for_test()
    claim = ladder.register(
        "evidence is sourced", owner="o",
        supports=[
            (Rung.EXISTS, "core/evidence/packet.py"),
            (Rung.WIRED, "core/knowledge/atomspace.py"),
            (Rung.CAUSAL, "tests/test_evidence_independence.py"),
        ],
        boundary="only on sourced paths",
    )
    assert claim.rung is Rung.CAUSAL
    assert claim.to_dict()["question_answered"].startswith("does lesioning")


def test_the_shipped_claims_stand_where_their_artifacts_reach():
    audit = get_ladder().audit()
    assert audit["ok"], audit["degraded"]
    assert audit["at_or_above_useful"] == 0, (
        "no artifact in this session establishes a task improvement, so nothing may "
        "claim USEFUL"
    )


def test_every_rung_states_the_question_it_answers():
    assert all(rung.question for rung in Rung)


# ── experiments ───────────────────────────────────────────────────────────

def _arm(name, role, score, **kw):
    return Arm(name, role, score, kw.pop("n", 20), seeds=kw.pop("seeds", (1, 2, 3)), budget=BUDGET, **kw)


def test_an_experiment_with_no_null_cannot_be_registered():
    registry = reset_experiment_registry_for_test()
    with pytest.raises(MalformedExperiment, match="cannot fail"):
        registry.register(
            "e", hypothesis="h", arms=[_arm("t", "treatment", 0.8)],
            task_families=("a",), claim_boundary="b",
        )


def test_an_experiment_with_one_sample_cannot_be_registered():
    registry = reset_experiment_registry_for_test()
    with pytest.raises(MalformedExperiment, match="settles nothing"):
        registry.register(
            "e", hypothesis="h",
            arms=[_arm("t", "treatment", 0.8, seeds=()), _arm("n", "null", 0.4)],
            task_families=("a",), claim_boundary="b",
        )


def test_arms_that_were_not_allowed_the_same_resources_are_refused():
    registry = reset_experiment_registry_for_test()
    rich = dict(BUDGET, max_output_tokens=4096)
    with pytest.raises(MalformedExperiment, match="same resources"):
        registry.register(
            "e", hypothesis="h",
            arms=[
                Arm("t", "treatment", 0.9, 20, seeds=(1,), budget=rich),
                Arm("n", "null", 0.1, 20, seeds=(1,), budget=BUDGET),
            ],
            task_families=("a",), claim_boundary="b",
        )


def test_an_experiment_with_no_stated_boundary_is_refused():
    registry = reset_experiment_registry_for_test()
    with pytest.raises(MalformedExperiment, match="general result"):
        registry.register(
            "e", hypothesis="h", arms=[_arm("t", "treatment", 0.8), _arm("n", "null", 0.4)],
            task_families=("a",), claim_boundary="",
        )


def test_a_well_formed_experiment_reports_its_separation_and_its_commit():
    registry = reset_experiment_registry_for_test()
    record = registry.register(
        "e", hypothesis="the treatment beats its null",
        arms=[_arm("t", "treatment", 0.82), _arm("n", "null", 0.41), _arm("b", "baseline", 0.5)],
        task_families=("recall", "planning"), claim_boundary="two families only",
        verdict="supported",
    )
    assert record.separation == pytest.approx(0.41)
    assert record.null_failed
    assert record.commit and record.content_hash


def test_a_null_that_beats_the_treatment_is_recorded_as_such():
    registry = reset_experiment_registry_for_test()
    record = registry.register(
        "e", hypothesis="h", arms=[_arm("t", "treatment", 0.3), _arm("n", "null", 0.5)],
        task_families=("a",), claim_boundary="b",
    )
    assert not record.null_failed
    assert record.separation < 0


def test_refusals_are_counted_so_a_pattern_of_them_is_visible():
    registry = reset_experiment_registry_for_test()
    for _ in range(3):
        with pytest.raises(MalformedExperiment):
            registry.register("e", hypothesis="h", arms=[_arm("t", "treatment", 0.8)],
                              task_families=("a",), claim_boundary="b")
    assert registry.report()["refused"] == 3


# ── parameters ────────────────────────────────────────────────────────────

def test_a_fitted_parameter_with_no_dataset_is_refused():
    with pytest.raises(UnprovenanceError, match="was not fitted"):
        Parameter("x", 1.0, Kind.FITTED, owner="o")


def test_a_policy_parameter_with_no_rationale_is_refused():
    with pytest.raises(UnprovenanceError, match="nobody can argue with"):
        Parameter("x", 1.0, Kind.POLICY, owner="o")


def test_best_of_three_cannot_support_a_calibration_claim():
    """The committed instance: three encoder widths, one campaign, twelve won."""
    registry = reset_parameter_registry_for_test()
    registry.fitted("grassmann.anchors", 12.0, owner="o",
                    dataset="one campaign over widths 8/12/16", n=1)
    result = registry.check_calibration_claim(["grassmann.anchors"])
    assert not result["ok"]
    assert "best-of-n is noise" in result["problems"][0]


def test_a_fitted_parameter_needs_an_interval_and_a_sensitivity_to_be_identifiable():
    registry = reset_parameter_registry_for_test()
    wide = registry.fitted("a", 1.0, owner="o", dataset="d", n=100, interval=(0.0, 5.0), sensitivity=0.5)
    tight = registry.fitted("b", 1.0, owner="o", dataset="d", n=100, interval=(0.9, 1.1), sensitivity=0.5)
    insensitive = registry.fitted("c", 1.0, owner="o", dataset="d", n=100, interval=(0.9, 1.1), sensitivity=0.0)
    assert not wide.identifiable
    assert tight.identifiable
    assert not insensitive.identifiable, "an interval alone is imprecision, not identifiability"


def test_a_policy_constant_never_pretends_to_support_calibration():
    registry = reset_parameter_registry_for_test()
    registry.policy("threshold", 0.7, owner="o", rationale="chosen to refuse rather than guess")
    assert not registry.check_calibration_claim(["threshold"])["ok"]


def test_the_constants_this_work_introduced_are_declared_as_policy():
    registry = reset_parameter_registry_for_test(known=True)
    report = registry.report()
    assert report["parameters"] >= 6
    assert report["by_kind"].get("fitted", 0) == 0, (
        "nothing here was fitted to data; calling any of it fitted would be the defect "
        "the registry exists to catch"
    )


# ── calibration ───────────────────────────────────────────────────────────

def test_two_sources_saying_zero_point_nine_do_not_mean_the_same_thing():
    layer = reset_calibration_layer_for_test()
    for i in range(200):
        layer.observe("cortex", 0.9, i % 10 < 6)
        layer.observe("rules", 0.9, i % 20 < 19)
    assert layer.read("cortex", 0.9).calibrated == pytest.approx(0.6, abs=0.02)
    assert layer.read("rules", 0.9).calibrated == pytest.approx(0.95, abs=0.02)


def test_a_source_with_too_little_history_passes_through_and_says_so():
    layer = reset_calibration_layer_for_test()
    for _ in range(MIN_OBSERVATIONS - 1):
        layer.observe("thin", 0.9, False)
    reading = layer.read("thin", 0.9)
    assert reading.status == "uncalibrated"
    assert reading.calibrated == 0.9
    assert not reading.usable_for_comparison


def test_an_uncalibrated_source_is_excluded_from_the_aggregate_not_averaged_in():
    layer = reset_calibration_layer_for_test()
    for i in range(100):
        layer.observe("known", 0.9, i % 2 == 0)
    combined = layer.combine([layer.read("known", 0.9), layer.read("unknown", 0.9)])
    assert combined["usable_sources"] == ["known"]
    assert combined["excluded_sources"] == ["unknown"]
    assert combined["combined"] == pytest.approx(0.5, abs=0.05)


def test_no_calibrated_source_gives_no_number_rather_than_a_guess():
    layer = reset_calibration_layer_for_test()
    assert layer.combine([layer.read("a", 0.9)])["combined"] is None


def test_overconfidence_is_reported_as_a_bias():
    layer = reset_calibration_layer_for_test()
    for i in range(100):
        layer.observe("optimist", 0.95, i % 4 == 0)
    assert layer.report()["by_source"]["optimist"]["bias"] > 0.5


# ── the learning audit ────────────────────────────────────────────────────

def test_a_complete_chain_is_accepted():
    audit = reset_learning_audit_for_test()
    audit.open("compiled a shortcut", observations=["obs-1", "obs-2"])
    audit.record_update("compiled a shortcut", "event:412")
    audit.record_artifact("compiled a shortcut", "procedure:p9")
    audit.record_retrieval("compiled a shortcut", "turn:88")
    audit.record_delta("compiled a shortcut", 0.31, comparator="same task before the artifact existed")
    assert audit.verify("compiled a shortcut")["accepted"]


@pytest.mark.parametrize(
    "skip,expected",
    [
        ("observations", Link.OBSERVATION),
        ("update", Link.UPDATE),
        ("artifact", Link.ARTIFACT),
        ("retrieval", Link.RETRIEVAL),
        ("delta", Link.DELTA),
    ],
)
def test_each_missing_link_is_named_rather_than_waved_through(skip, expected):
    audit = reset_learning_audit_for_test()
    audit.open("x", observations=[] if skip == "observations" else ["o"])
    if skip != "update":
        audit.record_update("x", "e1")
    if skip != "artifact":
        audit.record_artifact("x", "a1")
    if skip != "retrieval":
        audit.record_retrieval("x", "t1")
    if skip != "delta":
        audit.record_delta("x", 0.2, comparator="before")
    result = audit.verify("x")
    assert not result["accepted"]
    assert expected.value in result["missing"]


def test_a_delta_with_no_comparator_is_an_incomplete_link_not_a_small_effect():
    audit = reset_learning_audit_for_test()
    audit.open("x", observations=["o"])
    audit.record_update("x", "e")
    audit.record_artifact("x", "a")
    audit.record_retrieval("x", "t")
    audit.record_delta("x", 0.9, comparator="")
    assert Link.DELTA.value in audit.verify("x")["missing"]


def test_the_audit_says_which_link_fails_most_often():
    audit = reset_learning_audit_for_test()
    for i in range(3):
        audit.open(f"c{i}", observations=["o"])
        audit.record_update(f"c{i}", "e")
        audit.record_artifact(f"c{i}", "a")
    assert audit.report()["missing_links"][Link.RETRIEVAL.value] == 3


# ── the baseline portfolio ────────────────────────────────────────────────

def test_a_result_that_never_ran_the_cortex_arm_may_claim_nothing():
    from core.science.baseline_portfolio import (
        BaselineKind, BaselineResult, Contamination, compare,
    )

    verdict = compare(0.9, [BaselineResult(BaselineKind.SEARCH_ONLY, 0.4, 50)],
                      contamination=Contamination.PROCEDURAL)
    assert "cortex_only" in verdict.entitled_to_claim


def test_an_architecture_that_degrades_its_own_model_says_so_first():
    from core.science.baseline_portfolio import BaselineKind, BaselineResult, compare

    verdict = compare(0.60, [BaselineResult(BaselineKind.CORTEX_ONLY, 0.70, 50)])
    assert verdict.cortex_parity is False
    assert verdict.entitled_to_claim == "the architecture degrades the model it is built on"


def test_beating_everything_on_a_public_benchmark_is_not_a_novelty_claim():
    from core.science.baseline_portfolio import (
        BaselineKind, BaselineResult, Contamination, compare,
    )

    verdict = compare(
        0.9,
        [
            BaselineResult(BaselineKind.CORTEX_ONLY, 0.6, 50),
            BaselineResult(BaselineKind.CORTEX_WITH_TOOLS, 0.7, 50),
            BaselineResult(BaselineKind.SIMPLE_SCAFFOLD, 0.75, 50),
        ],
        contamination=Contamination.PUBLIC,
    )
    assert "no novelty claim" in verdict.entitled_to_claim


def test_the_strongest_sentence_needs_uncontaminated_tasks_and_every_arm():
    from core.science.baseline_portfolio import (
        BaselineKind, BaselineResult, Contamination, compare,
    )

    verdict = compare(
        0.9,
        [
            BaselineResult(BaselineKind.CORTEX_ONLY, 0.6, 50),
            BaselineResult(BaselineKind.CORTEX_WITH_TOOLS, 0.7, 50),
            BaselineResult(BaselineKind.SIMPLE_SCAFFOLD, 0.75, 50),
        ],
        contamination=Contamination.PROCEDURAL,
    )
    assert verdict.entitled_to_claim == "beats every baseline on tasks that cannot have been memorised"


def test_cortex_parity_finds_the_regression_an_internal_ab_cannot_see():
    from core.science.baseline_portfolio import check_parity

    report = check_parity(
        {"coding": 0.60, "recall": 0.90},
        {"coding": 0.70, "recall": 0.85, "multilingual": 0.40},
    )
    assert not report["parity_held"]
    assert report["regressions"][0]["capability"] == "coding"
    assert report["unmeasured"] == ["multilingual"]


def test_an_unclassified_task_is_treated_as_public():
    from core.science.baseline_portfolio import Contamination

    assert not Contamination.UNKNOWN.supports_a_novelty_claim


def test_every_baseline_states_the_question_only_it_answers():
    from core.science.baseline_portfolio import BaselineKind

    assert len({k.question for k in BaselineKind}) == len(list(BaselineKind))


# ── the latency law ───────────────────────────────────────────────────────

def _recalls(n, seed=7, noise=0.002):
    import random

    rng = random.Random(seed)
    return [
        __import__("core.science.retrieval_latency", fromlist=["x"]).RetrievalObservation(
            seconds=0.01 + 0.0004 * c + 0.002 * h + rng.gauss(0, noise),
            candidates=c, store_hops=h,
        )
        for c, h in ((rng.randint(5, 500), rng.randint(1, 4)) for _ in range(n))
    ]


def test_a_law_fitted_to_the_work_actually_done_explains_the_timing():
    from core.science.retrieval_latency import fit

    law = fit(_recalls(300))
    assert law.explains_anything
    assert law.per_candidate == pytest.approx(0.0004, abs=5e-5)


def test_a_law_fitted_to_noise_reports_that_it_explains_nothing():
    import random

    from core.science.retrieval_latency import RetrievalObservation, fit

    rng = random.Random(3)
    noise = [
        RetrievalObservation(seconds=rng.gauss(1, 0.5), candidates=rng.randint(5, 500))
        for _ in range(200)
    ]
    assert not fit(noise).explains_anything


def test_too_few_recalls_gives_no_law_rather_than_a_bad_one():
    from core.science.retrieval_latency import MIN_OBSERVATIONS, fit

    assert fit(_recalls(MIN_OBSERVATIONS - 1)) is None


def test_a_term_that_never_varies_is_reported_as_unknown_not_fitted_to_noise():
    from core.science.retrieval_latency import fit

    law = fit(_recalls(200))
    assert law.per_embedding_call == 0.0


def test_the_causal_claim_is_the_predicted_direction_under_intervention():
    from core.science.retrieval_latency import fit

    observations = _recalls(300)
    law = fit(observations)
    intervention = law.intervene(observations[0], candidates=observations[0].candidates * 4)
    assert intervention["direction"] == "up"
    assert intervention["predicted_delta"] > 0


def test_a_wrong_answer_with_a_near_neighbour_is_interference_not_fabrication():
    from core.science.retrieval_latency import ErrorKind, classify_error

    assert classify_error(returned="cat", expected="dog", nearest_similarity=0.8) is ErrorKind.COMMISSION
    assert classify_error(returned="xyzzy", expected="dog", nearest_similarity=0.05) is ErrorKind.FABRICATION
    assert classify_error(returned=None, expected="dog") is ErrorKind.OMISSION
    assert classify_error(returned="dog", expected="dog") is ErrorKind.CORRECT
