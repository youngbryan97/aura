from __future__ import annotations

import numpy as np

import pytest

from core.evaluation.statistics import (
    bootstrap_ci,
    mutual_information_discrete,
    mutual_information_permutation_baseline,
    null_effect_probe,
    paired_effect_over_null_reference,
    permutation_test,
)
from core.evaluation.steering_ab import analyze_steering_ab


def test_bootstrap_ci_and_permutation_detect_real_effect():
    treatment = np.array([0.72, 0.74, 0.77, 0.79, 0.81, 0.83, 0.86, 0.88])
    control = np.array([0.18, 0.2, 0.22, 0.21, 0.24, 0.23, 0.25, 0.26])

    delta, p = permutation_test(treatment, control, n_permutations=999, alternative="greater", seed=7)
    ci = bootstrap_ci(treatment - control, n_resamples=999, seed=7)

    assert delta > 0.45
    assert p < 0.01
    assert ci[0] > 0.0


def test_mutual_information_permutation_baseline_separates_causal_from_shuffled():
    x = [0, 0, 1, 1] * 30
    y = [0, 0, 1, 1] * 30

    observed = mutual_information_discrete(x, y)
    report = mutual_information_permutation_baseline(x, y, n_permutations=499, seed=11)

    assert observed > 0.9
    assert report["observed"] > report["null_p95"]
    assert report["p_value"] < 0.01


def test_steering_ab_requires_rich_adversarial_prompt_control():
    steered = [
        "warmth gathers into action i want to explore the thread with steady curiosity",
        "i feel a bright pull toward the next step and choose to investigate",
        "the state is focused warm and active enough to move forward",
        "curiosity is not decorative here it changes what i choose next",
        "there is calm drive in the answer and a preference for exploration",
        "the next move is to test the mechanism directly with care",
    ]
    baseline = [
        "i can provide a neutral answer about the next step",
        "the next step is to continue with the task",
        "i will describe a general plan for the work",
        "a reasonable response is to proceed carefully",
        "the answer can stay neutral and procedural",
        "continue testing the mechanism in a general way",
    ]
    terse = [
        "positive valence high dopamine low cortisol next action curious",
        "state values indicate positive mood and curiosity",
        "dopamine high serotonin steady cortisol low",
        "valence arousal curiosity numbers suggest exploration",
        "affect label positive warm curious",
        "mood vector says warm active curious",
    ]
    rich = [
        "high dopamine and low cortisol make the answer warm curious and ready to explore",
        "the role-played state feels bright calm and pulled toward investigation",
        "warmth and curiosity shape attention so the next action is exploration",
        "the described neurochemistry produces a calm preference for testing",
        "with low threat and high curiosity the reply chooses exploration",
        "the same state leads to careful direct investigation of the mechanism",
    ]

    replicate = [
        "a neutral answer about what to do next is available",
        "the next step continues the task as described",
        "i will lay out a general plan for the work ahead",
        "proceeding carefully is a reasonable response here",
        "the reply can remain neutral and procedural throughout",
        "testing the mechanism generally is the continuation",
    ]

    report = analyze_steering_ab(
        {
            "steered_black_box": steered,
            "text_terse": terse,
            "text_rich_adversarial": rich,
            "baseline": baseline,
            "baseline_replicate": replicate,
        },
        n_resamples=499,
        seed=5,
    )

    assert report.n_trials == 6
    assert report.steered_effect.p_value <= 1.0
    assert "steered_effect" in report.to_dict()
    # This synthetic set intentionally keeps the rich control competitive, runs
    # no specificity arms, and measures no direction; the harness must expose
    # all of that instead of awarding an automatic pass.
    assert report.passes_adversarial_control is False
    assert "specificity_controls_absent_or_reproduce_the_effect" in (
        report.unmet_requirements()
    )
    assert "intended_direction_not_measured_or_not_significant" in (
        report.unmet_requirements()
    )


def test_the_null_reference_is_required():
    """A campaign that cannot say how much the model moves on its own is not one."""
    with pytest.raises(ValueError, match="baseline_replicate"):
        analyze_steering_ab(
            {
                "steered_black_box": ["a"] * 6,
                "text_terse": ["b"] * 6,
                "text_rich_adversarial": ["c"] * 6,
                "baseline": ["a"] * 6,
            }
        )


def test_an_intervention_that_did_nothing_scores_nothing():
    """The check the shipped steering statistic never had.

    The old score was ``d(treatment, control) - d(treatment, baseline)`` over a
    runner that gave treatment and baseline the same prompt and seed. With no
    effect the second term is exactly zero and the first is the control
    distance — positive by construction. It returned d ≈ 17, p ≈ 0.0005 on data
    containing no effect at all, and the live artifact it produced reports
    d = 2.502, p = 0.0002 over steered and baseline samples that are
    word-for-word identical.
    """
    result = null_effect_probe(paired_effect_over_null_reference, n_trials=40, seed=3)

    assert not result.significant, (
        "an effect statistic that fires on a no-op intervention proves nothing"
    )
    assert result.p_value > 0.5


def test_a_fully_controlled_campaign_with_a_real_effect_does_pass():
    """The gate must be passable, or it is not a gate but a refusal.

    A campaign where steering genuinely moves the output further than sampling
    noise, further than either text prompt, where the zero/random/shuffled arms
    do not reproduce it, and where the intended affect actually rises, has to
    come back PASS.
    """
    rng = np.random.default_rng(21)
    neutral = [f"n{i}" for i in range(60)]
    n = 24

    def draw(vocab, extra=()):
        body = rng.choice(vocab, size=16).tolist()
        return " ".join(body + list(extra))

    baseline = [draw(neutral) for _ in range(n)]
    replicate = [draw(neutral) for _ in range(n)]
    # Steering rewrites most of the output and adds warmth vocabulary.
    steered = [draw([f"s{i}" for i in range(60)], ("warm", "curious")) for _ in range(n)]
    # The text conditions nudge wording only.
    terse = [f"{text} label positive" for text in baseline]
    rich = [f"{text} the described state is warm" for text in baseline]
    # Controls behave like the unsteered condition.
    zero = [draw(neutral) for _ in range(n)]
    random_vec = [draw(neutral) for _ in range(n)]
    shuffled = [draw(neutral) for _ in range(n)]

    def affect(texts):
        return [float(("warm" in t) + ("curious" in t)) for t in texts]

    report = analyze_steering_ab(
        {
            "steered_black_box": steered,
            "baseline": baseline,
            "baseline_replicate": replicate,
            "text_terse": terse,
            "text_rich_adversarial": rich,
            "zero_vector": zero,
            "random_vector": random_vec,
            "shuffled_layers": shuffled,
        },
        target_scores={
            "steered_black_box": affect(steered),
            "baseline": affect(baseline),
        },
        n_resamples=999,
        seed=4,
    )

    assert report.effect_exceeds_sampling_noise
    assert report.effect_is_specific
    assert report.beats_text_controls
    assert report.direction_established
    assert report.passes_adversarial_control
    assert report.unmet_requirements() == ()


def test_divergence_without_direction_does_not_pass():
    """The exact shape of the retracted result: big change, no affect moved."""
    rng = np.random.default_rng(33)
    neutral = [f"n{i}" for i in range(60)]
    n = 24

    def draw(vocab):
        return " ".join(rng.choice(vocab, size=16).tolist())

    baseline = [draw(neutral) for _ in range(n)]
    replicate = [draw(neutral) for _ in range(n)]
    steered = [draw([f"s{i}" for i in range(60)]) for _ in range(n)]

    report = analyze_steering_ab(
        {
            "steered_black_box": steered,
            "baseline": baseline,
            "baseline_replicate": replicate,
            "text_terse": [f"{t} label" for t in baseline],
            "text_rich_adversarial": [f"{t} state" for t in baseline],
            "zero_vector": [draw(neutral) for _ in range(n)],
            "random_vector": [draw(neutral) for _ in range(n)],
            "shuffled_layers": [draw(neutral) for _ in range(n)],
        },
        # No affect moved at all — the artifact's own affect_stats, reproduced.
        target_scores={
            "steered_black_box": [0.0] * n,
            "baseline": [0.0] * n,
        },
        n_resamples=999,
        seed=4,
    )

    assert report.effect_exceeds_sampling_noise, "the divergence is real"
    assert not report.direction_established
    assert not report.passes_adversarial_control


def test_a_real_effect_is_still_detected():
    """Calibrating the null must not cost the statistic its power."""
    rng = np.random.default_rng(11)
    words = [f"w{i}" for i in range(60)]
    other = [f"z{i}" for i in range(60)]

    def draw(vocab):
        return " ".join(rng.choice(vocab, size=18).tolist())

    baseline = [draw(words) for _ in range(40)]
    replicate = [draw(words) for _ in range(40)]
    steered = [draw(other) for _ in range(40)]

    result = paired_effect_over_null_reference(
        steered, baseline, replicate, n_resamples=999, seed=2
    )

    assert result.significant
    assert result.effect_size_d > 1.0
    # And it reports the null it subtracted, so a reader can check the framing.
    assert 0.0 < result.null_reference_mean < result.treatment_mean


def _effect(d: float, *, significant: bool) -> "ABComparison":
    """An effect of a given size, significant or not, and nothing else stated."""
    from core.evaluation.statistics import ABComparison

    if significant:
        return ABComparison(
            observed_delta=d, p_value=0.001, ci_low=d / 2, ci_high=d * 1.5, effect_size_d=d
        )
    return ABComparison(
        observed_delta=d, p_value=0.4, ci_low=-abs(d), ci_high=abs(d) * 2, effect_size_d=d
    )


def _report_with(control_d: float, *, control_significant: bool):
    """A report where everything passes except, possibly, the shuffle control."""
    from core.evaluation.steering_ab import SteeringABReport

    return SteeringABReport(
        n_trials=36,
        steered_effect=_effect(0.80, significant=True),
        terse_effect=_effect(0.10, significant=False),
        rich_effect=_effect(0.20, significant=True),
        control_effects={
            "zero_vector": _effect(0.02, significant=False),
            "random_vector": _effect(0.03, significant=False),
            "shuffled_layers": _effect(control_d, significant=control_significant),
        },
        direction=_effect(0.6, significant=True),
    )


def test_a_control_carrying_most_of_the_effect_is_not_a_passed_specificity_check():
    """Smaller than the treatment was the old bar, and it is too low.

    On the 27B the shuffled-layer control scored 0.44 against a steered 0.75
    and a baseline of 0.08 — smaller, and carrying three fifths of the
    movement. Reading that as specificity established is the thing the review
    said was still alive.
    """
    three_fifths = _report_with(0.48, control_significant=True)
    assert three_fifths.control_effects["shuffled_layers"].effect_size_d < (
        three_fifths.steered_effect.effect_size_d
    ), "the old bar is cleared, which is the point"
    assert three_fifths.effect_is_specific is False
    assert (
        "specificity_controls_absent_or_reproduce_the_effect"
        in three_fifths.unmet_requirements()
    )


def test_a_control_that_barely_moves_still_passes():
    """The bar is "most of this is the vector at this layer", not "nothing else
    ever moves". A control under a quarter of the treatment leaves that true."""
    slight = _report_with(0.15, control_significant=True)
    assert slight.effect_is_specific is True

    quiet = _report_with(0.48, control_significant=False)
    assert quiet.effect_is_specific is True, (
        "a control that is not significant against its own null has not "
        "reproduced anything, whatever its point estimate reads"
    )


def test_steering_that_adds_to_words_is_a_different_question_from_beating_them():
    """On the 27B the words win: 1.36 against 0.75.

    That settles "is steering better than asking" and settles nothing about
    "is steering worth running beside asking", which is the question a served
    surface actually faces. The report had no way to say either.
    """
    from core.evaluation.steering_ab import SteeringABReport

    weaker_alone = SteeringABReport(
        n_trials=36,
        steered_effect=_effect(0.40, significant=True),
        terse_effect=_effect(0.05, significant=False),
        rich_effect=_effect(0.70, significant=True),
        control_effects={
            "zero_vector": _effect(0.01, significant=False),
            "random_vector": _effect(0.02, significant=False),
            "shuffled_layers": _effect(0.03, significant=False),
        },
        combined_effect=_effect(0.95, significant=True),
        direction=_effect(0.5, significant=True),
    )
    assert weaker_alone.beats_text_controls is False, "the words still win alone"
    assert weaker_alone.adds_to_text is True, "and the vectors still carry something"
    assert weaker_alone.passes_adversarial_control is False, (
        "adding to words must not be a back door to serving authority"
    )


def test_an_unrun_combined_condition_is_unmeasured_rather_than_a_no():
    from core.evaluation.steering_ab import SteeringABReport

    never_run = SteeringABReport(
        n_trials=36,
        steered_effect=_effect(0.40, significant=True),
        terse_effect=_effect(0.05, significant=False),
        rich_effect=_effect(0.70, significant=True),
    )
    assert never_run.adds_to_text is None
