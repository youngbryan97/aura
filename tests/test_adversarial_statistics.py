from __future__ import annotations

import numpy as np

from core.evaluation.statistics import (
    bootstrap_ci,
    mutual_information_discrete,
    mutual_information_permutation_baseline,
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

    report = analyze_steering_ab(
        {
            "steered_black_box": steered,
            "text_terse": terse,
            "text_rich_adversarial": rich,
            "baseline": baseline,
        },
        n_resamples=499,
        seed=5,
    )

    assert report.n_trials == 6
    assert report.steered_vs_terse.p_value <= 1.0
    assert "steered_vs_rich" in report.to_dict()
    # This synthetic set intentionally keeps the rich control competitive; the
    # harness must expose that instead of awarding an automatic pass.
    assert report.passes_adversarial_control is False
