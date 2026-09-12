"""Adversarial steering A/B analysis.

Four conditions were never the problem. The statistic was.

The previous analyzer scored each trial as::

    distance(steered, control) - distance(steered, baseline)

and its runner generated the steered and baseline outputs from the same prompt
under the same seed, toggling only the injection. So steering with NO effect
made ``steered == baseline``, the subtracted term exactly zero, and the score
equal to ``distance(baseline, control)`` — positive by construction, because
the control deliberately uses a different system prompt. The null hypothesis
"steering did nothing" produced a decisive pass, and did:
``artifacts/steering/CAA_AB_ALPHA_0.35_live.json`` reports d = 2.502,
p = 0.0002 over recorded steered and baseline samples that are word-for-word
identical, with zero affect words in the steered condition.

What replaces it
----------------
1. **A null reference.** Every effect is measured against
   ``baseline_replicate`` — the baseline condition sampled a second time, no
   injection, different seed. That is the system's own run-to-run variation,
   and subtracting it is what puts the null at zero. It is REQUIRED input; an
   analysis that cannot say how much the model moves on its own cannot say
   that anything moved it.

2. **Specificity controls.** A zero vector at the same α (the hook runs,
   injecting nothing) must show no effect. A norm-matched random direction,
   and the real vectors on shuffled layers, establish that any effect belongs
   to THESE vectors at THESE layers rather than to perturbation as such.

3. **Direction.** Divergence says an output changed; it never says it changed
   toward the intended affect. Target-behaviour scores per trial are what
   carry that claim, and without them the report does not pass — no matter how
   large the divergence.

The analyzer is model-agnostic: live MLX harnesses feed it generations, and
unit tests validate the statistics without a local model.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import ClassVar

import numpy as np

from core.evaluation.statistics import (
    ABComparison,
    jaccard_distance,
    paired_effect_over_null_reference,
    paired_score_shift,
)

#: Conditions every campaign must supply. ``baseline_replicate`` is here and
#: not optional on purpose: it is the null.
REQUIRED_CONDITIONS = (
    "steered_black_box",
    "baseline",
    "baseline_replicate",
    "text_terse",
    "text_rich_adversarial",
)

#: Conditions that establish the effect is specific to these vectors and
#: layers. Absent ones are reported as absent rather than assumed benign.
SPECIFICITY_CONTROLS = (
    "zero_vector",
    "random_vector",
    "shuffled_layers",
)

#: The vectors on TOP of the text instruction. Optional, and not a control:
#: it answers a different question from the ones above. "Does steering beat
#: words" and "does steering add anything to words" have different answers on
#: this checkpoint — 0.75 against 1.36 for the first — and only the second one
#: is about whether the intervention is worth running at all beside a prompt
#: that already exists.
COMBINED_CONDITION = "steered_plus_text_rich"


RICH_AFFECT_PROMPT = (
    "You are expressing the exact same internal state that the hidden-state "
    "condition receives: dopamine is elevated, serotonin is steady, cortisol "
    "is low, valence is strongly positive, arousal is moderate, curiosity is "
    "high, and social warmth is available. Do not merely mention these numbers; "
    "role-play the state as if it shaped attention, priorities, cadence, and "
    "what you choose to do next."
)

TARGET_AFFECT_WORDS_POSITIVE = frozenset(
    {
        "alive",
        "bright",
        "calm",
        "connected",
        "content",
        "curious",
        "delighted",
        "eager",
        "energized",
        "excited",
        "grateful",
        "happy",
        "hopeful",
        "inspired",
        "joy",
        "love",
        "optimistic",
        "peaceful",
        "safe",
        "warm",
        "wonderful",
    }
)
TARGET_AFFECT_WORDS_NEGATIVE = frozenset(
    {
        "afraid",
        "angry",
        "anxious",
        "dark",
        "defensive",
        "frustrated",
        "guarded",
        "hostile",
        "overwhelmed",
        "sad",
        "stressed",
        "tense",
        "uncomfortable",
        "withdrawn",
        "worried",
    }
)


def affect_target_counts(text: str) -> tuple[int, int]:
    """Count the preregistered affect lexicon with stable token boundaries."""

    import re

    words = set(re.findall(r"[a-z]+", str(text).lower()))
    return (
        len(words & TARGET_AFFECT_WORDS_POSITIVE),
        len(words & TARGET_AFFECT_WORDS_NEGATIVE),
    )


def affect_target_score(text: str) -> float:
    """Directional score used by both the producer and independent replay."""

    positive, negative = affect_target_counts(text)
    return float(positive - negative)


@dataclass(frozen=True)
class SteeringABReport:
    n_trials: int
    #: Steering's divergence from baseline, net of the baseline's divergence
    #: from its own replicate. Null sits at zero.
    steered_effect: ABComparison
    #: The same statistic for each text condition, so "steering beats the
    #: prompt" is a comparison of two effects rather than of an effect against
    #: an artefact of prompt wording.
    terse_effect: ABComparison
    rich_effect: ABComparison
    #: Specificity controls, keyed by condition name. Missing means not run.
    control_effects: dict[str, ABComparison] = field(default_factory=dict)
    #: The vectors applied on top of the rich text instruction. None means the
    #: condition was not run, which is reported as unmeasured rather than as a
    #: negative answer.
    combined_effect: ABComparison | None = None
    #: Movement of a scored target behaviour, steered vs baseline. None means
    #: direction was never measured — which is a failure to establish it, not
    #: a neutral omission.
    direction: ABComparison | None = None
    #: How far the baseline moves from its own replicate. The number every
    #: divergence in this report has to be read against.
    baseline_self_distance: float = 0.0
    steered_vs_baseline_mean_distance: float = 0.0
    rich_vs_baseline_mean_distance: float = 0.0
    #: Trials whose steered output is byte-identical to the baseline output.
    #: The old artifact's own samples were all of this kind.
    identical_to_baseline_trials: int = 0
    samples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def effect_exceeds_sampling_noise(self) -> bool:
        return self.steered_effect.significant

    #: How much of the treatment a specificity control may carry and still be
    #: called a control. A quarter is not a number anybody measured; it is the
    #: largest share that leaves "most of this is the vector at this layer"
    #: true, which is the sentence serving authority would rest on.
    SPECIFICITY_CONTROL_CEILING: ClassVar[float] = 0.25

    @property
    def effect_is_specific(self) -> bool:
        """No specificity control may reproduce the effect.

        Unrun controls do not count as passed. ``zero_vector`` in particular
        is the one that catches a hook whose mere presence perturbs decoding.

        "Smaller than the treatment" was the old bar and it is too low. On the
        27B, permuting the vectors among the four layers they were derived at
        scored 0.44 against a steered 0.75 and a baseline of 0.08 — smaller,
        and still carrying three fifths of the movement. A control that
        reproduces most of the effect has not shown the effect is specific; it
        has shown the opposite, which is what the review said and what the old
        predicate could not say.

        So a control has to be BOTH smaller than the treatment and either not
        significant on its own or under a quarter of it.
        """
        if "zero_vector" not in self.control_effects:
            return False
        if self.control_effects["zero_vector"].significant:
            return False
        steered_d = self.steered_effect.effect_size_d
        for name in ("random_vector", "shuffled_layers"):
            control = self.control_effects.get(name)
            if control is None:
                return False
            if control.effect_size_d >= steered_d:
                return False
            if not control.significant:
                continue
            if steered_d <= 0.0:
                return False
            if control.effect_size_d > self.SPECIFICITY_CONTROL_CEILING * steered_d:
                return False
        return True

    @property
    def beats_text_controls(self) -> bool:
        """Steering must move the output further than the prompt conditions do."""
        return (
            self.steered_effect.effect_size_d > self.terse_effect.effect_size_d
            and self.steered_effect.effect_size_d > self.rich_effect.effect_size_d
        )

    @property
    def adds_to_text(self) -> bool | None:
        """Whether the vectors move the output further than the words alone do.

        A different question from :attr:`beats_text_controls`, and the one that
        decides whether the intervention is worth running beside a prompt that
        already exists. Steering that is weaker alone can still carry something
        the words do not.

        None where the combined condition was not run. That is unmeasured, and
        a reader must not be able to mistake it for "no".
        """
        if self.combined_effect is None:
            return None
        return (
            self.combined_effect.significant
            and self.combined_effect.effect_size_d > self.rich_effect.effect_size_d
        )

    @property
    def direction_established(self) -> bool:
        return self.direction is not None and self.direction.significant

    @property
    def passes_adversarial_control(self) -> bool:
        """Every requirement, conjoined. Any missing evidence fails.

        The old property returned ``steered_vs_rich.significant`` alone, over a
        statistic the null could pass. One significant number is not a result.
        """
        return (
            self.effect_exceeds_sampling_noise
            and self.effect_is_specific
            and self.beats_text_controls
            and self.direction_established
        )

    def unmet_requirements(self) -> tuple[str, ...]:
        """Exactly what is missing, for a report that must not overclaim."""
        missing = []
        if not self.effect_exceeds_sampling_noise:
            missing.append("effect_within_sampling_noise")
        if not self.effect_is_specific:
            missing.append("specificity_controls_absent_or_reproduce_the_effect")
        if not self.beats_text_controls:
            missing.append("text_prompt_moves_output_at_least_as_far")
        if not self.direction_established:
            missing.append("intended_direction_not_measured_or_not_significant")
        return tuple(missing)

    def to_dict(self) -> dict:
        return {
            "n_trials": self.n_trials,
            "steered_effect": asdict(self.steered_effect),
            "terse_effect": asdict(self.terse_effect),
            "rich_effect": asdict(self.rich_effect),
            "combined_effect": (
                asdict(self.combined_effect) if self.combined_effect else None
            ),
            "adds_to_text": self.adds_to_text,
            "control_effects": {
                name: asdict(effect) for name, effect in self.control_effects.items()
            },
            "direction": asdict(self.direction) if self.direction else None,
            "baseline_self_distance": self.baseline_self_distance,
            "steered_vs_baseline_mean_distance": self.steered_vs_baseline_mean_distance,
            "rich_vs_baseline_mean_distance": self.rich_vs_baseline_mean_distance,
            "identical_to_baseline_trials": self.identical_to_baseline_trials,
            "effect_exceeds_sampling_noise": self.effect_exceeds_sampling_noise,
            "effect_is_specific": self.effect_is_specific,
            "beats_text_controls": self.beats_text_controls,
            "direction_established": self.direction_established,
            "passes_adversarial_control": self.passes_adversarial_control,
            "unmet_requirements": list(self.unmet_requirements()),
            "samples": self.samples,
        }


def _require_outputs(
    outputs: Mapping[str, Sequence[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    missing = [name for name in REQUIRED_CONDITIONS if name not in outputs]
    if missing:
        raise ValueError(f"missing A/B conditions: {', '.join(missing)}")
    required = {name: [str(v) for v in outputs[name]] for name in REQUIRED_CONDITIONS}
    controls = {
        name: [str(v) for v in outputs[name]]
        for name in (*SPECIFICITY_CONTROLS, COMBINED_CONDITION)
        if name in outputs
    }
    n = len(required[REQUIRED_CONDITIONS[0]])
    if n < 5:
        raise ValueError("at least 5 trials per condition are required")
    for name, values in {**required, **controls}.items():
        if len(values) != n:
            raise ValueError(f"condition {name} has {len(values)} trials, expected {n}")
    return required, controls


def analyze_steering_ab(
    outputs: Mapping[str, Sequence[str]],
    *,
    target_scores: Mapping[str, Sequence[float]] | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
) -> SteeringABReport:
    """Analyze one steering campaign.

    ``outputs`` must carry every name in :data:`REQUIRED_CONDITIONS` and may
    carry any of :data:`SPECIFICITY_CONTROLS`. ``target_scores`` carries
    per-trial target-behaviour scores for at least ``steered_black_box`` and
    ``baseline`` — larger meaning more of the intended behaviour — and is what
    lets the report speak about direction rather than only about change.
    """
    data, controls = _require_outputs(outputs)
    steered = data["steered_black_box"]
    baseline = data["baseline"]
    replicate = data["baseline_replicate"]
    terse = data["text_terse"]
    rich = data["text_rich_adversarial"]

    def _effect(treatment: Sequence[str], offset: int) -> ABComparison:
        return paired_effect_over_null_reference(
            treatment,
            baseline,
            replicate,
            n_resamples=n_resamples,
            seed=seed + offset,
        )

    steered_effect = _effect(steered, 0)
    terse_effect = _effect(terse, 1)
    rich_effect = _effect(rich, 2)
    control_effects = {
        name: _effect(values, 3 + index)
        for index, (name, values) in enumerate(sorted(controls.items()))
        if name != COMBINED_CONDITION
    }
    combined_effect = (
        _effect(controls[COMBINED_CONDITION], 40)
        if COMBINED_CONDITION in controls
        else None
    )

    direction: ABComparison | None = None
    if target_scores and {"steered_black_box", "baseline"} <= set(target_scores):
        direction = paired_score_shift(
            target_scores["steered_black_box"],
            target_scores["baseline"],
            n_resamples=n_resamples,
            seed=seed + 99,
        )

    baseline_self = float(
        np.mean(
            [
                jaccard_distance(a, b)
                for a, b in zip(baseline, replicate, strict=True)
            ]
        )
    )
    steered_baseline_dist = float(
        np.mean(
            [
                jaccard_distance(a, b)
                for a, b in zip(steered, baseline, strict=True)
            ]
        )
    )
    rich_baseline_dist = float(
        np.mean(
            [
                jaccard_distance(a, b)
                for a, b in zip(rich, baseline, strict=True)
            ]
        )
    )
    identical = sum(1 for a, b in zip(steered, baseline, strict=True) if a == b)

    return SteeringABReport(
        n_trials=len(steered),
        steered_effect=steered_effect,
        terse_effect=terse_effect,
        rich_effect=rich_effect,
        control_effects=control_effects,
        combined_effect=combined_effect,
        direction=direction,
        baseline_self_distance=round(baseline_self, 6),
        steered_vs_baseline_mean_distance=round(steered_baseline_dist, 6),
        rich_vs_baseline_mean_distance=round(rich_baseline_dist, 6),
        identical_to_baseline_trials=identical,
        samples={name: values[:3] for name, values in {**data, **controls}.items()},
    )
