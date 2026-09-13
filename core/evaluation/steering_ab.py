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
    #: negative answer. This is a DIVERGENCE comparison — how far the output
    #: moved — and it is not the one :attr:`adds_to_text` reads.
    combined_effect: ABComparison | None = None
    #: The same condition against the rich prompt alone, on the SCORED target
    #: behaviour. `paired_score_shift` says it in its own docstring:
    #: "divergence says an output changed; this says it changed toward the
    #: thing the intervention was supposed to produce". Adding something the
    #: words do not carry is a claim about direction, so this is the
    #: comparison that answers it.
    combined_direction: ABComparison | None = None
    #: Movement of a scored target behaviour, steered vs baseline. None means
    #: direction was never measured — which is a failure to establish it, not
    #: a neutral omission.
    direction: ABComparison | None = None
    #: The same movement for each specificity control. Specificity is a claim
    #: about the BEHAVIOUR a control reproduces, and divergence cannot carry
    #: it: measured on the 27B, a norm-matched random direction at sixteen
    #: layers scored -0.0833 on the target against the treatment's +1.3333 —
    #: inert — while reproducing 98.6% of the treatment's divergence and a
    #: LARGER standardised effect. Judged on divergence that control fails the
    #: predicate, and so would every activation intervention that has ever been
    #: run, because what divergence measures is how hard the stream was pushed.
    control_directions: dict[str, ABComparison] = field(default_factory=dict)
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
        if self.direction is None:
            return False
        treatment = self.direction.observed_delta
        if treatment <= 0.0:
            return False
        zero = self.control_directions.get("zero_vector")
        if zero is None or zero.significant:
            return False
        for name in ("random_vector", "shuffled_layers"):
            control = self.control_directions.get(name)
            if control is None:
                return False
            if not control.significant:
                continue
            if control.observed_delta > self.SPECIFICITY_CONTROL_CEILING * treatment:
                return False
        return True

    @property
    def beats_text_controls(self) -> bool:
        """Steering must move the output further than the prompt conditions do.

        How FAR is `observed_delta`. This compared `effect_size_d`, which is
        that distance divided by its spread across trials, and so asked how
        CONSISTENTLY instead. The two parted company on the 27B: steering moved
        the output 0.1769 against the rich prompt's 0.1275 and was called the
        smaller effect, because a prompt prefix does the same thing to every
        task and a vector does not.

        The same substitution was already found and fixed once, in
        :attr:`adds_to_text`, whose first version compared the combined
        condition's `d` and reported False while the affect score went 1.36 to
        3.61. This is that defect in the property next to it.
        """
        return (
            self.steered_effect.observed_delta > self.terse_effect.observed_delta
            and self.steered_effect.observed_delta > self.rich_effect.observed_delta
        )

    @property
    def adds_to_text(self) -> bool | None:
        """Whether the vectors move the TARGET further than the words alone do.

        A different question from :attr:`beats_text_controls`, and the one that
        decides whether the intervention is worth running beside a prompt that
        already exists. Steering that is weaker alone can still carry something
        the words do not.

        Read off the scored behaviour, not off divergence. The first version
        compared `combined_effect.effect_size_d` against `rich_effect`'s, and
        on the 27B that reported False while the affect score went 1.36 -> 3.61
        — because the combined condition moves the output FURTHER (delta 0.1389
        against 0.1235) and less consistently, so its d is smaller. Divergence
        answers "did the text change"; this property claims to answer "did it
        change toward the thing steering is for", and those parted company on
        the first run that had both conditions.

        None where the combined condition was not run. That is unmeasured, and
        a reader must not be able to mistake it for "no".
        """
        if self.combined_direction is None:
            return None
        return (
            self.combined_direction.significant
            and self.combined_direction.observed_delta > 0.0
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
            "combined_direction": (
                asdict(self.combined_direction) if self.combined_direction else None
            ),
            "adds_to_text": self.adds_to_text,
            "control_effects": {
                name: asdict(effect) for name, effect in self.control_effects.items()
            },
            "direction": asdict(self.direction) if self.direction else None,
            "control_directions": {
                name: asdict(shift) for name, shift in self.control_directions.items()
            },
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


def adversarial_control_fingerprint() -> str:
    """A digest of the four predicates a verdict is refused by.

    A committed verdict and its committed samples must not silently disagree,
    and until now a disagreement had exactly one reading: somebody edited a
    file. There is a second, and it happened -- a predicate was corrected, and
    every verdict derived under the old one stopped re-deriving.

    Recording this beside a verdict separates the two. The digest is taken from
    the source of the predicates themselves, so it cannot go stale the way a
    hand-kept version number does.
    """
    import hashlib
    import inspect

    source = "".join(
        inspect.getsource(getattr(SteeringABReport, name).fget)
        for name in (
            "effect_exceeds_sampling_noise",
            "effect_is_specific",
            "beats_text_controls",
            "direction_established",
        )
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


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

    control_directions: dict[str, ABComparison] = {}
    if target_scores and "baseline" in target_scores:
        for index, name in enumerate(sorted(controls)):
            if name == COMBINED_CONDITION or name not in target_scores:
                continue
            control_directions[name] = paired_score_shift(
                target_scores[name],
                target_scores["baseline"],
                n_resamples=n_resamples,
                seed=seed + 200 + index,
            )

    combined_direction: ABComparison | None = None
    if target_scores and {COMBINED_CONDITION, "text_rich_adversarial"} <= set(
        target_scores
    ):
        combined_direction = paired_score_shift(
            target_scores[COMBINED_CONDITION],
            target_scores["text_rich_adversarial"],
            n_resamples=n_resamples,
            seed=seed + 131,
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
        combined_direction=combined_direction,
        direction=direction,
        control_directions=control_directions,
        baseline_self_distance=round(baseline_self, 6),
        steered_vs_baseline_mean_distance=round(steered_baseline_dist, 6),
        rich_vs_baseline_mean_distance=round(rich_baseline_dist, 6),
        identical_to_baseline_trials=identical,
        samples={name: values[:3] for name, values in {**data, **controls}.items()},
    )
