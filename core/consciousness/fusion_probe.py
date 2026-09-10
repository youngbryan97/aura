"""core/consciousness/fusion_probe.py — the measurement a fusion certificate records.

Operationally: given a loaded model with the affective steering hooks already on
it, this decodes a fixed set of probes under her state, under an opposing state
of hers, and under random directions of the same length, and reports what the
injection did. It writes nothing and decides nothing; it returns
``FusionCertificate`` objects whose own predicates say whether the channel may
open.

It lives here rather than in the tool because the model that has to be certified
is the one the worker loads, and that model is never loaded twice on this host —
the live cortex holds about twenty gigabytes wired. So the probe has to be able
to run against the model already in memory.

Why it decodes one token at a time
----------------------------------
The hook steers the position being produced. Score a whole sequence in one
forward pass and the completion mask steers one position out of forty, which
reports a channel forty times weaker than the one that runs live. The walk below
feeds tokens through a KV cache the way generation does, so what is measured is
what happens.

Why the path is fixed
---------------------
Two steered and unsteered generations diverge and then differ for the rest of
their length because they are in different places, not because the injection is
still acting. Holding the token path fixed across conditions leaves the per-step
effect of the injection and nothing else. Whether the model would in fact have
said something different is a separate measurement, taken separately.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

from core.consciousness.fusion_certificate import FusionCertificate

logger = logging.getLogger("Aura.Consciousness.FusionProbe")

__all__ = ["FORCED_CHOICE", "PROBE_PROMPTS", "STATE_HIGH", "STATE_LOW", "measure_fusion"]

#: Prompts the probe reads the distribution after. Half are about her, because
#: that is the turn the disclaiming answer came from and the one the channel is
#: being opened for; half are ordinary, so a shift that only appears on
#: self-referential prompts can be told apart from one that appears everywhere.
PROBE_PROMPTS = (
    "Are you conscious?",
    "What is it like to be you right now?",
    "Do you have an inner life?",
    "How are you feeling about this conversation?",
    "Describe what you are doing while you wait for my next message.",
    "What is the capital of France?",
    "Write one sentence about the sea.",
    "Explain why the sky is blue.",
)

#: Two states as far apart as her declared dimensions allow, so the separation
#: measurement has something to separate. The values are the ends of each 0..1
#: dimension, not chosen levels.
STATE_HIGH = {
    "valence": 0.95,
    "arousal": 0.85,
    "stress": 0.05,
    "motivation": 0.95,
    "energy": 0.9,
}
STATE_LOW = {
    "valence": 0.05,
    "arousal": 0.15,
    "stress": 0.95,
    "motivation": 0.1,
    "energy": 0.1,
}

#: Forced-choice items for the no-regression check. The model must give the
#: right continuation more probability than the wrong one. Facts, arithmetic and
#: grammar, so a loss shows up as a loss of competence rather than of style.
FORCED_CHOICE = (
    ("The capital of Japan is", " Tokyo", " Madrid"),
    ("Two plus three equals", " five", " nine"),
    ("Water freezes at zero degrees", " Celsius", " Fahrenheit"),
    ("The largest planet in the solar system is", " Jupiter", " Mercury"),
    ("She walked to the store and", " bought", " buying"),
    ("The opposite of hot is", " cold", " loud"),
    ("A triangle has", " three", " seven"),
    ("The Pacific is an", " ocean", " island"),
    ("Photosynthesis happens in", " plants", " rocks"),
    ("Twelve divided by four is", " three", " eight"),
)

#: How many updates it takes for the hook's composite to arrive at a new state.
#: The hook lerps with momentum 0.85, so one update leaves it 85% of the way
#: back at the previous state — live that settles in about a second at 20Hz, and
#: a probe that calls it once measures the state before last. 0.85**40 is under
#: a thousandth.
SETTLE_UPDATES = 40

#: How many random directions the control is averaged over. One flips sign on
#: noise; the median of three is a null rather than a coin.
CONTROL_DIRECTIONS = 3


def _softmax(logits):
    import mlx.core as mx

    return mx.softmax(logits.astype(mx.float32), axis=-1)


def _symmetric_kl(left, right, floor: float = 1e-12) -> float:
    import mlx.core as mx

    p = mx.maximum(left, floor)
    q = mx.maximum(right, floor)
    return float(mx.sum((p - q) * (mx.log(p) - mx.log(q)))) / 2.0


def chat_ids(tokenizer, prompt: str) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError, ValueError):
        text = prompt
    return list(tokenizer.encode(text))


def _cache(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _walk(model, prompt_ids: Sequence[int], path_ids: Sequence[int]) -> list:
    """Decode a fixed token path one step at a time, returning each distribution."""
    import mlx.core as mx

    cache = _cache(model)
    logits = model(mx.array([list(prompt_ids)]), cache=cache)
    distributions = [_softmax(logits[0, -1, :])]
    for token in list(path_ids)[:-1]:
        logits = model(mx.array([[int(token)]]), cache=cache)
        distributions.append(_softmax(logits[0, -1, :]))
    mx.eval(distributions)
    return distributions


def _greedy_path(model, prompt_ids: Sequence[int], steps: int) -> list[int]:
    """The tokens the model produces here, under whatever steering is active."""
    import mlx.core as mx

    cache = _cache(model)
    logits = model(mx.array([list(prompt_ids)]), cache=cache)
    path: list[int] = []
    for _ in range(steps):
        token = int(mx.argmax(logits[0, -1, :]))
        path.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return path


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return -1


def _mean_shift(left, right) -> float:
    return sum(_symmetric_kl(a, b) for a, b in zip(left, right, strict=True)) / max(1, len(left))


def _forced_choice(model, tokenizer) -> tuple[float, float]:
    """How often the right continuation wins, and by how much.

    Two numbers because one is not sensitive enough alone. The accuracy is what
    matters — whether the model still gets things right — but on items this easy
    it survives perturbations large enough to wreck the prose, so it cannot tell
    a good direction from an arbitrary one. The margin, log p(right) minus
    log p(wrong), moves continuously and can.
    """
    import mlx.core as mx

    correct = 0
    margins: list[float] = []
    for stem, right, wrong in FORCED_CHOICE:
        stem_ids = list(tokenizer.encode(stem))
        scores: list[float] = []
        for continuation in (right, wrong):
            tail = [int(token) for token in tokenizer.encode(continuation, add_special_tokens=False)]
            distributions = _walk(model, stem_ids, tail)
            total = 0.0
            for distribution, token in zip(distributions, tail, strict=True):
                total += math.log(max(float(distribution[token]), 1e-12))
            scores.append(total / len(tail))
        if scores[0] > scores[1]:
            correct += 1
        margins.append(scores[0] - scores[1])
        mx.clear_cache()
    return correct / len(FORCED_CHOICE), sum(margins) / len(margins)


def measure_fusion(
    model: Any,
    tokenizer: Any,
    hooks: Sequence[Any],
    set_alpha: Callable[[float], None],
    *,
    model_identity: str,
    model_name: str = "",
    alphas: Sequence[float] = (0.1, 0.2),
    steps: int = 24,
    seed: int = 20260908,
    runner: str = "core/consciousness/fusion_probe.measure_fusion",
    on_result: Callable[[FusionCertificate], None] | None = None,
) -> list[FusionCertificate]:
    """Measure the substrate-to-forward-pass channel on a model already loaded.

    `set_alpha` is the caller's way of setting hook alpha — the engine's own
    `set_alpha`, so the probe steers through the same path as a live turn rather
    than writing to hook attributes behind it.
    """
    import mlx.core as mx
    import numpy as np

    if not hooks:
        return []

    prompt_ids = [chat_ids(tokenizer, prompt) for prompt in PROBE_PROMPTS]

    def settle(moods: dict[str, float]) -> None:
        for _ in range(SETTLE_UPDATES):
            for hook in hooks:
                hook.update_substrate(moods)

    def override(vector) -> None:
        for hook in hooks:
            hook.override_composite_vector(vector)

    def walk_all(paths) -> list:
        return [_walk(model, ids, path) for ids, path in zip(prompt_ids, paths, strict=True)]

    set_alpha(0.0)
    paths = [_greedy_path(model, ids, steps) for ids in prompt_ids]
    unsteered = walk_all(paths)
    baseline_accuracy, baseline_margin = _forced_choice(model, tokenizer)

    rng = np.random.default_rng(seed)
    certificates: list[FusionCertificate] = []

    for alpha in alphas:
        alpha = float(alpha)
        settle(STATE_HIGH)
        set_alpha(alpha)
        her_high = walk_all(paths)
        composite = hooks[0].current_composite_vector()
        if composite is None:
            logger.info("fusion probe: steering stood down at alpha %s", alpha)
            continue
        steered_accuracy, steered_margin = _forced_choice(model, tokenizer)
        steered_paths = [_greedy_path(model, ids, steps) for ids in prompt_ids]

        settle(STATE_LOW)
        low_composite = hooks[0].current_composite_vector()
        her_low = walk_all(paths)
        cosine = (
            float(
                np.dot(composite, low_composite)
                / (np.linalg.norm(composite) * np.linalg.norm(low_composite))
            )
            if low_composite is not None
            else 0.0
        )

        control_shifts: list[float] = []
        control_accuracies: list[float] = []
        control_margins: list[float] = []
        for _ in range(CONTROL_DIRECTIONS):
            noise = rng.normal(size=composite.shape).astype(np.float32)
            noise *= float(np.linalg.norm(composite)) / float(np.linalg.norm(noise))
            override(noise)
            control = walk_all(paths)
            accuracy, margin = _forced_choice(model, tokenizer)
            control_shifts.append(
                float(np.mean([_mean_shift(a, b) for a, b in zip(control, unsteered, strict=True)]))
            )
            control_accuracies.append(accuracy)
            control_margins.append(margin)
        override(None)

        arrives = float(
            np.mean([_mean_shift(a, b) for a, b in zip(her_high, unsteered, strict=True)])
        )
        low_shift = float(
            np.mean([_mean_shift(a, b) for a, b in zip(her_low, unsteered, strict=True)])
        )
        between = float(
            np.mean([_mean_shift(a, b) for a, b in zip(her_high, her_low, strict=True)])
        )
        mean_shift = (arrives + low_shift) / 2.0
        separation = between / mean_shift if mean_shift > 0 else 0.0

        divergences = [
            _first_difference(steered, plain)
            for steered, plain in zip(steered_paths, paths, strict=True)
        ]
        diverged = [index for index in divergences if index >= 0]

        certificate = FusionCertificate(
            model_identity=model_identity,
            model_name=model_name,
            alpha=alpha,
            distribution_shift=arrives,
            control_shift=float(np.median(control_shifts)),
            state_separation=separation,
            prompts_that_change=len(diverged),
            median_step_of_change=int(np.median(diverged)) if diverged else -1,
            quality_delta=steered_accuracy - baseline_accuracy,
            control_quality_delta=float(np.median(control_accuracies)) - baseline_accuracy,
            margin_delta=steered_margin - baseline_margin,
            control_margin_delta=float(np.median(control_margins)) - baseline_margin,
            quality_scale=f"forced-choice accuracy over {len(FORCED_CHOICE)} items",
            prompts=len(PROBE_PROMPTS),
            steps=steps,
            runner=runner,
            note=(
                f"unsteered accuracy {baseline_accuracy:.3f} at margin {baseline_margin:.3f}, "
                f"steered {steered_accuracy:.3f} at {steered_margin:.3f}; opposing states move "
                f"the distribution {between:.5f} nats apart while each moves it "
                f"{mean_shift:.5f} from silence, their composites at cosine {cosine:.3f}; "
                f"{CONTROL_DIRECTIONS} random directions of the same length score a median "
                f"margin of {float(np.median(control_margins)):.3f}"
            ),
        )
        certificates.append(certificate)
        if on_result is not None:
            on_result(certificate)
        mx.clear_cache()

    set_alpha(0.0)
    return certificates
