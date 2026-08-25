"""Interventions on z_Aura, and whether anything downstream actually moves.

The interesting question about a cognitive substrate is not whether it exists
but whether it does anything. That question has a standard form:

    do(uncertainty.confidence = 0.9)

hold everything else, and see what changes. If the code moves from
``UNCERTAINTY high`` to ``UNCERTAINTY low``, if the vocabulary bias shifts
towards the words a confident state prefers, and if the reply follows — then
the dimension is load-bearing. If nothing moves, the dimension is telemetry.

Two rules keep an intervention from measuring itself.

**The live state is never touched.** ``EndogenousState.do`` returns a copy, so
the arm and its control are two objects, and the runtime keeps the state it
actually holds.

**Every effect is read against a matched null.** Intervening on one dimension
by some amount and finding a change proves nothing until the same-sized
intervention on the *other* dimensions is known to produce less. That
comparison is what separates "this dimension matters" from "this head is
sensitive to any perturbation", and it is why every function here returns a
null alongside its effect.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.brain.llm.cognitive_code import CognitiveCode, IntentHead, read_code
from core.brain.llm.endogenous_state import (
    CHANNELS,
    FEATURE_INDEX,
    FEATURES,
    EndogenousState,
)
from core.brain.llm.endogenous_vocab_head import (
    EndogenousVocabHead,
    HeadUnusableError,
)

logger = logging.getLogger("Aura.EndogenousIntervention")

#: How many tokens an effect report names. A list of every moved token is not
#: a finding.
TOP_TOKENS = 12


@dataclass(frozen=True)
class Arm:
    """One condition of an experiment: a state, and what it reads out as."""

    name: str
    state: EndogenousState
    code: CognitiveCode
    delta: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_digest": self.state.digest,
            "coverage": round(self.state.coverage, 4),
            "code": self.code.render(),
            "interventions": [i.as_dict() for i in self.state.interventions],
            "has_bias": self.delta is not None,
        }


@dataclass(frozen=True)
class Effect:
    """What one intervention did, and what the same-sized nulls did."""

    feature: str
    value: float
    code_lines_moved: tuple[str, ...]
    bias_shift: float
    top_promoted: tuple[int, ...]
    top_suppressed: tuple[int, ...]
    null_bias_shifts: tuple[float, ...]

    @property
    def null_ceiling(self) -> float:
        return float(max(self.null_bias_shifts)) if self.null_bias_shifts else 0.0

    @property
    def exceeds_null(self) -> bool:
        """Whether this dimension moved language more than its peers would.

        With no nulls this is False, not True. An effect with no comparison is
        not a small effect, it is an unmeasured one.
        """
        return bool(self.null_bias_shifts) and self.bias_shift > self.null_ceiling

    @property
    def moved_anything(self) -> bool:
        return bool(self.code_lines_moved) or self.bias_shift > 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round(self.value, 6),
            "code_lines_moved": list(self.code_lines_moved),
            "bias_shift_l2": round(self.bias_shift, 6),
            "top_promoted_tokens": list(self.top_promoted),
            "top_suppressed_tokens": list(self.top_suppressed),
            "null_bias_shifts": [round(v, 6) for v in self.null_bias_shifts],
            "null_ceiling": round(self.null_ceiling, 6),
            "exceeds_null": self.exceeds_null,
            "moved_anything": self.moved_anything,
        }


def _delta_or_none(
    head: EndogenousVocabHead | None, state: EndogenousState
) -> np.ndarray | None:
    if head is None:
        return None
    try:
        return head.delta_logits(state)
    except (HeadUnusableError, TypeError, ValueError) as exc:
        logger.debug("head produced no bias for an arm: %s", exc)
        return None


def build_arm(
    name: str,
    state: EndogenousState,
    *,
    head: EndogenousVocabHead | None = None,
    intent_head: IntentHead | None = None,
) -> Arm:
    return Arm(
        name=name,
        state=state,
        code=read_code(state, intent_head=intent_head, include_organ_lines=False),
        delta=_delta_or_none(head, state),
    )


def _null_features(feature: str, *, count: int) -> list[str]:
    """Peer dimensions to intervene on instead, for the matched null.

    Taken from other channels, because a same-channel neighbour often carries
    a correlated meaning and would understate the null.
    """
    channel = FEATURES[FEATURE_INDEX[feature]].channel
    peers = [f.name for f in FEATURES if f.channel != channel]
    if not peers:
        peers = [f.name for f in FEATURES if f.name != feature]
    stride = max(1, len(peers) // max(1, count))
    return peers[::stride][:count]


def measure_intervention(
    state: EndogenousState,
    feature: str,
    value: float,
    *,
    head: EndogenousVocabHead | None = None,
    intent_head: IntentHead | None = None,
    null_count: int = 8,
) -> Effect:
    """Apply ``do(feature = value)`` and report what moved, against its nulls."""
    if feature not in FEATURE_INDEX:
        raise KeyError(f"no such endogenous dimension: {feature}")
    baseline = build_arm("baseline", state, head=head, intent_head=intent_head)
    treated = build_arm(
        f"do({feature}={value})",
        state.do(**{feature: value}),
        head=head,
        intent_head=intent_head,
    )
    moved = tuple(baseline.code.diff(treated.code).keys())

    shift, promoted, suppressed = _bias_change(baseline.delta, treated.delta)

    nulls: list[float] = []
    for peer in _null_features(feature, count=null_count):
        peer_arm = build_arm(
            f"do({peer})", state.do(**{peer: value}), head=head, intent_head=intent_head
        )
        peer_shift, _, _ = _bias_change(baseline.delta, peer_arm.delta)
        nulls.append(peer_shift)

    return Effect(
        feature=feature,
        value=float(value),
        code_lines_moved=moved,
        bias_shift=shift,
        top_promoted=promoted,
        top_suppressed=suppressed,
        null_bias_shifts=tuple(nulls),
    )


def _bias_change(
    before: np.ndarray | None, after: np.ndarray | None
) -> tuple[float, tuple[int, ...], tuple[int, ...]]:
    if before is None or after is None or before.shape != after.shape:
        return 0.0, (), ()
    difference = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    magnitude = float(np.linalg.norm(difference))
    if magnitude <= 0.0:
        return 0.0, (), ()
    order = np.argsort(difference)
    return (
        magnitude,
        tuple(int(i) for i in order[::-1][:TOP_TOKENS]),
        tuple(int(i) for i in order[:TOP_TOKENS]),
    )


def measure_contrast(
    state: EndogenousState,
    feature: str,
    low: float,
    high: float,
    *,
    head: EndogenousVocabHead | None = None,
    intent_head: IntentHead | None = None,
    null_count: int = 8,
) -> Effect:
    """Compare ``do(feature=low)`` against ``do(feature=high)`` directly.

    Measuring one intervention against the state as it happens to be found
    reports nothing when the state already sat where the intervention put it —
    which is most of the time, and it looks identical to a dimension that does
    not matter. A contrast between two pinned values asks the question the
    experiment is actually about, and its nulls are the same contrast on peer
    dimensions.
    """
    if feature not in FEATURE_INDEX:
        raise KeyError(f"no such endogenous dimension: {feature}")
    low_arm = build_arm(
        f"{feature}={low}", state.do(**{feature: low}), head=head, intent_head=intent_head
    )
    high_arm = build_arm(
        f"{feature}={high}", state.do(**{feature: high}), head=head, intent_head=intent_head
    )
    shift, promoted, suppressed = _bias_change(low_arm.delta, high_arm.delta)
    nulls: list[float] = []
    for peer in _null_features(feature, count=null_count):
        peer_low = build_arm(
            f"{peer}={low}", state.do(**{peer: low}), head=head, intent_head=intent_head
        )
        peer_high = build_arm(
            f"{peer}={high}", state.do(**{peer: high}), head=head, intent_head=intent_head
        )
        peer_shift, _, _ = _bias_change(peer_low.delta, peer_high.delta)
        nulls.append(peer_shift)
    return Effect(
        feature=feature,
        value=float(high),
        code_lines_moved=tuple(low_arm.code.diff(high_arm.code).keys()),
        bias_shift=shift,
        top_promoted=promoted,
        top_suppressed=suppressed,
        null_bias_shifts=tuple(nulls),
    )


def measure_ablation(
    state: EndogenousState,
    channel: str,
    *,
    head: EndogenousVocabHead | None = None,
    intent_head: IntentHead | None = None,
) -> Effect:
    """Remove a whole channel — the "take the memory away" experiment.

    Its null is the same operation on each other channel, which is the fair
    comparison: a head that reacts to losing any channel equally has not shown
    that this one carries anything special.
    """
    if channel not in CHANNELS:
        raise KeyError(f"no such endogenous channel: {channel}")
    baseline = build_arm("baseline", state, head=head, intent_head=intent_head)
    treated = build_arm(
        f"ablate({channel})", state.ablate(channel), head=head, intent_head=intent_head
    )
    shift, promoted, suppressed = _bias_change(baseline.delta, treated.delta)
    nulls = []
    for peer in CHANNELS:
        if peer == channel:
            continue
        peer_arm = build_arm(
            f"ablate({peer})", state.ablate(peer), head=head, intent_head=intent_head
        )
        peer_shift, _, _ = _bias_change(baseline.delta, peer_arm.delta)
        nulls.append(peer_shift)
    return Effect(
        feature=f"channel:{channel}",
        value=0.0,
        code_lines_moved=tuple(baseline.code.diff(treated.code).keys()),
        bias_shift=shift,
        top_promoted=promoted,
        top_suppressed=suppressed,
        null_bias_shifts=tuple(nulls),
    )


def sweep_dimension(
    state: EndogenousState,
    feature: str,
    values: Sequence[float],
    *,
    head: EndogenousVocabHead | None = None,
    intent_head: IntentHead | None = None,
) -> dict[str, Any]:
    """Walk one dimension across a range and record the readout at each step.

    This is the trajectory z1 → z2 → z3 the architecture argument is about,
    produced by moving one named number rather than by replaying prose back
    into a model.
    """
    steps = []
    previous: CognitiveCode | None = None
    for value in values:
        arm = build_arm(
            f"{feature}={value}",
            state.do(**{feature: float(value)}),
            head=head,
            intent_head=intent_head,
        )
        steps.append(
            {
                "value": float(value),
                "code": arm.code.render(),
                "moved_from_previous": (
                    list(previous.diff(arm.code).keys()) if previous else []
                ),
                "bias_norm": (
                    round(float(np.linalg.norm(arm.delta)), 6)
                    if arm.delta is not None
                    else None
                ),
            }
        )
        previous = arm.code
    return {"feature": feature, "steps": steps}


def channel_influence_map(
    state: EndogenousState, head: EndogenousVocabHead
) -> dict[str, Any]:
    """Which channels this head actually listens to.

    A head can be trained, attach cleanly, and still be reading three of nine
    channels. The map says which, so "the substrate steers language" never
    stands in for "one channel steers language".
    """
    rows = []
    for channel in CHANNELS:
        effect = measure_ablation(state, channel, head=head)
        rows.append(
            {
                "channel": channel,
                "bias_shift_l2": round(effect.bias_shift, 6),
                "exceeds_null": effect.exceeds_null,
                "code_lines_moved": list(effect.code_lines_moved),
            }
        )
    rows.sort(key=lambda row: row["bias_shift_l2"], reverse=True)
    listened = [row["channel"] for row in rows if row["bias_shift_l2"] > 0.0]
    return {
        "channels": rows,
        "channels_with_influence": listened,
        "channels_ignored": [c for c in CHANNELS if c not in listened],
    }


__all__ = [
    "TOP_TOKENS",
    "Arm",
    "Effect",
    "build_arm",
    "channel_influence_map",
    "measure_ablation",
    "measure_contrast",
    "measure_intervention",
    "sweep_dimension",
]


# ──────────────────────────────────────────────────────────────────────────
# The text arm: does an intervention on z change what she actually says?
#
# Everything above is measurable with no model at all, which is the point —
# the code and the bias move before generation. But the claim that matters is
# about language, and that needs the model. This runs against a client the
# runtime already holds. It never loads a second one.
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TextArm:
    """One condition of a text experiment: the state, and what came out."""

    name: str
    state: EndogenousState
    text: str
    receipt: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_digest": self.state.digest,
            "interventions": [i.as_dict() for i in self.state.interventions],
            "text": self.text,
            "endogenous_bias": self.receipt,
        }


def _token_set(text: str) -> set[str]:
    return {piece for piece in str(text or "").lower().split() if piece}


def text_distance(left: str, right: str) -> float:
    """One minus the Jaccard overlap of the two word sets.

    A blunt measure, deliberately. Anything finer would need a model to
    score it, and a model scoring its own outputs is a confound rather than a
    measurement.
    """
    a, b = _token_set(left), _token_set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / max(1, len(a | b))


async def run_text_arm(
    client: Any,
    prompt: str,
    state: EndogenousState,
    *,
    name: str = "",
    max_tokens: int = 160,
    temp: float = 0.0,
) -> TextArm:
    """Generate one reply under a given z_Aura, with the prompt held fixed.

    The state is put on the job directly rather than assembled from the live
    organs, which is what makes this an intervention: everything about the
    turn is identical except the named dimension that was moved.
    """
    from core.brain.llm.endogenous_decode import JOB_STATE_KEY

    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temp": float(temp),
        JOB_STATE_KEY: state.to_payload(),
    }
    from core.brain.llm.endogenous_decode import pathway_health

    text = await client.generate(prompt, **kwargs)
    # The receipt comes from the parent's own view of the pathway rather than
    # from the client: the bias is decided in the worker, and this is where
    # its answer lands.
    receipt = dict(pathway_health().get("last_receipt") or {})
    return TextArm(
        name=name or state.digest, state=state, text=str(text or ""), receipt=receipt
    )


async def causal_text_experiment(
    client: Any,
    prompt: str,
    state: EndogenousState,
    feature: str,
    low: float,
    high: float,
    *,
    null_features: Sequence[str] | None = None,
    max_tokens: int = 160,
) -> dict[str, Any]:
    """Two arms on one dimension, plus matched nulls on the others.

    The verdict rests on the comparison, not on the two arms: if moving any
    dimension changes the reply by as much as moving this one, the finding is
    that generation is sensitive to perturbation, which is a different and far
    weaker claim.
    """
    peers = list(null_features or _null_features(feature, count=4))
    low_arm = await run_text_arm(
        client, prompt, state.do(**{feature: low}), name=f"{feature}={low}",
        max_tokens=max_tokens,
    )
    high_arm = await run_text_arm(
        client, prompt, state.do(**{feature: high}), name=f"{feature}={high}",
        max_tokens=max_tokens,
    )
    treated = text_distance(low_arm.text, high_arm.text)

    nulls = []
    for peer in peers:
        peer_low = await run_text_arm(
            client, prompt, state.do(**{peer: low}), name=f"{peer}={low}",
            max_tokens=max_tokens,
        )
        peer_high = await run_text_arm(
            client, prompt, state.do(**{peer: high}), name=f"{peer}={high}",
            max_tokens=max_tokens,
        )
        nulls.append(
            {
                "feature": peer,
                "distance": round(text_distance(peer_low.text, peer_high.text), 6),
                "low": peer_low.text,
                "high": peer_high.text,
            }
        )

    ceiling = max((n["distance"] for n in nulls), default=0.0)
    return {
        "prompt": prompt,
        "feature": feature,
        "low": low,
        "high": high,
        "arms": [low_arm.as_dict(), high_arm.as_dict()],
        "treated_distance": round(treated, 6),
        "null_distances": nulls,
        "null_ceiling": round(ceiling, 6),
        "exceeds_null": bool(nulls) and treated > ceiling,
        "verdict": (
            "no_verdict_no_null"
            if not nulls
            else ("causal" if treated > ceiling else "within_null")
        ),
    }


__all__ += ["TextArm", "causal_text_experiment", "run_text_arm", "text_distance"]
