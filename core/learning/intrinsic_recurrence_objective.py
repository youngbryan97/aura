"""Train the checkpoint to survive — and then use — its own recurrence (CP227).

CP226 measured what an untrained retrofit does to the resident 32B:

    T=1    64 layers   reasoning 12%   answered 79%   (== vanilla)
    T=2    96 layers   reasoning  8%   answered 71%
    T=4   160 layers   reasoning  0%   answered  4%
    T=8   288 layers   reasoning  0%   answered  0%

Nothing diverged; every state stayed finite and the loop kept moving. The
model simply stopped emitting parseable output. That localizes the defect:
``layers[coda:]`` has never in its life received a state that passed
through ``layers[prelude:coda]`` eight times, so its output distribution
collapses on input it was never trained to accept.

Three consequences shape this objective, each traceable to that finding
rather than to taste:

* **The coda is adapted, not just the window.** The window produces the
  out-of-distribution state; the coda is what breaks on it. Training only
  the window would leave the failing component untouched.
* **T=1 stays in the mix as an anchor.** Without it, the model is free to
  purchase depth-tolerance with base ability and the ladder would look
  like progress while the model got worse. The anchor makes that trade
  visible as a loss increase instead of hiding it.
* **Depth is PRICED, not forced.** ``adaptive_depth_loss`` (v4, wired in
  CP211) selects depth by softmin over CE + price*depth. The earlier
  monotone hinge demanded improvement on families where measured depth
  response is negative (khop -33.1%, modular +15.1%), which is an
  unwinnable objective and produced branch collapse.

Gradient checkpointing is mandatory, not an optimization: backprop through
T passes of a 32-layer window retains T*32 layer activations, and at T=4 on
a 32B that is the difference between fitting in the envelope and repeating
the 103 GB incident.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

INTRINSIC_OBJECTIVE_SCHEMA = "aura.intrinsic_recurrence_objective.v1"


@dataclass(frozen=True)
class IntrinsicTrainingSpec:
    """What to train, at which depths, under what stabilizer."""

    prelude_end: int
    coda_start: int
    # T=1 must be present: it is the anchor that keeps base ability honest.
    depths: tuple[int, ...] = (1, 2, 4)
    anchor_injection: float = 0.0
    renormalize: bool = True
    compute_price: float = 0.01
    depth_temperature: float = 0.15
    # Weight on the T=1 anchor's own CE, over and above its role in the
    # priced selection. Base ability is not a free variable.
    anchor_weight: float = 1.0
    # Direct pressure on CP226's measured obstacle: cos(pass1, pass2) =
    # 0.9994 — the loop moves 42% in magnitude but barely rotates, so each
    # re-entry re-computes the previous increment instead of taking a NEW
    # algorithm step. This term penalizes cos² between consecutive window-
    # pass increments, punishing both idempotence (cos→+1) and the period-2
    # oscillation CP210 caught (cos→−1); orthogonal increments — what
    # distinct computation steps look like — cost nothing. 0.0 = off.
    rotation_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.prelude_end >= self.coda_start:
            raise ValueError("prelude_end must precede coda_start")
        if not self.depths:
            raise ValueError("depths must not be empty")
        if any(type(d) is not int or d < 1 for d in self.depths):
            raise ValueError("every depth must be a positive integer")
        if len(set(self.depths)) != len(self.depths):
            raise ValueError("depths must be distinct")
        if 1 not in self.depths:
            raise ValueError(
                "depth 1 must be present: without the base-ability anchor the "
                "model can buy depth tolerance by getting worse, and the "
                "ladder would show that as progress"
            )
        if (
            isinstance(self.anchor_weight, bool)
            or not isinstance(self.anchor_weight, (int, float))
            or not 0.0 <= float(self.anchor_weight) <= 10.0
        ):
            raise ValueError("anchor_weight must be inside [0, 10]")
        if (
            isinstance(self.rotation_weight, bool)
            or not isinstance(self.rotation_weight, (int, float))
            or not 0.0 <= float(self.rotation_weight) <= 10.0
        ):
            raise ValueError("rotation_weight must be inside [0, 10]")

    def plan_at(self, depth: int):
        from core.learning.intrinsic_recurrence import RecurrentDepthPlan

        return RecurrentDepthPlan(
            prelude_end=self.prelude_end,
            coda_start=self.coda_start,
            iterations=depth,
            anchor_injection=self.anchor_injection,
            renormalize=self.renormalize,
        )

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": INTRINSIC_OBJECTIVE_SCHEMA,
            "prelude_end": self.prelude_end,
            "coda_start": self.coda_start,
            "depths": list(self.depths),
            "anchor_injection": float(self.anchor_injection),
            "renormalize": self.renormalize,
            "compute_price": self.compute_price,
            "depth_temperature": self.depth_temperature,
            "anchor_weight": self.anchor_weight,
            "rotation_weight": self.rotation_weight,
        }


def adapted_layer_indices(spec: IntrinsicTrainingSpec, total_layers: int) -> list[int]:
    """Which layers get LoRA: the window AND the coda.

    CP226 localized the collapse to the coda -- it receives a state that
    went through the window T times and its output distribution fails.
    Adapting only the window would train everything except the component
    that actually broke.
    """
    if type(total_layers) is not int or total_layers < spec.coda_start:
        raise ValueError("total_layers is smaller than the spec's window")
    window = list(range(spec.prelude_end, spec.coda_start))
    coda = list(range(spec.coda_start, total_layers))
    if not coda:
        raise ValueError(
            "the spec leaves no coda layers to adapt, so the component that "
            "collapsed in CP226 would go untrained"
        )
    return window + coda


def answer_cross_entropy(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    plan: Any,
    *,
    caches: dict[str, Any] | None = None,
) -> Any:
    """CE of the answer under an intrinsically-recurrent forward pass.

    This is the whole point of CP226: the loss flows through the ANSWER's
    own deepened computation. The previous objectives scored an answer that
    had traversed the window exactly once regardless of depth, so no
    gradient could ever teach depth to help.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from core.learning.intrinsic_recurrence import recurrent_hidden_states

    answer_count = int(answer_tokens.shape[-1])
    if answer_count < 1:
        raise ValueError("answer_tokens must contain at least one token")
    full = mx.concatenate([tokens, answer_tokens], axis=1)
    hidden, trajectory = recurrent_hidden_states(model, full, plan, caches=caches)
    if getattr(model, "lm_head", None) is not None:
        logits = model.lm_head(hidden)
    else:
        logits = model.model.embed_tokens.as_linear(hidden)
    start = int(full.shape[1]) - answer_count - 1
    predicted = logits[:, start : start + answer_count, :]
    losses = nn.losses.cross_entropy(
        predicted.astype(mx.float32),
        answer_tokens,
        reduction="none",
    )
    return mx.mean(losses), trajectory


def rotation_pressure(trajectory: Sequence[Any]) -> tuple[Any, dict[str, Any]]:
    """Penalize consecutive window-pass increments that point the same way.

    CP226 reduced the obstacle to one number: cos(pass1, pass2) = 0.9994.
    The state moves 42% in magnitude and barely rotates — re-entering the
    window adds another increment along nearly the same ray, so extra depth
    re-computes the last step instead of taking a new one. This term is the
    first DIRECT training pressure on that geometry:

        loss = mean over consecutive increment pairs of cos²(Δt, Δt+1)

    cos² punishes both failure shapes we have measured: idempotence
    (cos→+1, CP226) and the period-2 limit cycle (cos→−1, CP210).
    Orthogonal increments — what distinct steps of an algorithm look like —
    cost nothing. Computed in float32; fp16 reductions on residual streams
    overflow (the CP226 meter bug).

    Needs at least three trajectory states (two increments); below that it
    returns zero loss with the receipt saying why.
    """
    import mlx.core as mx

    states = list(trajectory)
    if len(states) < 3:
        return mx.zeros(()), {
            "pairs": 0,
            "mean_cos": None,
            "reason": "needs_at_least_two_increments",
        }
    increments = [
        (states[index + 1] - states[index]).astype(mx.float32)
        for index in range(len(states) - 1)
    ]
    cosines = []
    for first, second in zip(increments, increments[1:], strict=False):
        flat_first = mx.reshape(first, (-1,))
        flat_second = mx.reshape(second, (-1,))
        denominator = mx.maximum(
            mx.linalg.norm(flat_first) * mx.linalg.norm(flat_second), 1e-6
        )
        cosines.append(mx.sum(flat_first * flat_second) / denominator)
    stacked = mx.stack(cosines)
    loss = mx.mean(mx.square(stacked))
    return loss, {
        "pairs": len(cosines),
        "mean_cos": round(float(mx.mean(stacked)), 6),
        "mean_cos_sq": round(float(loss), 6),
    }


def latent_step_answer_ce(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    plan: Any,
) -> list[float]:
    """Answer CE decoded from EACH intermediate window state via the coda.

    The per-step score curve latent-trajectory credit assignment needs:
    which internal passes actually moved the state toward the answer. The
    final entry equals the ordinary recurrent forward's CE by construction.
    Costs one coda pass per iteration — bounded by the plan's depth.
    """
    import mlx.core as mx
    import mlx.nn as nn

    from core.learning.intrinsic_recurrence import recurrent_hidden_states

    answer_count = int(answer_tokens.shape[-1])
    if answer_count < 1:
        raise ValueError("answer_tokens must contain at least one token")
    full = mx.concatenate([tokens, answer_tokens], axis=1)
    _, trajectory = recurrent_hidden_states(model, full, plan)
    inner = model.model
    layers = inner.layers
    start = int(full.shape[1]) - answer_count - 1
    from core.learning.intrinsic_recurrence import _run

    scores: list[float] = []
    for state in trajectory:
        hidden = _run(layers[plan.coda_start :], state)
        hidden = inner.norm(hidden)
        if getattr(model, "lm_head", None) is not None:
            logits = model.lm_head(hidden)
        else:
            logits = inner.embed_tokens.as_linear(hidden)
        predicted = logits[:, start : start + answer_count, :]
        losses = nn.losses.cross_entropy(
            predicted.astype(mx.float32), answer_tokens, reduction="none"
        )
        scores.append(float(mx.mean(losses)))
    return scores


def intrinsic_depth_loss(
    model: Any,
    tokens: Any,
    answer_tokens: Any,
    spec: IntrinsicTrainingSpec,
) -> tuple[Any, dict[str, Any]]:
    """Priced depth selection over intrinsically-recurrent forward passes.

    Returns ``(loss, telemetry)``. Telemetry carries the per-depth CE and
    the selected depth so a training run can be read for what it actually
    learned rather than a single scalar -- the failure that let a period-2
    limit cycle and a fixed-point collapse both go unnoticed.
    """
    import mlx.core as mx

    from core.learning.recurrence_native_objective_v4 import adaptive_depth_loss

    depth_losses: list[Any] = []
    per_depth: dict[str, float] = {}
    anchor_loss: Any = None
    rotation_terms: list[Any] = []
    rotation_evidence: dict[str, Any] = {}
    for depth in spec.depths:
        loss, trajectory = answer_cross_entropy(
            model, tokens, answer_tokens, spec.plan_at(depth)
        )
        depth_losses.append(loss)
        per_depth[f"T{depth}"] = float(loss)
        if depth == 1:
            anchor_loss = loss
        if spec.rotation_weight > 0.0 and depth >= 3:
            term, evidence = rotation_pressure(trajectory)
            if evidence.get("pairs"):
                rotation_terms.append(term)
                rotation_evidence[f"T{depth}"] = evidence

    # adaptive_depth_loss returns (loss, priced costs, selected DEPTH).
    # The third value is the depth itself, not an index into spec.depths --
    # indexing with it silently mislabels the selection when it happens to
    # be in range, and only raises when it is not.
    selected, priced_costs, selected_depth = adaptive_depth_loss(
        depth_losses,
        list(spec.depths),
        compute_price=spec.compute_price,
        temperature=spec.depth_temperature,
    )
    # The anchor is added OUTSIDE the softmin. Inside, a model could win by
    # making every depth equally bad; outside, base ability has to hold.
    total = selected + spec.anchor_weight * anchor_loss
    rotation_total: Any = None
    if rotation_terms:
        import mlx.core as mx

        rotation_total = mx.mean(mx.stack(rotation_terms))
        total = total + spec.rotation_weight * rotation_total
    telemetry: dict[str, Any] = {
        "schema": INTRINSIC_OBJECTIVE_SCHEMA,
        "per_depth_ce": per_depth,
        "priced_costs": [round(float(c), 4) for c in priced_costs],
        "selected_depth": int(selected_depth),
        "anchor_ce": float(anchor_loss),
        "priced_ce": float(selected),
        "total": float(total),
    }
    if rotation_total is not None:
        telemetry["rotation"] = {
            "weight": spec.rotation_weight,
            "loss": round(float(rotation_total), 6),
            "per_depth": rotation_evidence,
        }
    return total, telemetry


def depth_tolerance(per_depth_ce: dict[str, float]) -> dict[str, Any]:
    """Has the model stopped collapsing at depth?

    The CP226 signature was not a worse CE -- it was output collapse, CE
    running away as the coda failed. This reports the shape directly so a
    training run can be stopped when the collapse is fixed, rather than
    inferred from the total loss.
    """
    if not per_depth_ce:
        raise ValueError("per_depth_ce must not be empty")
    ordered = sorted(
        per_depth_ce.items(), key=lambda item: int(item[0].lstrip("T"))
    )
    values = [value for _, value in ordered]
    base = values[0]
    if base <= 0.0:
        raise ValueError("anchor CE must be positive")
    worst_ratio = max(value / base for value in values)
    return {
        "schema": INTRINSIC_OBJECTIVE_SCHEMA,
        "depths": [name for name, _ in ordered],
        "ce": [round(v, 5) for v in values],
        "anchor_ce": round(base, 5),
        "worst_relative_ce": round(worst_ratio, 4),
        # CP226 collapsed to unparseable output; CE within ~1.5x of the
        # anchor across the ladder means the coda is accepting iterated
        # states at all, which is the precondition for depth helping.
        "collapse_repaired": bool(worst_ratio <= 1.5),
        "depth_helps": bool(min(values) < base and values.index(min(values)) > 0),
    }


# Gradient checkpointing lives in intrinsic_recurrence, because it has to
# wrap THAT module's forward. v2's layer_checkpoint_scope is bound to v2's
# own _causal_layers and would have been a no-op here -- a mechanism present
# in name only, the defect CP211 had to repair in v4.
from core.learning.intrinsic_recurrence import (  # noqa: E402
    checkpointed_window,
)


__all__ = [
    "INTRINSIC_OBJECTIVE_SCHEMA",
    "IntrinsicTrainingSpec",
    "adapted_layer_indices",
    "answer_cross_entropy",
    "checkpointed_window",
    "depth_tolerance",
    "intrinsic_depth_loss",
    "latent_step_answer_ce",
    "rotation_pressure",
]
