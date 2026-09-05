"""Recurrence-native training objective v3 (CP181).

The CP179 pilot refuted the v2-trained adapter with four diagnoses; this
objective answers the two that live in the loss itself:

- ``recurrence_training_failed_directional_gain_gate`` — v2's monotonic
  hinge only fired when a deeper pass was WORSE than a shallower one, so
  the optimum tolerated exact equality: recurrence that does nothing. v3
  demands a positive depth advantage: each deeper pass must beat the
  (gradient-detached) shallower pass by at least ``depth_margin`` nats of
  answer cross-entropy before the hinge releases.
- ``adapter_virtual_width_functionally_collapsed`` — v2's mean-answer CE
  pushed every branch toward one function; the pilot measured exchange
  cosines pinned in [0.99979, 1.0] and byte-identical branch scores. v3
  adds a differentiable diversity term over the branches' final latent
  states: pairwise cosine above ``diversity_target_cos`` is penalized
  quadratically, so roles may AGREE on the answer while being pushed to
  reach it through decorrelated latent trajectories. Post-exchange
  pairwise cosines are returned as telemetry floats every step, so
  training logs show whether diversity actually held (the post-jitter
  telemetry the tracker requires).

Both terms ride the SAME live-path forward as v2 (`live_path_forward`) —
training and live execution share one execution graph, one bridge, one
norm discipline. Bridge parity and held-out validation are trainer-side
obligations; this module carries the loss and its telemetry receipt.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v2 import (
    LivePathForward,
    branch_mean_answer_loss,
    live_path_forward,
)

RECURRENCE_NATIVE_SCHEMA_V3 = "aura.recurrence_native_objective.v3"

_MAX_WEIGHT = 10.0


def _validate_unit_weight(name: str, value: Any, upper: float = _MAX_WEIGHT) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= upper
    ):
        raise ValueError(f"{name} must be inside [0, {upper}]")
    return float(value)


def depth_margin_penalty(losses: Sequence[Any], *, margin: float) -> Any:
    """Hinge that releases only when deeper beats shallower by ``margin``.

    relu(deep − stop_gradient(shallow) + margin): zero exactly when
    ``deep <= shallow − margin``. The shallow side is gradient-detached so
    the optimizer cannot satisfy the hinge by damaging shallow competence —
    the only way out is genuinely better deep computation.
    """
    import mlx.core as mx

    if len(losses) < 2:
        raise ValueError("depth margin penalty needs at least two depths")
    margin_value = _validate_unit_weight("depth_margin", margin, upper=2.0)
    penalty = mx.zeros(())
    for shallow, deep in zip(losses, losses[1:]):
        penalty = penalty + mx.maximum(
            deep - mx.stop_gradient(shallow) + margin_value, 0.0
        )
    return penalty


def _pairwise_state_cosines(forward: LivePathForward) -> list[Any]:
    """Differentiable pairwise cosines between branches' final states."""
    import mlx.core as mx

    flats = [mx.reshape(state, (-1,)) for state in forward.branch_states]
    cosines: list[Any] = []
    for left_index in range(len(flats)):
        for right_index in range(left_index + 1, len(flats)):
            left, right = flats[left_index], flats[right_index]
            denom = mx.maximum(
                mx.linalg.norm(left) * mx.linalg.norm(right), 1e-9
            )
            cosines.append(mx.sum(left * right) / denom)
    return cosines


def branch_diversity_penalty(
    forward: LivePathForward,
    *,
    target_cos: float = 0.98,
) -> tuple[Any, list[float]]:
    """Quadratic penalty above ``target_cos`` + telemetry cosines.

    Returns (differentiable penalty, detached per-pair cosines). A single
    branch yields a zero penalty and empty telemetry — diversity is only a
    demand where width exists to diversify.
    """
    import mlx.core as mx

    target = _validate_unit_weight("diversity_target_cos", target_cos, upper=1.0)
    cosines = _pairwise_state_cosines(forward)
    if not cosines:
        return mx.zeros(()), []
    penalty = mx.zeros(())
    telemetry: list[float] = []
    for cosine in cosines:
        excess = mx.maximum(cosine - target, 0.0)
        penalty = penalty + excess * excess
    telemetry = [float(value) for value in cosines]
    return penalty / len(cosines), telemetry


def depth_curriculum_loss_v3(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depths: tuple[int, ...] = (1, 2, 4),
    monotonicity_weight: float = 0.5,
    depth_margin: float = 0.05,
    diversity_weight: float = 0.25,
    diversity_target_cos: float = 0.98,
    bridge_tokens: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    """v3 composite: mean answer CE over the depth ladder
    + margin-bearing monotonic hinge + branch-diversity penalty.

    Returns ``(loss, telemetry)`` where telemetry carries detached floats
    (per-depth CE, per-depth post-exchange pairwise cosines, penalty
    values) for the trainer's step receipt — every step's diversity and
    depth structure is auditable after the fact.
    """
    if (
        len(depths) < 2
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or tuple(sorted(set(depths))) != depths
    ):
        raise ValueError("depths must be a strictly increasing tuple")
    weight = _validate_unit_weight("monotonicity_weight", monotonicity_weight)
    diversity_scale = _validate_unit_weight("diversity_weight", diversity_weight)

    losses: list[Any] = []
    diversity_penalties: list[Any] = []
    telemetry_cosines: dict[str, list[float]] = {}
    for depth in depths:
        forward = live_path_forward(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec.with_depth(depth),
            bridge_tokens=bridge_tokens,
        )
        losses.append(branch_mean_answer_loss(forward, answer_tokens))
        penalty, cosines = branch_diversity_penalty(
            forward, target_cos=diversity_target_cos
        )
        diversity_penalties.append(penalty)
        telemetry_cosines[str(depth)] = [round(value, 6) for value in cosines]

    margin_penalty = depth_margin_penalty(losses, margin=depth_margin)
    diversity_penalty = sum(diversity_penalties) / len(diversity_penalties)
    loss = (
        sum(losses) / len(losses)
        + weight * margin_penalty
        + diversity_scale * diversity_penalty
    )
    telemetry = {
        "schema": RECURRENCE_NATIVE_SCHEMA_V3,
        "depth_losses": [round(float(value), 6) for value in losses],
        "depth_margin": float(depth_margin),
        "margin_penalty": round(float(margin_penalty), 6),
        "diversity_weight": float(diversity_scale),
        "diversity_target_cos": float(diversity_target_cos),
        "diversity_penalty": round(float(diversity_penalty), 6),
        "post_exchange_pairwise_cos": telemetry_cosines,
    }
    return loss, telemetry


__all__ = [
    "RECURRENCE_NATIVE_SCHEMA_V3",
    "branch_diversity_penalty",
    "depth_curriculum_loss_v3",
    "depth_margin_penalty",
]
