"""Recurrence-native training objective v4 (CP209).

v3 shipped a branch-diversity term that could not work, and the live
CP195 run proved it: at the observed operating point (pairwise cosine
0.9998, answer CE 0.263) the penalty contributed 9.8e-5 — **0.037% of the
loss**, gradient 0.0099. Branch collapse was not a malfunction; it was the
optimum of the objective as posed. Three separate defects caused it:

1. **Vanishing penalty shape.** ``(max(cos - target, 0))**2`` goes to zero
   quadratically exactly in the regime that needs pressure. v4 penalizes a
   LINEAR hinge on normalized separation, whose gradient is constant.

2. **Ill-conditioned and partly self-inflicted metric.** Raw cosine over
   the flattened workspace is dominated by the shared prompt-derived bulk
   every branch is seeded from, and it *included the communication slot* —
   which ``_exchange_and_decorrelate`` deliberately drags toward consensus.
   v4 measures **normalized separation** ``||f_i - f_j|| / mean||f||`` over
   the NON-communication slots. For equal norms separation is
   ``sqrt(2(1-cos))``, which expands the collapse region ~10x (cos 0.9998
   -> 0.020; cos 0.98 -> 0.200). Centering is deliberately NOT used: with
   two branches the deviation vectors are antiparallel by construction, so
   a centered cosine is identically -1 and carries no signal.

3. **No reason to differ.** ``branch_mean_answer_loss`` trains every branch
   on the same answer tokens, so a ``counterexample_search`` branch is
   optimized to emit exactly what ``constructive_solution`` emits. But the
   engine SELECTS one winning branch at inference. Training the mean is a
   train/inference mismatch that rewards identical generalists. v4 trains a
   **softmin** over branch losses — gradient concentrates on whichever
   branch is currently best, matching selection semantics and leaving the
   others free to specialize — with an MoE-style **load-balancing** term so
   no branch is permanently starved into drift.

Everything here is differentiable through the SAME live-path forward as
v2/v3; only the loss composition changes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v2 import (
    LivePathForward,
    _advance_recurrent_states,
    _persist_and_score,
    _prepare_live_path,
    live_path_forward,
)

RECURRENCE_NATIVE_SCHEMA_V4 = "aura.recurrence_native_objective.v4"

# Separation below which branches are treated as collapsed. 0.30 separation
# corresponds to pairwise cosine ~0.955 for equal-norm states — genuinely
# distinct trajectories, not the 0.9998 the v3 run settled into.
DEFAULT_TARGET_SEPARATION = 0.30


def _validate_scalar(name: str, value: Any, *, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= float(value) <= high
    ):
        raise ValueError(f"{name} must be inside [{low}, {high}]")
    return float(value)


def _diversity_slots(state: Any, comm_slot: int) -> Any:
    """Drop the communication slot: exchange homogenizes it BY DESIGN.

    Including it conflates intended consensus with unintended collapse —
    a defect in the v3 metric that inflated every reading.
    """
    import mlx.core as mx

    slot_count = state.shape[1]
    if slot_count <= 1:
        return state
    if not 0 <= comm_slot < slot_count:
        return state
    keep = [index for index in range(slot_count) if index != comm_slot]
    return mx.take(state, mx.array(keep), axis=1)


def pairwise_separations(
    forward: LivePathForward,
    *,
    comm_slot: int = 0,
) -> list[Any]:
    """Normalized separations ``||f_i - f_j|| / mean||f||`` per branch pair.

    Well-conditioned where cosine is not: identical states give 0, and the
    near-collapse region that cosine compresses into [0.98, 1.0] is spread
    across [0, 0.2].
    """
    import mlx.core as mx

    flats = [
        mx.reshape(_diversity_slots(state, comm_slot), (-1,))
        for state in forward.branch_states
    ]
    separations: list[Any] = []
    for left_index in range(len(flats)):
        for right_index in range(left_index + 1, len(flats)):
            left, right = flats[left_index], flats[right_index]
            scale = mx.maximum(
                0.5 * (mx.linalg.norm(left) + mx.linalg.norm(right)), 1e-9
            )
            separations.append(mx.linalg.norm(left - right) / scale)
    return separations


def branch_decorrelation_penalty(
    forward: LivePathForward,
    *,
    target_separation: float = DEFAULT_TARGET_SEPARATION,
    comm_slot: int = 0,
) -> tuple[Any, list[float]]:
    """Linear hinge BELOW the target separation; constant gradient.

    Returns ``(penalty, detached separations)``. A single branch makes no
    diversity demand. Unlike v3's quadratic, the gradient does not decay as
    the collapsed state is approached.
    """
    import mlx.core as mx

    target = _validate_scalar(
        "target_separation", target_separation, low=0.0, high=2.0
    )
    separations = pairwise_separations(forward, comm_slot=comm_slot)
    if not separations:
        return mx.zeros(()), []
    penalty = mx.zeros(())
    for separation in separations:
        penalty = penalty + mx.maximum(target - separation, 0.0)
    return penalty / len(separations), [
        float(separation) for separation in separations
    ]


def softmin_answer_loss(
    forward: LivePathForward,
    answer_tokens: Sequence[int],
    *,
    temperature: float = 0.5,
) -> tuple[Any, list[float], list[float]]:
    """Selection-matched answer loss: gradient concentrates on the best branch.

    The engine picks ONE winning branch at inference; training the mean
    optimizes every branch toward the same generalist function, which is
    precisely what collapsed virtual width. Softmin weights (softmax of
    ``-loss/temperature``) give most gradient to whichever branch is
    currently best while keeping the others alive, so specialization is
    permitted instead of penalized. Temperature -> 0 approaches hard
    best-of; large temperature approaches the v2/v3 mean.

    Returns ``(loss, detached per-branch losses, detached weights)``.
    """
    import mlx.core as mx
    import mlx.nn as nn

    tau = _validate_scalar("temperature", temperature, low=0.01, high=10.0)
    targets = mx.array(list(answer_tokens))[None, :]
    losses = [
        nn.losses.cross_entropy(logits, targets, reduction="mean")
        for logits in forward.branch_logits
    ]
    if len(losses) == 1:
        return losses[0], [float(losses[0])], [1.0]
    stacked = mx.stack(losses)
    weights = mx.softmax(-stacked / tau, axis=0)
    # Weights are treated as a selection decision, not a gradient path:
    # differentiating through them would reward inflating other branches'
    # losses to win weight, which is the opposite of the intent.
    weights = mx.stop_gradient(weights)
    loss = mx.sum(weights * stacked)
    return (
        loss,
        [float(value) for value in losses],
        [float(value) for value in weights],
    )


def load_balance_penalty(weights: Sequence[float], branch_count: int) -> Any:
    """MoE-style aux loss: keep any branch from being permanently starved.

    Softmin alone can let one branch win every sample, leaving the rest
    without gradient until they drift into uselessness. Penalizing squared
    deviation from uniform usage keeps every role trained.
    """
    import mlx.core as mx

    if branch_count < 2 or not weights:
        return mx.zeros(())
    uniform = 1.0 / branch_count
    return mx.array(
        sum((float(weight) - uniform) ** 2 for weight in weights) / branch_count
    )


def trajectory_answer_losses(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int,
    probe_steps: Sequence[int] | None = None,
    bridge_tokens: Sequence[int] = (),
) -> tuple[list[Any], list[Any]]:
    """Answer CE decoded from intermediate states WITHIN one trajectory.

    Every earlier objective scored only the final state, or compared
    SEPARATE runs at different depths. Neither can see what the state does
    step to step, which is how a period-2 limit cycle went unnoticed:
    measured on the untrained 1.5B at depth 16, answer CE alternates and
    both parities degrade monotonically (khop odd steps 2.00->2.25, even
    steps 1.92->2.12), so the best answer arrives at step 2 and every step
    after that makes it worse.

    This returns the per-step answer losses and the per-step states, which
    is the only signal that can force later steps to think rather than
    spin. Probing every step costs one persist+score per step, so
    ``probe_steps`` selects a subset for large models.

    Returns ``(losses at probe steps, states at probe steps)``.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be a positive integer")
    wanted = (
        sorted({int(step) for step in probe_steps})
        if probe_steps is not None
        else list(range(1, depth + 1))
    )
    if not wanted or any(step < 1 or step > depth for step in wanted):
        raise ValueError("probe_steps must be inside [1, depth]")

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec.with_depth(depth),
        bridge_tokens=tuple(bridge_tokens),
    )
    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    losses: list[Any] = []
    probed: list[Any] = []
    for step in range(depth):
        states = _advance_recurrent_states(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            spec.with_depth(depth),
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        if (step + 1) not in wanted:
            continue
        logits = _persist_and_score(
            model,
            prepared.prompt_embeddings,
            prepared.seeds[0],
            states[0],
            prepared.tail_embeddings,
            bridge_count=prepared.bridge_count,
            answer_count=prepared.answer_count,
            prelude_end=prepared.prelude_end,
            coda_start=prepared.coda_start,
        )
        losses.append(nn.losses.cross_entropy(logits, targets, reduction="mean"))
        probed.append(states[0])
    return losses, probed


def monotone_improvement_penalty(
    step_losses: Sequence[Any],
    *,
    margin: float = 0.02,
) -> Any:
    """Require each probed step to beat the previous one by ``margin``.

    This is the objective that separates thinking from spinning. The
    previous step is gradient-detached so the optimizer cannot satisfy the
    constraint by making EARLIER steps worse -- the only way through is to
    make later steps genuinely better.
    """
    import mlx.core as mx

    if len(step_losses) < 2:
        return mx.zeros(())
    step_margin = _validate_scalar("margin", margin, low=0.0, high=2.0)
    penalty = mx.zeros(())
    for previous, current in zip(step_losses, step_losses[1:]):
        penalty = penalty + mx.maximum(
            current - mx.stop_gradient(previous) + step_margin, 0.0
        )
    return penalty / (len(step_losses) - 1)


def oscillation_penalty(states: Sequence[Any]) -> tuple[Any, list[float]]:
    """Penalize period-2 ping-pong in the state trajectory.

    Measured signature of the failure: successive update directions are
    anti-correlated, so the state alternates between two phases instead of
    advancing. Penalizing negative cosine between consecutive deltas
    targets exactly that, leaving consistent motion (cos > 0) free.

    Returns ``(penalty, detached consecutive-delta cosines)``.
    """
    import mlx.core as mx

    if len(states) < 3:
        return mx.zeros(()), []
    deltas = [
        mx.reshape(current - previous, (-1,))
        for previous, current in zip(states, states[1:])
    ]
    penalty = mx.zeros(())
    cosines: list[float] = []
    for previous, current in zip(deltas, deltas[1:]):
        denominator = mx.maximum(
            mx.linalg.norm(previous) * mx.linalg.norm(current), 1e-9
        )
        cosine = mx.sum(previous * current) / denominator
        penalty = penalty + mx.maximum(-cosine, 0.0)
        cosines.append(float(cosine))
    return penalty / len(cosines), cosines


def adaptive_depth_loss(
    depth_losses: Sequence[Any],
    depths: Sequence[int],
    *,
    compute_price: float = 0.01,
    temperature: float = 0.15,
) -> tuple[Any, list[float], int]:
    """Compute-priced depth SELECTION instead of forced monotone improvement.

    Measured on the untrained resident-class base (1.5B probe, task depth 8,
    3 samples per family), recurrent depth 1 -> 8 moves answer CE by:

        khop      -33.1%   (monotone in every sample)
        boolean    -6.9%
        register   -6.1%
        code       -4.3%
        modular   **+15.1%** (ANTI-monotone in every sample)

    Depth is not uniformly good; on modular arithmetic it is reliably
    destructive. v2/v3's hinge nonetheless demanded ``deep <= shallow -
    margin`` on EVERY sample, so the modular cells were permanently in
    violation. The cheapest way for the optimizer to satisfy a constraint
    it cannot win is to drive the recurrent transformation toward the
    identity — which also destroys the -33% khop signal. That conflict,
    not the width term, is the primary reason recurrence trains itself
    inert.

    This objective prices depth instead of mandating it:
    ``softmin_tau(CE(d) + compute_price * d)``. Gradient flows to whichever
    depth actually pays for the sample, so khop learns to use depth and
    modular learns to stay shallow — the adaptive-computation behavior the
    architecture exists to provide, rather than one compromise policy
    averaged over families that disagree.

    Returns ``(loss, detached priced costs, selected depth)``.
    """
    import mlx.core as mx

    if len(depth_losses) != len(depths) or not depths:
        raise ValueError("depth_losses and depths must align and be non-empty")
    price = _validate_scalar("compute_price", compute_price, low=0.0, high=1.0)
    tau = _validate_scalar("temperature", temperature, low=0.01, high=10.0)
    if len(depths) == 1:
        return depth_losses[0], [float(depth_losses[0])], int(depths[0])
    priced = [
        loss + price * float(depth)
        for loss, depth in zip(depth_losses, depths, strict=True)
    ]
    stacked = mx.stack(priced)
    weights = mx.stop_gradient(mx.softmax(-stacked / tau, axis=0))
    loss = mx.sum(weights * stacked)
    detached = [float(value) for value in priced]
    selected = int(depths[detached.index(min(detached))])
    return loss, detached, selected


def depth_curriculum_loss_v4(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depths: tuple[int, ...] = (1, 2, 4, 8),
    compute_price: float = 0.01,
    depth_temperature: float = 0.15,
    diversity_weight: float = 1.0,
    target_separation: float = DEFAULT_TARGET_SEPARATION,
    softmin_temperature: float = 0.5,
    load_balance_weight: float = 0.1,
    bridge_tokens: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    """v4 composite: softmin answer CE over the depth ladder + margin hinge
    + separation-based decorrelation + load balancing.

    Returns ``(loss, telemetry)``; telemetry carries detached per-depth
    losses, per-branch losses and selection weights, and pairwise
    separations, so a training step's width behavior is auditable rather
    than inferred.
    """

    if (
        len(depths) < 2
        or any(type(depth) is not int or depth < 1 for depth in depths)
        or tuple(sorted(set(depths))) != depths
    ):
        raise ValueError("depths must be a strictly increasing tuple")
    diversity_scale = _validate_scalar(
        "diversity_weight", diversity_weight, low=0.0, high=10.0
    )
    balance_scale = _validate_scalar(
        "load_balance_weight", load_balance_weight, low=0.0, high=10.0
    )

    answer_losses: list[Any] = []
    diversity_penalties: list[Any] = []
    balance_penalties: list[Any] = []
    telemetry_separations: dict[str, list[float]] = {}
    telemetry_branch_losses: dict[str, list[float]] = {}
    telemetry_weights: dict[str, list[float]] = {}
    for depth in depths:
        forward = live_path_forward(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec.with_depth(depth),
            bridge_tokens=bridge_tokens,
        )
        loss, branch_losses, weights = softmin_answer_loss(
            forward, answer_tokens, temperature=softmin_temperature
        )
        answer_losses.append(loss)
        penalty, separations = branch_decorrelation_penalty(
            forward,
            target_separation=target_separation,
            comm_slot=spec.comm_slot,
        )
        diversity_penalties.append(penalty)
        balance_penalties.append(
            load_balance_penalty(weights, len(forward.branch_states))
        )
        key = str(depth)
        telemetry_separations[key] = [round(v, 6) for v in separations]
        telemetry_branch_losses[key] = [round(v, 6) for v in branch_losses]
        telemetry_weights[key] = [round(v, 6) for v in weights]

    # Compute-priced depth SELECTION, not a uniform improvement mandate.
    # The measured family split (khop -33.1% monotone, modular +15.1%
    # anti-monotone) makes a global hinge unwinnable on modular cells, and
    # the cheapest escape is to neutralize recurrence everywhere. Selection
    # lets each sample pick the depth that actually pays.
    depth_selected_loss, priced, selected_depth = adaptive_depth_loss(
        answer_losses,
        depths,
        compute_price=compute_price,
        temperature=depth_temperature,
    )
    diversity_penalty = sum(diversity_penalties) / len(diversity_penalties)
    balance_penalty = sum(balance_penalties) / len(balance_penalties)
    loss = (
        depth_selected_loss
        + diversity_scale * diversity_penalty
        + balance_scale * balance_penalty
    )
    telemetry = {
        "schema": RECURRENCE_NATIVE_SCHEMA_V4,
        "depth_losses": [round(float(v), 6) for v in answer_losses],
        "selected_depth": selected_depth,
        "priced_depth_costs": [round(value, 6) for value in priced],
        "compute_price": float(compute_price),
        "diversity_penalty": round(float(diversity_penalty), 6),
        "diversity_weight": diversity_scale,
        "target_separation": float(target_separation),
        "load_balance_penalty": round(float(balance_penalty), 6),
        "pairwise_separation": telemetry_separations,
        "branch_losses": telemetry_branch_losses,
        "branch_weights": telemetry_weights,
    }
    return loss, telemetry


__all__ = [
    "DEFAULT_TARGET_SEPARATION",
    "RECURRENCE_NATIVE_SCHEMA_V4",
    "branch_decorrelation_penalty",
    "depth_curriculum_loss_v4",
    "load_balance_penalty",
    "pairwise_separations",
    "monotone_improvement_penalty",
    "oscillation_penalty",
    "softmin_answer_loss",
    "trajectory_answer_losses",
    "trajectory_loss_v4",
]


def trajectory_loss_v4(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int = 8,
    probe_steps: Sequence[int] | None = None,
    improvement_weight: float = 1.0,
    improvement_margin: float = 0.02,
    oscillation_weight: float = 0.5,
    final_weight: float = 1.0,
    bridge_tokens: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    """The objective that trains recurrence to THINK rather than spin.

    Three terms, each answering a measured failure:

    * ``final``       — the endpoint answer must be good (what v2/v3 had).
    * ``improvement`` — every probed step must beat the previous one by a
      margin. Measured failure: answer CE peaks at step 2 and degrades
      monotonically thereafter (khop best->last +10.3%), so extra depth was
      cost without cognition.
    * ``oscillation`` — consecutive update directions must not be
      anti-correlated. Measured failure: a period-2 limit cycle in which
      the state ping-pongs between two phases while both degrade.

    Together these make "later steps produce measurably better answers" a
    trained property instead of an assumption. Telemetry returns the full
    per-step curve so the property is auditable every step, not inferred.
    """

    improve_scale = _validate_scalar(
        "improvement_weight", improvement_weight, low=0.0, high=10.0
    )
    oscillate_scale = _validate_scalar(
        "oscillation_weight", oscillation_weight, low=0.0, high=10.0
    )
    final_scale = _validate_scalar(
        "final_weight", final_weight, low=0.0, high=10.0
    )
    step_losses, states = trajectory_answer_losses(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec,
        depth=depth,
        probe_steps=probe_steps,
        bridge_tokens=bridge_tokens,
    )
    improvement = monotone_improvement_penalty(
        step_losses, margin=improvement_margin
    )
    oscillation, cosines = oscillation_penalty(states)
    final = step_losses[-1]
    loss = (
        final_scale * final
        + improve_scale * improvement
        + oscillate_scale * oscillation
    )
    detached = [float(value) for value in step_losses]
    improving = sum(
        1 for index in range(1, len(detached)) if detached[index] < detached[index - 1]
    )
    telemetry = {
        "schema": RECURRENCE_NATIVE_SCHEMA_V4,
        "depth": int(depth),
        "step_losses": [round(value, 6) for value in detached],
        "best_step_index": int(min(range(len(detached)), key=detached.__getitem__)),
        "improving_steps": improving,
        "probed_steps": len(detached),
        "improvement_penalty": round(float(improvement), 6),
        "oscillation_penalty": round(float(oscillation), 6),
        "delta_cosines": [round(value, 6) for value in cosines],
        "final_loss": round(float(final), 6),
    }
    return loss, telemetry
