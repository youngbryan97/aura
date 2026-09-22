"""SPARK-061: train later latent states to be BETTER, and prove it was real.

The objective half of this item already exists — v4's `trajectory_loss_v4`
asks each probed step to beat the previous one by a margin, with the previous
step gradient-detached so the constraint cannot be satisfied by making earlier
steps worse. What does not exist is the part that decides whether a descending
loss curve means the thing the curve is supposed to mean.

v4's own docstring records the trap, measured on the untrained resident-class
base: depth is not uniformly good (khop -33.1% monotone, modular **+15.1%
anti-monotone**), so a global improvement mandate is unwinnable on some
families, and

    the cheapest way for the optimizer to satisfy a constraint it cannot win
    is to drive the recurrent transformation toward the identity.

An identity operator satisfies "later states do not get worse" perfectly. Every
step loss is equal, the hinge is silent, the loss curve descends because the
answer head is still learning, and recurrence is dead.

That claim has been carried as a citation for several checkpoints, so
`tools/measure_progressive_collapse.py` measured it. The recurrent update is
``z' = (1-alpha)*z + alpha*RMSMatch(window(z), anchor)``, so ``alpha -> 0`` *is*
the identity operator, exactly and continuously; sweeping alpha walks the
objective along the collapse axis. On the untrained Qwen2.5-1.5B at depth 4 over
khop and modular tasks (receipt under
`artifacts/closeout/latent_cortex/spark061_progressive_objective/`), the result
**refines rather than confirms** the inherited claim:

* Collapse is **not** the global optimum. v4's objective is minimized at the
  honest end (alpha 0.5, loss 2.980).
* Collapse **is a local basin**. alpha 0.01 scores 3.3435, and the two steps
  toward real motion cost **+0.1446** and **+0.0455** before the loss falls
  away — a ~0.19-nat barrier. An optimizer that drifts into low motion is
  trapped there, which is the falsifiable form of the concern.
* The barrier is worst exactly where the state has stopped moving: at alpha
  0.01 the mean per-step displacement is 0.0117.

The same measurement found a defect in this module's own first calibration: a
detector floor of 0.01 does not fire at 0.0117, so the penalty was exactly zero
at the local minimum it exists to remove. Detecting collapse and pricing it out
are different jobs with different constants, and the pricing constants are
operating-point specific — the resident 32B will not share the 1.5B's
displacement scale. `solve_collapse_barrier` therefore derives them from
measured data instead of assuming them; on the 1.5B it returns weight 4.0 and
floor 0.103, which makes the penalized objective strictly decreasing in alpha
with no basin left.

So this module is an *instrument with the authority to refuse*, plus the loss
term that closes the escape hatch it detects.

Four ways a progressive claim can be false, each with its own measurement:

* **Identity collapse.** The operator stops moving the state. Measured as
  per-step normalized displacement ‖z_t − z_{t−1}‖ / ‖z_{t−1}‖. Below the
  floor, "improvement" is vacuous no matter what the losses did.
* **Early-step sabotage.** Later steps look better because step 1 was made
  worse. v4's `stop_gradient` blocks the *gradient* path to that solution but
  measures nothing, so it cannot see the same outcome arriving through the
  auxiliary terms or a curriculum change. Measured against an independent
  depth-1 reference loss.
* **Length confound.** "Not merely imitate long solutions": improvement that
  tracks answer length across a batch is a style effect. Measured as the
  Pearson correlation between per-task improvement and answer token count.
* **Causal idleness.** The honest form of "verify monotonic quality" is not
  that the numbers descend — it is that removing a step *hurts*. Each probed
  step's update is replaced by the identity, the trajectory is completed, and
  the final answer CE is re-measured. A step whose removal costs nothing did
  nothing, whatever the curve says.

Plus the gradient leg SPARK-061 names explicitly. `state_gradient_norms`
measures the norm of ∂(final answer CE)/∂z_t for each probed step — the actual
backpropagated signal reaching that depth. Exponential decay backwards is a
vanishing-gradient chain that cannot train its early steps; explosive growth is
the chain that will not survive a real learning rate.

`progressive_objective_loss` is the trainable form. It keeps v4's improvement
and oscillation terms and adds the **displacement floor**: a linear hinge that
charges the optimizer for letting the operator go quiet. The collapse solution
stops being free, which is the only thing that makes the improvement term mean
what it says.

Nothing here runs a capability campaign or grants a capability claim. It
decides whether one is worth running.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec
from core.learning.recurrence_native_objective_v2 import (
    _advance_recurrent_states,
    _persist_and_score,
    _prepare_live_path,
)
from core.learning.recurrence_native_objective_v4 import (
    monotone_improvement_penalty,
    oscillation_penalty,
)

PROGRESSIVE_OBJECTIVE_SCHEMA = "aura.spark061.progressive_recurrent_objective.v1"
PROGRESSIVE_TRAJECTORY_SET_SCHEMA = "aura.spark061.progressive_trajectory_set.v1"

# DETECTOR floor: normalized per-step displacement below which a *report*
# calls the operator collapsed. Calibrated against the contraction CP210
# measured on the untrained 1.5B (residual 0.302 -> 0.026 while still
# computing), so it sits below the asymptote of a working contraction and above
# numerical noise: it flags a dead operator without flagging convergence.
#
# This is deliberately NOT the training floor. See the module docstring: at the
# measured local basin the displacement is 0.0117, above this value, so using
# it to *price* collapse produces a penalty of exactly zero where the pressure
# is needed. Use `solve_collapse_barrier` for training constants.
DEFAULT_DISPLACEMENT_FLOOR = 0.01

# Training constants derived by `solve_collapse_barrier` from the sweep on the
# untrained Qwen2.5-1.5B at depth 4 (khop + modular, 6 tasks). They are
# recorded as a REFERENCE, not a default: the solver must be re-run against the
# resident model's own displacement scale before a 32B campaign, because a
# floor calibrated on a 1.5B has no claim on a 32B's activation geometry.
MEASURED_1P5B_COLLAPSE_SOLUTION = {
    "model": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "depth": 4,
    "weight": 4.0,
    "floor": 0.103043,
    "barrier_removed": True,
}

# A step must change the final answer CE by at least this much (in nats) for
# its update to count as causally necessary. Below it, the step is idle.
DEFAULT_NECESSITY_MARGIN = 1e-3

# |Pearson r| between per-task improvement and answer length above which the
# improvement is reported as length-confounded rather than clean.
DEFAULT_LENGTH_CONFOUND_LIMIT = 0.5

# Ratio between the largest and smallest per-step state-gradient norm beyond
# which the unrolled chain is reported as vanishing or exploding.
DEFAULT_GRADIENT_RATIO_LIMIT = 100.0

VERDICTS = (
    "real_progress",
    "degenerate_identity_collapse",
    "degenerate_early_sabotage",
    "degenerate_length_confound",
    "causally_idle",
    "vacuous_no_improvement",
    "unmeasured",
)


class ProgressiveObjectiveError(ValueError):
    """A progressive-objective claim could not be measured or does not hold."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _round(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    # not a failure: a value that is not a number is not one this can read.
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _validate_scalar(name: str, value: Any, *, low: float, high: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not low <= float(value) <= high
    ):
        raise ProgressiveObjectiveError(f"{name} must be a finite number inside [{low}, {high}]")
    return float(value)


def _probe_steps(depth: int, probe_steps: Sequence[int] | None) -> list[int]:
    if type(depth) is not int or not 1 <= depth <= 64:
        raise ProgressiveObjectiveError("depth must be an integer inside [1, 64]")
    wanted = (
        sorted({int(step) for step in probe_steps})
        if probe_steps is not None
        else list(range(1, depth + 1))
    )
    if not wanted or any(step < 1 or step > depth for step in wanted):
        raise ProgressiveObjectiveError("probe_steps must be inside [1, depth]")
    return wanted


@dataclass(frozen=True)
class ProgressiveTrajectory:
    """One measured trajectory: what each step cost and how far it moved.

    ``step_losses`` and ``displacements`` are aligned to ``probe_steps``.
    ``displacements`` are normalized (‖Δz‖ / ‖z_prev‖), so they are comparable
    across models and depths; a raw norm would make the floor model-specific.
    """

    depth: int
    probe_steps: tuple[int, ...]
    step_losses: tuple[float, ...]
    displacements: tuple[float, ...]
    anchor_drifts: tuple[float, ...]
    answer_token_count: int

    def __post_init__(self) -> None:
        if type(self.depth) is not int or self.depth < 1:
            raise ProgressiveObjectiveError("trajectory depth must be positive")
        count = len(self.probe_steps)
        if (
            count < 1
            or len(self.step_losses) != count
            or len(self.displacements) != count
            or len(self.anchor_drifts) != count
            or tuple(sorted(set(self.probe_steps))) != self.probe_steps
            or any(step < 1 or step > self.depth for step in self.probe_steps)
        ):
            raise ProgressiveObjectiveError(
                "trajectory probes, losses, displacement, and drift must align"
            )
        if type(self.answer_token_count) is not int or self.answer_token_count < 1:
            raise ProgressiveObjectiveError("answer token count must be positive")
        if any(
            not math.isfinite(float(value))
            for values in (
                self.step_losses,
                self.displacements,
                self.anchor_drifts,
            )
            for value in values
        ):
            raise ProgressiveObjectiveError("trajectory measurements must be finite")

    @property
    def improvement(self) -> float:
        """First probed loss minus last: positive means later states are better."""
        if len(self.step_losses) < 2:
            return 0.0
        return self.step_losses[0] - self.step_losses[-1]

    @property
    def best_step_index(self) -> int:
        return min(range(len(self.step_losses)), key=self.step_losses.__getitem__)

    @property
    def min_displacement(self) -> float:
        return min(self.displacements) if self.displacements else 0.0

    def to_dict(self) -> dict[str, Any]:
        step_losses = [round(value, 6) for value in self.step_losses]
        displacements = [round(value, 6) for value in self.displacements]
        anchor_drifts = [round(value, 6) for value in self.anchor_drifts]
        return {
            "depth": self.depth,
            "probe_steps": list(self.probe_steps),
            "step_losses": step_losses,
            "displacements": displacements,
            "anchor_drifts": anchor_drifts,
            "answer_token_count": self.answer_token_count,
            "improvement": round(step_losses[0] - step_losses[-1], 6),
            "best_step_index": min(range(len(step_losses)), key=step_losses.__getitem__),
            "min_displacement": min(displacements),
        }


@dataclass(frozen=True)
class ProgressiveTrajectorySet:
    """Branch-complete measurement of one exchange-coupled recurrent graph."""

    execution_spec_sha256: str
    branch_roles: tuple[str, ...]
    trajectories: tuple[ProgressiveTrajectory, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.execution_spec_sha256, str)
            or len(self.execution_spec_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.execution_spec_sha256)
        ):
            raise ProgressiveObjectiveError("execution spec digest is invalid")
        if (
            not self.branch_roles
            or len(set(self.branch_roles)) != len(self.branch_roles)
            or any(not isinstance(role, str) or not role for role in self.branch_roles)
        ):
            raise ProgressiveObjectiveError("branch roles must be unique identifiers")
        if len(self.branch_roles) != len(self.trajectories):
            raise ProgressiveObjectiveError("branch roles and progressive trajectories must align")
        if not self.trajectories:
            raise ProgressiveObjectiveError("trajectory set cannot be empty")
        reference = self.trajectories[0]
        for trajectory in self.trajectories:
            if (
                trajectory.depth != reference.depth
                or trajectory.probe_steps != reference.probe_steps
                or trajectory.answer_token_count != reference.answer_token_count
            ):
                raise ProgressiveObjectiveError(
                    "all branch trajectories must share depth, probes, and answer"
                )

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema": PROGRESSIVE_TRAJECTORY_SET_SCHEMA,
            "execution_spec_sha256": self.execution_spec_sha256,
            "branch_roles": list(self.branch_roles),
            "branch_count": len(self.branch_roles),
            "trajectories": [
                {
                    "branch_index": index,
                    "branch_role": role,
                    **trajectory.to_dict(),
                }
                for index, (role, trajectory) in enumerate(
                    zip(self.branch_roles, self.trajectories, strict=True)
                )
            ],
        }
        return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_progressive_trajectory_set(value: Any) -> dict[str, Any]:
    """Validate and replay the canonical branch-complete measurement receipt."""

    if not isinstance(value, dict):
        raise ProgressiveObjectiveError("trajectory-set receipt must be a mapping")
    required = {
        "schema",
        "execution_spec_sha256",
        "branch_roles",
        "branch_count",
        "trajectories",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ProgressiveObjectiveError("trajectory-set receipt fields do not match")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ProgressiveObjectiveError("trajectory-set receipt commitment mismatch")
    if value["schema"] != PROGRESSIVE_TRAJECTORY_SET_SCHEMA:
        raise ProgressiveObjectiveError("unsupported trajectory-set schema")
    roles = value["branch_roles"]
    rows = value["trajectories"]
    if (
        not isinstance(roles, list)
        or not isinstance(rows, list)
        or value["branch_count"] != len(roles)
        or len(rows) != len(roles)
    ):
        raise ProgressiveObjectiveError("trajectory-set branches do not align")
    trajectories: list[ProgressiveTrajectory] = []
    for index, (role, row) in enumerate(zip(roles, rows, strict=True)):
        if (
            not isinstance(row, dict)
            or row.get("branch_index") != index
            or row.get("branch_role") != role
        ):
            raise ProgressiveObjectiveError("trajectory branch identity is invalid")
        try:
            trajectory = ProgressiveTrajectory(
                depth=int(row["depth"]),
                probe_steps=tuple(int(step) for step in row["probe_steps"]),
                step_losses=tuple(float(item) for item in row["step_losses"]),
                displacements=tuple(float(item) for item in row["displacements"]),
                anchor_drifts=tuple(float(item) for item in row["anchor_drifts"]),
                answer_token_count=int(row["answer_token_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProgressiveObjectiveError("trajectory branch measurement is malformed") from exc
        expected = {
            "branch_index": index,
            "branch_role": role,
            **trajectory.to_dict(),
        }
        if row != expected:
            raise ProgressiveObjectiveError("trajectory branch summary does not replay")
        trajectories.append(trajectory)
    reconstructed = ProgressiveTrajectorySet(
        execution_spec_sha256=str(value["execution_spec_sha256"]),
        branch_roles=tuple(str(role) for role in roles),
        trajectories=tuple(trajectories),
    )
    if reconstructed.receipt() != value:
        raise ProgressiveObjectiveError("trajectory-set receipt does not replay canonically")
    return dict(value)


def _normalized_displacement(previous: Any, current: Any) -> float:
    import mlx.core as mx

    numerator = mx.linalg.norm(mx.reshape(current - previous, (-1,)))
    denominator = mx.maximum(mx.linalg.norm(mx.reshape(previous, (-1,))), 1e-9)
    return float(numerator / denominator)


def measure_progressive_trajectory(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int = 8,
    probe_steps: Sequence[int] | None = None,
    bridge_tokens: Sequence[int] = (),
) -> ProgressiveTrajectory:
    """Unroll the real live path and record loss AND motion at every step.

    v4's `trajectory_answer_losses` records only the losses, which is exactly
    the blind spot: an identity operator and a working operator produce
    different motion and can produce identical loss curves.
    """
    import mlx.core as mx
    import mlx.nn as nn

    wanted = _probe_steps(depth, probe_steps)
    if len(spec.branch_roles) != 1:
        raise ProgressiveObjectiveError(
            "progressive measurement requires a single-branch spec; width is "
            "SPARK-016/017's subject, and exchange would confound displacement"
        )
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec.with_depth(depth),
        bridge_tokens=tuple(bridge_tokens),
    )
    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    anchor = prepared.anchors[0]
    losses: list[float] = []
    displacements: list[float] = []
    drifts: list[float] = []
    for step in range(depth):
        previous = states[0]
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
        losses.append(float(nn.losses.cross_entropy(logits, targets, reduction="mean")))
        displacements.append(_normalized_displacement(previous, states[0]))
        drifts.append(_normalized_displacement(anchor, states[0]))
    return ProgressiveTrajectory(
        depth=depth,
        probe_steps=tuple(wanted),
        step_losses=tuple(losses),
        displacements=tuple(displacements),
        anchor_drifts=tuple(drifts),
        answer_token_count=len(list(answer_tokens)),
    )


def measure_progressive_trajectories(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int = 8,
    probe_steps: Sequence[int] | None = None,
    bridge_tokens: Sequence[int] = (),
) -> ProgressiveTrajectorySet:
    """Measure every branch inside one real exchange-coupled recurrent unroll."""

    import mlx.core as mx
    import mlx.nn as nn

    wanted = _probe_steps(depth, probe_steps)
    resolved = spec.with_depth(depth)
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=resolved,
        bridge_tokens=tuple(bridge_tokens),
    )
    if len(prepared.states) != len(spec.branch_roles):
        raise ProgressiveObjectiveError("prepared branch states do not match the execution spec")
    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    losses: list[list[float]] = [[] for _role in spec.branch_roles]
    displacements: list[list[float]] = [[] for _role in spec.branch_roles]
    drifts: list[list[float]] = [[] for _role in spec.branch_roles]
    for step in range(depth):
        previous = list(states)
        states = _advance_recurrent_states(
            model,
            prepared.prompts_at_window,
            states,
            prepared.anchors,
            resolved,
            step,
            prepared.prelude_end,
            prepared.coda_start,
        )
        if (step + 1) not in wanted:
            continue
        for index, state in enumerate(states):
            logits = _persist_and_score(
                model,
                prepared.prompt_embeddings,
                prepared.seeds[index],
                state,
                prepared.tail_embeddings,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
            mx.eval(loss)
            losses[index].append(float(loss))
            displacements[index].append(_normalized_displacement(previous[index], state))
            drifts[index].append(_normalized_displacement(prepared.anchors[index], state))
    return ProgressiveTrajectorySet(
        execution_spec_sha256=resolved.sha256,
        branch_roles=tuple(spec.branch_roles),
        trajectories=tuple(
            ProgressiveTrajectory(
                depth=depth,
                probe_steps=tuple(wanted),
                step_losses=tuple(losses[index]),
                displacements=tuple(displacements[index]),
                anchor_drifts=tuple(drifts[index]),
                answer_token_count=len(tuple(answer_tokens)),
            )
            for index in range(len(spec.branch_roles))
        ),
    )


def step_necessity(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int,
    lesion_steps: Sequence[int],
    bridge_tokens: Sequence[int] = (),
) -> dict[int, float]:
    """Replace one step's update with the identity and re-measure the answer.

    This is the causal form of "verify monotonic quality". A descending loss
    curve is consistent with steps that contribute nothing; a step whose
    removal leaves the final answer CE unchanged did not participate in
    producing that answer, and no amount of curve-watching reveals it.

    Returns ``{step: intact_loss - lesioned_loss}``. NEGATIVE means removing
    the step made the answer worse, i.e. the step was doing useful work.
    """
    import mlx.core as mx
    import mlx.nn as nn

    targets = mx.array(list(answer_tokens))[None, :]
    lesions = sorted({int(step) for step in lesion_steps})
    if not lesions or any(step < 1 or step > depth for step in lesions):
        raise ProgressiveObjectiveError("lesion_steps must be inside [1, depth]")

    def run(skip: int | None) -> float:
        prepared = _prepare_live_path(
            model,
            prompt_tokens,
            answer_tokens,
            spec=spec.with_depth(depth),
            bridge_tokens=tuple(bridge_tokens),
        )
        states = list(prepared.states)
        for step in range(depth):
            if skip is not None and step + 1 == skip:
                # Identity update: the step is executed as a no-op. Compute
                # parity is deliberately NOT preserved here — this measures
                # whether the transform mattered, not an equal-compute arm.
                continue
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
        return float(nn.losses.cross_entropy(logits, targets, reduction="mean"))

    intact = run(None)
    return {step: intact - run(step) for step in lesions}


def state_gradient_norms(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int,
    probe_steps: Sequence[int] | None = None,
    bridge_tokens: Sequence[int] = (),
) -> dict[int, float]:
    """‖∂(final answer CE)/∂z_t‖ for each probed step — the signal that arrives.

    SPARK-061 asks for "useful gradients at the resident architecture". The
    useful measurement is not the parameter-gradient norm (which mixes every
    depth together) but the sensitivity of the final loss to each intermediate
    state: that is exactly the quantity that decays in a vanishing chain, and
    a chain whose early steps receive nothing cannot train those steps no
    matter how long the campaign runs.
    """
    import mlx.core as mx
    import mlx.nn as nn

    wanted = _probe_steps(depth, probe_steps)
    targets = mx.array(list(answer_tokens))[None, :]
    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec.with_depth(depth),
        bridge_tokens=tuple(bridge_tokens),
    )
    # Forward once, retaining the state entering each probed step.
    entering: dict[int, Any] = {}
    states = list(prepared.states)
    for step in range(depth):
        if (step + 1) in wanted:
            entering[step + 1] = states[0]
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

    norms: dict[int, float] = {}
    for probe in wanted:

        def tail_loss(state: Any, start: int = probe) -> Any:
            current = [state]
            for step in range(start - 1, depth):
                current = _advance_recurrent_states(
                    model,
                    prepared.prompts_at_window,
                    current,
                    prepared.anchors,
                    spec.with_depth(depth),
                    step,
                    prepared.prelude_end,
                    prepared.coda_start,
                )
            logits = _persist_and_score(
                model,
                prepared.prompt_embeddings,
                prepared.seeds[0],
                current[0],
                prepared.tail_embeddings,
                bridge_count=prepared.bridge_count,
                answer_count=prepared.answer_count,
                prelude_end=prepared.prelude_end,
                coda_start=prepared.coda_start,
            )
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        gradient = mx.grad(tail_loss)(entering[probe])
        mx.eval(gradient)
        norms[probe] = float(mx.linalg.norm(mx.reshape(gradient, (-1,))))
    return norms


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    if variance_left <= 1e-12 or variance_right <= 1e-12:
        return None
    return covariance / math.sqrt(variance_left * variance_right)


def build_progressive_report(
    trajectories: Sequence[ProgressiveTrajectory],
    *,
    necessity: dict[int, float] | None = None,
    gradient_norms: dict[int, float] | None = None,
    depth_one_reference: float | None = None,
    displacement_floor: float = DEFAULT_DISPLACEMENT_FLOOR,
    necessity_margin: float = DEFAULT_NECESSITY_MARGIN,
    length_confound_limit: float = DEFAULT_LENGTH_CONFOUND_LIMIT,
    gradient_ratio_limit: float = DEFAULT_GRADIENT_RATIO_LIMIT,
    improvement_margin: float = 0.0,
) -> dict[str, Any]:
    """Decide what the measured trajectories actually support.

    The verdict order matters and is deliberate: a collapsed operator is
    reported as collapsed even when its losses descend beautifully, because
    the descent belongs to the answer head rather than to recurrence. Only a
    trajectory set that survives every degeneracy check earns
    ``real_progress``, and even that is a statement about the objective's
    behavior, never about task capability.
    """
    floor = _validate_scalar("displacement_floor", displacement_floor, low=0.0, high=1.0)
    margin = _validate_scalar("necessity_margin", necessity_margin, low=0.0, high=10.0)
    confound_limit = _validate_scalar(
        "length_confound_limit", length_confound_limit, low=0.0, high=1.0
    )
    ratio_limit = _validate_scalar("gradient_ratio_limit", gradient_ratio_limit, low=1.0, high=1e9)
    improve_margin = _validate_scalar("improvement_margin", improvement_margin, low=0.0, high=10.0)
    if not trajectories:
        raise ProgressiveObjectiveError("a report needs at least one trajectory")
    if any(len(item.step_losses) < 2 for item in trajectories):
        raise ProgressiveObjectiveError(
            "each trajectory needs at least two probed steps to have a trend"
        )

    improvements = [item.improvement for item in trajectories]
    lengths = [float(item.answer_token_count) for item in trajectories]
    mean_improvement = sum(improvements) / len(improvements)
    improved_count = sum(1 for value in improvements if value > improve_margin)
    min_displacement = min(item.min_displacement for item in trajectories)
    mean_displacement = sum(
        sum(item.displacements) / max(1, len(item.displacements)) for item in trajectories
    ) / len(trajectories)
    length_correlation = _pearson(improvements, lengths)

    necessity_rows = (
        [
            {"step": step, "delta": _round(value)}
            for step, value in sorted((necessity or {}).items())
        ]
        if necessity
        else []
    )
    # A step is necessary when removing it made the answer WORSE, i.e. the
    # intact loss was lower than the lesioned loss => delta < -margin.
    idle_steps = [
        row["step"] for row in necessity_rows if row["delta"] is None or row["delta"] > -margin
    ]
    gradient_rows = (
        [
            {"step": step, "norm": _round(value, 9)}
            for step, value in sorted((gradient_norms or {}).items())
        ]
        if gradient_norms
        else []
    )
    finite_norms = [
        row["norm"] for row in gradient_rows if row["norm"] is not None and row["norm"] > 0.0
    ]
    gradient_ratio = max(finite_norms) / min(finite_norms) if len(finite_norms) >= 2 else None
    gradient_health = "unmeasured"
    if gradient_rows and len(finite_norms) != len(gradient_rows):
        gradient_health = "dead_gradient"
    elif gradient_ratio is not None:
        gradient_health = "healthy" if gradient_ratio <= ratio_limit else "ill_conditioned"

    sabotage = (
        depth_one_reference is not None
        and math.isfinite(float(depth_one_reference))
        and any(
            item.step_losses[0] > float(depth_one_reference) + improve_margin
            for item in trajectories
        )
    )

    if min_displacement < floor:
        verdict = "degenerate_identity_collapse"
    elif sabotage:
        verdict = "degenerate_early_sabotage"
    elif (
        length_correlation is not None
        and abs(length_correlation) > confound_limit
        and mean_improvement > improve_margin
    ):
        verdict = "degenerate_length_confound"
    elif necessity_rows and len(idle_steps) == len(necessity_rows):
        verdict = "causally_idle"
    elif mean_improvement <= improve_margin:
        verdict = "vacuous_no_improvement"
    else:
        verdict = "real_progress"

    payload = {
        "schema": PROGRESSIVE_OBJECTIVE_SCHEMA,
        "verdict": verdict,
        "trajectory_count": len(trajectories),
        "trajectories": [item.to_dict() for item in trajectories],
        "mean_improvement": _round(mean_improvement),
        "improved_trajectories": improved_count,
        "min_displacement": _round(min_displacement),
        "mean_displacement": _round(mean_displacement),
        "displacement_floor": round(floor, 6),
        "length_correlation": _round(length_correlation),
        "length_confound_limit": round(confound_limit, 6),
        "necessity": necessity_rows,
        "necessity_margin": round(margin, 6),
        "idle_steps": idle_steps,
        "gradient_norms": gradient_rows,
        "gradient_ratio": _round(gradient_ratio, 6),
        "gradient_ratio_limit": round(ratio_limit, 6),
        "gradient_health": gradient_health,
        "depth_one_reference": _round(depth_one_reference),
        "improvement_margin": round(improve_margin, 6),
        "supports_training": verdict == "real_progress",
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_progressive_report(value: Any) -> dict[str, Any]:
    """Independently replay the verdict from the report's own measurements."""

    if not isinstance(value, dict):
        raise ProgressiveObjectiveError("progressive report must be a mapping")
    required = {
        "schema",
        "verdict",
        "trajectory_count",
        "trajectories",
        "mean_improvement",
        "improved_trajectories",
        "min_displacement",
        "mean_displacement",
        "displacement_floor",
        "length_correlation",
        "length_confound_limit",
        "necessity",
        "necessity_margin",
        "idle_steps",
        "gradient_norms",
        "gradient_ratio",
        "gradient_ratio_limit",
        "gradient_health",
        "depth_one_reference",
        "improvement_margin",
        "supports_training",
        "receipt_sha256",
    }
    if set(value) != required:
        raise ProgressiveObjectiveError("progressive report fields do not match")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise ProgressiveObjectiveError("progressive report commitment mismatch")
    if value["schema"] != PROGRESSIVE_OBJECTIVE_SCHEMA:
        raise ProgressiveObjectiveError("unsupported progressive report schema")
    if value["verdict"] not in VERDICTS:
        raise ProgressiveObjectiveError("unknown progressive verdict")
    if value["trajectory_count"] != len(value["trajectories"]):
        raise ProgressiveObjectiveError("trajectory count differs from rows")
    if value["supports_training"] is not (value["verdict"] == "real_progress"):
        raise ProgressiveObjectiveError("training support must follow the verdict exactly")

    # Rebuild the ENTIRE report from its own trajectory rows and require
    # equality. Checking only the summary fields was a real defect in the
    # first version of this validator: a forger who edited the trajectories
    # and the summary consistently, then resealed, passed every aggregate
    # check while the rows underneath said "collapsed". A commitment proves
    # the bytes were not altered after signing; it proves nothing about
    # whether the signer's arithmetic was honest, so the arithmetic is redone.
    try:
        rebuilt_trajectories = [
            ProgressiveTrajectory(
                depth=int(row["depth"]),
                probe_steps=tuple(int(step) for step in row["probe_steps"]),
                step_losses=tuple(float(item) for item in row["step_losses"]),
                displacements=tuple(float(item) for item in row["displacements"]),
                anchor_drifts=tuple(float(item) for item in row["anchor_drifts"]),
                answer_token_count=int(row["answer_token_count"]),
            )
            for row in value["trajectories"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProgressiveObjectiveError("progressive trajectory rows are malformed") from exc
    rebuilt = build_progressive_report(
        rebuilt_trajectories,
        # A recorded None is an unmeasured value, not an absent row. Feeding
        # NaN back preserves the row and re-normalizes to None, so an honest
        # report carrying an unmeasured step still round-trips exactly.
        necessity={
            int(row["step"]): (
                float(row["delta"]) if row.get("delta") is not None else float("nan")
            )
            for row in value["necessity"]
        }
        or None,
        gradient_norms={
            int(row["step"]): (float(row["norm"]) if row.get("norm") is not None else float("nan"))
            for row in value["gradient_norms"]
        }
        or None,
        depth_one_reference=value["depth_one_reference"],
        displacement_floor=float(value["displacement_floor"]),
        necessity_margin=float(value["necessity_margin"]),
        length_confound_limit=float(value["length_confound_limit"]),
        gradient_ratio_limit=float(value["gradient_ratio_limit"]),
        improvement_margin=float(value["improvement_margin"]),
    )
    if rebuilt != dict(value):
        differing = sorted(
            key for key in set(rebuilt) | set(value) if rebuilt.get(key) != value.get(key)
        )
        raise ProgressiveObjectiveError(
            f"progressive report does not replay from its own rows: {differing}"
        )
    return dict(value)


def solve_collapse_barrier(
    measurements: Sequence[tuple[float, float, float]],
    *,
    weight_grid: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0),
    floor_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Find the (weight, floor) that removes the collapse basin, from data.

    ``measurements`` is ``[(alpha, base_loss, displacement), ...]`` in
    increasing alpha, as produced by ``tools/measure_progressive_collapse.py``.

    The sweep on the untrained 1.5B found something more precise than the
    inherited claim that collapse is the optimum: at depth 4 the global minimum
    of v4's objective is at the *honest* end (alpha 0.5), but the low-motion
    region is a **local basin** — alpha 0.01 scores 3.3435 while the two steps
    toward real motion cost +0.1446 and +0.0455 before the loss falls away. An
    optimizer that drifts into that region is behind a ~0.19-nat barrier, which
    is the operational form of the concern and is falsifiable in a way "collapse
    is cheapest" is not.

    It also found that this module's *detector* floor is far too low to serve as
    a *training* floor: at alpha 0.01 the measured displacement is 0.0117, above
    the 0.01 detector floor, so the penalty is exactly zero at the local minimum
    it exists to eliminate. Detecting collapse and pricing it out are different
    jobs with different constants, and the training constants are operating-point
    specific — the resident 32B will not share the 1.5B's displacement scale.

    So this solver derives them rather than assuming them: the smallest weight
    on the grid, and the smallest floor, for which the penalized objective is
    strictly decreasing in alpha across the measured range — i.e. from anywhere
    in the collapse region there is a downhill path out.
    """

    rows = [
        (
            _validate_scalar("alpha", alpha, low=0.0, high=1.0),
            _validate_scalar("base_loss", loss, low=-1e6, high=1e6),
            _validate_scalar("displacement", displacement, low=0.0, high=1e6),
        )
        for alpha, loss, displacement in measurements
    ]
    if len(rows) < 3:
        raise ProgressiveObjectiveError(
            "a barrier solve needs at least three measured operating points"
        )
    if [row[0] for row in rows] != sorted(row[0] for row in rows):
        raise ProgressiveObjectiveError("measurements must be sorted by alpha")

    displacements = [row[2] for row in rows]
    candidates = (
        list(floor_grid)
        if floor_grid is not None
        else sorted({round(value * 1.15, 6) for value in displacements})
    )
    barriers = [
        {
            "from_alpha": rows[index][0],
            "to_alpha": rows[index + 1][0],
            "base_gap": round(rows[index + 1][1] - rows[index][1], 6),
        }
        for index in range(len(rows) - 1)
        if rows[index + 1][1] > rows[index][1]
    ]

    def monotone(weight: float, floor: float) -> bool:
        penalized = [
            loss + weight * max(floor - displacement, 0.0) for _alpha, loss, displacement in rows
        ]
        return all(
            later < earlier for earlier, later in zip(penalized[:-1], penalized[1:], strict=True)
        )

    solution: dict[str, Any] | None = None
    for weight in sorted(float(value) for value in weight_grid):
        for floor in sorted(float(value) for value in candidates):
            if monotone(weight, floor):
                solution = {
                    "weight": round(weight, 6),
                    "floor": round(floor, 6),
                }
                break
        if solution is not None:
            break

    payload = {
        "schema": PROGRESSIVE_OBJECTIVE_SCHEMA,
        "analysis": "collapse_barrier_solve_v1",
        "measurements": [
            {
                "alpha": round(alpha, 6),
                "base_loss": round(loss, 6),
                "displacement": round(displacement, 6),
            }
            for alpha, loss, displacement in rows
        ],
        "barriers": barriers,
        "barrier_total": round(sum(row["base_gap"] for row in barriers), 6),
        "collapse_is_global_optimum": rows[0][1] == min(row[1] for row in rows),
        "collapse_is_local_basin": bool(barriers),
        "solution": solution,
        "solved": solution is not None,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def displacement_floor_penalty(
    states: Sequence[Any],
    *,
    floor: float = DEFAULT_DISPLACEMENT_FLOOR,
) -> tuple[Any, list[float]]:
    """Charge the optimizer for letting the recurrent operator go quiet.

    This is the term that closes v4's escape hatch. Without it, "do nothing"
    is a globally optimal solution to an improvement mandate on families where
    depth is destructive, and the optimizer finds it. The hinge is linear, so
    its gradient does not vanish as collapse is approached — the same reason
    v4 replaced its quadratic diversity penalty.

    Returns ``(penalty, detached normalized displacements)``.
    """
    import mlx.core as mx

    band = _validate_scalar("floor", floor, low=0.0, high=1.0)
    if len(states) < 2:
        return mx.zeros(()), []
    penalty = mx.zeros(())
    measured: list[float] = []
    for previous, current in zip(states[:-1], states[1:], strict=True):
        numerator = mx.linalg.norm(mx.reshape(current - previous, (-1,)))
        denominator = mx.maximum(mx.linalg.norm(mx.reshape(previous, (-1,))), 1e-9)
        displacement = numerator / denominator
        penalty = penalty + mx.maximum(band - displacement, 0.0)
        measured.append(float(displacement))
    return penalty / (len(states) - 1), measured


def progressive_objective_loss(
    model: Any,
    prompt_tokens: Sequence[int],
    answer_tokens: Sequence[int],
    *,
    spec: RLCExecutionSpec,
    depth: int = 8,
    probe_steps: Sequence[int] | None = None,
    final_weight: float = 1.0,
    improvement_weight: float = 1.0,
    improvement_margin: float = 0.02,
    oscillation_weight: float = 0.5,
    displacement_weight: float = 1.0,
    displacement_floor: float = DEFAULT_DISPLACEMENT_FLOOR,
    bridge_tokens: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    """v4's trajectory objective with the collapse escape hatch closed.

    Four terms, and the fourth is the point:

    * ``final``        — the endpoint answer must be good.
    * ``improvement``  — each probed step must beat the previous by a margin,
      with the previous detached so earlier steps cannot be sabotaged.
    * ``oscillation``  — consecutive updates must not be anti-correlated.
    * ``displacement`` — the operator must keep moving the state. This is what
      makes the other three mean what they say.

    Telemetry carries the per-step curve AND the per-step motion, so the
    identity-collapse failure is visible in every training step's record
    rather than discoverable only after the campaign.
    """
    import mlx.core as mx
    import mlx.nn as nn

    wanted = _probe_steps(depth, probe_steps)
    final_scale = _validate_scalar("final_weight", final_weight, low=0.0, high=10.0)
    improve_scale = _validate_scalar("improvement_weight", improvement_weight, low=0.0, high=10.0)
    oscillate_scale = _validate_scalar("oscillation_weight", oscillation_weight, low=0.0, high=10.0)
    displace_scale = _validate_scalar(
        "displacement_weight", displacement_weight, low=0.0, high=10.0
    )
    if len(spec.branch_roles) != 1:
        raise ProgressiveObjectiveError(
            "the progressive objective is single-branch; width is SPARK-016's"
        )

    prepared = _prepare_live_path(
        model,
        prompt_tokens,
        answer_tokens,
        spec=spec.with_depth(depth),
        bridge_tokens=tuple(bridge_tokens),
    )
    targets = mx.array(list(answer_tokens))[None, :]
    states = list(prepared.states)
    # Every state on the trajectory feeds the displacement term; only probed
    # states are decoded, because a persist+score per step is the expensive
    # part on a resident-scale model.
    trajectory_states: list[Any] = [states[0]]
    step_losses: list[Any] = []
    probed_states: list[Any] = []
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
        trajectory_states.append(states[0])
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
        step_losses.append(nn.losses.cross_entropy(logits, targets, reduction="mean"))
        probed_states.append(states[0])

    improvement = monotone_improvement_penalty(step_losses, margin=improvement_margin)
    oscillation, cosines = oscillation_penalty(probed_states)
    displacement, measured = displacement_floor_penalty(trajectory_states, floor=displacement_floor)
    final = step_losses[-1]
    loss = (
        final_scale * final
        + improve_scale * improvement
        + oscillate_scale * oscillation
        + displace_scale * displacement
    )
    detached = [float(value) for value in step_losses]
    telemetry = {
        "schema": PROGRESSIVE_OBJECTIVE_SCHEMA,
        "depth": int(depth),
        "probe_steps": list(wanted),
        "step_losses": [round(value, 6) for value in detached],
        "best_step_index": int(min(range(len(detached)), key=detached.__getitem__)),
        "improving_steps": sum(
            1 for index in range(1, len(detached)) if detached[index] < detached[index - 1]
        ),
        "final_loss": round(detached[-1], 6),
        "improvement_penalty": _round(improvement),
        "oscillation_penalty": _round(oscillation),
        "displacement_penalty": _round(displacement),
        "displacements": [round(value, 6) for value in measured],
        "min_displacement": round(min(measured), 6) if measured else None,
        "displacement_floor": round(float(displacement_floor), 6),
        "delta_cosines": [round(value, 6) for value in cosines],
        "collapse_pressure_active": bool(measured and min(measured) < float(displacement_floor)),
    }
    return loss, telemetry


__all__ = [
    "DEFAULT_DISPLACEMENT_FLOOR",
    "DEFAULT_GRADIENT_RATIO_LIMIT",
    "MEASURED_1P5B_COLLAPSE_SOLUTION",
    "DEFAULT_LENGTH_CONFOUND_LIMIT",
    "DEFAULT_NECESSITY_MARGIN",
    "PROGRESSIVE_OBJECTIVE_SCHEMA",
    "PROGRESSIVE_TRAJECTORY_SET_SCHEMA",
    "VERDICTS",
    "ProgressiveObjectiveError",
    "ProgressiveTrajectory",
    "ProgressiveTrajectorySet",
    "build_progressive_report",
    "canonical_sha256",
    "displacement_floor_penalty",
    "measure_progressive_trajectory",
    "measure_progressive_trajectories",
    "progressive_objective_loss",
    "solve_collapse_barrier",
    "state_gradient_norms",
    "step_necessity",
    "validate_progressive_report",
    "validate_progressive_trajectory_set",
]
