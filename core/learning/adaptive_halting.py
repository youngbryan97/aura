"""Spend thought where it is needed (CP230).

Anima Rationis makes the compute-scaling curve the central success
criterion -- accuracy should rise with recurrent steps on problems that
need them (line 403), and the Will should allocate depth by expected
difficulty, uncertainty, stakes and expected value of further computation
(line 765). Neither is possible with a fixed T: a constant depth is not an
allocation policy, and a flat curve measured at constant depth says nothing
about whether depth COULD have helped.

The document also names the failure this must avoid: latent "overthinking"
(line 425), where excessive recurrence degrades results. So halting is not
merely an efficiency optimization. Running a contracting loop past its
fixed point burns compute and can make answers worse, and CP226 measured
exactly that shape -- deltas 0.55, 0.50, 0.32 with accuracy falling.

The head is zero-initialized, so an untrained model never halts early and
behaves exactly as it did at fixed depth. Allocation is something the model
earns, not something imposed on it at attach time.

The claim this module makes falsifiable: **allocated depth should
correlate with required depth.** A halting policy that spends uniformly, or
spends by prompt length, is not allocating thought -- and that is
measurable rather than assertable.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.verify.invariants import invariant

ADAPTIVE_HALTING_SCHEMA = "aura.adaptive_halting.v1"
VERIFIED_STOPPING_TEACHER_SCHEMA = "aura.verified_stopping_teacher.v1"


class HaltingHead:
    """Reads the latent state, emits P(this is enough thinking).

    Deliberately tiny. A large halting network would be a second model
    whose own errors are indistinguishable from the cortex's, and it would
    have to be trained on the same scarce verified outcomes.
    """

    def __init__(self, hidden_size: int, *, threshold: float = 0.5) -> None:
        if type(hidden_size) is not int or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0.0 < float(threshold) < 1.0
        ):
            raise ValueError("threshold must be inside (0, 1)")
        import mlx.core as mx

        self.hidden_size = hidden_size
        self.threshold = float(threshold)
        # Zero weight => logit 0 => p = 0.5 everywhere. Callers explicitly
        # treat this identity parameterization as inert; relying on threshold
        # comparison alone is incorrect when threshold == 0.5.
        self.weight = mx.zeros((hidden_size, 1))
        self.bias = mx.zeros((1,))

    def is_identity(self) -> bool:
        import mlx.core as mx

        return bool(mx.all(self.weight == 0) and mx.all(self.bias == 0))

    def halt_probability(self, state: Any) -> Any:
        """P(stop) from the pooled latent state."""
        import mlx.core as mx

        if state.shape[-1] != self.hidden_size:
            raise ValueError("state width does not match the halting head")
        pooled = mx.mean(state.astype(mx.float32), axis=tuple(range(state.ndim - 1)))
        logit = mx.reshape(pooled, (1, -1)) @ self.weight + self.bias
        return mx.sigmoid(mx.reshape(logit, ()))

    def parameters(self) -> dict[str, Any]:
        return {"weight": self.weight, "bias": self.bias}

    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.parameters().values()))

    # ── Persistence (the live engine loads a TRAINED head from disk) ────
    def save(self, path: Any) -> None:
        """Serialize weights + threshold so a trained head can go live."""
        from io import BytesIO
        from pathlib import Path

        import numpy as np

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        target = Path(path).expanduser()
        if not str(target).endswith(".npz"):
            target = Path(str(target) + ".npz")
        with BytesIO() as buffer:
            np.savez(
                buffer,
                weight=np.array(self.weight, dtype=np.float32),
                bias=np.array(self.bias, dtype=np.float32),
                threshold=np.array([self.threshold], dtype=np.float32),
                hidden_size=np.array([self.hidden_size], dtype=np.int64),
            )
            payload = buffer.getvalue()
        with local_internal_governed_scope("adaptive_halting.save"):
            get_file_write_gateway().write_bytes(target, payload, source="adaptive_halting.save")

    @classmethod
    def load(cls, path: Any) -> HaltingHead:
        """Rebuild a head exactly as saved; malformed files fail loudly."""
        from pathlib import Path

        import numpy as np

        source = Path(path).expanduser()
        with np.load(source) as payload:
            return cls._from_payload(payload, source=source)

    @classmethod
    def _from_payload(cls, payload: Any, *, source: Any) -> HaltingHead:
        import numpy as np

        required = {"weight", "bias", "threshold", "hidden_size"}
        if set(payload) != required:
            raise ValueError(
                f"halting head file {source} missing fields: "
                f"{sorted(required - set(payload))}"
            )
        hidden_size = int(payload["hidden_size"][0])
        weight = payload["weight"].astype(np.float32)
        bias = payload["bias"].astype(np.float32)
        if weight.shape != (hidden_size, 1) or bias.shape != (1,):
            raise ValueError(f"halting head file {source} has wrong shapes")

        import mlx.core as mx

        head = cls(hidden_size, threshold=float(payload["threshold"][0]))
        head.weight = mx.array(weight)
        head.bias = mx.array(bias)
        return head


@invariant("learning.halting_artifact_shape_before_allocation", scope="learning",
           owner="core/learning/adaptive_halting.py", observational=False)
def _halting_head_shape_invariant() -> tuple:
    import numpy as np

    class AllocationTrap(HaltingHead):
        def __init__(self, *args, **kwargs):
            raise AssertionError("the unchecked halting artifact reached head allocation")

    payload = {
        "weight": np.zeros((4, 1), dtype=np.float32),
        "bias": np.zeros(1, dtype=np.float32),
        "threshold": np.array([0.7]),
        "hidden_size": np.array([2**40]),
    }
    try:
        AllocationTrap._from_payload(payload, source="shape canary")
    except ValueError:
        return ()
    raise AssertionError("the mismatched halting artifact was accepted")


@dataclass(frozen=True)
class HaltingPolicy:
    """Bounds on how much thought may be spent."""

    min_steps: int = 1
    max_steps: int = 8
    # Price per step, in units of loss. Anima Rationis line 765 wants
    # allocation by expected value of computation; a price is how that
    # trade-off becomes explicit instead of implicit.
    ponder_cost: float = 0.01

    def __post_init__(self) -> None:
        for name in ("min_steps", "max_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if (
            isinstance(self.ponder_cost, bool)
            or not isinstance(self.ponder_cost, (int, float))
            or not 0.0 <= float(self.ponder_cost) <= 1.0
        ):
            raise ValueError("ponder_cost must be inside [0, 1]")


@dataclass(frozen=True, slots=True)
class VerifiedStoppingTeacher:
    """Detached optimal-stopping target for base-weight training.

    The production stop head cannot inspect an answer key. Training can, but
    only after the completion has independent positive-verifier credit. This
    teacher converts those per-depth answer losses into a bounded stop
    distribution. The probabilities are detached before the exact adjoint
    uses them, so the model can lower answer risk at useful depths but cannot
    manipulate the teacher itself.
    """

    steps: tuple[int, ...]
    losses: tuple[float, ...]
    risks: tuple[float, ...]
    probabilities: tuple[float, ...]
    selected_step: int
    expected_loss: float
    expected_steps: float
    expected_risk: float
    ponder_cost: float
    temperature: float

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema": VERIFIED_STOPPING_TEACHER_SCHEMA,
            "steps": list(self.steps),
            "losses": list(self.losses),
            "risks": list(self.risks),
            "probabilities": list(self.probabilities),
            "selected_step": self.selected_step,
            "expected_loss": self.expected_loss,
            "expected_steps": self.expected_steps,
            "expected_risk": self.expected_risk,
            "ponder_cost": self.ponder_cost,
            "temperature": self.temperature,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return {
            **payload,
            "receipt_sha256": hashlib.sha256(encoded).hexdigest(),
        }


def verified_stopping_teacher(
    step_losses: Sequence[float],
    steps: Sequence[int],
    *,
    ponder_cost: float,
    temperature: float,
) -> VerifiedStoppingTeacher:
    """Build a stable cost-aware soft optimal-stopping target.

    Accuracy remains primary because ``ponder_cost`` is bounded to one loss
    unit per step. Temperature controls how sharply the teacher concentrates
    on the lowest answer-loss-plus-compute-risk depth. The earliest depth wins
    exact ties, which makes the corresponding hard target deterministic.
    """

    normalized_steps = tuple(steps)
    raw_losses = tuple(step_losses)
    if (
        len(normalized_steps) < 2
        or len(normalized_steps) != len(raw_losses)
        or any(type(step) is not int or step < 1 for step in normalized_steps)
        or tuple(sorted(set(normalized_steps))) != normalized_steps
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in raw_losses
        )
    ):
        raise ValueError("verified stopping steps and losses are invalid")
    normalized_losses = tuple(float(value) for value in raw_losses)
    if (
        isinstance(ponder_cost, bool)
        or not isinstance(ponder_cost, (int, float))
        or not math.isfinite(float(ponder_cost))
        or not 0.0 <= float(ponder_cost) <= 1.0
    ):
        raise ValueError("verified stopping ponder_cost must be inside [0, 1]")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or not 1e-4 <= float(temperature) <= 10.0
    ):
        raise ValueError("verified stopping temperature must be inside [1e-4, 10]")
    cost = float(ponder_cost)
    resolved_temperature = float(temperature)
    risks = tuple(
        loss + cost * step for step, loss in zip(normalized_steps, normalized_losses, strict=True)
    )
    minimum = min(risks)
    exponentials = tuple(
        math.exp(max(-80.0, min(0.0, -(risk - minimum) / resolved_temperature))) for risk in risks
    )
    denominator = sum(exponentials)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("verified stopping distribution is non-finite")
    probabilities = tuple(value / denominator for value in exponentials)
    selected_index = min(range(len(risks)), key=lambda index: (risks[index], index))
    expected_loss = sum(
        probability * loss
        for probability, loss in zip(probabilities, normalized_losses, strict=True)
    )
    expected_steps = sum(
        probability * step
        for probability, step in zip(probabilities, normalized_steps, strict=True)
    )
    expected_risk = expected_loss + cost * expected_steps
    if not all(
        math.isfinite(value)
        for value in (*probabilities, expected_loss, expected_steps, expected_risk)
    ):
        raise FloatingPointError("verified stopping teacher produced non-finite values")
    return VerifiedStoppingTeacher(
        steps=normalized_steps,
        losses=normalized_losses,
        risks=risks,
        probabilities=probabilities,
        selected_step=normalized_steps[selected_index],
        expected_loss=expected_loss,
        expected_steps=expected_steps,
        expected_risk=expected_risk,
        ponder_cost=cost,
        temperature=resolved_temperature,
    )


def validate_verified_stopping_teacher_receipt(value: Any) -> dict[str, Any]:
    """Recompute teacher arithmetic over producer-sealed per-depth loss atoms."""

    fields = {
        "schema",
        "steps",
        "losses",
        "risks",
        "probabilities",
        "selected_step",
        "expected_loss",
        "expected_steps",
        "expected_risk",
        "ponder_cost",
        "temperature",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("verified stopping teacher receipt fields differ")
    payload = {key: value[key] for key in fields - {"receipt_sha256"}}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if value["receipt_sha256"] != hashlib.sha256(encoded).hexdigest():
        raise ValueError("verified stopping teacher receipt commitment mismatch")
    if value["schema"] != VERIFIED_STOPPING_TEACHER_SCHEMA:
        raise ValueError("verified stopping teacher schema is unsupported")
    replayed = verified_stopping_teacher(
        value["losses"],
        value["steps"],
        ponder_cost=value["ponder_cost"],
        temperature=value["temperature"],
    ).receipt()
    if replayed != value:
        raise ValueError("verified stopping teacher arithmetic does not replay")
    return dict(value)


def decide_steps(
    head: HaltingHead,
    states: Sequence[Any],
    policy: HaltingPolicy,
) -> dict[str, Any]:
    """Walk the trajectory and stop when the head says enough.

    Returns the chosen depth plus the per-step probabilities, because a
    halting decision nobody can inspect is indistinguishable from a
    constant.
    """
    if not states:
        raise ValueError("no states to evaluate")
    probabilities: list[float] = []
    chosen = min(len(states), policy.max_steps)
    halted_early = False
    identity = head.is_identity()
    for index, state in enumerate(states[: policy.max_steps], start=1):
        probability = float(head.halt_probability(state))
        probabilities.append(round(probability, 6))
        if not identity and index >= policy.min_steps and probability >= head.threshold:
            chosen = index
            halted_early = index < min(len(states), policy.max_steps)
            break
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "steps": int(chosen),
        "halt_probabilities": probabilities,
        "halted_early": bool(halted_early),
        "min_steps": policy.min_steps,
        "max_steps": policy.max_steps,
    }


def ponder_loss(
    step_losses: Sequence[Any],
    halt_probabilities: Sequence[Any],
    policy: HaltingPolicy,
) -> tuple[Any, dict[str, Any]]:
    """Expected loss under the halting distribution, plus a price on depth.

    The head learns to stop where stopping is cheap in ACCURACY, not where
    stopping is cheap in compute -- the ponder cost is a tiebreaker, not the
    objective. An objective dominated by the compute term produces a model
    that always halts immediately and reports excellent efficiency.
    """
    import mlx.core as mx

    if len(step_losses) != len(halt_probabilities):
        raise ValueError("step losses and halt probabilities must align")
    if not step_losses:
        raise ValueError("nothing to score")

    # Probability of stopping AT step i = p_i * prod(1 - p_j) for j < i.
    remaining = mx.array(1.0)
    weights: list[Any] = []
    for index, probability in enumerate(halt_probabilities):
        last = index == len(halt_probabilities) - 1
        stop_here = remaining if last else remaining * probability
        weights.append(stop_here)
        remaining = remaining * (1.0 - probability)

    expected = mx.stack(
        [weight * loss for weight, loss in zip(weights, step_losses, strict=True)]
    ).sum()
    expected_steps = mx.stack(
        [weight * float(index + 1) for index, weight in enumerate(weights)]
    ).sum()
    total = expected + policy.ponder_cost * expected_steps
    return total, {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "expected_loss": float(expected),
        "expected_steps": round(float(expected_steps), 4),
        "ponder_cost": policy.ponder_cost,
        "total": float(total),
    }


def allocation_report(
    allocations: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Does spent depth track REQUIRED depth?

    ``allocations`` is (required_depth, allocated_steps). This is the
    falsifiable core of the whole halting claim: a policy that spends
    uniformly, or spends by prompt length, is not allocating thought. The
    correlation says which one is happening, and a flat allocation is
    reported as such rather than defended.
    """
    if len(allocations) < 3:
        raise ValueError("need at least three allocations to judge a policy")
    required = [float(r) for r, _ in allocations]
    spent = [float(s) for _, s in allocations]
    n = len(allocations)
    mean_r, mean_s = sum(required) / n, sum(spent) / n
    cov = sum((r - mean_r) * (s - mean_s) for r, s in zip(required, spent, strict=True)) / n
    var_r = sum((r - mean_r) ** 2 for r in required) / n
    var_s = sum((s - mean_s) ** 2 for s in spent) / n
    if var_s < 1e-12:
        return {
            "schema": ADAPTIVE_HALTING_SCHEMA,
            "correlation": 0.0,
            "allocates_by_difficulty": False,
            "reason": "constant allocation is not a policy",
            "mean_steps": round(mean_s, 4),
        }
    correlation = cov / math.sqrt(max(var_r * var_s, 1e-12))
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "correlation": round(correlation, 4),
        "mean_steps": round(mean_s, 4),
        "mean_required": round(mean_r, 4),
        # A weak positive correlation is not allocation; it is noise with a
        # sign. The bar is deliberately not "greater than zero".
        "allocates_by_difficulty": bool(correlation >= 0.3),
    }


def overthinking_report(
    step_losses: Sequence[float],
) -> dict[str, Any]:
    """Did extra steps make it worse?

    Anima Rationis line 425 names latent overthinking as a real observed
    failure. CP226 measured its shape: the loop kept moving (deltas 0.55,
    0.50, 0.32) while accuracy fell. Reported explicitly so 'the state is
    still changing' is never mistaken for 'it is still thinking'.
    """
    values = [float(v) for v in step_losses]
    if len(values) < 2:
        raise ValueError("need at least two steps to detect overthinking")
    best = min(values)
    best_step = values.index(best) + 1
    return {
        "schema": ADAPTIVE_HALTING_SCHEMA,
        "best_step": best_step,
        "best_loss": round(best, 6),
        "final_loss": round(values[-1], 6),
        "overthinks": bool(values[-1] > best + 1e-9),
        "wasted_steps": len(values) - best_step,
    }


__all__ = [
    "ADAPTIVE_HALTING_SCHEMA",
    "HaltingHead",
    "HaltingPolicy",
    "VERIFIED_STOPPING_TEACHER_SCHEMA",
    "VerifiedStoppingTeacher",
    "allocation_report",
    "decide_steps",
    "overthinking_report",
    "ponder_loss",
    "validate_verified_stopping_teacher_receipt",
    "verified_stopping_teacher",
]
