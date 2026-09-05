"""Protected recurrent memory for the latent cortex (CP213).

Measured problem: the recurrent update
``z_{t+1} = (1-a) z_t + a * rms_match(R(z_t), anchor)`` is a contraction.
Its residual decays 0.302 -> 0.026 and asymptotes, so the state reaches a
single attractor and stops computing. That behavior is CORRECT for
semantic refinement -- converging on an interpretation is what refinement
means. It is fatal for computational control, because a contraction has
exactly one fixed point and every direction in the tensor is dragged into
it, including the directions that were holding:

    intermediate conclusions, variable bindings, unresolved assumptions,
    subgoal stacks, evidence for/against, provenance, and the position in
    an internal procedure.

One dynamical regime cannot serve both jobs. This module gives the
workspace a second lane whose DEFAULT transition is identity, so
information persists unless the model explicitly decides to overwrite it:

    M_{t+1} = (1 - w_t) * carry(M_t) + w_t * candidate_t

where ``carry`` is identity (optionally an orthogonal/norm-preserving
transport) and ``w_t`` is a per-slot, per-step write gate. With w_t = 0
the slot is bit-preserved across arbitrarily many steps; nothing about the
semantic lane's convergence can erase it.

The lanes are addressed by slot index, so the memory lane occupies real
KV positions exactly like semantic slots and remains causally available to
the decode and to ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

PROTECTED_MEMORY_SCHEMA = "aura.latent_protected_memory.v1"


@dataclass(frozen=True)
class MemoryLayout:
    """Which slot indices are semantic, protected memory, and control.

    Slots are real sequence positions, so a layout is just a partition of
    ``range(n_slots)``. Disjointness is enforced: a slot cannot be both
    contracted and protected.
    """

    n_slots: int
    memory_slots: tuple[int, ...]
    control_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.n_slots) is not int or self.n_slots < 1:
            raise ValueError("n_slots must be a positive integer")
        protected = tuple(self.memory_slots) + tuple(self.control_slots)
        if any(
            type(index) is not int or not 0 <= index < self.n_slots
            for index in protected
        ):
            raise ValueError("protected slot indices must be inside [0, n_slots)")
        if len(set(protected)) != len(protected):
            raise ValueError("a slot cannot be both memory and control")
        if len(protected) >= self.n_slots:
            raise ValueError("at least one semantic slot must remain")

    @property
    def protected(self) -> tuple[int, ...]:
        return tuple(sorted(self.memory_slots + self.control_slots))

    @property
    def semantic(self) -> tuple[int, ...]:
        blocked = set(self.protected)
        return tuple(i for i in range(self.n_slots) if i not in blocked)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROTECTED_MEMORY_SCHEMA,
            "n_slots": self.n_slots,
            "memory_slots": list(self.memory_slots),
            "control_slots": list(self.control_slots),
            "semantic_slots": list(self.semantic),
        }


def _protected_mask(layout: MemoryLayout) -> Any:
    """1.0 on protected slots, 0.0 elsewhere, shape (1, n_slots, 1)."""
    import mlx.core as mx

    if not layout.protected:
        return mx.zeros((1, layout.n_slots, 1))
    positions = mx.arange(layout.n_slots)
    hit = mx.sum(
        mx.stack([positions == index for index in layout.protected]), axis=0
    ) > 0
    return mx.reshape(
        mx.where(hit, mx.ones((layout.n_slots,)), mx.zeros((layout.n_slots,))),
        (1, layout.n_slots, 1),
    )


def write_gates(
    previous: Any,
    candidate: Any,
    layout: MemoryLayout,
    *,
    gate_bias: float = -2.0,
    gate_scale: float = 1.0,
) -> Any:
    """Per-slot write gates over the protected lane, shape (1, n_slots, 1).

    The gate is driven by how much the candidate DISAGREES with what is
    already stored: a candidate that merely restates memory should not
    consume a write. ``gate_bias`` is negative so the default is to
    preserve -- writing is the exception in a protected lane.
    """
    import mlx.core as mx

    disagreement = mx.mean(
        mx.abs(candidate - previous), axis=-1, keepdims=True
    )
    scale = mx.maximum(mx.mean(mx.abs(previous)) , 1e-6)
    logit = gate_scale * (disagreement / scale) + gate_bias
    return mx.sigmoid(logit) * _protected_mask(layout)


def apply_protected_transition(
    previous: Any,
    contracted: Any,
    layout: MemoryLayout,
    *,
    write_gate: Any = None,
    carry: Any = None,
) -> tuple[Any, Any]:
    """Blend the contracted update ONLY into semantic slots.

    Protected slots take ``carry(previous)`` -- exact identity by default --
    adjusted by ``write_gate``. Semantic slots take the contracted update
    unchanged, so refinement is untouched.

    ``write_gate`` is an EXPLICIT decision, shape broadcastable to
    (1, n_slots, 1). Omitting it means preserve exactly: gate 0. That
    default is load-bearing rather than conservative styling -- any
    per-step leak compounds, and a mere 0.18 leak erases a slot to 0.2% of
    its value over 32 steps. Deciding when to write is a learned policy;
    this function is the mechanism it acts through.

    Returns ``(next_state, per-slot write gate actually applied)``.
    """
    import mlx.core as mx

    if previous.shape != contracted.shape:
        raise ValueError("previous and contracted states must share a shape")
    if int(previous.shape[1]) != layout.n_slots:
        raise ValueError("state slot count does not match the layout")
    carried = previous if carry is None else carry(previous)
    if write_gate is None:
        gate = mx.zeros((1, layout.n_slots, 1))
    else:
        gate = mx.broadcast_to(
            mx.reshape(write_gate, (1, layout.n_slots, 1)),
            (1, layout.n_slots, 1),
        ) * _protected_mask(layout)
    protected_state = (1.0 - gate) * carried + gate * contracted
    mask = _protected_mask(layout)
    return mask * protected_state + (1.0 - mask) * contracted, gate


def memory_retention(initial: Any, final: Any, layout: MemoryLayout) -> dict[str, float]:
    """How much of the protected lane survived the trajectory.

    ``cosine`` near 1.0 and ``relative_drift`` near 0.0 mean the lane held
    its contents; this is the measurement that decides whether a counter or
    binding is still recoverable at depth 32.
    """
    import mlx.core as mx

    if not layout.protected:
        return {"cosine": 1.0, "relative_drift": 0.0, "slots": 0}
    indices = mx.array(list(layout.protected))
    start = mx.reshape(mx.take(initial, indices, axis=1), (-1,))
    end = mx.reshape(mx.take(final, indices, axis=1), (-1,))
    denominator = mx.maximum(mx.linalg.norm(start) * mx.linalg.norm(end), 1e-9)
    cosine = float(mx.sum(start * end) / denominator)
    drift = float(
        mx.linalg.norm(end - start) / mx.maximum(mx.linalg.norm(start), 1e-9)
    )
    return {
        "cosine": cosine,
        "relative_drift": drift,
        "slots": len(layout.protected),
    }


def semantic_convergence(states: Sequence[Any], layout: MemoryLayout) -> list[float]:
    """Residual trail of the SEMANTIC lane only.

    Convergence here is desirable and should remain visible; conflating it
    with the protected lane is what hid the erasure in the first place.
    """
    import mlx.core as mx

    if len(states) < 2 or not layout.semantic:
        return []
    indices = mx.array(list(layout.semantic))
    residuals: list[float] = []
    for previous, current in zip(states, states[1:]):
        before = mx.reshape(mx.take(previous, indices, axis=1), (-1,))
        after = mx.reshape(mx.take(current, indices, axis=1), (-1,))
        scale = mx.maximum(mx.linalg.norm(after), 1e-9)
        residuals.append(float(mx.linalg.norm(after - before) / scale))
    return residuals


__all__ = [
    "PROTECTED_MEMORY_SCHEMA",
    "MemoryLayout",
    "_protected_mask",
    "apply_protected_transition",
    "memory_retention",
    "semantic_convergence",
    "write_gates",
]
