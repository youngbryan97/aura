"""Depth-conditioned recurrent operators (CP219).

Measured here, independently: injecting a per-step phase code into the
recurrence INPUT improves answer CE (-6.2% at best depth, gain growing
with depth) but does not break the contraction -- the residual ratio moved
only 0.1142 -> 0.1048 and the state still converges. A fixed input
perturbation is re-absorbed by alpha-interpolation; to make step t compute
something genuinely different from step t+1, the OPERATOR must differ, not
its input.

That conclusion is externally corroborated. Think-at-Hard (Tsinghua)
reports gains across math, QA and coding using depth-aware LoRA modules
applied selectively, and looped-transformer expressivity work finds that
timestep encoding and loop-conditioned scaling increase expressive power.
This module is that mechanism for Aura's existing adapter stack.

Design:

* One shared base LoRA plus a small bank of per-depth deltas. Storing K
  full operators would cost K times the parameters; a shared operator with
  cheap per-depth modulation keeps the budget near a single adapter while
  making every step's effective transform distinct.
* Deltas initialize at ZERO, so an untrained depth-conditioned adapter is
  bit-identical to the plain shared adapter. Training can only add
  differentiation; it cannot start by destroying a working operator.
  (This is the identity-biased-initialization discipline that
  "Thinking Deeper, Not Longer" credits for stability past 20 steps.)
* Depths beyond the bank reuse the last entry rather than failing, so a
  model trained to depth 8 still runs at depth 32 -- depth extrapolation
  is testable instead of blocked.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

DEPTH_CONDITIONED_SCHEMA = "aura.depth_conditioned_lora.v1"

# Current recurrent step, published by the training/inference loop so every
# wrapped projection sees the same depth without threading an argument
# through the whole transformer stack.
_CURRENT_DEPTH: ContextVar[int] = ContextVar(
    "aura_recurrent_depth_index", default=0
)


@contextmanager
def recurrent_depth_index(step: int) -> Iterator[None]:
    """Publish the recurrent step for the duration of one window pass."""
    if type(step) is not int or step < 0:
        raise ValueError("recurrent depth index must be a non-negative integer")
    token = _CURRENT_DEPTH.set(step)
    try:
        yield
    finally:
        _CURRENT_DEPTH.reset(token)


def current_depth_index() -> int:
    return _CURRENT_DEPTH.get()


class DepthConditionedLoRA:
    """Shared LoRA plus zero-initialized per-depth deltas.

    Wraps an existing ``ScopedLoRALinear`` so the recurrence adapter's scope
    gating, receipts, and load compatibility are preserved unchanged; only
    the effective (A, B) factors vary with the recurrent step.
    """

    def __init__(self, scoped: Any, *, depths: int, delta_scale: float = 1.0):
        if type(depths) is not int or depths < 1:
            raise ValueError("depths must be a positive integer")
        if (
            isinstance(delta_scale, bool)
            or not isinstance(delta_scale, (int, float))
            or not 0.0 <= float(delta_scale) <= 10.0
        ):
            raise ValueError("delta_scale must be inside [0, 10]")
        import mlx.core as mx

        self.scoped = scoped
        self.depths = depths
        self.delta_scale = float(delta_scale)
        if any(
            hasattr(scoped, name)
            for name in ("depth_a", "depth_b", "depth_conditioned_steps")
        ):
            raise ValueError("scoped adapter already has a depth-conditioned bank")
        # These tensors live directly on the MLX Module. A plain helper object
        # is not traversed by ``trainable_parameters()``, which made the former
        # bank visible to the forward pass but invisible to the optimizer and
        # checkpoint writer. Zero deltas preserve exact shared-adapter parity.
        scoped.depth_a = [mx.zeros_like(scoped.lora_a) for _ in range(depths)]
        scoped.depth_b = [mx.zeros_like(scoped.lora_b) for _ in range(depths)]
        scoped.depth_conditioned_steps = depths
        scoped.depth_delta_scale = self.delta_scale
        self.depth_a = scoped.depth_a
        self.depth_b = scoped.depth_b

    def factors_for(self, step: int) -> tuple[Any, Any]:
        """Effective (A, B) at a recurrent step.

        Steps beyond the trained bank reuse the final entry so depth
        extrapolation runs rather than raising -- the whole question of
        whether a depth-8-trained operator generalizes to depth 32 is only
        askable if the forward pass is defined there.
        """
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        index = min(step, self.depths - 1)
        return (
            self.scoped.lora_a + self.delta_scale * self.depth_a[index],
            self.scoped.lora_b + self.delta_scale * self.depth_b[index],
        )

    def is_identity_at(self, step: int) -> bool:
        """True when this depth's EFFECTIVE operator equals the shared one.

        A LoRA delta is ``dW = A @ B``, so it vanishes when EITHER factor is
        zero -- not only when both are. Checking both would report a delta
        as active while it contributes exactly nothing, which is the same
        class of error as reporting an unchecked property as verified.
        (It is also why zero-initializing B alone is the standard way to
        make an adapter exactly identity at attach.)
        """
        import mlx.core as mx

        index = min(max(step, 0), self.depths - 1)
        return bool(
            mx.all(self.depth_a[index] == 0) or mx.all(self.depth_b[index] == 0)
        )

    def differentiation(self) -> list[float]:
        """Per-depth delta magnitude relative to the shared operator.

        The telemetry that answers 'did the operator actually specialize by
        depth, or is this depth-conditioning in name only?'
        """
        import mlx.core as mx

        shared = float(
            mx.linalg.norm(mx.reshape(self.scoped.lora_a, (-1,)))
            * mx.linalg.norm(mx.reshape(self.scoped.lora_b, (-1,)))
        )
        scale = max(shared, 1e-9)
        magnitudes: list[float] = []
        for index in range(self.depths):
            delta = float(
                mx.linalg.norm(mx.reshape(self.depth_a[index], (-1,)))
                * mx.linalg.norm(mx.reshape(self.depth_b[index], (-1,)))
            )
            magnitudes.append(round(self.delta_scale * delta / scale, 6))
        return magnitudes

    def trainable(self) -> dict[str, Any]:
        """Per-depth parameters for the optimizer, keyed for checkpointing."""
        parameters: dict[str, Any] = {}
        for index in range(self.depths):
            parameters[f"depth_{index}.lora_a"] = self.depth_a[index]
            parameters[f"depth_{index}.lora_b"] = self.depth_b[index]
        return parameters

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": DEPTH_CONDITIONED_SCHEMA,
            "depths": self.depths,
            "delta_scale": self.delta_scale,
            "differentiation": self.differentiation(),
            "identity_at_init": all(
                self.is_identity_at(step) for step in range(self.depths)
            ),
        }


def wrap_depth_conditioned(
    model: Any,
    *,
    depths: int,
    delta_scale: float = 1.0,
) -> dict[str, DepthConditionedLoRA]:
    """Attach depth conditioning to every ScopedLoRALinear already present.

    Runs AFTER the ordinary recurrence wrapping, so the scope gating and
    identity receipts of the existing adapter stack are untouched.
    """
    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    wrapped: dict[str, DepthConditionedLoRA] = {}
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) or []
    for layer_index, layer in enumerate(layers):
        for parent_name in ("self_attn", "linear_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in ("o_proj", "v_proj", "q_proj", "k_proj", "down_proj"):
                projection = getattr(parent, target, None)
                if isinstance(projection, ScopedLoRALinear):
                    key = f"model.layers.{layer_index}.{parent_name}.{target}"
                    bank = DepthConditionedLoRA(
                        projection, depths=depths, delta_scale=delta_scale
                    )
                    # Attach so the projection's forward consults the bank.
                    # Without this the bank exists but nothing reads it --
                    # a mechanism that is present in name only, which is
                    # exactly the defect CP211 had to repair in v4.
                    projection.depth_bank = bank
                    wrapped[key] = bank
    if not wrapped:
        raise RuntimeError(
            "no ScopedLoRALinear projections found; wrap the recurrent "
            "window before adding depth conditioning"
        )
    return wrapped


class ContinuousDepthConditionedLoRA:
    """A bounded continuous operator family defined at every recurrent depth.

    Discrete depth banks either memorize their training steps or clamp unseen
    depths to the final operator. This parameterization uses the same bounded
    rational basis as the unified controller. Step zero has no basis features,
    preserving the shared T=1 anchor exactly; later and unseen steps interpolate
    or extrapolate through one learned function rather than a lookup table.
    """

    def __init__(
        self,
        scoped: Any,
        *,
        basis_size: int = 4,
        delta_scale: float = 1.0,
    ) -> None:
        if type(basis_size) is not int or basis_size < 1:
            raise ValueError("continuous depth basis_size must be positive")
        if (
            isinstance(delta_scale, bool)
            or not isinstance(delta_scale, (int, float))
            or not 0.0 <= float(delta_scale) <= 10.0
        ):
            raise ValueError("continuous depth delta_scale must be inside [0, 10]")
        if hasattr(scoped, "depth_bank"):
            raise ValueError("scoped adapter already has depth conditioning")
        import mlx.core as mx

        self.scoped = scoped
        self.basis_size = basis_size
        self.delta_scale = float(delta_scale)
        # A starts from the shared random projection while B starts at zero.
        # The complete correction is therefore exact identity at attach, but B
        # receives a first-step gradient instead of the zero-times-zero dead
        # gradient produced by initializing both factors to zero.
        scoped.continuous_depth_a = [
            scoped.lora_a + 0 for _ in range(basis_size)
        ]
        scoped.continuous_depth_b = [
            mx.zeros_like(scoped.lora_b) for _ in range(basis_size)
        ]
        scoped.continuous_depth_basis_size = basis_size
        scoped.continuous_depth_delta_scale = self.delta_scale
        self.depth_a = scoped.continuous_depth_a
        self.depth_b = scoped.continuous_depth_b

    def features_for(self, step: int) -> tuple[float, ...]:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        u = float(step) / float(step + 1) if step else 0.0
        return tuple(u ** (index + 1) for index in range(self.basis_size))

    def factors_for(self, step: int) -> tuple[Any, Any]:
        features = self.features_for(step)
        effective_a = self.scoped.lora_a
        effective_b = self.scoped.lora_b
        for feature, basis_a, basis_b in zip(
            features,
            self.depth_a,
            self.depth_b,
            strict=True,
        ):
            effective_a = effective_a + self.delta_scale * feature * basis_a
            effective_b = effective_b + self.delta_scale * feature * basis_b
        return effective_a, effective_b

    def identity_initialized(self) -> bool:
        import mlx.core as mx

        return bool(
            mx.all(self.scoped.lora_b == 0)
            and all(mx.all(value == 0) for value in self.depth_b)
        )

    def trainable(self) -> dict[str, Any]:
        parameters = {}
        for index in range(self.basis_size):
            parameters[f"basis_{index}.lora_a"] = self.depth_a[index]
            parameters[f"basis_{index}.lora_b"] = self.depth_b[index]
        return parameters

    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": "aura.continuous_depth_conditioned_lora.v1",
            "basis": "bounded_rational_polynomial",
            "basis_size": self.basis_size,
            "delta_scale": self.delta_scale,
            "step_zero_anchor_exact": True,
            "unseen_depths_defined": True,
            "identity_initialized": self.identity_initialized(),
        }


def wrap_continuous_depth_conditioned(
    model: Any,
    *,
    basis_size: int = 4,
    delta_scale: float = 1.0,
) -> dict[str, ContinuousDepthConditionedLoRA]:
    """Attach one continuous depth operator to every prepared projection."""

    from core.brain.llm.latent_cortex.recurrence_adapter import ScopedLoRALinear

    wrapped: dict[str, ContinuousDepthConditionedLoRA] = {}
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    for layer_index, layer in enumerate(layers):
        for parent_name in ("self_attn", "linear_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in ("o_proj", "v_proj", "q_proj", "k_proj", "down_proj"):
                projection = getattr(parent, target, None)
                if not isinstance(projection, ScopedLoRALinear):
                    continue
                key = f"model.layers.{layer_index}.{parent_name}.{target}"
                bank = ContinuousDepthConditionedLoRA(
                    projection,
                    basis_size=basis_size,
                    delta_scale=delta_scale,
                )
                projection.depth_bank = bank
                wrapped[key] = bank
    if not wrapped:
        raise RuntimeError("no prepared projections accepted continuous depth")
    return wrapped


__all__ = [
    "ContinuousDepthConditionedLoRA",
    "DEPTH_CONDITIONED_SCHEMA",
    "DepthConditionedLoRA",
    "current_depth_index",
    "recurrent_depth_index",
    "wrap_depth_conditioned",
    "wrap_continuous_depth_conditioned",
]
