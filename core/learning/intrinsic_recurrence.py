"""Make the checkpoint itself recurrent (CP226).

The RLC as built was a model that TALKS to a recurrent workspace. Read
``_persist_and_score``: the answer tokens traverse ``layers[prelude:coda]``
exactly ONCE, at every depth setting. Only the four slot positions were
ever recurred. So the computation that produces the answer always received
the base checkpoint's 64 layers -- identical to vanilla -- and "depth"
changed nothing except what got written into a scratchpad the answer
attends to.

The measurement follows from the architecture and should have been
predicted from it:

    RLC depth 1 / 2 / 4 / 8 : 25 / 29 / 25 / 25 %   (8x compute, flat)
    vanilla greedy          : 21 %

Slots are causal and worth a few points as a prior. Depth is worth nothing
because no depth was ever applied to the answer's own computation.

This module applies it. The real token stream re-enters the middle block
``T`` times:

    h = layers[:prelude](h)                     # prelude, once
    for t in range(T):  h = layers[prelude:coda](h)
    h = layers[coda:](h)                        # coda, once

Effective depth becomes ``prelude + T*(coda-prelude) + (L-coda)``: a 64
layer checkpoint runs 160 layers deep at T=4, with the SAME weights. This
is the Ouro/LoopLM architecture, retrofitted onto a checkpoint that was not
pretrained for it -- which is the only reason the stabilizers below exist.

The load-bearing safety property: at ``T=1`` this is bit-identical to the
base forward pass. Recurrence is added by increasing T from a known-good
starting point, never by a cutover.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

INTRINSIC_RECURRENCE_SCHEMA = "aura.intrinsic_recurrence.v1"


@dataclass(frozen=True)
class RecurrentDepthPlan:
    """Which layers loop, how many times, and how the loop is stabilized.

    A checkpoint pretrained without recurrence has no reason for its middle
    block to be a stable map. Iterating it can drift in norm until the coda
    receives activations outside anything it was trained on -- the retrofit
    failure mode. ``anchor_injection`` and ``renormalize`` exist for that,
    and both default to OFF so the plain loop is what gets measured first.
    """

    prelude_end: int
    coda_start: int
    iterations: int = 1
    # Re-inject the prelude output at each RE-entry (never the first), the
    # retrofitted-recurrence stabilizer. Keeps deep loops anchored to the
    # problem instead of drifting into a self-referential fixed point.
    anchor_injection: float = 0.0
    # Rescale to the anchor's RMS before re-entry. Controls norm blowup
    # without changing direction.
    renormalize: bool = False
    # Deterministic isotropic perturbation at each RE-entry, relative to the
    # state's RMS. The divergence lever for the cos(pass1,pass2)=0.9994
    # obstacle: a damped iteration of one fixed map aligns successive
    # increments with its dominant eigendirection (power iteration), so the
    # loop re-computes the same step. Seeded noise kicks the state off that
    # ray so later passes can explore directions the contraction never
    # visits. Never applied at the first pass — T=1 stays bit-identical.
    interpass_noise: float = 0.0
    noise_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("prelude_end", "coda_start", "iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.prelude_end >= self.coda_start:
            raise ValueError(
                "recurrent window is empty: prelude_end must precede coda_start"
            )
        if (
            isinstance(self.anchor_injection, bool)
            or not isinstance(self.anchor_injection, (int, float))
            or not 0.0 <= float(self.anchor_injection) <= 1.0
        ):
            raise ValueError("anchor_injection must be inside [0, 1]")
        if type(self.renormalize) is not bool:
            raise ValueError("renormalize must be a bool")
        if (
            isinstance(self.interpass_noise, bool)
            or not isinstance(self.interpass_noise, (int, float))
            or not 0.0 <= float(self.interpass_noise) <= 1.0
        ):
            raise ValueError("interpass_noise must be inside [0, 1]")
        if type(self.noise_seed) is not int or self.noise_seed < 0:
            raise ValueError("noise_seed must be a non-negative integer")

    def window_size(self) -> int:
        return self.coda_start - self.prelude_end

    def effective_depth(self, total_layers: int) -> int:
        """Layer applications per token -- the honest 'how deep is this'."""
        if type(total_layers) is not int or total_layers < self.coda_start:
            raise ValueError("total_layers is smaller than the plan's window")
        return (
            self.prelude_end
            + self.iterations * self.window_size()
            + (total_layers - self.coda_start)
        )

    def is_base_equivalent(self) -> bool:
        """True when this plan reduces to the unmodified forward pass."""
        return self.iterations == 1

    def to_receipt(self, total_layers: int) -> dict[str, Any]:
        return {
            "schema": INTRINSIC_RECURRENCE_SCHEMA,
            "prelude_end": self.prelude_end,
            "coda_start": self.coda_start,
            "window_size": self.window_size(),
            "iterations": self.iterations,
            "anchor_injection": float(self.anchor_injection),
            "renormalize": self.renormalize,
            "interpass_noise": float(self.interpass_noise),
            "noise_seed": self.noise_seed,
            "total_layers": total_layers,
            "effective_depth": self.effective_depth(total_layers),
            "base_equivalent": self.is_base_equivalent(),
        }


from contextlib import contextmanager  # noqa: E402
from contextvars import ContextVar  # noqa: E402


def _rms(tensor: Any) -> Any:
    """RMS in float32.

    Transformer residual streams carry famously large outlier activations;
    squaring them in fp16 overflows to inf and the renormalizer then hands
    the model NaN. Measured: every renormalize=True setting produced a
    non-finite state at iteration 1, which looked like the checkpoint being
    unstable and was actually this.
    """
    import mlx.core as mx

    wide = tensor.astype(mx.float32)
    return mx.sqrt(mx.mean(mx.square(wide), axis=-1, keepdims=True) + 1e-6)


class _CheckpointState:
    """Gradient checkpointing over the recurrent window.

    Not an optimization. Backprop through T passes retains T*window layer
    activations; at T=4 over a 32-layer window on a 32B that is 128 layers
    of retained activation, which is the road back to the 103 GB incident.
    """

    def __init__(self, model: Any, group_size: int) -> None:
        self.model = model
        self.group_size = group_size
        self.wrappers: dict[Any, Any] = {}

    def traced_parameters(self) -> Any:
        """The TRAINABLE parameters, read at forward time.

        Two constraints meet here:

        * Read at forward time, never at scope entry. mx.checkpoint
          recomputes its group from the arrays handed to it, so those must
          be the ones value_and_grad is tracing. A snapshot taken when the
          scope opened is disconnected from the trace and the gradient
          comes back all zeros -- silently, because a zero gradient looks
          like a converged step rather than a broken one.
        * Trainable only. mx.checkpoint differentiates with respect to its
          inputs, so handing it the full tree makes it try to differentiate
          the frozen 4-bit base weights: "[QuantizedMatmul::vjp] no
          gradient wrt the quantized weights."
        """
        return self.model.trainable_parameters()


_CHECKPOINTS: ContextVar[Any] = ContextVar(
    "intrinsic_recurrence_checkpoints", default=None
)


@contextmanager
def checkpointed_window(model: Any, *, group_size: int = 4):
    """Trade recompute for memory inside the recurrent forward."""
    if type(group_size) is not int or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    token = _CHECKPOINTS.set(_CheckpointState(model, group_size))
    try:
        yield
    finally:
        _CHECKPOINTS.reset(token)


def _run(layers: Any, hidden: Any, caches: Any = None) -> Any:
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    layer_list = list(layers)
    if not layer_list:
        return hidden
    state = _CHECKPOINTS.get()
    if caches is None and state is not None:
        # Checkpointing and KV caches are mutually exclusive: a cache is
        # mutated in place, so recomputing a checkpointed group would append
        # its keys a second time and silently corrupt the attention history.
        # Training has no cache; decode has no checkpointing.
        for offset in range(0, len(layer_list), state.group_size):
            group = tuple(layer_list[offset : offset + state.group_size])
            key = tuple(id(layer) for layer in group)
            call = state.wrappers.get(key)
            if call is None:

                def group_call(
                    all_parameters: Any,
                    value: Any,
                    _group: tuple[Any, ...] = group,
                ) -> Any:
                    state.model.update(all_parameters)
                    for member in _group:
                        value = member(
                            value, create_attention_mask(value, None), None
                        )
                    return value

                call = mx.checkpoint(group_call)
                state.wrappers[key] = call
            hidden = call(state.traced_parameters(), hidden)
        return hidden
    if caches is None:
        mask = create_attention_mask(hidden, None)
        for layer in layer_list:
            hidden = layer(hidden, mask, None)
        return hidden
    if state is not None:
        raise ValueError(
            "gradient checkpointing cannot be combined with KV caches: "
            "recomputing a checkpointed group would append its keys twice"
        )
    if len(caches) != len(layer_list):
        raise ValueError("cache count does not match the layer count")
    mask = create_attention_mask(hidden, caches)
    for layer, cache in zip(layer_list, caches):
        hidden = layer(hidden, mask, cache)
    return hidden


def model_layer_caches(model: Any) -> list[Any]:
    """One cache object per layer, of the type that layer actually needs.

    A hybrid checkpoint interleaves attention layers, which hold K/V, with
    linear-attention layers, which hold a recurrent state instead. Qwen3.8-27B
    is 16 of the former and 48 of the latter, and ``mlx_lm`` defers to the
    model's own ``make_cache`` for exactly that reason. Building 64 plain
    ``KVCache`` objects hands the loop the wrong object at three layers in
    four -- the shape that reads as a working decode right up until the
    linear layers silently carry no state.
    """
    factory = getattr(model, "make_cache", None)
    if callable(factory):
        return list(factory())
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) or ()
    from mlx_lm.models.cache import KVCache

    return [KVCache() for _ in range(len(layers))]


def make_recurrent_caches(model: Any, plan: RecurrentDepthPlan) -> dict[str, Any]:
    """Per-iteration KV caches for an intrinsically-recurrent decode.

    Each pass through the window is a DIFFERENT computation over the same
    weights, so pass ``t`` must attend to the keys previous tokens produced
    at pass ``t`` -- one shared cache would let iteration 4 read iteration
    1's history and silently corrupt the recurrence. This is also what
    makes decode O(n) instead of O(n^2); the unbounded quadratic version is
    what drove this host to 103 GB.
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        raise ValueError("model has no transformer layers")
    if plan.coda_start > len(layers):
        raise ValueError("plan's coda_start exceeds the model's layer count")
    return {
        "prelude": model_layer_caches(model)[: plan.prelude_end],
        "window": [
            model_layer_caches(model)[plan.prelude_end : plan.coda_start]
            for _ in range(plan.iterations)
        ],
        "coda": model_layer_caches(model)[plan.coda_start :],
    }


def recurrent_hidden_states(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    *,
    caches: dict[str, Any] | None = None,
) -> tuple[Any, list[Any]]:
    """Run the intrinsically-recurrent forward pass.

    Returns ``(final_hidden, per_iteration_states)``. The trajectory is
    returned rather than discarded because every honest question about
    recurrence -- is it converging, oscillating, or improving -- is a
    question about the intermediate states, and outcome-only scoring is
    exactly how a period-2 limit cycle went unnoticed before.
    """
    import mlx.core as mx

    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not layers:
        raise ValueError("model has no transformer layers")
    if plan.coda_start > len(layers):
        raise ValueError("plan's coda_start exceeds the model's layer count")

    if caches is not None and set(caches) != {"prelude", "window", "coda"}:
        raise ValueError("caches must come from make_recurrent_caches")
    if caches is not None and len(caches["window"]) != plan.iterations:
        raise ValueError("cache iteration count does not match the plan")

    hidden = inner.embed_tokens(tokens)
    hidden = _run(
        layers[: plan.prelude_end],
        hidden,
        caches["prelude"] if caches else None,
    )
    anchor = hidden
    anchor_rms = _rms(anchor) if plan.renormalize else None

    trajectory: list[Any] = []
    window = layers[plan.prelude_end : plan.coda_start]
    for iteration in range(plan.iterations):
        if iteration > 0:
            # Re-entry only. The first pass must be untouched, which is
            # what keeps T=1 bit-identical to the base model.
            if plan.anchor_injection > 0.0:
                hidden = hidden + plan.anchor_injection * anchor
            if plan.interpass_noise > 0.0:
                # Deterministic per (seed, iteration): the same plan always
                # produces the same trajectory, so results replay. Scaled by
                # the CURRENT state's RMS so the kick is proportional at
                # every depth instead of vanishing as norms grow.
                key = mx.random.key(plan.noise_seed * 1_000_003 + iteration)
                kick = mx.random.normal(hidden.shape, key=key).astype(
                    mx.float32
                ) * (plan.interpass_noise * _rms(hidden))
                hidden = hidden + kick.astype(hidden.dtype)
            if plan.renormalize:
                scale = (anchor_rms / _rms(hidden)).astype(hidden.dtype)
                hidden = hidden * scale
        with recurrent_iteration(iteration):
            hidden = _run(
                window, hidden, caches["window"][iteration] if caches else None
            )
        trajectory.append(hidden)

    hidden = _run(
        layers[plan.coda_start :], hidden, caches["coda"] if caches else None
    )
    hidden = inner.norm(hidden)
    return hidden, trajectory


def recurrent_logits(
    model: Any,
    tokens: Any,
    plan: RecurrentDepthPlan,
    *,
    caches: dict[str, Any] | None = None,
) -> Any:
    """Logits from the intrinsically-recurrent pass."""
    hidden, _ = recurrent_hidden_states(model, tokens, plan, caches=caches)
    if getattr(model, "lm_head", None) is not None:
        return model.lm_head(hidden)
    return model.model.embed_tokens.as_linear(hidden)


# ── Per-iteration identity, so the loop can know where it is ────────────

_ITERATION: ContextVar[int] = ContextVar("intrinsic_recurrence_iteration", default=0)


def current_iteration() -> int:
    """Which pass through the window is executing.

    Published so depth-conditioned adapters can specialize per iteration:
    the same weights doing genuinely different work at pass 1 and pass 4 is
    the difference between deepening and repeating.
    """
    return _ITERATION.get()


@contextmanager
def recurrent_iteration(index: int):
    """Publish the iteration index on BOTH recurrence clocks.

    ``depth_conditioned_lora`` keys its per-depth weight deltas off its own
    ContextVar, which until now was set only by the slot-recurrence loop.
    Leaving the two clocks separate would mean the depth bank silently
    reports depth 0 for every pass here -- a mechanism present in name
    only, which is the same defect CP211 had to repair in v4. So this sets
    both, and a test pins that the bank sees the intrinsic index.

    This matters specifically because of what the 1.5B sweep measured:
    cos(pass 1, pass 2) = 0.9994. Applying the SAME map repeatedly barely
    rotates the state, so depth buys little. Per-iteration weight deltas
    make each pass a different function, which is what turns repetition
    into deepening.
    """
    if type(index) is not int or index < 0:
        raise ValueError("iteration index must be a non-negative integer")
    from core.learning.depth_conditioned_lora import recurrent_depth_index

    token = _ITERATION.set(index)
    try:
        with recurrent_depth_index(index):
            yield index
    finally:
        _ITERATION.reset(token)


# ── Does the extra depth actually compute anything? ─────────────────────


def trajectory_dynamics(trajectory: list[Any]) -> dict[str, Any]:
    """Is the loop converging, spinning, or still moving?

    Measured because "the state changed" is not evidence of thinking. A
    contracting loop reaches a fixed point and every further iteration is
    wasted compute; an oscillating one alternates between two states
    forever. Both look like activity and neither is progress.
    """
    import mlx.core as mx

    if len(trajectory) < 2:
        return {
            "schema": INTRINSIC_RECURRENCE_SCHEMA,
            "iterations": len(trajectory),
            "measurable": False,
            "diverged": False,
            "reason": "need at least two iterations to measure motion",
        }
    # A retrofitted loop can overflow: the middle block of a checkpoint
    # never pretrained to iterate is not a stable map, and in fp16 the
    # norm can run away inside a few passes. Detect it FIRST -- every
    # statistic below is meaningless once the state is non-finite, and an
    # accuracy scored from NaN logits is noise wearing a number's clothes.
    finite = all(
        bool(mx.all(mx.isfinite(state.astype(mx.float32))))
        for state in trajectory
    )
    if not finite:
        return {
            "schema": INTRINSIC_RECURRENCE_SCHEMA,
            "iterations": len(trajectory),
            "measurable": False,
            "diverged": True,
            "reason": "non-finite hidden state: the loop overflowed",
        }

    # float32 throughout: an fp16 mean over a long sequence overflows, and
    # the resulting 0.0/nan reads as "the loop is not moving" when it is
    # actually the measurement that broke.
    wide = [state.astype(mx.float32) for state in trajectory]
    deltas = []
    for previous, current in zip(wide, wide[1:]):
        deltas.append(
            float(mx.mean(mx.abs(current - previous)))
            / max(float(mx.mean(mx.abs(previous))), 1e-9)
        )
    contracting = all(
        later <= earlier * 1.05 for earlier, later in zip(deltas, deltas[1:])
    )
    oscillating = False
    if len(trajectory) >= 3:
        flat = [mx.reshape(state, (-1,)) for state in wide]
        steps = [b - a for a, b in zip(flat, flat[1:])]
        cosines = [
            float(
                mx.sum(a * b)
                / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-9)
            )
            for a, b in zip(steps, steps[1:])
        ]
        oscillating = bool(cosines) and all(c < -0.5 for c in cosines)
    return {
        "schema": INTRINSIC_RECURRENCE_SCHEMA,
        "iterations": len(trajectory),
        "measurable": True,
        "diverged": False,
        "relative_deltas": [round(d, 6) for d in deltas],
        "final_delta": round(deltas[-1], 6),
        "contracting": bool(contracting),
        # A loop that has stopped moving has stopped computing, whatever
        # the compute budget says.
        "at_fixed_point": bool(deltas[-1] < 0.01),
        "oscillating": oscillating,
    }


__all__ = [
    "INTRINSIC_RECURRENCE_SCHEMA",
    "RecurrentDepthPlan",
    "checkpointed_window",
    "current_iteration",
    "make_recurrent_caches",
    "recurrent_hidden_states",
    "recurrent_iteration",
    "recurrent_logits",
    "trajectory_dynamics",
]
