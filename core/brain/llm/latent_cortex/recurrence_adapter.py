"""LoRA weights that are active only during latent-slot computation.

A recurrence-native adapter is not a personality adapter. It must not alter
prompt prefill, ordinary generation, the lexical answer decoder, or any other
resident-model caller. ``ScopedLoRALinear`` therefore returns the wrapped base
projection unless an explicit, task-local activation scope is open.

The optional position span supports the differentiable training view of the
live cache path: prompt, slots, and teacher-forced answer tokens can share one
causal sequence while the learned delta is applied only to the slot positions.
Live RLC calls contain slots only and use the full-span scope.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from mlx_lm.tuner.lora import LoRALinear

from core.runtime.model_layers import require_model_layers


@dataclass
class RecurrenceAdapterActivation:
    """Mutable receipt for one lexical or latent execution boundary.

    ``calls`` and ``adapted_positions`` are aggregates: they prove *something*
    fired, which is what the CP227 repair needed. They cannot distinguish "all
    sixteen wrapped projections fired" from "one did and fifteen were silently
    never wrapped", and that difference is a treatment that is really a
    quarter of a treatment. ``applied_blocks`` and ``applied_sites`` record
    identity per application so the caller can compare what fired against what
    attachment claimed to wrap.
    """

    start: int | None = None
    stop: int | None = None
    calls: int = 0
    adapted_positions: int = 0
    observed_positions: int = 0
    applied_blocks: dict[int, int] = field(default_factory=dict)
    applied_sites: dict[str, int] = field(default_factory=dict)

    def record_application(self, *, block_index: int | None, site: str | None) -> None:
        """Note that one identified projection actually applied its delta."""

        if block_index is not None:
            self.applied_blocks[block_index] = self.applied_blocks.get(block_index, 0) + 1
        if site is not None:
            self.applied_sites[site] = self.applied_sites.get(site, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "stop": self.stop,
            "calls": self.calls,
            "adapted_positions": self.adapted_positions,
            "observed_positions": self.observed_positions,
            "applied_blocks": dict(sorted(self.applied_blocks.items())),
            "applied_sites": dict(sorted(self.applied_sites.items())),
        }

    def activated_blocks(self) -> list[int]:
        """Block indices that actually applied a delta, in order."""

        return sorted(self.applied_blocks)

    def unfired_sites(self, expected_sites: Iterable[str]) -> list[str]:
        """Attachment sites that were wrapped but never applied anything.

        A non-empty result is the CP227 shape caught while it is still a
        measurement rather than a published verdict.
        """

        return sorted(set(expected_sites) - set(self.applied_sites))

    def absorb(self, nested: RecurrenceAdapterActivation) -> None:
        """Aggregate a completed nested scope into this enclosing receipt."""

        if nested is self:
            raise ValueError("recurrence adapter activation cannot absorb itself")
        self.calls += nested.calls
        self.adapted_positions += nested.adapted_positions
        self.observed_positions += nested.observed_positions
        for block, count in nested.applied_blocks.items():
            self.applied_blocks[block] = self.applied_blocks.get(block, 0) + count
        for site, count in nested.applied_sites.items():
            self.applied_sites[site] = self.applied_sites.get(site, 0) + count


_ACTIVE_SCOPE: ContextVar[RecurrenceAdapterActivation | None] = ContextVar(
    "aura_recurrence_adapter_scope",
    default=None,
)
_ACTIVE_CODA_SCOPE: ContextVar[RecurrenceAdapterActivation | None] = ContextVar(
    "aura_coda_adapter_scope",
    default=None,
)
_ACTIVATION_COLLECTOR: ContextVar[RecurrenceAdapterActivation | None] = ContextVar(
    "aura_recurrence_adapter_activation_collector",
    default=None,
)
_DISABLE_DEPTH: ContextVar[int] = ContextVar(
    "aura_recurrence_adapter_disable_depth",
    default=0,
)
_CODA_DISABLE_DEPTH: ContextVar[int] = ContextVar(
    "aura_coda_adapter_disable_depth",
    default=0,
)

_KNOWN_SCOPED_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def current_recurrence_adapter_scope() -> RecurrenceAdapterActivation | None:
    """Return the current task-local activation, if one is open."""

    return _ACTIVE_SCOPE.get()


def current_coda_adapter_scope() -> RecurrenceAdapterActivation | None:
    """Return the active RLC-only coda interpretation scope, if any."""

    return _ACTIVE_CODA_SCOPE.get()


@contextmanager
def recurrence_adapter_activation_collector() -> Iterator[RecurrenceAdapterActivation]:
    """Collect nested activation receipts without activating other calls."""

    parent = _ACTIVATION_COLLECTOR.get()
    collector = RecurrenceAdapterActivation()
    token = _ACTIVATION_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        _ACTIVATION_COLLECTOR.reset(token)
        if parent is not None:
            parent.absorb(collector)


def scoped_recurrence_adapter_sites(
    model: Any,
    *,
    layer_indices: Iterable[int],
) -> tuple[str, ...]:
    """Enumerate and verify every recurrence-scoped projection in a layer set."""

    layers = require_model_layers(model).layers
    sites: list[str] = []
    for index in layer_indices:
        if type(index) is not int or not 0 <= index < len(layers):
            raise ValueError("recurrence adapter layer inventory is invalid")
        layer = layers[index]
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in _KNOWN_SCOPED_PROJECTIONS:
                projection = getattr(parent, target, None)
                if not isinstance(projection, ScopedLoRALinear):
                    continue
                expected = f"model.layers.{index}.{parent_name}.{target}"
                if (
                    getattr(projection, "recurrence_block_index", None) != index
                    or getattr(projection, "recurrence_site", None) != expected
                ):
                    raise ValueError(
                        f"recurrence adapter identity is incomplete: {expected}"
                    )
                sites.append(expected)
    if not sites:
        raise ValueError("no recurrence-scoped projection is attached")
    return tuple(sorted(sites))


def scoped_coda_adapter_sites(
    model: Any,
    *,
    layer_indices: Iterable[int],
) -> tuple[str, ...]:
    """Enumerate and verify every RLC-only coda projection in a layer set."""

    layers = require_model_layers(model).layers
    sites: list[str] = []
    for index in layer_indices:
        if type(index) is not int or not 0 <= index < len(layers):
            raise ValueError("coda adapter layer inventory is invalid")
        layer = layers[index]
        for parent_name in ("self_attn", "mlp"):
            parent = getattr(layer, parent_name, None)
            if parent is None:
                continue
            for target in _KNOWN_SCOPED_PROJECTIONS:
                projection = getattr(parent, target, None)
                if not isinstance(projection, ScopedCodaLoRALinear):
                    continue
                expected = f"model.layers.{index}.{parent_name}.{target}"
                if (
                    getattr(projection, "recurrence_block_index", None) != index
                    or getattr(projection, "recurrence_site", None) != expected
                ):
                    raise ValueError(f"coda adapter identity is incomplete: {expected}")
                sites.append(expected)
    if not sites:
        raise ValueError("no coda-scoped projection is attached")
    return tuple(sorted(sites))


@contextmanager
def recurrence_adapter_disabled() -> Iterator[None]:
    """Run the same recurrent graph against the frozen base projection.

    Inner recurrence scopes still open and preserve graph structure, but their
    LoRA deltas remain disabled. Nesting is counted so independent callers can
    compose reference-policy work without accidentally re-enabling an outer
    boundary.
    """

    token = _DISABLE_DEPTH.set(_DISABLE_DEPTH.get() + 1)
    try:
        yield
    finally:
        _DISABLE_DEPTH.reset(token)


@contextmanager
def coda_adapter_disabled() -> Iterator[None]:
    """Lesion only the RLC coda interpreter while recurrence remains active."""

    token = _CODA_DISABLE_DEPTH.set(_CODA_DISABLE_DEPTH.get() + 1)
    try:
        yield
    finally:
        _CODA_DISABLE_DEPTH.reset(token)


@contextmanager
def recurrence_adapter_scope(
    *,
    start: int | None = None,
    stop: int | None = None,
) -> Iterator[RecurrenceAdapterActivation]:
    """Activate recurrent LoRA deltas for all or a slice of sequence positions.

    ``start`` and ``stop`` follow normal non-negative slice semantics over the
    sequence axis. Both omitted means every position. Nested scopes restore the
    parent exactly, and ``ContextVar`` keeps concurrent requests isolated.
    """

    if (start is None) != (stop is None):
        raise ValueError("recurrence adapter start and stop must be supplied together")
    if start is not None and (
        type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
    ):
        raise ValueError("recurrence adapter span must be a non-empty positive slice")
    parent = _ACTIVE_SCOPE.get()
    activation = RecurrenceAdapterActivation(start=start, stop=stop)
    token = _ACTIVE_SCOPE.set(activation)
    try:
        yield activation
    finally:
        _ACTIVE_SCOPE.reset(token)
        if parent is not None:
            parent.absorb(activation)
        else:
            collector = _ACTIVATION_COLLECTOR.get()
            if collector is not None:
                collector.absorb(activation)


@contextmanager
def coda_adapter_scope(
    *,
    start: int | None = None,
    stop: int | None = None,
) -> Iterator[RecurrenceAdapterActivation]:
    """Activate interpretation tissue only for an RLC persistence/decode path.

    This scope is deliberately independent from ``recurrence_adapter_scope``.
    The recurrent operator learns how to transform latent slots; the coda
    operator learns how to interpret the resulting state.  Ordinary model
    calls open neither scope and therefore remain exact base-checkpoint
    inference.
    """

    if (start is None) != (stop is None):
        raise ValueError("coda adapter start and stop must be supplied together")
    if start is not None and (
        type(start) is not int
        or type(stop) is not int
        or start < 0
        or stop <= start
    ):
        raise ValueError("coda adapter span must be a non-empty positive slice")
    parent = _ACTIVE_CODA_SCOPE.get()
    activation = RecurrenceAdapterActivation(start=start, stop=stop)
    token = _ACTIVE_CODA_SCOPE.set(activation)
    try:
        yield activation
    finally:
        _ACTIVE_CODA_SCOPE.reset(token)
        if parent is not None:
            parent.absorb(activation)
        else:
            collector = _ACTIVATION_COLLECTOR.get()
            if collector is not None:
                collector.absorb(activation)


class _ScopedLoRALinearBase(LoRALinear):  # type: ignore[misc]
    """Common implementation for independently scoped cognitive adapters."""

    @classmethod
    def from_base(
        cls,
        linear: Any,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        *,
        block_index: int | None = None,
        site: str | None = None,
    ) -> ScopedLoRALinear:
        """Wrap ``linear`` without the base class factory erasing our subtype.

        ``block_index`` and ``site`` are optional identity. When an attachment
        site supplies them, every application is attributable, so a projection
        that was wrapped but never fired can be named instead of hiding inside
        an aggregate call count.
        """

        from core.brain.llm.latent_cortex.fast_weights import _linear_dims

        output_dims, input_dims = _linear_dims(linear)
        scoped = cls(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
        )
        scoped.linear = linear
        scoped.exact_episodic_operation = False
        if block_index is not None:
            if type(block_index) is not int or block_index < 0:
                raise ValueError("recurrence adapter block index must be a non-negative int")
            scoped.recurrence_block_index = block_index
        if site is not None:
            if not isinstance(site, str) or not site.strip():
                raise ValueError("recurrence adapter site must be a non-empty string")
            scoped.recurrence_site = site
        return scoped

    def __call__(self, x: Any) -> Any:
        activation = self._active_scope()
        y = self.linear(x)
        if activation is None or _DISABLE_DEPTH.get() > 0:
            return y

        sequence_length = int(x.shape[-2])
        activation.calls += 1
        activation.observed_positions += sequence_length
        # Depth conditioning (CP219): when a per-depth bank is attached, the
        # EFFECTIVE operator varies with the recurrent step, so step t and
        # step t+1 compute different functions. A phase code injected into
        # the input is re-absorbed by alpha-interpolation; changing the
        # operator is what actually differentiates the steps. Absent a
        # bank this is bit-identical to the shared adapter.
        bank = getattr(self, "depth_bank", None)
        if bank is None:
            lora_a, lora_b = self.lora_a, self.lora_b
        else:
            from core.learning.depth_conditioned_lora import current_depth_index

            lora_a, lora_b = bank.factors_for(current_depth_index())
        role_bank = getattr(self, "role_bank", None)
        if role_bank is not None:
            from core.learning.role_conditioned_lora import current_branch_index

            lora_a, lora_b = role_bank.factors_for(
                lora_a,
                lora_b,
                current_branch_index(),
            )
        z = (self.dropout(x) @ lora_a) @ lora_b
        block_index = getattr(self, "recurrence_block_index", None)
        site = getattr(self, "recurrence_site", None)
        if activation.start is None or activation.stop is None:
            activation.adapted_positions += sequence_length
            activation.record_application(block_index=block_index, site=site)
            correction = self.scale * z
            if not bool(getattr(self, "exact_episodic_operation", False)):
                correction = correction.astype(x.dtype)
            return y + correction

        start = int(activation.start)
        stop = int(activation.stop)
        if stop > sequence_length:
            raise ValueError(
                "recurrence adapter span exceeds sequence length: "
                f"span=[{start}:{stop}) sequence={sequence_length}"
            )
        import mlx.core as mx

        positions = mx.arange(sequence_length)
        mask = ((positions >= start) & (positions < stop)).astype(x.dtype)
        shape = (1,) * max(0, x.ndim - 2) + (sequence_length, 1)
        activation.adapted_positions += stop - start
        activation.record_application(block_index=block_index, site=site)
        correction = self.scale * z * mx.reshape(mask, shape)
        if not bool(getattr(self, "exact_episodic_operation", False)):
            correction = correction.astype(x.dtype)
        return y + correction

    def _active_scope(self) -> RecurrenceAdapterActivation | None:
        raise NotImplementedError


class ScopedLoRALinear(_ScopedLoRALinearBase):
    """LoRA projection active only in the recurrent latent-slot window."""

    def _active_scope(self) -> RecurrenceAdapterActivation | None:
        return _ACTIVE_SCOPE.get()


class ScopedCodaLoRALinear(_ScopedLoRALinearBase):
    """LoRA projection active only while interpreting an RLC-derived state."""

    def _active_scope(self) -> RecurrenceAdapterActivation | None:
        if _CODA_DISABLE_DEPTH.get() > 0:
            return None
        return _ACTIVE_CODA_SCOPE.get()


__all__ = [
    "RecurrenceAdapterActivation",
    "ScopedCodaLoRALinear",
    "ScopedLoRALinear",
    "coda_adapter_disabled",
    "coda_adapter_scope",
    "current_coda_adapter_scope",
    "current_recurrence_adapter_scope",
    "recurrence_adapter_activation_collector",
    "recurrence_adapter_disabled",
    "recurrence_adapter_scope",
    "scoped_coda_adapter_sites",
    "scoped_recurrence_adapter_sites",
]
