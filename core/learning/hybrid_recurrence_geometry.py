"""Recurrence geometry for a checkpoint whose layers are not all the same.

Qwen2.5-32B was 64 attention layers, so "the window is layers 16 to 47" said
everything. Qwen3.8-27B is `qwen3_5`: `full_attention_interval` 4, meaning a
layer carries ``self_attn`` only when ``(index + 1) % 4 == 0`` and otherwise
carries ``linear_attn``, a gated-delta recurrence with a state instead of a
K/V cache.

Two things go quietly wrong when the old geometry meets the new checkpoint.

The window stops being aligned. Each group of four layers ends in the
attention layer that mixes across tokens, so a window that starts or ends
mid-group brackets a partial group and the last thing an iteration does is a
purely positionwise update. The default fractions happen to give 16 and 48,
which are aligned; nothing was checking, so the next fraction anybody picks
is a coin flip.

The adapters thin out. ``_attach_window_adapters`` walks the window asking
each layer for ``self_attn``, and a layer that does not have one is skipped.
On the 32B a 32-layer window yielded 64 sites. On the 27B the same window
yields 16, because only 8 of those layers have attention at all. It does not
raise -- the site list is merely shorter, and a campaign trains a quarter of
the capacity it declared.

Both are geometry, so both are decidable before a model loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

HYBRID_GEOMETRY_SCHEMA: Final = "aura.hybrid_recurrence_geometry.v1"

#: Projections that live under an attention block, and under nothing else.
ATTENTION_PROJECTIONS: Final = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})

#: Projections that live under the feed-forward block. Present on every layer
#: of both checkpoints, which is what makes them the portable target.
MLP_PROJECTIONS: Final = frozenset({"gate_proj", "up_proj", "down_proj"})


@dataclass(frozen=True)
class LayerGeometry:
    """Which layers carry attention, decided from the checkpoint's own config."""

    num_hidden_layers: int
    full_attention_interval: int | None = None

    def __post_init__(self) -> None:
        if type(self.num_hidden_layers) is not int or self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be a positive integer")
        interval = self.full_attention_interval
        if interval is not None and (type(interval) is not int or interval < 1):
            raise ValueError("full_attention_interval must be a positive integer")

    @property
    def is_hybrid(self) -> bool:
        return self.full_attention_interval is not None

    def attention_layers(self) -> tuple[int, ...]:
        """Indices carrying ``self_attn``.

        Mirrors ``mlx_lm.models.qwen3_5.DecoderLayer``, which sets
        ``is_linear = (index + 1) % interval != 0``. A dense checkpoint has no
        interval and every layer qualifies.
        """
        if not self.is_hybrid:
            return tuple(range(self.num_hidden_layers))
        interval = int(self.full_attention_interval)
        return tuple(
            index
            for index in range(self.num_hidden_layers)
            if (index + 1) % interval == 0
        )

    def linear_layers(self) -> tuple[int, ...]:
        attention = set(self.attention_layers())
        return tuple(
            index for index in range(self.num_hidden_layers) if index not in attention
        )

    def carries_attention(self, index: int) -> bool:
        if not self.is_hybrid:
            return 0 <= index < self.num_hidden_layers
        return (index + 1) % int(self.full_attention_interval) == 0

    @classmethod
    def from_config(cls, config: Any) -> LayerGeometry:
        """Read the geometry out of a checkpoint config, nested or flat."""
        if not isinstance(config, dict):
            raise ValueError("config must be a mapping")
        text = config.get("text_config")
        source = text if isinstance(text, dict) else config
        layers = source.get("num_hidden_layers")
        if type(layers) is not int:
            raise ValueError("config declares no integer num_hidden_layers")
        interval = source.get("full_attention_interval")
        return cls(
            num_hidden_layers=layers,
            full_attention_interval=interval if isinstance(interval, int) else None,
        )

    @classmethod
    def from_model(cls, model: Any) -> LayerGeometry:
        """Read it off a loaded model, which knows its own layer kinds.

        ``is_linear`` is set per layer at construction, so this needs no
        config and cannot disagree with the object that will actually run.
        """
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None)
        if not layers:
            raise ValueError("model has no transformer layers")
        kinds = [bool(getattr(layer, "is_linear", False)) for layer in layers]
        if not any(kinds):
            return cls(num_hidden_layers=len(layers))
        attention = [index for index, linear in enumerate(kinds) if not linear]
        if not attention:
            raise ValueError("model has no attention layers")
        interval = attention[0] + 1
        expected = [i for i in range(len(layers)) if (i + 1) % interval == 0]
        if expected != attention:
            raise ValueError(
                "attention layers are not on a fixed interval; "
                f"found {attention[:8]}"
            )
        return cls(num_hidden_layers=len(layers), full_attention_interval=interval)


def window_alignment_errors(
    geometry: LayerGeometry, prelude_end: int, coda_start: int
) -> list[str]:
    """Reasons a recurrence window is wrong for this checkpoint.

    On a hybrid checkpoint the attention layer is the last of each group, so an
    aligned window starts on a group boundary and ends immediately after one.
    Both bounds are then multiples of the interval, and every iteration ends
    having just mixed across tokens.
    """
    errors: list[str] = []
    if not 0 <= prelude_end < coda_start <= geometry.num_hidden_layers:
        errors.append("window_bounds_outside_the_checkpoint")
        return errors
    if not geometry.is_hybrid:
        return errors
    interval = int(geometry.full_attention_interval)
    if prelude_end % interval:
        errors.append(
            f"prelude_end {prelude_end} is not a multiple of the "
            f"full-attention interval {interval}"
        )
    if coda_start % interval:
        errors.append(
            f"coda_start {coda_start} is not a multiple of the "
            f"full-attention interval {interval}"
        )
    if not any(
        geometry.carries_attention(index) for index in range(prelude_end, coda_start)
    ):
        errors.append("window contains no attention layer")
    return errors


def expected_adapter_sites(
    geometry: LayerGeometry,
    prelude_end: int,
    coda_start: int,
    targets: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Every site an attachment over this window is entitled to produce.

    Declared before the model loads so the trainer can compare what it got
    against what the campaign asked for. A short list is a silently smaller
    experiment, and this is what turns it into a refusal.
    """
    sites: list[str] = []
    for index in range(prelude_end, coda_start):
        for target in targets:
            if target in ATTENTION_PROJECTIONS:
                if geometry.carries_attention(index):
                    sites.append(f"model.layers.{index}.self_attn.{target}")
            elif target in MLP_PROJECTIONS:
                sites.append(f"model.layers.{index}.mlp.{target}")
    return tuple(sites)


def portable_targets(
    geometry: LayerGeometry, targets: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    """Targets that reach every layer of the window, attention or not.

    ``o_proj,v_proj`` was a complete choice while every layer had attention.
    On a hybrid checkpoint it reaches one layer in four, so a campaign that
    wants the whole window has to name a feed-forward projection too. This
    reports the gap rather than choosing for the operator.
    """
    if not geometry.is_hybrid:
        return tuple(targets)
    return tuple(target for target in targets if target in MLP_PROJECTIONS)


def geometry_receipt(
    geometry: LayerGeometry,
    prelude_end: int,
    coda_start: int,
    targets: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """What the campaign commits about the shape it is about to train."""
    sites = expected_adapter_sites(geometry, prelude_end, coda_start, targets)
    attention_in_window = [
        index
        for index in range(prelude_end, coda_start)
        if geometry.carries_attention(index)
    ]
    return {
        "schema": HYBRID_GEOMETRY_SCHEMA,
        "num_hidden_layers": geometry.num_hidden_layers,
        "full_attention_interval": geometry.full_attention_interval,
        "is_hybrid": geometry.is_hybrid,
        "window": [prelude_end, coda_start],
        "window_size": coda_start - prelude_end,
        "attention_layers_in_window": attention_in_window,
        "linear_layers_in_window": [
            index
            for index in range(prelude_end, coda_start)
            if not geometry.carries_attention(index)
        ],
        "targets": list(targets),
        "targets_reaching_every_window_layer": list(
            portable_targets(geometry, targets)
        ),
        "expected_adapter_sites": list(sites),
        "expected_adapter_site_count": len(sites),
        "alignment_errors": window_alignment_errors(geometry, prelude_end, coda_start),
    }
