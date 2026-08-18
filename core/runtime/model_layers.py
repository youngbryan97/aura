"""Validated access to transformer layers across supported model wrappers.

Model topology is runtime data, not a naming convention.  MLX model families
and loader versions expose the same transformer stack through different wrapper
paths; consumers must resolve that structure once rather than each assuming a
different ``model.model.layers`` shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ModelLayerView:
    """A validated transformer stack and the object that owns its forward pass."""

    owner: Any
    layers: Sequence[Any]
    path: str


_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    # Prefer nested forward owners over convenience properties on an outer
    # language-model wrapper.  Qwen's top-level ``Model.layers`` forwards to
    # ``Model.model.layers``, but only the nested owner has ``embed_tokens``
    # and ``norm`` and can safely receive a forward-pass patch.
    ("model", "layers"),
    ("model", "transformer", "layers"),
    ("language_model", "model", "layers"),
    ("language_model", "layers"),
    ("transformer", "layers"),
    ("layers",),
)


def resolve_model_layers(model: Any) -> ModelLayerView | None:
    """Return a validated transformer layer view for a known model topology.

    The resolver is deliberately bounded to explicit model-family layouts.  It
    does not recursively inspect arbitrary attributes, which could select a
    draft model, vision tower, or decoder-like auxiliary stack by accident.
    """

    if model is None:
        return None

    for path in _LAYER_PATHS:
        current = model
        owner = model
        try:
            for part in path:
                owner = current
                current = getattr(current, part)
        except (AttributeError, RuntimeError, TypeError):
            continue

        layers = current
        if layers is None or isinstance(layers, (str, bytes, bytearray)):
            continue
        try:
            count = len(layers)
        except (TypeError, RuntimeError, AttributeError):
            continue
        if count <= 0:
            continue
        try:
            layers[0]
        except (TypeError, RuntimeError, AttributeError, IndexError, KeyError):
            continue
        return ModelLayerView(owner=owner, layers=layers, path=".".join(path))

    return None


def require_model_layers(model: Any) -> ModelLayerView:
    """Resolve transformer layers or raise a stable, diagnostic error."""

    view = resolve_model_layers(model)
    if view is None:
        model_type = f"{type(model).__module__}.{type(model).__qualname__}"
        raise ValueError(f"unsupported_model_layer_topology:{model_type}")
    return view
