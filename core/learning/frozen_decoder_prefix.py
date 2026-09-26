"""Reuse an immutable decoder prefix while training its native suffix."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from core.brain.llm.decoder_topology import (
    decoder_backbone,
    decoder_backbone_owner,
    decoder_layer_masks,
)


class NativeDecoderSuffix(nn.Module):
    """The original blocks, normalization and vocabulary projection after a split.

    Modules are shared with the loaded model. No projection is reinitialized.
    This object does not grant serving authority or modify the frozen prefix.
    """

    def __init__(self, model: Any, *, split_at: int):
        super().__init__()
        backbone = decoder_backbone(model)
        owner = decoder_backbone_owner(model)
        if (type(split_at) is not int or not 0 < split_at < len(backbone.layers)
                or not callable(getattr(backbone, "norm", None))):
            raise ValueError("native suffix requires a nonempty decoder split")
        self.layers = list(backbone.layers[split_at:])
        self.norm = backbone.norm
        self.tied_output = bool(getattr(owner.args, "tie_word_embeddings", False))
        self.output = backbone.embed_tokens if self.tied_output else owner.lm_head

    def __call__(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] < 1:
            raise ValueError("native suffix requires a complete hidden sequence")
        masks = decoder_layer_masks(self, hidden)
        for layer, mask in zip(self.layers, masks, strict=True):
            hidden = layer(hidden, mask=mask, cache=None)
        hidden = self.norm(hidden)
        return self.output.as_linear(hidden) if self.tied_output else self.output(hidden)


class FrozenDecoderPrefix:
    """Execute frozen embeddings and blocks with the decoder's own layer masks.

    Capture is valid only while this prefix has no trainable parameters and
    runs in evaluation mode. Inputs include the whole teacher-forced sequence;
    shortening them would change the causal token alignment at the boundary.
    Callers own checkpoint identity and any persistence of captured states.
    """

    def __init__(self, model: Any, *, split_at: int):
        self.backbone = decoder_backbone(model)
        if type(split_at) is not int or not 0 < split_at < len(self.backbone.layers):
            raise ValueError("frozen prefix requires a nonempty decoder split")
        self.split_at = split_at
        self.embedding = self.backbone.embed_tokens
        self.layers = tuple(self.backbone.layers[:split_at])
        self._assert_frozen()

    def _assert_frozen(self):
        from mlx.utils import tree_flatten

        for module in (self.embedding, *self.layers):
            if module.training or tree_flatten(module.trainable_parameters()):
                raise ValueError("captured prefix must be frozen and in evaluation mode")

    def capture(self, token_ids):
        self._assert_frozen()
        if (token_ids.ndim != 2 or min(token_ids.shape) < 1
                or token_ids.dtype not in (mx.int32, mx.int64, mx.uint32, mx.uint64)):
            raise ValueError("frozen prefix requires integer token sequences")
        hidden = self.embedding(token_ids)
        masks = decoder_layer_masks(self.backbone, hidden, start=0, end=self.split_at)
        for layer, mask in zip(self.layers, masks, strict=True):
            hidden = layer(hidden, mask=mask, cache=None)
        hidden = mx.stop_gradient(hidden)
        mx.eval(hidden)
        return hidden
