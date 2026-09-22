"""Experimental full-request features for the existing semantic decoder.

This module has no serving authority. Inputs are frozen resident features,
not target programs or generated answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class RequestContextConfig:
    input_width: int
    width: int = 64
    heads: int = 4
    layers: int = 2
    position_mode: str = "absolute"
    feature_scaling: str = "none"

    def __post_init__(self) -> None:
        values = (self.input_width, self.width, self.heads, self.layers)
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("request context dimensions must be positive integers")
        if self.width % self.heads or self.width % 2:
            raise ValueError("request context width must be even and divisible by heads")
        if self.position_mode not in {"absolute", "relative", "none"}:
            raise ValueError("unknown request position representation")
        if self.feature_scaling not in {"none", "unit_variance"}:
            raise ValueError("unknown source feature scaling")


class RequestContextBlock(nn.Module):
    """Pre-norm attention with the same real-valued mask in fit and inference.

    Parameter names match the original encoder for checkpoint continuity.
    Explicit public submodules avoid the fused encoder's boolean-mask path.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, state: torch.Tensor, *, src_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(state)
        attended, _ = self.self_attn(normalized, normalized, normalized,
                                    attn_mask=src_mask, need_weights=False)
        state = state + attended
        return state + self.linear2(functional.relu(self.linear1(self.norm2(state))))


class SemanticRequestContext(nn.Module):
    """Reconsider each source token using the complete public request."""

    def __init__(self, config: RequestContextConfig):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.input_width, config.width)
        self.blocks = nn.ModuleList([
            RequestContextBlock(config.width, config.heads) for _ in range(config.layers)
        ])
        self.restore = nn.Linear(config.width, config.input_width, bias=False)
        self.relative_bias = (nn.Linear(3, config.heads, bias=False)
                              if config.position_mode == "relative" else None)

    def forward(
        self, features: torch.Tensor, valid: torch.Tensor, *, cross_token: bool = True,
    ) -> torch.Tensor:
        """Return unit features; padded tokens never affect valid tokens.

        ``cross_token=False`` keeps the same parameters and token-wise work,
        but blocks communication between tokens for the context lesion.
        """
        if (features.ndim != 3 or features.shape[-1] != self.config.input_width
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or valid.device != features.device or not features.is_floating_point()
                or features.shape[0] == 0 or features.shape[1] == 0):
            raise ValueError("invalid request context input geometry")
        if not valid.any(dim=1).all().item():
            raise ValueError("each request needs at least one source token")
        if not torch.isfinite(features[valid]).all().item():
            raise ValueError("source features must be finite")
        if not torch.allclose(features[valid].norm(dim=-1),
                              torch.ones_like(features[valid, 0]), atol=1e-4):
            raise ValueError("source features must be unit normalized")
        clean = features.masked_fill(~valid.unsqueeze(-1), 0)
        # Unit-length features have mean square 1/input_width. Restore the
        # fan-in scale expected by Linear without changing the residual.
        scale = math.sqrt(self.config.input_width) if self.config.feature_scaling == "unit_variance" else 1.0
        state = self.project(clean * scale)
        # Count source tokens, not padding positions, including left padding.
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        if self.config.position_mode == "absolute":
            frequencies = torch.exp(torch.arange(
                0, self.config.width, 2, device=features.device, dtype=state.dtype,
            ) * (-math.log(10000.0) / self.config.width))
            angles = positions.unsqueeze(-1) * frequencies
            position = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
            state = state + position
        size = valid.shape[1]
        mask = state.new_zeros((valid.shape[0], self.config.heads, size, size))
        if self.relative_bias is not None:
            distance = positions[:, None, :] - positions[:, :, None]
            magnitude = distance.abs().to(state.dtype).log1p()
            relation = torch.stack((distance.sign() * magnitude, magnitude,
                                    (distance == 0).to(state.dtype)), dim=-1)
            mask = self.relative_bias(relation).permute(0, 3, 1, 2)
        mask = mask.masked_fill(~valid[:, None, None, :], -torch.inf)
        if not cross_token:
            # Padded queries may attend to real tokens to avoid an all-masked
            # softmax. Their outputs are discarded and cannot become keys.
            diagonal = torch.eye(size, device=valid.device, dtype=torch.bool)
            excluded = (~diagonal)[None, None] & valid[:, None, :, None]
            mask = mask.masked_fill(excluded, -torch.inf)
        mask = mask.reshape(-1, size, size)
        for block in self.blocks:
            state = block(state, src_mask=mask)
        result = functional.normalize(clean + self.restore(state), dim=-1)
        return result.masked_fill(~valid.unsqueeze(-1), 0)


class ContextualSpanRecognizer(nn.Module):
    """Score proposed source spans without accepting a target graph at inference."""

    def __init__(self, config: RequestContextConfig, labels: tuple[str, ...]):
        super().__init__()
        if not labels or len(set(labels)) != len(labels) or any(not x for x in labels):
            raise ValueError("span labels must be nonempty and unique")
        self.labels = labels
        self.context = SemanticRequestContext(config)
        self.classifier = nn.Linear(3 * config.input_width, len(labels))

    def forward(self, features: torch.Tensor, valid: torch.Tensor,
                spans: torch.Tensor, *, cross_token: bool = True) -> torch.Tensor:
        """Score [batch index, start, exclusive end] proposals in supplied order."""
        if spans.ndim != 2 or spans.shape[1] != 3 or spans.dtype != torch.long:
            raise ValueError("span proposals must be an integer matrix with three columns")
        if spans.device != features.device:
            raise ValueError("span proposals and features must share a device")
        context = self.context(features, valid, cross_token=cross_token)
        if not spans.numel():
            return context.new_empty((0, len(self.labels)))
        batch, start, end = spans.unbind(dim=1)
        if ((batch < 0).any() or (batch >= features.shape[0]).any()
                or (start < 0).any() or (end <= start).any()
                or (end > features.shape[1]).any()):
            raise ValueError("span proposal exceeds source bounds")
        counts = functional.pad(valid.long().cumsum(dim=1), (1, 0))
        if not ((counts[batch, end] - counts[batch, start]) == end - start).all():
            raise ValueError("span proposal includes padding")
        # Apply linear maps before gathering spans. A full resident feature
        # vector per candidate would multiply memory by span width and labels.
        start_weight, end_weight, mean_weight = self.classifier.weight.chunk(3, dim=1)
        starts = functional.linear(context, start_weight)
        ends = functional.linear(context, end_weight)
        means = functional.linear(context, mean_weight)
        prefix = functional.pad(means.cumsum(dim=1), (0, 0, 1, 0))
        mean = (prefix[batch, end] - prefix[batch, start]) / (end - start).unsqueeze(-1)
        return starts[batch, start] + ends[batch, end - 1] + mean + self.classifier.bias
