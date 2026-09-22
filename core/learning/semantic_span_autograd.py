"""Differentiate the existing exact labeled span-set partition."""

import torch
from torch.autograd.function import once_differentiable

from core.learning.semantic_labeled_span_learning import labeled_span_partition


class _LabeledPartition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, max_spans):
        partition, marginal = labeled_span_partition(
            scores.detach().cpu().double().numpy(), max_spans,
        )
        ctx.save_for_backward(torch.as_tensor(marginal, device=scores.device, dtype=scores.dtype))
        return scores.new_tensor(partition)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (marginal,) = ctx.saved_tensors
        return upstream * marginal, None


def labeled_partition(scores: torch.Tensor, max_spans: int) -> torch.Tensor:
    """Reuse the audited DP and its marginals for first-order neural training."""
    if not scores.is_floating_point():
        raise ValueError("labeled scores must be floating point")
    return _LabeledPartition.apply(scores, max_spans)
