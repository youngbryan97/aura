"""Exact structured operation loss over target-independent source proposals."""

import torch

from core.learning.semantic_request_context import ContextualSpanRecognizer
from core.learning.semantic_span_autograd import labeled_partition


def best_operation_set(scores: torch.Tensor, *, max_operations: int) -> tuple[tuple[int, int, int], ...]:
    """Exact operation-only MAP, including the empty set, without gold counts."""
    from types import SimpleNamespace

    from core.learning.semantic_operation_search import OperationChartSearch
    from core.learning.semantic_program_ir import TokenSpan

    # The shared partition validates exclusions and score geometry.
    labeled_partition(scores.detach(), max_operations)
    values, labels = scores.detach().cpu().max(dim=-1)
    nodes = []
    for start, column in torch.isfinite(values).nonzero().tolist():
        label = int(labels[start, column])
        nodes.append(SimpleNamespace(span=TokenSpan(start, start + column + 1),
                     operation=str(label), score=float(values[start, column])))
    search = OperationChartSearch(nodes, max_steps=max_operations, length_penalty=0.)
    selected = next(search, ())
    if sum(node.score for node in selected) <= 0:
        return ()
    return tuple((node.span.start, node.span.end, int(node.operation)) for node in selected)


def operation_scores(model: ContextualSpanRecognizer, hidden: torch.Tensor, *,
                     span_width: int, cross_token: bool = True) -> torch.Tensor:
    """Enumerate every in-bound span, with no gold spans or counts as inputs."""
    if hidden.ndim != 2 or hidden.shape[0] == 0:
        raise ValueError("operation evidence must be a nonempty token matrix")
    if type(span_width) is not int or span_width < 1:
        raise ValueError("span width must be a positive integer")
    n = hidden.shape[0]
    width = min(n, span_width)
    valid = torch.arange(n, device=hidden.device)[:, None] + torch.arange(
        1, width + 1, device=hidden.device) <= n
    start, column = valid.nonzero(as_tuple=True)
    spans = torch.stack((torch.zeros_like(start), start, start + column + 1), dim=1)
    logits = model(hidden.unsqueeze(0), torch.ones((1, n), device=hidden.device,
                   dtype=torch.bool), spans, cross_token=cross_token)
    chart = logits.new_full((n, width, len(model.labels)), -torch.inf)
    chart[start, column] = logits
    return chart


def operation_set_loss(scores: torch.Tensor, target: tuple[tuple[int, int, int], ...],
                       *, max_operations: int) -> torch.Tensor:
    """Negative log probability of a non-overlapping labeled span set.

    Target entries are (start, exclusive end, label index). The capacity is
    a declared architecture limit, never the example's gold operation count.
    """
    if scores.ndim != 3 or not all(scores.shape):
        raise ValueError("invalid operation score geometry")
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("invalid operation capacity")
    if len(target) > max_operations:
        raise ValueError("target exceeds operation capacity")
    ordered = sorted(target)
    previous = 0
    terms = []
    for start, end, label in ordered:
        if (any(type(value) is not int for value in (start, end, label))
                or start < previous or end <= start or end > scores.shape[0]
                or end - start > scores.shape[1] or not 0 <= label < scores.shape[2]):
            raise ValueError("target is outside the non-overlapping operation grammar")
        term = scores[start, end - start - 1, label]
        if not torch.isfinite(term):
            raise ValueError("target operation was excluded from proposals")
        terms.append(term)
        previous = end
    target_score = torch.stack(terms).sum() if terms else scores.new_zeros(())
    return labeled_partition(scores, max_operations) - target_score
