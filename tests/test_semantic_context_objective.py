"""Structured learning uses public proposals and exact set probabilities."""

import pytest
import torch

from core.learning.semantic_context_objective import (
    best_operation_set,
    operation_scores,
    operation_set_loss,
)
from core.learning.semantic_request_context import ContextualSpanRecognizer, RequestContextConfig


def test_loss_matches_exhaustive_single_operation_distribution():
    scores = torch.tensor([[[2., 3.]], [[5., 7.]]], dtype=torch.double).log().requires_grad_()
    loss = operation_set_loss(scores, ((1, 2, 1),), max_operations=1)
    torch.testing.assert_close(loss, torch.tensor(18. / 7., dtype=torch.double).log())
    loss.backward()
    expected = scores.detach().exp() / 18.
    expected[1, 0, 1] -= 1
    torch.testing.assert_close(scores.grad, expected)


def test_proposal_chart_and_structured_gradient():
    torch.manual_seed(8)
    model = ContextualSpanRecognizer(RequestContextConfig(4, 8, 2, 1), ("a", "b"))
    hidden = torch.eye(4)
    scores = operation_scores(model, hidden, span_width=3)
    assert scores.shape == (4, 3, 2)
    assert torch.isneginf(scores[3, 1:]).all()
    loss = operation_set_loss(scores, ((0, 2, 1), (3, 4, 0)), max_operations=3)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("target", [((0, 2, 0), (1, 2, 1)), ((0, 3, 0),),
                                     ((0, 1, 2),), ((0, 1, 0), (0, 1, 0))])
def test_invalid_gold_cannot_silently_become_a_training_target(target):
    with pytest.raises(ValueError):
        operation_set_loss(torch.zeros(2, 2, 2), target, max_operations=2)


def test_map_selects_count_and_labels_without_gold():
    scores = torch.tensor([[[2., 3.], [4., 2.]], [[5., 1.], [-torch.inf, -torch.inf]]])
    assert best_operation_set(scores, max_operations=2) == ((0, 1, 1), (1, 2, 0))
    assert best_operation_set(scores, max_operations=1) == ((1, 2, 0),)
    assert best_operation_set(scores - 10., max_operations=2) == ()
