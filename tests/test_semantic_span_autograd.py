"""The neural loss uses the same partition and gradient as the exact decoder."""

import torch

from core.learning.semantic_span_autograd import labeled_partition


def test_exact_partition_and_gradient():
    scores = torch.tensor([[[2., 3.]], [[5., 7.]]], dtype=torch.double).log().requires_grad_()
    loss = labeled_partition(scores, 1)
    torch.testing.assert_close(loss, torch.tensor(18., dtype=torch.double).log())
    loss.backward()
    torch.testing.assert_close(scores.grad, scores.detach().exp() / 18.)


def test_finite_difference_gradient():
    torch.manual_seed(92)
    scores = torch.randn(4, 2, 3, dtype=torch.double, requires_grad=True)
    valid = torch.arange(4)[:, None] + torch.arange(1, 3) <= 4
    assert torch.autograd.gradcheck(
        lambda x: labeled_partition(x.masked_fill(~valid.unsqueeze(-1), -torch.inf), 2),
        (scores,),
    )
