"""Context reach, isolation, and gradients of the experimental adapter."""

import pytest
import torch
from torch.nn import functional as functional

from core.learning.semantic_request_context import (
    ContextualSpanRecognizer,
    RequestContextBlock,
    RequestContextConfig,
    SemanticRequestContext,
)


def setup(position_mode="absolute"):
    torch.manual_seed(73)
    model = SemanticRequestContext(RequestContextConfig(12, width=8, heads=2, layers=1,
                                                       position_mode=position_mode))
    x = functional.normalize(torch.randn(2, 5, 12), dim=-1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    return model, x, valid


def test_scaled_projection_receives_unit_mean_square_features():
    model = SemanticRequestContext(RequestContextConfig(
        5120, width=8, heads=2, layers=1, feature_scaling="unit_variance"))
    seen = []
    hook = model.project.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    x = functional.normalize(torch.randn(1, 3, 5120), dim=-1)
    result = model(x, torch.ones(1, 3, dtype=torch.bool))
    hook.remove()
    torch.testing.assert_close(seen[0].square().mean(dim=-1), torch.ones(1, 3))
    assert torch.isfinite(result).all()


def test_default_scaling_preserves_checkpoint_input_contract():
    model, x, valid = setup()
    seen = []
    hook = model.project.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    model(x, valid)
    hook.remove()
    torch.testing.assert_close(seen[0], x)


def test_unknown_scaling_is_rejected():
    with pytest.raises(ValueError, match="scaling"):
        RequestContextConfig(12, feature_scaling="guess")


def test_later_clause_changes_earlier_features_only_with_context():
    model, x, valid = setup()
    changed = x.clone()
    changed[:, -1] = -changed[:, -1]
    assert not torch.allclose(model(x, valid)[:, 0], model(changed, valid)[:, 0])
    torch.testing.assert_close(model(x, valid, cross_token=False)[:, 0],
                               model(changed, valid, cross_token=False)[:, 0])


@pytest.mark.parametrize("cross_token", [True, False])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("position_mode", ["absolute", "relative", "none"])
def test_padding_and_other_requests_cannot_change_features(cross_token, left, position_mode):
    model, x, valid = setup(position_mode)
    expected = model(x[:1], valid[:1], cross_token=cross_token)
    pad = torch.full((1, 2, 12), float("nan"))
    padded = torch.cat((pad, x[:1]) if left else (x[:1], pad), dim=1)
    mask = torch.isfinite(padded).all(dim=-1)
    actual = model(padded, mask, cross_token=cross_token)
    torch.testing.assert_close(actual[mask], expected.reshape(-1, 12))
    assert torch.count_nonzero(actual[~mask]) == 0
    torch.testing.assert_close(model(x, valid, cross_token=cross_token)[:1], expected)


def test_gradient_reaches_context_parameters_and_output_is_normalized():
    model, x, valid = setup()
    output = model(x, valid)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(2, 5))
    output[..., 0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_state_roundtrip():
    model, x, valid = setup()
    copy = SemanticRequestContext(model.config)
    copy.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x, valid), copy(x, valid))


@pytest.mark.parametrize("position_mode", ["absolute", "relative", "none"])
@pytest.mark.parametrize("cross_token", [True, False])
def test_eval_preserves_real_valued_attention_bias(position_mode, cross_token):
    model, x, valid = setup(position_mode)
    with torch.no_grad():
        expected = model(x, valid, cross_token=cross_token)
        model.eval()
        actual = model(x, valid, cross_token=cross_token)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_block_preserves_legacy_weights_outputs_and_gradients():
    torch.manual_seed(11)
    legacy = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=32, dropout=0.,
                                              batch_first=True, norm_first=True)
    block = RequestContextBlock(8, 2)
    block.load_state_dict(legacy.state_dict(), strict=True)
    state = torch.randn(2, 5, 8)
    mask = torch.randn(4, 5, 5)
    expected = legacy(state, src_mask=mask)
    actual = block(state, src_mask=mask)
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    for name, parameter in block.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(legacy.named_parameters())[name].grad)


def test_invalid_evidence_rejected():
    model, x, valid = setup()
    with pytest.raises(ValueError, match="at least one"):
        model(x, torch.zeros_like(valid))
    with pytest.raises(ValueError, match="unit normalized"):
        model(x * 3, valid)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(x, valid)


def test_span_scores_are_proposal_order_equivariant():
    _, x, valid = setup()
    model = ContextualSpanRecognizer(RequestContextConfig(12, width=8, heads=2, layers=1),
                                     ("background", "operation"))
    spans = torch.tensor([[0, 0, 2], [1, 2, 5], [0, 3, 4]])
    torch.testing.assert_close(model(x, valid, spans.flip(0)), model(x, valid, spans).flip(0))
    assert model(x, valid, spans[:0]).shape == (0, 2)
    valid[0, 1] = False
    with pytest.raises(ValueError, match="padding"):
        model(x, valid, spans)


def test_projection_before_pooling_preserves_dense_logits_and_gradients():
    _, x, valid = setup()
    model = ContextualSpanRecognizer(RequestContextConfig(12, 8, 2, 1), ("a", "b"))
    spans = torch.tensor([[0, 0, 2], [1, 1, 5], [0, 2, 3]])
    actual = model(x, valid, spans)
    context = model.context(x, valid)
    pooled = torch.stack([torch.cat((context[b, start], context[b, end - 1],
                                    context[b, start:end].mean(dim=0)))
                          for b, start, end in spans.tolist()])
    expected = model.classifier(pooled)
    torch.testing.assert_close(actual, expected)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters)
    expected_gradients = torch.autograd.grad(expected.square().sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=1e-4)


def test_learns_suffix_dependent_span_labels_without_target_input():
    torch.manual_seed(81)
    model = ContextualSpanRecognizer(RequestContextConfig(4, width=8, heads=2, layers=1),
                                     ("left", "right"))
    # Identical first tokens: a token-local readout cannot separate these.
    x = torch.tensor([[[1., 0, 0, 0], [0, 1., 0, 0]],
                      [[1., 0, 0, 0], [0, 0, 1., 0]]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    spans = torch.tensor([[0, 0, 1], [1, 0, 1]])
    target = torch.tensor([0, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(60):
        optimizer.zero_grad()
        loss = functional.cross_entropy(model(x, valid, spans), target)
        loss.backward()
        optimizer.step()
    assert torch.equal(model(x, valid, spans).argmax(dim=-1), target)
    local = model(x, valid, spans, cross_token=False)
    torch.testing.assert_close(local[0], local[1])
