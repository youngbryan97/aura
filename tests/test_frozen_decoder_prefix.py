"""Cached frozen states must preserve the loaded decoder computation and gradients."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
from core.learning.semantic_native_program import NativeProgramSequence
from tools.train_semantic_native_program import native_loss


def _model(*, tied=False, hybrid=False, width=16):
    from mlx_lm.models.qwen2 import Model as DenseModel
    from mlx_lm.models.qwen2 import ModelArgs as DenseArgs
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    mx.random.seed(7)
    if hybrid:
        args = TextModelArgs(model_type="qwen3_5_text", hidden_size=width,
                             intermediate_size=2 * width, num_hidden_layers=4,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=width // 2,
                             vocab_size=32, full_attention_interval=4,
                             linear_num_key_heads=2, linear_num_value_heads=2,
                             linear_key_head_dim=32, linear_value_head_dim=32,
                             tie_word_embeddings=tied)
        model = TextModel(args)
    else:
        model = DenseModel(DenseArgs(model_type="qwen2", hidden_size=width,
                                     intermediate_size=2 * width, num_hidden_layers=3,
                                     num_attention_heads=2, num_key_value_heads=1,
                                     rms_norm_eps=1e-6, vocab_size=32,
                                     tie_word_embeddings=tied))
    model.freeze()
    model.eval()
    return model


@pytest.mark.parametrize("hybrid,tied", [(False, False), (False, True), (True, False), (True, True)])
def test_split_reuses_native_logits_at_every_token(hybrid, tied):
    model = _model(hybrid=hybrid, tied=tied)
    split = len(model.layers) - 1
    prefix = FrozenDecoderPrefix(model, split_at=split)
    suffix = NativeDecoderSuffix(model, split_at=split)
    tokens = mx.array([[1, 4, 2, 9, 5], [5, 3, 7, 1, 2]])
    assert suffix.layers[-1] is model.layers[-1]
    assert mx.allclose(model(tokens), suffix(prefix.capture(tokens)), atol=1e-5).item()
    batched = prefix.capture(tokens)
    individual = mx.concatenate([prefix.capture(tokens[index:index + 1])
                                 for index in range(tokens.shape[0])])
    assert mx.allclose(batched, individual, atol=1e-5).item()


def test_prefix_rejects_training_or_unfrozen_parameters_after_construction():
    model = _model()
    prefix = FrozenDecoderPrefix(model, split_at=2)
    model.layers[0].train()
    with pytest.raises(ValueError, match="evaluation"):
        prefix.capture(mx.array([[1, 2]]))
    model.eval()
    model.layers[0].unfreeze()
    with pytest.raises(ValueError, match="frozen"):
        prefix.capture(mx.array([[1, 2]]))


def test_native_suffix_gradients_change_the_same_loaded_projection():
    model = _model(hybrid=True)
    prefix = FrozenDecoderPrefix(model, split_at=3)
    suffix = NativeDecoderSuffix(model, split_at=3)
    shared = model.layers[3].mlp.down_proj
    shared.unfreeze()
    hidden = prefix.capture(mx.array([[1, 2, 3]]))
    loss, gradients = nn.value_and_grad(suffix, lambda tail: mx.sum(tail(hidden) ** 2))(suffix)
    assert mx.isfinite(loss).item()
    weight_gradient = gradients["layers"][0]["mlp"]["down_proj"]["weight"]
    assert mx.sum(mx.abs(weight_gradient)).item() > 0
    before = mx.array(shared.weight)
    suffix.update({"layers": [{"mlp": {"down_proj": {
        "weight": shared.weight - 1e-5 * weight_gradient}}}]})
    assert not mx.array_equal(shared.weight, before).item()
    assert mx.allclose(model(mx.array([[1, 2, 3]])), suffix(hidden), atol=1e-5).item()


@pytest.mark.parametrize("split", [0, -1, 3, True])
def test_invalid_split_has_no_prefix_capture(split):
    model = _model()
    for factory in (FrozenDecoderPrefix, NativeDecoderSuffix):
        with pytest.raises(ValueError, match="split"):
            factory(model, split_at=split)


def test_outer_wrapper_uses_the_existing_topology_owner():
    model = _model()
    wrapper = SimpleNamespace(language_model=model)
    tokens = mx.array([[3, 1, 9]])
    assert mx.allclose(model(tokens), NativeDecoderSuffix(wrapper, split_at=2)(
        FrozenDecoderPrefix(wrapper, split_at=2).capture(tokens))).item()


@pytest.mark.parametrize("hybrid", [False, True])
def test_quantized_native_adapter_update_and_control_restore_share_the_loaded_model(hybrid):
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model = _model(hybrid=hybrid, width=32)
    nn.quantize(model, group_size=32, bits=4)
    model.freeze()
    model.eval()
    split = len(model.layers) - 1
    prefix = FrozenDecoderPrefix(model, split_at=split)
    suffix = NativeDecoderSuffix(model, split_at=split)
    linear_to_lora_layers(model, 1, {
        "rank": 2, "scale": 4., "dropout": 0.,
        "keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.down_proj"]})
    assert len(tree_flatten(suffix.trainable_parameters())) == 8
    assert all("lora_" in name for name, _value in tree_flatten(model.trainable_parameters()))
    baseline_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
    tokens = mx.array([[1, 2, 3, 4]])
    captured = prefix.capture(tokens)
    baseline_logits = suffix(captured)
    mx.eval(baseline_logits, baseline_weights)
    sequence = NativeProgramSequence((1, 2, 3, 4, 5), 2)
    optimizer = optim.AdamW(learning_rate=.01)
    for _step in range(2):
        loss, gradient = nn.value_and_grad(suffix, native_loss)(suffix, captured, sequence)
        optimizer.update(suffix, gradient)
        mx.eval(suffix.parameters(), loss)
        assert mx.isfinite(loss).item()
    trained_weights = tree_map(lambda value: mx.array(value), suffix.trainable_parameters())
    trained_logits = suffix(captured)
    mx.eval(trained_logits, trained_weights)
    assert not mx.allclose(trained_logits, baseline_logits).item()
    assert mx.allclose(model(tokens), trained_logits, atol=1e-5).item()
    assert mx.array_equal(prefix.capture(tokens), captured).item()
    suffix.update(baseline_weights)
    assert mx.array_equal(suffix(captured), baseline_logits).item()
    suffix.update(trained_weights)
    assert mx.array_equal(model(tokens), trained_logits).item()


@pytest.mark.parametrize("hybrid,tied", [(False, False), (False, True), (True, False), (True, True)])
def test_selected_projection_preserves_full_logits_and_suffix_gradients(hybrid, tied):
    from mlx.utils import tree_flatten
    model = _model(hybrid=hybrid, tied=tied)
    split = len(model.layers) - 1
    prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
    model.layers[-1].mlp.down_proj.unfreeze()
    hidden = prefix.capture(mx.array([[1, 3, 4, 2, 9], [5, 7, 8, 2, 3]]))
    positions = (0, 2, 4)
    index = mx.array(positions)
    full = suffix(hidden)
    assert mx.allclose(suffix(hidden, logit_positions=positions), full[:, index], atol=1e-5).item()
    full_loss, full_gradient = nn.value_and_grad(suffix, lambda tail:
        mx.sum(tail(hidden)[:, index] ** 2))(suffix)
    selected_loss, selected_gradient = nn.value_and_grad(suffix, lambda tail:
        mx.sum(tail(hidden, logit_positions=positions) ** 2))(suffix)
    assert mx.allclose(full_loss, selected_loss, atol=1e-5).item()
    full_items, selected_items = tree_flatten(full_gradient), tree_flatten(selected_gradient)
    assert [name for name, _value in full_items] == [name for name, _value in selected_items]
    assert all(mx.allclose(left, right, atol=1e-5).item()
               for (_left_name, left), (_right_name, right) in zip(
                   full_items, selected_items, strict=True))


def test_vocabulary_projection_only_receives_requested_rows_after_all_causal_layers():
    model = _model(hybrid=True)
    suffix = NativeDecoderSuffix(model, split_at=3)
    class ObservedOutput(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original
            self.observed = []
        def __call__(self, value):
            self.observed.append(value.shape)
            return self.original(value)
    output = ObservedOutput(suffix.output)
    suffix.output = output
    hidden = FrozenDecoderPrefix(model, split_at=3).capture(mx.array([[1, 3, 5, 6, 7]]))
    assert suffix(hidden, logit_positions=(1, 4)).shape == (1, 2, 32)
    assert output.observed == [(1, 2, 16)]
    for invalid in ((), (True,), (-1,), (5,), (1, 1)):
        with pytest.raises(ValueError, match="logit positions"):
            suffix(hidden, logit_positions=invalid)
