"""A hybrid decoder is priced by what each of its layers actually is.

Charging 48 gated-delta layers at the attention rate overstates the model's own
cost, which inflates an equal-FLOP control and hands the treatment arm a budget
it did not earn. Nothing raises when that happens.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.resource_accounting import (
    ESTIMATOR_VERSION,
    HYBRID_ESTIMATOR_VERSION,
    HYBRID_MODEL_PROFILE_SCHEMA,
    MODEL_PROFILE_SCHEMA,
    ModelComputeProfile,
)

QWEN35 = json.loads(
    json.dumps(
        {
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 8,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 32,
                "vocab_size": 512,
                "full_attention_interval": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 8,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
            },
        }
    )
)


def _args(**overrides):
    base = dict(QWEN35["text_config"])
    base.update(overrides)
    return SimpleNamespace(**base)


def _layer(linear: bool):
    if linear:
        return SimpleNamespace(is_linear=True, linear_attn=object(), mlp=object())
    return SimpleNamespace(is_linear=False, self_attn=object(), mlp=object())


def _hybrid(layers: int = 8, interval: int = 4):
    blocks = [_layer((i + 1) % interval != 0) for i in range(layers)]
    text = SimpleNamespace(
        args=_args(num_hidden_layers=layers), model=SimpleNamespace(layers=blocks)
    )
    return SimpleNamespace(
        args=SimpleNamespace(model_type="qwen3_5"), language_model=text
    )


def _dense(layers: int = 8):
    blocks = [_layer(False) for _ in range(layers)]
    args = _args(model_type="qwen2", num_hidden_layers=layers, head_dim=64,
                 num_attention_heads=4, num_key_value_heads=2)
    return SimpleNamespace(args=args, model=SimpleNamespace(layers=blocks))


# ── The failure that started this ───────────────────────────────────────


def test_a_hybrid_wrapper_no_longer_reports_no_profile():
    # The live error was: ValueError("model does not expose a decoder compute
    # profile"), raised because the walk looked for `model.model.args`.
    profile = ModelComputeProfile.from_model(_hybrid())
    assert profile.num_hidden_layers == 8
    assert profile.is_hybrid is True


def test_a_query_width_wider_than_the_residual_stream_is_accepted():
    # 24 heads of 32 is 768 against a 256 stream. The old invariant asserted
    # equality and refused the checkpoint outright.
    profile = ModelComputeProfile.from_model(_hybrid())
    assert profile.num_attention_heads * profile.head_dim != profile.hidden_size


def test_an_object_with_no_decoder_still_fails_honestly():
    with pytest.raises(ValueError, match="decoder compute profile"):
        ModelComputeProfile.from_model(SimpleNamespace(nothing=True))


# ── Layer inventory and pricing ─────────────────────────────────────────


def test_the_profile_counts_both_layer_kinds():
    profile = ModelComputeProfile.from_model(_hybrid())
    assert profile.attention_layer_count == 2
    assert profile.linear_layer_count == 6
    assert len(profile.layer_kinds) == 8


def test_a_linear_layer_is_not_priced_as_an_attention_layer():
    profile = ModelComputeProfile.from_model(_hybrid())
    assert profile.linear_flops_per_token_layer > 0
    assert profile.linear_flops_per_token_layer != profile.dense_flops_per_token_layer


def test_the_full_stack_is_the_sum_of_its_actual_layers():
    profile = ModelComputeProfile.from_model(_hybrid())
    expected = (
        profile.attention_layer_count * profile.dense_flops_per_token_layer
        + profile.linear_layer_count * profile.linear_flops_per_token_layer
    )
    assert profile.flops_per_token_full_stack == expected


def test_charging_a_hybrid_at_the_dense_rate_is_a_different_number():
    # The whole point: the two disagree, and the wrong one is not an error.
    profile = ModelComputeProfile.from_model(_hybrid())
    dense_reading = profile.num_hidden_layers * profile.dense_flops_per_token_layer
    assert dense_reading != profile.flops_per_token_full_stack


def test_a_dense_profile_has_no_linear_layers_to_price():
    profile = ModelComputeProfile.from_model(_dense())
    assert profile.is_hybrid is False
    assert profile.linear_flops_per_token_layer == 0
    with pytest.raises(ValueError, match="no linear-attention layers"):
        profile.flops_per_token_layer("linear_attention")


def test_an_unknown_layer_kind_is_refused():
    profile = ModelComputeProfile.from_model(_hybrid())
    with pytest.raises(ValueError, match="not a recognised decoder layer"):
        profile.flops_per_token_layer("quantum_attention")


# ── Receipts stay compatible ────────────────────────────────────────────


def test_a_dense_profile_still_emits_the_v1_schema():
    # Existing certificates and preregistered digests must keep verifying.
    receipt = ModelComputeProfile.from_model(_dense()).to_receipt()
    assert receipt["schema"] == MODEL_PROFILE_SCHEMA
    assert receipt["estimator_version"] == ESTIMATOR_VERSION
    assert "layer_kinds" not in receipt


def test_a_hybrid_profile_emits_the_v2_schema_and_its_own_estimator():
    receipt = ModelComputeProfile.from_model(_hybrid()).to_receipt()
    assert receipt["schema"] == HYBRID_MODEL_PROFILE_SCHEMA
    assert receipt["estimator_version"] == HYBRID_ESTIMATOR_VERSION
    assert receipt["attention_layer_count"] == 2
    assert receipt["linear_layer_count"] == 6


def test_both_schemas_round_trip():
    for model in (_dense(), _hybrid()):
        profile = ModelComputeProfile.from_model(model)
        assert ModelComputeProfile.from_receipt(profile.to_receipt()) == profile


def test_a_hybrid_receipt_missing_its_hybrid_fields_is_refused():
    receipt = dict(ModelComputeProfile.from_model(_hybrid()).to_receipt())
    receipt.pop("layer_kinds")
    with pytest.raises(ValueError, match="schema is invalid"):
        ModelComputeProfile.from_receipt(receipt)


def test_a_v1_receipt_carrying_hybrid_fields_is_refused():
    receipt = dict(ModelComputeProfile.from_model(_hybrid()).to_receipt())
    receipt["schema"] = MODEL_PROFILE_SCHEMA
    with pytest.raises(ValueError, match="schema is invalid"):
        ModelComputeProfile.from_receipt(receipt)


def test_a_tampered_hybrid_receipt_is_refused():
    receipt = dict(ModelComputeProfile.from_model(_hybrid()).to_receipt())
    receipt["linear_layer_count"] = 0
    with pytest.raises(ValueError):
        ModelComputeProfile.from_receipt(receipt)


def test_a_hybrid_profile_cannot_claim_the_dense_estimator():
    profile = ModelComputeProfile.from_model(_hybrid())
    with pytest.raises(ValueError, match="hybrid estimator"):
        ModelComputeProfile(
            model_type=profile.model_type,
            hidden_size=profile.hidden_size,
            intermediate_size=profile.intermediate_size,
            num_hidden_layers=profile.num_hidden_layers,
            num_attention_heads=profile.num_attention_heads,
            num_key_value_heads=profile.num_key_value_heads,
            vocab_size=profile.vocab_size,
            head_dim=profile.head_dim,
            estimator_version=ESTIMATOR_VERSION,
            layer_kinds=profile.layer_kinds,
        )


def test_layer_kinds_must_cover_every_layer():
    with pytest.raises(ValueError, match="cover every hidden layer"):
        ModelComputeProfile(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=8,
            num_attention_heads=24,
            num_key_value_heads=4,
            vocab_size=512,
            head_dim=32,
            estimator_version=HYBRID_ESTIMATOR_VERSION,
            layer_kinds=("linear_attention", "full_attention"),
        )


def test_a_profile_with_no_attention_layer_is_refused():
    with pytest.raises(ValueError, match="at least one attention layer"):
        ModelComputeProfile(
            model_type="qwen3_5_text",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=24,
            num_key_value_heads=4,
            vocab_size=512,
            head_dim=32,
            estimator_version=HYBRID_ESTIMATOR_VERSION,
            layer_kinds=("linear_attention", "linear_attention"),
        )
