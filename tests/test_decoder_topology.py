"""A decoder's shape is read from the model, not assumed from its era.

Live Latent Cortex raised `model does not expose a decoder compute profile` on
the 27B. Three assumptions failed at once and none announced itself: the
wrapper attribute moved, `hidden_size == heads * head_dim` stopped holding, and
three layers in four stopped being attention layers.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.brain.llm.decoder_topology import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    DecoderTopologyError,
    decoder_layer_masks,
    decoder_layers,
    resolve_language_model,
    topology_disagreements,
    topology_from_config,
    topology_from_model,
)


def _args(**overrides):
    base = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "intermediate_size": 17408,
        "vocab_size": 248320,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _attention_layer():
    return SimpleNamespace(is_linear=False, self_attn=object(), mlp=object())


def _linear_layer():
    return SimpleNamespace(is_linear=True, linear_attn=object(), mlp=object())


def _hybrid_model(layers: int = 64, interval: int = 4):
    blocks = [
        _attention_layer() if (i + 1) % interval == 0 else _linear_layer()
        for i in range(layers)
    ]
    inner = SimpleNamespace(layers=blocks)
    text = SimpleNamespace(args=_args(), model=inner, layers=blocks)
    # The multimodal wrapper: language_model, not model.
    return SimpleNamespace(args=SimpleNamespace(model_type="qwen3_5"), language_model=text)


def _dense_model(layers: int = 64):
    blocks = [_attention_layer() for _ in range(layers)]
    inner = SimpleNamespace(layers=blocks)
    return SimpleNamespace(args=_args(model_type="qwen2"), model=inner)


# ── Resolving the language model through its wrapper ────────────────────


def test_a_multimodal_wrapper_is_walked_to_its_language_model():
    # This is the exact failure: the old walk looked for `model.model.args`,
    # found nothing on a wrapper exposing `language_model`, and reported the
    # checkpoint as having no profile at all.
    resolved = resolve_language_model(_hybrid_model())
    assert getattr(resolved.args, "hidden_size", None) == 5120


def test_a_plain_model_is_already_the_language_model():
    resolved = resolve_language_model(_dense_model())
    assert resolved.args.model_type == "qwen2"


def test_an_object_that_is_not_a_decoder_is_refused():
    with pytest.raises(DecoderTopologyError, match="compute profile"):
        resolve_language_model(SimpleNamespace(nothing=True))


def test_a_cyclic_wrapper_does_not_hang():
    node = SimpleNamespace()
    node.model = node
    with pytest.raises(DecoderTopologyError):
        resolve_language_model(node)


def test_layers_are_found_through_either_holder():
    assert len(decoder_layers(_hybrid_model())) == 64
    assert len(decoder_layers(_dense_model())) == 64


@pytest.mark.hardware
def test_a_real_hybrid_layer_walk_receives_two_mask_contracts():
    """The latent engine's direct walk must not send ``"causal"`` to SSM layers."""

    import mlx.core as mx
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    text_config = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 2,
        "full_attention_interval": 2,
        "head_dim": 16,
        "tie_word_embeddings": False,
    }
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text_config))
    decoder = model.language_model.model
    cache = model.make_cache()
    hidden = decoder.embed_tokens(mx.array([[1, 2, 3, 4]], dtype=mx.int32))
    masks = decoder_layer_masks(decoder, hidden, cache)

    assert len(masks) == 4
    assert masks[0] is None or not isinstance(masks[0], str)
    assert masks[1] == "causal"
    for index, layer in enumerate(decoder.layers):
        hidden = layer(hidden, masks[index], cache[index])
    mx.eval(hidden)
    assert hidden.shape == (1, 4, 64)


def test_a_mask_window_accepts_only_its_matching_cache_slice():
    decoder = SimpleNamespace(layers=[_linear_layer(), _attention_layer()] * 2)
    hidden = SimpleNamespace(shape=(1, 1, 8))
    masks = decoder_layer_masks(decoder, hidden, [None, None], start=1, end=3)
    assert masks == (None, None)
    with pytest.raises(DecoderTopologyError, match="cache length"):
        decoder_layer_masks(decoder, hidden, [None], start=1, end=3)


def test_mask_topology_accepts_a_structural_linear_layer_signal():
    decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(linear_attn=object()),
            SimpleNamespace(self_attn=object()),
        ]
    )
    hidden = SimpleNamespace(shape=(1, 1, 8))

    assert decoder_layer_masks(decoder, hidden, [None, None]) == (None, None)


def test_a_decoder_without_layers_is_refused():
    with pytest.raises(DecoderTopologyError, match="layers"):
        decoder_layers(SimpleNamespace(args=_args(), model=SimpleNamespace(layers=[])))


# ── Layer inventory ─────────────────────────────────────────────────────


def test_a_hybrid_reports_sixteen_attention_and_forty_eight_linear():
    topology = topology_from_model(_hybrid_model())
    assert topology.is_hybrid is True
    assert topology.num_hidden_layers == 64
    assert topology.kv_layer_count == 16
    assert len(topology.linear_layer_indices) == 48
    assert topology.attention_layer_indices[:4] == (3, 7, 11, 15)


def test_a_dense_model_reports_every_layer_as_attention():
    topology = topology_from_model(_dense_model())
    assert topology.is_hybrid is False
    assert topology.kv_layer_count == 64
    assert topology.linear_layer_indices == ()


def test_a_silent_stub_is_read_as_dense_not_as_having_no_attention():
    # A bare double gives no signal either way. Reading that as "no attention
    # layers" would refuse every existing test model; dense is the reading
    # every checkpoint had before hybrids existed.
    blocks = [object() for _ in range(8)]
    model = SimpleNamespace(args=_args(), model=SimpleNamespace(layers=blocks))
    topology = topology_from_model(model)
    assert topology.is_hybrid is False
    assert topology.kv_layer_count == 8


def test_a_model_with_only_linear_layers_is_refused():
    blocks = [_linear_layer() for _ in range(4)]
    model = SimpleNamespace(args=_args(), model=SimpleNamespace(layers=blocks))
    with pytest.raises(DecoderTopologyError, match="no attention layers"):
        topology_from_model(model)


def test_kind_at_names_the_layer_kind():
    topology = topology_from_model(_hybrid_model())
    assert topology.kind_at(3) == FULL_ATTENTION
    assert topology.kind_at(16) == LINEAR_ATTENTION
    with pytest.raises(DecoderTopologyError):
        topology.kind_at(999)


# ── Honest refusal ──────────────────────────────────────────────────────


def test_a_mechanism_needing_full_layer_kv_is_refused_on_a_hybrid():
    topology = topology_from_model(_hybrid_model())
    with pytest.raises(DecoderTopologyError) as excinfo:
        topology.require_full_layer_kv("thought-slot KV persistence")
    message = str(excinfo.value)
    # The refusal has to say how far short it falls, or the reader cannot tell
    # a hybrid from a broken load.
    assert "16 of 64" in message
    assert "48" in message


def test_the_same_mechanism_is_allowed_on_a_dense_decoder():
    topology_from_model(_dense_model()).require_full_layer_kv("slot persistence")


# ── Config reading and disagreement ─────────────────────────────────────


def test_config_layer_types_are_used_when_present():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        },
    }
    topology = topology_from_config(config)
    assert topology.attention_layer_indices == (3, 7)


def test_the_interval_is_used_when_layer_types_are_absent():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 8,
            "hidden_size": 5120,
            "full_attention_interval": 4,
        },
    }
    assert topology_from_config(config).attention_layer_indices == (3, 7)


def test_a_config_with_neither_reads_as_dense():
    config = {"model_type": "qwen2", "num_hidden_layers": 4, "hidden_size": 128}
    topology = topology_from_config(config)
    assert topology.is_hybrid is False
    assert topology.kv_layer_count == 4


def test_a_malformed_config_is_refused():
    with pytest.raises(DecoderTopologyError):
        topology_from_config({"text_config": {"num_hidden_layers": 0}})
    with pytest.raises(DecoderTopologyError):
        topology_from_config("not a mapping")


def test_agreement_between_a_model_and_its_config_is_reported_as_none():
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "full_attention_interval": 4,
        },
    }
    assert topology_disagreements(
        topology_from_model(_hybrid_model()), topology_from_config(config)
    ) == []


def test_a_layer_count_disagreement_names_itself():
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 32, "hidden_size": 5120,
                        "full_attention_interval": 4},
    }
    problems = topology_disagreements(
        topology_from_model(_hybrid_model()), topology_from_config(config)
    )
    assert any("num_hidden_layers" in problem for problem in problems)


def test_a_hidden_size_disagreement_names_itself():
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 64, "hidden_size": 4096,
                        "full_attention_interval": 4},
    }
    problems = topology_disagreements(
        topology_from_model(_hybrid_model()), topology_from_config(config)
    )
    assert any("hidden_size" in problem for problem in problems)


def test_an_attention_placement_disagreement_names_itself():
    # A config edited after a fuse, or a loader that reorders blocks, produces
    # a model whose receipts cite geometry it does not have.
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 64, "hidden_size": 5120,
                        "full_attention_interval": 8},
    }
    problems = topology_disagreements(
        topology_from_model(_hybrid_model()), topology_from_config(config)
    )
    assert any("attention layer placement" in problem for problem in problems)


def test_the_receipt_carries_the_inventory():
    receipt = topology_from_model(_hybrid_model()).to_receipt()
    assert receipt["is_hybrid"] is True
    assert len(receipt["attention_layer_indices"]) == 16
    assert len(receipt["linear_attention_layer_indices"]) == 48


def test_the_live_checkpoint_config_reads_as_the_expected_hybrid():
    from pathlib import Path

    manifest = Path("/Users/bryan/.aura/live-source/training/fused-model/active.json")
    if not manifest.exists():
        pytest.skip("no active model manifest")
    model_dir = Path(json.loads(manifest.read_text())["active_model_path"])
    if not (model_dir / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    topology = topology_from_config(json.loads((model_dir / "config.json").read_text()))
    assert topology.num_hidden_layers == 64
    assert topology.kv_layer_count == 16
    assert len(topology.linear_layer_indices) == 48
    assert topology.hidden_size == 5120
