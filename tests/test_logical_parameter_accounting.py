"""Parameter counts are computed from the topology, or reported unsupported.

The estimator assumed one decoder shape: square Q/O projections, every layer an
attention layer, and embeddings doubled whether or not they were tied. On a
Qwen3.5 hybrid all three are wrong, and none of them raise — the function would
have returned a confident number for a model it had never seen.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.brain.llm.model_artifact_profile import (
    _attention_layer_parameters,
    _estimate_parameters_from_config,
    _linear_attention_layer_parameters,
    parameter_cross_check,
)

INSTALL = Path("/Users/bryan/.aura/live-source")
RESIDENT = INSTALL / "training/fused-model/active.json"
LEGACY = (
    INSTALL
    / "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118/config.json"
)


def _resident_config():
    if not RESIDENT.exists():
        pytest.skip("no active model manifest")
    model = Path(json.loads(RESIDENT.read_text())["active_model_path"])
    if not (model / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    return json.loads((model / "config.json").read_text()), model


# ── Against the real checkpoints ────────────────────────────────────────


def test_the_hybrid_formula_matches_the_checkpoints_own_metadata():
    config, model = _resident_config()
    index = model / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip("no weight index")
    declared = json.loads(index.read_text()).get("metadata", {}).get("total_parameters")
    if not declared:
        pytest.skip("index declares no parameter total")
    estimate = _estimate_parameters_from_config(config)
    assert estimate > 0
    # Norms and biases are the only omission, so this is tight, not coarse.
    assert abs(estimate - int(declared)) / int(declared) < 1e-5


def test_the_dense_formula_still_lands_on_the_right_model():
    if not LEGACY.exists():
        pytest.skip("legacy checkpoint is not installed")
    estimate = _estimate_parameters_from_config(json.loads(LEGACY.read_text()))
    # Qwen2.5-32B is ~32.8B; a decoder counted as dense must not drift a class.
    assert 30e9 < estimate < 35e9


def test_a_hybrid_is_not_counted_as_a_dense_decoder():
    config, _ = _resident_config()
    hybrid = _estimate_parameters_from_config(config)
    dense = json.loads(json.dumps(config))
    text = dense.get("text_config", dense)
    text.pop("layer_types", None)
    text.pop("full_attention_interval", None)
    assert _estimate_parameters_from_config(dense) != hybrid


# ── Unsupported geometry is reported, not guessed ───────────────────────


def test_a_hybrid_missing_its_recurrent_geometry_is_unsupported():
    config, _ = _resident_config()
    broken = json.loads(json.dumps(config))
    broken.get("text_config", broken).pop("linear_key_head_dim", None)
    # Falling back to the dense formula would report a number for a model this
    # code has never seen, and nothing downstream could tell.
    assert _estimate_parameters_from_config(broken) == 0


def test_each_recurrent_field_is_individually_required():
    config, _ = _resident_config()
    for field in (
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_conv_kernel_dim",
    ):
        broken = json.loads(json.dumps(config))
        broken.get("text_config", broken)[field] = 0
        assert _estimate_parameters_from_config(broken) == 0, field


@pytest.mark.parametrize(
    "missing",
    ["hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
     "intermediate_size"],
)
def test_missing_core_geometry_is_unsupported(missing):
    config = {
        "model_type": "qwen2",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 27648,
        "vocab_size": 152064,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    }
    config[missing] = 0
    assert _estimate_parameters_from_config(config) == 0


def test_a_non_numeric_config_is_unsupported():
    assert _estimate_parameters_from_config({"hidden_size": "wide"}) == 0


# ── The exact rules the old formula got wrong ───────────────────────────


def test_tied_embeddings_are_counted_once():
    base = {
        "model_type": "qwen2",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    untied = _estimate_parameters_from_config({**base, "tie_word_embeddings": False})
    tied = _estimate_parameters_from_config({**base, "tie_word_embeddings": True})
    assert untied - tied == 1000 * 128


def test_a_query_width_wider_than_the_residual_stream_is_counted_as_such():
    # 24 heads of 256 is 6144 against a 5120 stream. hidden*hidden understates
    # both the query and the output projection.
    square = _attention_layer_parameters(
        hidden=5120, heads=40, kv_heads=8, head_dim=128, gated_output=False
    )
    wide = _attention_layer_parameters(
        hidden=5120, heads=24, kv_heads=4, head_dim=256, gated_output=False
    )
    assert wide != square


def test_an_output_gate_doubles_the_query_projection():
    plain = _attention_layer_parameters(
        hidden=5120, heads=24, kv_heads=4, head_dim=256, gated_output=False
    )
    gated = _attention_layer_parameters(
        hidden=5120, heads=24, kv_heads=4, head_dim=256, gated_output=True
    )
    assert gated - plain == 5120 * 24 * 256


def test_a_recurrent_block_shares_nothing_with_an_attention_block():
    attention = _attention_layer_parameters(
        hidden=5120, heads=24, kv_heads=4, head_dim=256, gated_output=True
    )
    recurrent = _linear_attention_layer_parameters(
        hidden=5120,
        key_head_dim=128,
        value_head_dim=128,
        num_key_heads=16,
        num_value_heads=48,
        conv_kernel=4,
    )
    assert recurrent != attention
    assert recurrent > 0


def test_the_depthwise_convolution_is_not_counted_as_dense():
    # groups == channels, so the kernel is per-channel, not channels squared.
    small = _linear_attention_layer_parameters(
        hidden=64, key_head_dim=8, value_head_dim=8, num_key_heads=2,
        num_value_heads=4, conv_kernel=4,
    )
    key_dim, value_dim = 2 * 8, 4 * 8
    conv_dim = 2 * key_dim + value_dim
    dense_conv_would_be = conv_dim * conv_dim * 4
    assert small < dense_conv_would_be


# ── Storage is a different question from logical parameters ─────────────


def test_quantization_artifacts_are_excluded_from_the_tensor_inventory():
    config, model = _resident_config()
    index = model / "model.safetensors.index.json"
    if not index.exists():
        pytest.skip("no weight index")
    weight_map = json.loads(index.read_text())["weight_map"]
    report = parameter_cross_check(config, weight_map)
    assert report["supported"] is True
    assert report["quantization_artifacts_excluded"] > 0
    assert report["stored_tensor_count"] < len(weight_map)


def test_scales_and_biases_never_count_as_weights():
    weight_map = {
        "layers.0.q_proj.weight": "a",
        "layers.0.q_proj.scales": "a",
        "layers.0.q_proj.biases": "a",
    }
    report = parameter_cross_check({"hidden_size": 0}, weight_map)
    assert report["stored_tensor_count"] == 1
    assert report["quantization_artifacts_excluded"] == 2


def test_an_absent_inventory_reports_unknown_rather_than_zero():
    config, _ = _resident_config()
    report = parameter_cross_check(config, None)
    assert report["stored_tensor_count"] is None
    assert report["agreement"] is None


def test_an_unsupported_geometry_reports_unsupported_in_the_cross_check():
    report = parameter_cross_check({"hidden_size": 0}, {"a.weight": "x"})
    assert report["supported"] is False
    assert report["config_parameters"] is None
