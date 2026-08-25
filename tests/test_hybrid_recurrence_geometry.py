"""The recurrence window has to know which layers the checkpoint actually has.

Qwen2.5-32B was 64 attention layers. Qwen3.8-27B gives a layer ``self_attn``
only when ``(index + 1) % 4 == 0``. The same window declaration therefore means
two different experiments, and the smaller one does not raise on its own.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.learning.hybrid_recurrence_geometry import (
    LayerGeometry,
    expected_adapter_sites,
    geometry_receipt,
    portable_targets,
    window_alignment_errors,
)

DENSE = LayerGeometry(num_hidden_layers=64)
HYBRID = LayerGeometry(num_hidden_layers=64, full_attention_interval=4)


def test_a_dense_checkpoint_carries_attention_everywhere():
    assert DENSE.attention_layers() == tuple(range(64))
    assert DENSE.linear_layers() == ()
    assert not DENSE.is_hybrid


def test_a_hybrid_checkpoint_carries_attention_every_fourth_layer():
    # Matches mlx_lm.models.qwen3_5.DecoderLayer: the attention layer is the
    # LAST of each group, not the first.
    assert HYBRID.attention_layers()[:4] == (3, 7, 11, 15)
    assert len(HYBRID.attention_layers()) == 16
    assert len(HYBRID.linear_layers()) == 48
    assert not HYBRID.carries_attention(16)
    assert not HYBRID.carries_attention(17)


def test_the_default_window_collapses_to_a_quarter_of_its_sites():
    # This is the whole point. Same window, same targets, same shapes — and a
    # quarter of the adapted capacity, with nothing raising to say so.
    targets = ("o_proj", "v_proj")
    dense = expected_adapter_sites(DENSE, 16, 48, targets)
    hybrid = expected_adapter_sites(HYBRID, 16, 48, targets)
    assert len(dense) == 64
    assert len(hybrid) == 16
    assert set(hybrid) <= set(dense)


def test_attention_targets_land_only_on_attention_layers():
    sites = expected_adapter_sites(HYBRID, 16, 48, ("o_proj",))
    indices = sorted(int(site.split(".")[2]) for site in sites)
    assert indices == [19, 23, 27, 31, 35, 39, 43, 47]
    assert all(".self_attn." in site for site in sites)


def test_a_feed_forward_target_reaches_every_layer_of_the_window():
    sites = expected_adapter_sites(HYBRID, 16, 48, ("down_proj",))
    assert len(sites) == 32
    assert all(".mlp." in site for site in sites)


def test_the_default_targets_reach_no_layer_universally_on_a_hybrid():
    assert portable_targets(HYBRID, ("o_proj", "v_proj")) == ()
    assert portable_targets(DENSE, ("o_proj", "v_proj")) == ("o_proj", "v_proj")
    assert portable_targets(HYBRID, ("o_proj", "down_proj")) == ("down_proj",)


def test_the_shipped_window_is_aligned_and_a_neighbouring_one_is_not():
    assert window_alignment_errors(HYBRID, 16, 48) == []
    assert window_alignment_errors(HYBRID, 15, 48)
    assert window_alignment_errors(HYBRID, 16, 47)


def test_alignment_is_not_imposed_on_a_dense_checkpoint():
    assert window_alignment_errors(DENSE, 15, 47) == []


def test_a_window_with_no_attention_layer_is_refused():
    # Interval 8 over a short stack: layers 0..3 are all linear.
    sparse = LayerGeometry(num_hidden_layers=16, full_attention_interval=8)
    errors = window_alignment_errors(sparse, 0, 4)
    assert any("no attention layer" in error for error in errors)


def test_bounds_outside_the_checkpoint_are_refused():
    assert window_alignment_errors(HYBRID, 16, 65)
    assert window_alignment_errors(HYBRID, 48, 16)


def test_geometry_reads_from_a_nested_config():
    config = {
        "model_type": "qwen3_5",
        "text_config": {"num_hidden_layers": 64, "full_attention_interval": 4},
    }
    assert LayerGeometry.from_config(config) == HYBRID


def test_geometry_reads_from_a_flat_config():
    assert LayerGeometry.from_config(
        {"num_hidden_layers": 64}
    ) == DENSE


def test_geometry_reads_from_a_loaded_model():
    class _Layer:
        def __init__(self, index):
            self.is_linear = (index + 1) % 4 != 0

    model = SimpleNamespace(
        model=SimpleNamespace(layers=[_Layer(i) for i in range(64)])
    )
    assert LayerGeometry.from_model(model) == HYBRID


def test_a_model_with_irregular_attention_placement_is_refused():
    class _Layer:
        def __init__(self, linear):
            self.is_linear = linear

    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[_Layer(x) for x in (True, False, True, True, True, False)]
        )
    )
    with pytest.raises(ValueError, match="fixed interval"):
        LayerGeometry.from_model(model)


def test_the_receipt_commits_what_the_campaign_is_about_to_train():
    receipt = geometry_receipt(HYBRID, 16, 48, ("o_proj", "v_proj"))
    assert receipt["expected_adapter_site_count"] == 16
    assert receipt["attention_layers_in_window"] == [19, 23, 27, 31, 35, 39, 43, 47]
    assert len(receipt["linear_layers_in_window"]) == 24
    assert receipt["alignment_errors"] == []
    assert receipt["targets_reaching_every_window_layer"] == []


def test_the_live_checkpoints_are_the_two_this_module_was_written_for():
    """Read the geometry off the checkpoints on disk when they are present."""
    import json
    from pathlib import Path

    install = Path(__file__).resolve().parents[1]
    legacy = (
        install
        / "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
        / "config.json"
    )
    manifest = install / "training/fused-model/active.json"
    if not (legacy.exists() and manifest.exists()):
        pytest.skip("checkpoints are not installed in this environment")

    assert LayerGeometry.from_config(json.loads(legacy.read_text())) == DENSE
    active = Path(json.loads(manifest.read_text())["active_model_path"])
    if not (active / "config.json").exists():
        pytest.skip("active checkpoint is not installed")
    assert LayerGeometry.from_config(json.loads((active / "config.json").read_text())) == HYBRID


# ── Resolving a projection's parent block ────────────────────────────────
# `hasattr(layer.self_attn, target)` raised on a hybrid checkpoint: the
# attribute lookup happens before hasattr can guard it. On a dense model every
# layer answered and the expression looked total.


class _Projection:
    pass


def _attention_layer():
    return SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=_Projection(), v_proj=_Projection()),
        mlp=SimpleNamespace(down_proj=_Projection(), gate_proj=_Projection()),
    )


def _linear_layer():
    return SimpleNamespace(
        linear_attn=SimpleNamespace(in_proj_qkvz=_Projection()),
        mlp=SimpleNamespace(down_proj=_Projection(), gate_proj=_Projection()),
    )


def test_a_linear_layer_reports_no_attention_projection_instead_of_raising():
    from core.learning.hybrid_recurrence_geometry import resolve_projection_parent

    assert resolve_projection_parent(_linear_layer(), "o_proj") is None


def test_an_attention_layer_resolves_its_own_block():
    from core.learning.hybrid_recurrence_geometry import resolve_projection_parent

    name, parent = resolve_projection_parent(_attention_layer(), "o_proj")
    assert name == "self_attn"
    assert parent is not None


def test_a_feed_forward_projection_resolves_on_either_layer_kind():
    from core.learning.hybrid_recurrence_geometry import resolve_projection_parent

    for layer in (_attention_layer(), _linear_layer()):
        name, _ = resolve_projection_parent(layer, "down_proj")
        assert name == "mlp"


def test_a_linear_attention_projection_is_reachable():
    from core.learning.hybrid_recurrence_geometry import resolve_projection_parent

    name, _ = resolve_projection_parent(_linear_layer(), "in_proj_qkvz")
    assert name == "linear_attn"


def test_adaptable_sites_agree_with_the_declared_geometry_on_a_hybrid_stack():
    from core.learning.hybrid_recurrence_geometry import (
        adaptable_sites,
        expected_adapter_sites,
    )

    layers = [
        _attention_layer() if (i + 1) % 4 == 0 else _linear_layer()
        for i in range(64)
    ]
    targets = ("o_proj", "down_proj")
    walked = adaptable_sites(layers, range(16, 48), targets)
    declared = expected_adapter_sites(HYBRID, 16, 48, targets)
    assert sorted(walked) == sorted(declared)
    assert len(walked) == 8 + 32


def test_adaptable_sites_agree_on_a_dense_stack():
    from core.learning.hybrid_recurrence_geometry import (
        adaptable_sites,
        expected_adapter_sites,
    )

    layers = [_attention_layer() for _ in range(64)]
    targets = ("o_proj", "down_proj")
    walked = adaptable_sites(layers, range(16, 48), targets)
    declared = expected_adapter_sites(DENSE, 16, 48, targets)
    assert sorted(walked) == sorted(declared)
    assert len(walked) == 64


def test_the_old_expression_is_the_one_that_raised():
    """Pin the defect so nobody restores the shorter spelling."""
    layer = _linear_layer()
    with pytest.raises(AttributeError):
        hasattr(layer.self_attn, "o_proj")  # noqa: B018
