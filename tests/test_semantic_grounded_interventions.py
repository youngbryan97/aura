from types import SimpleNamespace

import pytest

from tools.semantic_grounded_development_archive import digest
from tools.semantic_grounded_interventions import (
    MODES,
    IntervenedGroundedDecoder,
    grounded_evidence_intervention,
)


def owner(kind="lora"):
    from tests.test_semantic_native_adapters import plan, tiny_model
    from tools.semantic_native_adapters import install_native_adapters

    model = tiny_model()
    value = plan(rank=2, layers=2, kind=kind, sites="attention_mlp_v1", alpha=4.,
        layer_kinds=[kind, kind], scaling="alpha_over_sqrt_rank_v1")
    install_native_adapters(model, value)
    return SimpleNamespace(suffix=SimpleSuffix(model.layers[-2:]), plan=value,
        engine=SimpleNamespace(evidence_weight=1., pointer=model.layers[0]),
        verification={"learned_checkpoint_selected": True, "selected_step": 4,
            "fit_receipt_sha256": "a" * 64, "weights_sha256": "b" * 64})


def SimpleSuffix(layers):
    import mlx.nn as nn

    suffix = nn.Module()
    suffix.layers = layers
    return suffix


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["lora", "product", "silu", "dora", "routed", "square", "dense"])
def test_lesions_act_on_installed_mlx_sites_and_restore_intact_arithmetic(mode, kind):
    import mlx.core as mx

    item = owner(kind)
    site = item.suffix.layers[-1].self_attn.q_proj
    field = {"square": "lora_square", "dense": "lora_full"}.get(kind, "lora_b")
    setattr(site, field, mx.ones_like(getattr(site, field)))
    x = mx.ones((1, 4))
    intact, base = site(x), site.linear(x)
    mx.eval(intact, base)
    original = site.scale
    with grounded_evidence_intervention(item, mode) as receipt:
        expected_scale = 0. if mode in {"adapter_off", "both_off"} else original
        assert site.scale == expected_scale
        assert item.engine.evidence_weight == (0. if mode in {"relation_off", "both_off"} else 1.)
        result = site(x)
        mx.eval(result)
        assert mx.array_equal(result, base if expected_scale == 0. else intact).item()
        assert receipt["restored"] is False
    assert site.scale == original and item.engine.evidence_weight == 1.
    assert receipt["restored"] is receipt["parameter_objects_unchanged"] is True
    assert receipt["receipt_sha256"] == digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    assert not receipt["qualification_evidence"] and not receipt["serving_authority"]


def test_mixed_hybrid_control_resolves_all_linear_mixing_and_full_attention_sites():
    import mlx.nn as nn

    from tests.test_semantic_native_adapters import plan, tiny_model
    from tools.semantic_native_adapters import install_native_adapters

    model = tiny_model()
    del model.layers[-2].self_attn
    linear = nn.Module()
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
        setattr(linear, name, nn.Linear(4, 4))
    model.layers[-2].linear_attn = linear
    model.freeze()
    value = plan(rank=2, layers=2, sites="native_topology_v1", layer_kinds=["product", "silu"])
    install_native_adapters(model, value)
    item = SimpleNamespace(suffix=SimpleSuffix(model.layers[-2:]), plan=value,
        engine=SimpleNamespace(evidence_weight=1., pointer=model.layers[0]),
        verification={"learned_checkpoint_selected": True, "selected_step": 4,
            "fit_receipt_sha256": "a" * 64, "weights_sha256": "b" * 64})
    with grounded_evidence_intervention(item, "both_off") as receipt:
        assert len(receipt["sites"]) == 15
        assert {row["kind"] for row in receipt["sites"]} == {"product", "silu"}
        assert all(row["applied_scale"] == 0. for row in receipt["sites"])
    assert receipt["restored"]


def test_exception_during_decode_restores_all_sites_and_allows_an_intact_rescue():
    item = owner()
    original = item.suffix.layers[-1].self_attn.q_proj.scale
    with pytest.raises(RuntimeError, match="decode failed"):
        with grounded_evidence_intervention(item, "both_off"):
            raise RuntimeError("decode failed")
    assert item.suffix.layers[-1].self_attn.q_proj.scale == original
    assert item.engine.evidence_weight == 1.
    with grounded_evidence_intervention(item, "intact") as receipt:
        assert item.suffix.layers[-1].self_attn.q_proj.scale == original
    assert receipt["restored"]


@pytest.mark.parametrize("fault", ["mode", "zero_step", "unverified", "gate", "weight", "depth", "missing_site"])
def test_interventions_reject_drift_before_mutating_any_site(fault):
    item = owner()
    mode = "both_off"
    site = item.suffix.layers[-1].self_attn.q_proj
    original = site.scale
    if fault == "mode":
        mode = "random"
    elif fault == "zero_step":
        item.verification["selected_step"] = 0
    elif fault == "unverified":
        item.verification["learned_checkpoint_selected"] = False
    elif fault == "gate":
        site.scale *= .5
        original = site.scale
    elif fault == "weight":
        item.engine.evidence_weight = 0.
    elif fault == "depth":
        item.suffix.layers.pop()
    else:
        del item.suffix.layers[0].self_attn.k_proj
    with pytest.raises(ValueError):
        with grounded_evidence_intervention(item, mode):
            pytest.fail("changed intervention admitted")
    assert site.scale == original


def test_nested_intervention_cannot_mix_two_mutation_owners():
    item = owner()
    with grounded_evidence_intervention(item, "adapter_off"):
        with pytest.raises(ValueError, match="ownership"):
            with grounded_evidence_intervention(item, "intact"):
                pytest.fail("nested mutation owner admitted")


@pytest.mark.parametrize("fault", ["gate", "parameter"])
def test_unexpected_decode_mutation_is_reported_and_never_receipted_as_intact(fault):
    import mlx.core as mx

    item = owner()
    site = item.suffix.layers[-1].self_attn.q_proj
    original = site.scale
    original_parameter = site.lora_a
    with pytest.raises(ValueError, match="during decode"):
        with grounded_evidence_intervention(item, "adapter_off"):
            if fault == "gate":
                site.scale = .5
            else:
                site.lora_a = mx.ones_like(site.lora_a)
    assert site.scale == original and item.engine.evidence_weight == 1.
    assert site.lora_a is original_parameter
    with grounded_evidence_intervention(item, "intact") as rescue:
        assert site.lora_a is original_parameter
    assert rescue["restored"]


def test_failed_parameter_recovery_taints_owner_and_requires_reload(monkeypatch):
    import mlx.core as mx

    item = owner()
    monkeypatch.setattr(item.suffix, "update", lambda _values: None)
    with pytest.raises(ValueError, match="reload required"):
        with grounded_evidence_intervention(item, "both_off"):
            item.suffix.layers[-1].self_attn.q_proj.lora_a = mx.ones_like(
                item.suffix.layers[-1].self_attn.q_proj.lora_a)
    assert item._grounded_intervention_tainted
    with pytest.raises(ValueError, match="ownership"):
        with grounded_evidence_intervention(item, "intact"):
            pytest.fail("tainted owner reused")


def test_operation_field_parameters_are_restored_and_not_confused_with_relation_lesion():
    import mlx.core as mx

    from core.learning.semantic_native_operation_field import NativeOperationField

    item = owner()
    field = item.engine.operation_field = NativeOperationField(4, depths=2, labels=("add", "sub"))
    original = field.output.weight
    with pytest.raises(ValueError, match="during decode"):
        with grounded_evidence_intervention(item, "relation_off") as receipt:
            assert receipt["operation_field_retained"]
            field.output.weight = mx.ones_like(original)
    assert field.output.weight is original
    with grounded_evidence_intervention(item, "intact") as rescue:
        assert item.engine.operation_field is field
    assert rescue["restored"]


def test_replacing_operation_module_taints_intervention_owner():
    from core.learning.semantic_native_operation_field import NativeOperationField

    item = owner()
    item.engine.operation_field = NativeOperationField(4, depths=2, labels=("add", "sub"))
    with pytest.raises(ValueError, match="module ownership"):
        with grounded_evidence_intervention(item, "intact"):
            item.engine.operation_field = NativeOperationField(4, depths=2, labels=("add", "sub"))
    assert item._grounded_intervention_tainted


def test_public_wrapper_preserves_request_and_actual_selection_receipt():
    item = owner()
    calls = []
    def decode(**public):
        calls.append(public)
        assert item.engine.evidence_weight == 0.
        decoder.last_receipt = {"schema": "fixture", "source_id": "c" * 64, "selected_chart": {"status": "bound"}}
        return "actual output"
    decoder = SimpleNamespace(owner=item, decode=decode, last_receipt=None)
    wrapped = IntervenedGroundedDecoder(decoder, mode="relation_off")
    public = {"source_token_ids": (1, 2), "search_time_limit_s": 30., "hidden_states": None,
        "public_inputs": (3,), "source_text_sha256": "c" * 64, "model_basis_sha256": "d" * 64}
    assert wrapped.decode(**public) == "actual output"
    assert calls == [public]
    assert wrapped.last_receipt["selected_chart"] == decoder.last_receipt["selected_chart"]
    assert wrapped.last_receipt["evidence_intervention"]["restored"]
    assert item.engine.evidence_weight == 1.


def test_public_wrapper_refuses_target_information_before_entering_intervention():
    wrapped = IntervenedGroundedDecoder(SimpleNamespace(owner=owner()), mode="intact")
    with pytest.raises(ValueError, match="public request"):
        wrapped.decode(source_token_ids=(1, 2), expected_answer=42)
