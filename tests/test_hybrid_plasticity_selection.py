from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from core.brain.llm.latent_cortex.fast_weights import EpisodicFastWeights
from core.brain.llm.latent_cortex.plasticity_sites import (
    PLASTICITY_SITE_REGISTRY,
    select_compatible_plasticity_layers,
)
from core.brain.llm.latent_cortex.runtime_integrity import (
    adapted_layer_fingerprint,
)
from core.brain.llm.latent_cortex.types import FastWeightsConfig


class _Linear:
    def __init__(self) -> None:
        self.weight = mx.eye(4)

    def __call__(self, value):
        return value @ self.weight.T

    def parameters(self):
        return {"weight": self.weight}


def _hybrid_layers(count: int = 8):
    layers = []
    for index in range(count):
        attributes = {
            "mlp": SimpleNamespace(down_proj=_Linear()),
        }
        if index % 4 == 0:
            attributes["self_attn"] = SimpleNamespace(o_proj=_Linear())
        else:
            attributes["linear_attn"] = SimpleNamespace(out_proj=_Linear())
        layers.append(SimpleNamespace(**attributes))
    return layers


def test_hybrid_selection_qualifies_projection_before_placement() -> None:
    layers = _hybrid_layers(12)

    assert select_compatible_plasticity_layers(
        layers, 1, 11, 1, target="o_proj", placement="early"
    ) == (4,)
    assert select_compatible_plasticity_layers(
        layers, 1, 11, 1, target="o_proj", placement="late"
    ) == (8,)
    assert select_compatible_plasticity_layers(
        layers, 1, 11, 2, target="o_proj", placement="distributed"
    ) == (4, 8)
    assert select_compatible_plasticity_layers(
        layers, 1, 11, 2, target="down_proj", placement="early"
    ) == (1, 2)


def test_registered_site_and_fast_weight_attach_share_hybrid_layer_set() -> None:
    layers = _hybrid_layers()
    model = SimpleNamespace(layers=layers)
    site = PLASTICITY_SITE_REGISTRY.resolve("o_proj", "early")
    expected = site.compatible_layer_indices(layers, 0, len(layers), 2)
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(
            enabled=True,
            target="o_proj",
            layer_placement="early",
            max_wrapped_layers=2,
        )
    )
    measured = adapted_layer_fingerprint(
        model,
        layer_indices=expected,
        target="o_proj",
    )

    try:
        assert fast_weights.attach(
            model,
            (0, len(layers)),
            seed_stat=0.4,
            episode_id="hybrid-attach",
        ) == len(expected)
        assert tuple(handle.layer_index for handle in fast_weights.handles) == expected
        assert measured["layer_ids"] == [
            f"layers.{index}.o_proj" for index in expected
        ]
        assert measured["tensor_count"] == len(expected)
    finally:
        fast_weights.detach()


def test_absent_target_refuses_before_acquiring_model_lease() -> None:
    model = SimpleNamespace(layers=_hybrid_layers(3))
    fast_weights = EpisodicFastWeights(
        FastWeightsConfig(enabled=True, target="o_proj", max_wrapped_layers=2)
    )

    with pytest.raises(ValueError, match="target o_proj is absent"):
        fast_weights.attach(
            model,
            (1, 3),
            seed_stat=0.4,
            episode_id="hybrid-no-target",
        )

    assert fast_weights.lease_receipt()["acquired"] is False


def test_integrity_rejects_a_layer_outside_qualified_target_set() -> None:
    model = SimpleNamespace(layers=_hybrid_layers())

    with pytest.raises(ValueError, match="absent from layer 1"):
        adapted_layer_fingerprint(
            model,
            layer_indices=(1,),
            target="o_proj",
        )
