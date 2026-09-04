from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import core.learning.semantic_program_compositional_campaign as campaign
from tests.test_semantic_program_shared_transducer import _examples, _grounding


def test_withheld_family_cannot_choose_the_training_session_basis(monkeypatch) -> None:
    examples = _examples()
    manifests = {
        "fit_a": {"manifest_sha256": "1" * 64},
        "fit_b": {"manifest_sha256": "2" * 64},
        "held": {"manifest_sha256": "3" * 64},
    }
    bundles = {
        family: SimpleNamespace(manifest=manifest, examples=examples)
        for family, manifest in manifests.items()
    }
    target_basis = examples[0].ir.model_basis_receipt_sha256
    observed: dict[str, object] = {}

    observed_required_splits: dict[str, frozenset[str]] = {}

    def convert_bundle(bundle, *, required_splits: frozenset[str]):
        family = next(name for name, candidate in bundles.items() if candidate is bundle)
        observed_required_splits[family] = required_splits
        return bundle.examples

    monkeypatch.setattr(campaign, "training_examples_from_feature_bundle", convert_bundle)

    def establish_fit(selected):
        observed["fit_manifest_names"] = set(selected)
        return {
            "target_training_session_basis_sha256": target_basis,
            "source_session_basis_sha256s": {
                "fit_a": [target_basis],
                "fit_b": ["f" * 64],
            },
            "receipt_sha256": "4" * 64,
        }

    monkeypatch.setattr(
        campaign,
        "establish_semantic_training_representation_compatibility",
        establish_fit,
    )

    def bind_fit(grouped, *, compatibility):
        assert compatibility["target_training_session_basis_sha256"] == target_basis
        return tuple(
            replace(
                item,
                construction_id=f"{family}:{item.construction_id}",
                topology_id=f"{family}:{item.topology_id}",
            )
            for family in sorted(grouped)
            for item in grouped[family]
        )

    monkeypatch.setattr(
        campaign,
        "bind_training_examples_to_shared_representation",
        bind_fit,
    )

    model = SimpleNamespace(
        model_basis_sha256=target_basis,
        receipt_sha256="5" * 64,
    )

    def fit(received, *, input_grounding):
        assert input_grounding is not None
        observed["fit_constructions"] = {
            item.construction_id.partition(":")[0] for item in received
        }
        return model

    monkeypatch.setattr(
        campaign,
        "fit_compositional_semantic_program_transducer",
        fit,
    )

    held_compatibility = {
        "training_session_basis_sha256": target_basis,
        "receipt_sha256": "6" * 64,
    }

    def establish_held(*, model, training_manifest, replication_manifest):
        observed["anchor_manifest"] = training_manifest
        observed["replication_manifest"] = replication_manifest
        return held_compatibility

    monkeypatch.setattr(
        campaign,
        "establish_semantic_representation_compatibility",
        establish_held,
    )
    monkeypatch.setattr(
        campaign,
        "bind_examples_to_compatible_training_session",
        lambda received, *, compatibility: tuple(received),
    )
    monkeypatch.setattr(
        campaign,
        "_family_report",
        lambda _model, received: {"example_count": len(received)},
    )

    result = campaign.run_compositional_leave_family_out_campaign(
        bundles,
        held_out_family="held",
        input_grounding=_grounding(),
        evaluation_families=("held",),
    )

    assert observed["fit_manifest_names"] == {"fit_a", "fit_b"}
    assert observed_required_splits == {
        "fit_a": frozenset({"train", "validation", "test"}),
        "fit_b": frozenset({"train", "validation", "test"}),
        "held": frozenset({"validation", "test"}),
    }
    assert observed["fit_constructions"] == {"fit_a", "fit_b"}
    assert observed["anchor_manifest"] is manifests["fit_a"]
    assert observed["replication_manifest"] is manifests["held"]
    assert result.report["representation_compatibility"][
        "target_training_session_basis_sha256"
    ] == target_basis
    assert result.report["held_out_representation_compatibility"] == (
        held_compatibility
    )
    assert result.report["families"]["held"]["example_count"] == len(examples)
    assert result.report["evaluated_families"] == ["held"]
    assert set(result.report["families"]) == {"held"}
