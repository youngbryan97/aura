"""Contracts for candidate-cortex tissue compatibility inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning.model_tissue_migration_inventory import (
    MAX_METADATA_BYTES,
    MAX_PROBES,
    TissueInventoryError,
    TissueProbe,
    build_tissue_migration_inventory,
    classify_tissue_probe,
    default_tissue_probes,
    load_candidate_descriptor,
    validate_tissue_migration_inventory,
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _descriptor(tmp_path: Path, *, marker: str = "a") -> dict:
    model = tmp_path / f"candidate-{marker}"
    model.mkdir(exist_ok=True)
    descriptor = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": str(model.resolve()),
        "repository_id": "test/candidate",
        "revision": marker * 40,
        "artifact_profile": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "vocab_size": 248320,
            "layer_types": ["linear_attention", "full_attention"],
        },
        "weight_identity": {
            "files": 1,
            "fingerprint": marker * 64,
            "method": "sha256",
        },
        "behavior_identity": {
            "bundle_sha256": ("b" if marker != "b" else "c") * 64,
            "file_count": 1,
            "files": [
                {
                    "path": "config.json",
                    "sha256": ("c" if marker != "c" else "d") * 64,
                    "size_bytes": 1,
                }
            ],
        },
    }
    descriptor["descriptor_sha256"] = hashlib.sha256(_canonical(descriptor)).hexdigest()
    return descriptor


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _probe(path: Path, *, basis: str = "checkpoint_basis", mismatch: str = "retrain") -> TissueProbe:
    return TissueProbe(
        family="persona_crsm",
        artifact_id=path.stem,
        artifact_kind="test_artifact",
        path=path,
        basis_class=basis,
        mismatch_outcome=mismatch,
    )


def test_exact_candidate_descriptor_qualifies_basis_artifact(tmp_path):
    descriptor = _descriptor(tmp_path)
    artifact = tmp_path / "adapter.json"
    _write_json(
        artifact,
        {"model_descriptor_sha256": descriptor["descriptor_sha256"]},
    )

    result = classify_tissue_probe(_probe(artifact), descriptor)

    assert result["outcome"] == "qualified"
    assert result["candidate_load_authorized"] is True
    assert result["identity"]["strength"] == "exact_descriptor"


def test_width_match_never_qualifies_wrong_checkpoint(tmp_path):
    descriptor = _descriptor(tmp_path)
    artifact = tmp_path / "adapter.json"
    _write_json(
        artifact,
        {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "model_descriptor_sha256": "f" * 64,
        },
    )

    result = classify_tissue_probe(_probe(artifact), descriptor)

    assert result["outcome"] == "retrain"
    assert result["candidate_load_authorized"] is False
    assert "candidate_basis_mismatch" in result["reason_codes"]


def test_legacy_weight_and_behavior_pair_is_exact_identity(tmp_path):
    descriptor = _descriptor(tmp_path)
    artifact = tmp_path / "legacy.json"
    _write_json(
        artifact,
        {
            "base_checkpoint": {
                "method": "sha256",
                "fingerprint": descriptor["weight_identity"]["fingerprint"],
            },
            "model_behavior_bundle": {
                "bundle_sha256": descriptor["behavior_identity"]["bundle_sha256"]
            },
        },
    )

    result = classify_tissue_probe(_probe(artifact), descriptor)

    assert result["outcome"] == "qualified"
    assert result["identity"]["strength"] == "exact_weight_and_behavior_bundle"


@pytest.mark.parametrize("identity_key", ["model", "model_config_sha256", "checkpoint_fingerprint"])
def test_partial_identity_is_not_enough_for_candidate_load(tmp_path, identity_key):
    descriptor = _descriptor(tmp_path)
    values = {
        "model": descriptor["canonical_path"],
        "model_config_sha256": descriptor["behavior_identity"]["files"][0]["sha256"],
        "checkpoint_fingerprint": descriptor["weight_identity"]["fingerprint"],
    }
    artifact = tmp_path / f"partial-{identity_key}.json"
    _write_json(artifact, {identity_key: values[identity_key]})

    result = classify_tissue_probe(_probe(artifact), descriptor)

    assert result["outcome"] == "refuse"
    assert result["candidate_load_authorized"] is False
    assert "exact_candidate_identity_absent" in result["reason_codes"]


def test_architecture_independent_corpus_is_reusable_by_content_digest(tmp_path):
    descriptor = _descriptor(tmp_path)
    corpus = tmp_path / "train.jsonl"
    corpus.write_text('{"prompt":"x","answer":"y"}\n', encoding="utf-8")

    result = classify_tissue_probe(
        TissueProbe(
            family="persona_crsm",
            artifact_id="corpus",
            artifact_kind="training_corpus",
            path=corpus,
            basis_class="architecture_independent",
            qualification_scope="retraining_input",
        ),
        descriptor,
    )

    assert result["outcome"] == "qualified"
    assert result["qualification_scope"] == "retraining_input"
    assert result["candidate_load_authorized"] is False
    assert result["integrity"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()


def test_tokenized_dataset_preserves_text_but_requires_retokenization(tmp_path):
    descriptor = _descriptor(tmp_path)
    dataset = tmp_path / "dataset.json"
    _write_json(
        dataset,
        {
            "examples": [
                {
                    "prompt": "x",
                    "answer": "y",
                    "prompt_tokens": list(range(60_000)),
                    "answer_tokens": [1],
                }
            ]
        },
    )
    probe = TissueProbe(
        family="recurrent_tissue",
        artifact_id="dataset",
        artifact_kind="mixed_dataset",
        path=dataset,
        basis_class="mixed_tokenized_data",
        portable_fields=("prompt", "answer"),
        basis_bound_fields=("prompt_tokens", "answer_tokens"),
        qualification_scope="retraining_input",
    )

    result = classify_tissue_probe(probe, descriptor)

    assert result["outcome"] == "retrain"
    assert result["portable_fields"] == ["prompt", "answer"]
    assert result["field_inventory"]["complete"] is True
    assert result["field_inventory"]["portable"] == {"prompt": 1, "answer": 1}
    assert "portable_source_requires_candidate_retokenization" in result["reason_codes"]


def test_architecture_independent_tissue_verifies_manifest_and_linked_weights(tmp_path):
    descriptor = _descriptor(tmp_path)
    weights = tmp_path / "weights.npz"
    weights.write_bytes(b"portable auxiliary tissue")
    manifest = {
        "schema": "aura.portable_tissue.v1",
        "weights_file": weights.name,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    probe = TissueProbe(
        family="recurrent_tissue",
        artifact_id="portable",
        artifact_kind="architecture_independent_auxiliary_tissue",
        path=manifest_path,
        basis_class="architecture_independent",
        qualification_scope="cross_model_auxiliary",
        require_self_digest=True,
        linked_file_field="weights_file",
        linked_sha256_field="weights_sha256",
    )

    qualified = classify_tissue_probe(probe, descriptor)

    assert qualified["outcome"] == "qualified"
    assert qualified["linked_artifact"]["status"] == "verified"
    weights.write_bytes(b"tampered")
    refused = classify_tissue_probe(probe, descriptor)
    assert refused["outcome"] == "refuse"
    assert refused["linked_artifact"]["status"] == "digest_mismatch"


def test_vector_bundle_count_mismatch_refuses_even_with_exact_descriptor(tmp_path):
    descriptor = _descriptor(tmp_path)
    meta = tmp_path / "caa.json"
    _write_json(
        meta,
        {
            "model_descriptor_sha256": descriptor["descriptor_sha256"],
            "total_vectors": 2,
        },
    )
    (tmp_path / "one.npz").write_bytes(b"vector")
    probe = TissueProbe(
        family="caa_steering",
        artifact_id="vectors",
        artifact_kind="activation_vectors",
        path=meta,
        basis_class="activation_basis",
        related_patterns=("*.npz",),
        declared_count_field="total_vectors",
    )

    result = classify_tissue_probe(probe, descriptor)

    assert result["outcome"] == "refuse"
    assert result["related_bundle"]["count_matches"] is False
    assert result["candidate_load_authorized"] is False


def test_missing_optional_expert_registry_is_retired(tmp_path):
    descriptor = _descriptor(tmp_path)
    probe = TissueProbe(
        family="expert_adapters",
        artifact_id="registry",
        artifact_kind="registry",
        path=tmp_path / "missing.json",
        basis_class="checkpoint_basis",
        mismatch_outcome="retire",
    )

    result = classify_tissue_probe(probe, descriptor)

    assert result["outcome"] == "retire"
    assert result["reason_codes"] == ["artifact_absent"]


def test_default_inventory_declares_absent_durable_fast_weights(tmp_path):
    probes = default_tissue_probes(repo_root=tmp_path, state_root=tmp_path)
    durable = [
        probe
        for probe in probes
        if probe.artifact_id == "durable_fast_weight_artifacts"
    ]

    assert len(durable) == 1
    assert durable[0].basis_class == "checkpoint_basis"
    assert durable[0].mismatch_outcome == "retire"


def test_inventory_is_machine_validated_and_family_actions_are_explicit(tmp_path):
    descriptor = _descriptor(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    old = tmp_path / "old.json"
    _write_json(old, {"model_descriptor_sha256": "f" * 64})
    missing = tmp_path / "missing.json"
    probes = [
        TissueProbe(
            "persona_crsm",
            "data",
            "corpus",
            corpus,
            "architecture_independent",
        ),
        TissueProbe(
            "persona_crsm",
            "adapter",
            "adapter",
            old,
            "checkpoint_basis",
        ),
        TissueProbe(
            "expert_adapters",
            "registry",
            "registry",
            missing,
            "checkpoint_basis",
            mismatch_outcome="retire",
        ),
    ]
    for family in ("caa_steering", "recurrent_tissue", "fast_weight_adapters"):
        recipe = tmp_path / f"{family}.py"
        recipe.write_text("pass\n", encoding="utf-8")
        probes.append(
            TissueProbe(
                family,
                "recipe",
                "recipe",
                recipe,
                "architecture_independent",
            )
        )

    inventory = build_tissue_migration_inventory(descriptor, probes, generated_at=1.0)

    assert validate_tissue_migration_inventory(inventory) == inventory
    summaries = {item["family"]: item for item in inventory["families"]}
    assert summaries["persona_crsm"]["outcome"] == "retrain"
    assert summaries["expert_adapters"]["outcome"] == "retire"
    assert inventory["promotion_ready"] is False
    tampered = json.loads(json.dumps(inventory))
    tampered["promotion_ready"] = True
    with pytest.raises(TissueInventoryError, match="inventory_digest_invalid"):
        validate_tissue_migration_inventory(tampered)


def test_family_refusal_is_not_hidden_by_retrain_entry(tmp_path):
    descriptor = _descriptor(tmp_path)
    mismatched = tmp_path / "mismatched.json"
    _write_json(mismatched, {"model_descriptor_sha256": "f" * 64})
    unbound = tmp_path / "unbound.json"
    _write_json(unbound, {"adapter_id": "unknown-origin"})
    probes = [
        TissueProbe(
            "persona_crsm",
            "mismatch",
            "adapter",
            mismatched,
            "checkpoint_basis",
            mismatch_outcome="retrain",
        ),
        TissueProbe(
            "persona_crsm",
            "unbound",
            "adapter",
            unbound,
            "checkpoint_basis",
            mismatch_outcome="retrain",
        ),
    ]
    for family in ("caa_steering", "recurrent_tissue", "fast_weight_adapters"):
        recipe = tmp_path / f"{family}.py"
        recipe.write_text("pass\n", encoding="utf-8")
        probes.append(
            TissueProbe(
                family,
                "recipe",
                "recipe",
                recipe,
                "architecture_independent",
            )
        )
    probes.append(
        TissueProbe(
            "expert_adapters",
            "registry",
            "registry",
            tmp_path / "missing.json",
            "checkpoint_basis",
            mismatch_outcome="retire",
        )
    )

    inventory = build_tissue_migration_inventory(descriptor, probes)
    summary = next(
        item for item in inventory["families"] if item["family"] == "persona_crsm"
    )

    assert summary["outcome"] == "refuse"
    assert summary["outcome_counts"]["retrain"] == 1
    assert summary["outcome_counts"]["refuse"] == 1


def test_probe_and_metadata_bounds_fail_closed(tmp_path):
    descriptor = _descriptor(tmp_path)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b"x" * (MAX_METADATA_BYTES + 1))
    result = classify_tissue_probe(_probe(oversized), descriptor)
    assert result["outcome"] == "refuse"
    assert any("artifact_size_out_of_bounds" in reason for reason in result["reason_codes"])

    probe = TissueProbe(
        "persona_crsm",
        "same",
        "test",
        tmp_path / "missing",
        "checkpoint_basis",
    )
    with pytest.raises(TissueInventoryError, match="tissue_probe_count_out_of_bounds"):
        build_tissue_migration_inventory(descriptor, [probe] * (MAX_PROBES + 1))


def test_candidate_descriptor_load_is_strict_and_bounded(tmp_path):
    descriptor = _descriptor(tmp_path)
    path = tmp_path / "descriptor.json"
    _write_json(path, descriptor)

    assert load_candidate_descriptor(path) == descriptor
    path.write_bytes(b"x" * (MAX_METADATA_BYTES + 1))
    with pytest.raises(TissueInventoryError, match="artifact_size_out_of_bounds"):
        load_candidate_descriptor(path)


def test_cli_inventory_does_not_import_model_runtime(tmp_path, monkeypatch):
    import tools.inventory_model_tissue_migration as cli

    descriptor = _descriptor(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    _write_json(descriptor_path, descriptor)
    source = tmp_path / "recipe.py"
    source.write_text("pass\n", encoding="utf-8")
    probe = TissueProbe(
        "fast_weight_adapters",
        "recipe",
        "recipe",
        source,
        "architecture_independent",
    )
    monkeypatch.setattr(cli, "default_tissue_probes", lambda **_kwargs: [probe])
    monkeypatch.setitem(
        __import__("sys").modules,
        "mlx_lm",
        SimpleNamespace(load=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model load"))),
    )
    out = tmp_path / "inventory.json"

    inventory = cli.run(
        SimpleNamespace(
            descriptor=descriptor_path,
            repo_root=tmp_path,
            state_root=tmp_path,
            out=out,
        )
    )

    assert out.is_file()
    assert inventory["limits"]["model_loaded"] is False
    assert json.loads(out.read_text())["inventory_sha256"] == inventory["inventory_sha256"]
