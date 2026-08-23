from __future__ import annotations

import builtins
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any

import pytest

from tools import preflight_candidate_persona_crsm as preflight


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lines(path: Path) -> int:
    return len(path.read_bytes().splitlines())


def _conversation(prompt: str, answer: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def _write_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    header: dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for name, shape in sorted(tensors.items()):
        nbytes = math.prod(shape) * 2
        header[name] = {
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _model_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "head_dim": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "vocab_size": 128,
            "max_position_embeddings": 1024,
        },
    }


def _descriptor_profile(config: dict[str, Any]) -> dict[str, Any]:
    text = config["text_config"]
    return {
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "vocab_size": text["vocab_size"],
        "native_context_window": text["max_position_embeddings"],
        "layer_types": text["layer_types"],
    }


def _rewrite_descriptor(descriptor_path: Path, model_root: Path) -> str:
    descriptor = json.loads(descriptor_path.read_text()) if descriptor_path.exists() else {}
    config = json.loads((model_root / "config.json").read_text())
    files = []
    for relative in ("config.json", "model.safetensors.index.json"):
        path = model_root / relative
        files.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    material = {
        "schema": "aura.model_artifact_descriptor.v1",
        "canonical_path": str(model_root.resolve()),
        "repository_id": descriptor.get("repository_id", "test/Qwen3.8-27B-4bit"),
        "revision": descriptor.get("revision", "test-revision"),
        "artifact_profile": _descriptor_profile(config),
        "weight_identity": {"method": "synthetic", "files": 1},
        "behavior_identity": {"file_count": len(files), "files": files},
    }
    digest = preflight._document_sha256(material)
    _write_json(descriptor_path, {**material, "descriptor_sha256": digest})
    return digest


def _manifest_record(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "lines": _lines(path)}


def _refresh_manifests(data_root: Path) -> None:
    source = data_root / "data/synthetic_training/lora_dataset.jsonl"
    train = data_root / "training/data/train.jsonl"
    valid = data_root / "training/data/valid.jsonl"
    delta_train = data_root / "training/data/crsm_delta/train.jsonl"
    delta_valid = data_root / "training/data/crsm_delta/valid.jsonl"
    source_record = {"source_sha256": _sha256(source), "source_lines": _lines(source)}
    _write_json(
        data_root / "training/data/crsm_integration_manifest.json",
        {
            **source_record,
            "output": {
                "train": _manifest_record(train),
                "valid": _manifest_record(valid),
            },
        },
    )
    _write_json(
        data_root / "training/data/crsm_delta_manifest.json",
        {
            **source_record,
            "output": {
                "train": _manifest_record(delta_train),
                "valid": _manifest_record(delta_valid),
            },
        },
    )


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    config = _model_config()
    _write_json(model_root / "config.json", config)
    specs = preflight._module_specs(config["text_config"])
    weight_map = {
        f"{spec['path']}.{suffix}": "model-00001-of-00001.safetensors"
        for spec in specs
        for suffix in ("weight", "scales", "biases")
    }
    _write_json(model_root / "model.safetensors.index.json", {"weight_map": weight_map})

    descriptor_path = tmp_path / "descriptor.json"
    digest = _rewrite_descriptor(descriptor_path, model_root)

    data_root = tmp_path / "data-root"
    _write_jsonl(
        data_root / "training/data/train.jsonl",
        [_conversation("train one", "answer one"), _conversation("train two", "answer two")],
    )
    _write_jsonl(
        data_root / "training/data/valid.jsonl",
        [_conversation("validation one", "held out")],
    )
    _write_jsonl(
        data_root / "data/synthetic_training/lora_dataset.jsonl",
        [{"text": "causally grounded CRSM example", "_quality": 0.9}],
    )
    _write_jsonl(
        data_root / "training/data/crsm_delta/train.jsonl",
        [_conversation("delta train", "delta result")],
    )
    _write_jsonl(
        data_root / "training/data/crsm_delta/valid.jsonl",
        [_conversation("delta validation", "delta held out")],
    )
    _refresh_manifests(data_root)

    source_root = tmp_path / "source"
    for relative in (
        "training/build_dataset_v3.py",
        "training/finetune_lora.py",
        "training/train_and_fuse.py",
    ):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n")

    legacy = data_root / "training/adapters/aura-personality"
    _write_json(
        legacy / "adapter_config.json",
        {
            "model": str(tmp_path / "models/Qwen2.5-32B-Instruct-4bit"),
            "lora_parameters": {"rank": 32, "scale": 20.0, "dropout": 0.0},
        },
    )
    _write_safetensors(
        legacy / "adapters.safetensors",
        {"model.layers.0.self_attn.q_proj.lora_a": [64, 32]},
    )
    return {
        "model_root": model_root,
        "descriptor": descriptor_path,
        "descriptor_sha256": digest,
        "data_root": data_root,
        "source_root": source_root,
        "output_root": tmp_path / "candidate-runs",
        "legacy": legacy,
    }


def _build(fixture: dict[str, Path | str]) -> dict[str, Any]:
    return preflight.build_preflight(
        descriptor_path=Path(fixture["descriptor"]),
        data_repo_root=Path(fixture["data_root"]),
        source_repo_root=Path(fixture["source_root"]),
        output_root=Path(fixture["output_root"]),
        expected_descriptor_sha256=str(fixture["descriptor_sha256"]),
        legacy_adapter=Path(fixture["legacy"]),
    )


def test_preflight_binds_dataset_topology_and_never_imports_model_stacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".", 1)[0] in {"mlx", "mlx_lm", "transformers"}:
            raise AssertionError(f"model stack imported during preflight: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = _build(fixture)

    assert result["no_model_load"] is True
    assert result["checks"]["descriptor_exact"] is True
    assert result["checks"]["persona_dataset_schema_valid"] is True
    assert result["checks"]["persona_split_disjoint"] is True
    assert result["target_modules"]["compatible"] is True
    assert result["target_modules"]["wrapped_projection_count"] == 15
    assert "linear_attn.in_proj_qkv" in result["target_modules"]["relative_keys"]
    assert "self_attn.q_proj" in result["target_modules"]["relative_keys"]
    assert result["verdict"] == "READY_FOR_PERSONA"


def test_dataset_content_identity_is_location_independent_and_mutation_bound(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _build(fixture)
    copied_root = tmp_path / "copied-data-root"
    shutil.copytree(Path(fixture["data_root"]), copied_root)
    copied_fixture = {**fixture, "data_root": copied_root}
    copied = _build(copied_fixture)

    assert (
        copied["datasets"]["persona"]["identity_sha256"]
        == first["datasets"]["persona"]["identity_sha256"]
    )
    assert copied["paths"]["run_id"] == first["paths"]["run_id"]

    train_path = copied_root / "training/data/train.jsonl"
    with train_path.open("a") as handle:
        handle.write(json.dumps(_conversation("new train", "new answer")) + "\n")
    _refresh_manifests(copied_root)
    mutated = _build(copied_fixture)

    assert (
        mutated["datasets"]["persona"]["identity_sha256"]
        != first["datasets"]["persona"]["identity_sha256"]
    )
    assert mutated["paths"]["run_id"] != first["paths"]["run_id"]
    assert mutated["paths"]["run_root"].startswith(
        str(Path(fixture["output_root"]).resolve())
    )


def test_missing_candidate_projection_blocks_target_module_compatibility(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    index_path = Path(fixture["model_root"]) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    missing_key = "language_model.model.layers.0.linear_attn.in_proj_a.weight"
    index["weight_map"].pop(missing_key)
    _write_json(index_path, index)
    fixture["descriptor_sha256"] = _rewrite_descriptor(
        Path(fixture["descriptor"]), Path(fixture["model_root"])
    )

    result = _build(fixture)

    assert result["target_modules"]["compatible"] is False
    assert result["target_modules"]["missing_weight_keys"] == [missing_key]
    assert "candidate_target_modules_incompatible" in result["readiness"][
        "persona_blockers"
    ]
    assert result["verdict"] == "BLOCKED"


def test_candidate_checkpoint_resume_is_accepted_and_qwen25_adapter_is_refused(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    initial = _build(fixture)
    adapter_dir = Path(initial["paths"]["persona_adapter_dir"])
    _write_json(
        adapter_dir / "adapter_config.json",
        {
            "model": str(Path(fixture["model_root"]).resolve()),
            "lora_parameters": {"rank": 32, "scale": 20.0, "dropout": 0.0},
        },
    )
    _write_json(
        adapter_dir / "candidate_binding.json",
        initial["resume"]["candidate_binding_template"],
    )
    config = json.loads((Path(fixture["model_root"]) / "config.json").read_text())
    specs = preflight._module_specs(config["text_config"])
    tensors = preflight._expected_adapter_tensors(specs, preflight.LORA_RANK)
    _write_safetensors(adapter_dir / "0000002_adapters.safetensors", tensors)
    latest = adapter_dir / "0000010_adapters.safetensors"
    _write_safetensors(latest, tensors)

    result = _build(fixture)
    resume = result["resume"]["candidate_persona"]
    legacy = result["resume"]["legacy_adapter_audit"]

    assert resume["accepted"] is True
    assert resume["checkpoint_step"] == 10
    assert resume["tensor_path"] == str(latest.resolve())
    assert resume["tensor_sha256"] == _sha256(latest)
    assert result["verdict"] == "READY_FOR_PERSONA_AND_CRSM"
    assert legacy["accepted"] is False
    assert "adapter_model_mismatch" in legacy["reasons"]
    assert "adapter_outside_candidate_run" in legacy["reasons"]
    assert "candidate_binding_missing" in legacy["reasons"]
    assert result["checks"]["legacy_qwen25_adapter_refused"] is True


def test_missing_crsm_delta_output_cannot_be_reported_current(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (Path(fixture["data_root"]) / "training/data/crsm_delta/valid.jsonl").unlink()

    result = _build(fixture)

    assert result["datasets"]["crsm"]["delta_valid"] is None
    assert result["datasets"]["crsm"]["delta_manifest"]["current"] is False
    assert "crsm_delta_dataset_requires_rebuild" in result["readiness"][
        "crsm_blockers"
    ]


def test_truncated_adapter_payload_is_refused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    initial = _build(fixture)
    adapter_dir = Path(initial["paths"]["persona_adapter_dir"])
    _write_json(
        adapter_dir / "adapter_config.json",
        {
            "model": str(Path(fixture["model_root"]).resolve()),
            "lora_parameters": {"rank": 32, "scale": 20.0, "dropout": 0.0},
        },
    )
    _write_json(
        adapter_dir / "candidate_binding.json",
        initial["resume"]["candidate_binding_template"],
    )
    tensor_path = adapter_dir / "adapters.safetensors"
    _write_safetensors(tensor_path, {"only.tensor": [64, 32]})
    tensor_path.write_bytes(tensor_path.read_bytes()[:-1])

    with pytest.raises(
        preflight.CandidatePreflightError,
        match="adapter_tensor_entry_invalid|adapter_tensor_layout_invalid",
    ):
        _build(fixture)
