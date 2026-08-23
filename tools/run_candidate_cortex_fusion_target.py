#!/usr/bin/env python3
"""Fuse one authenticated adaptive checkpoint into a new candidate model."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.model_artifact_profile import (  # noqa: E402
    build_model_artifact_descriptor,
)
from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_fusion import (  # noqa: E402
    FUSION_PROVENANCE_FILE,
    FUSION_RECEIPT_SCHEMA,
    CandidateCortexFusionError,
    available_bytes,
    build_fusion_provenance,
    copy_file,
    ensure_clean_directory,
    load_and_validate_fusion_plan,
    move_directory,
    publish_json,
    validate_fusion_provenance,
    validate_fusion_receipt,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    document_sha256,
    file_sha256,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

SOURCE = "candidate_cortex_fusion.target"


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateCortexFusionError("fusion_artifact_duplicate_key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise CandidateCortexFusionError("fusion_artifact_non_finite")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CandidateCortexFusionError("fusion_artifact_json_invalid") from exc
    if not isinstance(value, dict):
        raise CandidateCortexFusionError("fusion_artifact_json_invalid")
    return value


def _publish_exact(path: Path, payload: Mapping[str, Any], *, source: str) -> None:
    if path.exists():
        if path.is_symlink() or _read_json(path) != dict(payload):
            raise CandidateCortexFusionError("fusion_existing_artifact_mismatch")
        return
    publish_json(path, payload, source=source)


def _release_model_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _remove_adapter_input(plan: Mapping[str, Any]) -> None:
    path = Path(str(plan["output"]["fusion_root"])) / "adapter-input"
    if not os.path.lexists(path):
        return
    with local_internal_governed_scope(
        f"{SOURCE}.adapter_input_cleanup", domain="file_write"
    ):
        get_file_write_gateway().delete_path(
            path,
            recursive=True,
            source=f"{SOURCE}.adapter_input_cleanup",
        )


def _prepare_adapter_input(plan: Mapping[str, Any]) -> Path:
    fusion_root = Path(str(plan["output"]["fusion_root"]))
    adapter_input = ensure_clean_directory(
        fusion_root / "adapter-input",
        source=f"{SOURCE}.adapter_input",
    )
    copy_file(
        Path(str(plan["adaptive"]["checkpoint"]["path"])),
        adapter_input / "adapters.safetensors",
        source=f"{SOURCE}.adapter_checkpoint",
    )
    copy_file(
        Path(str(plan["adaptive"]["adapter_config"]["path"])),
        adapter_input / "adapter_config.json",
        source=f"{SOURCE}.adapter_config",
    )
    return adapter_input


def _fuse_with_mlx(
    base_model: str,
    adapter_input: Path,
    staging_path: Path,
    *,
    owner_id: str,
    metadata: Mapping[str, Any],
) -> int:
    with standalone_model_lane(
        owner_id=owner_id,
        model_path=base_model,
        purpose="fuse",
        priority=100,
        preemptible=False,
        require_exclusive=True,
        allow_owner_eviction=True,
        metadata=metadata,
    ):
        model = None
        tokenizer = None
        config = None
        fused_linears = None
        try:
            from mlx.utils import tree_unflatten
            from mlx_lm.utils import load, save

            model, tokenizer, config = load(
                base_model,
                adapter_path=str(adapter_input),
                return_config=True,
            )
            fused_linears = [
                (name, module.fuse(dequantize=False))
                for name, module in model.named_modules()
                if hasattr(module, "fuse")
            ]
            if not fused_linears:
                raise CandidateCortexFusionError("fusion_applied_zero_modules")
            model.update_modules(tree_unflatten(fused_linears))
            save(
                staging_path,
                base_model,
                model,
                tokenizer,
                config,
                donate_model=False,
            )
            return len(fused_linears)
        finally:
            fused_linears = None
            model = None
            tokenizer = None
            config = None
            _release_model_memory()


def _fuse(plan: Mapping[str, Any]) -> Path:
    output = plan["output"]
    final_path = Path(str(output["path"]))
    staging_path = Path(str(output["staging_path"]))
    if final_path.exists() or final_path.is_symlink():
        if final_path.is_symlink() or not final_path.is_dir():
            raise CandidateCortexFusionError("fusion_output_invalid")
        validate_fusion_provenance(
            plan,
            _read_json(final_path / FUSION_PROVENANCE_FILE),
        )
        return final_path
    if available_bytes(Path(str(output["root"]))) < int(output["minimum_free_bytes"]):
        raise CandidateCortexFusionError("fusion_disk_budget_unavailable")
    adapter_input = _prepare_adapter_input(plan)
    if os.path.lexists(staging_path):
        with local_internal_governed_scope(
            f"{SOURCE}.staging_cleanup", domain="file_write"
        ):
            get_file_write_gateway().delete_path(
                staging_path,
                recursive=True,
                source=f"{SOURCE}.staging_cleanup",
            )

    base_model = str(plan["base_model"]["canonical_path"])
    fused_module_count = _fuse_with_mlx(
        base_model,
        adapter_input,
        staging_path,
        owner_id=f"candidate-cortex-fusion:{output['generation_id']}",
        metadata={
            "tool": "run_candidate_cortex_fusion_target",
            "fusion_plan_sha256": plan["fusion_plan_sha256"],
        },
    )

    provenance = build_fusion_provenance(
        plan,
        fused_module_count=fused_module_count,
    )
    _publish_exact(
        staging_path / FUSION_PROVENANCE_FILE,
        provenance,
        source=f"{SOURCE}.provenance",
    )
    validate_fusion_provenance(
        plan,
        _read_json(staging_path / FUSION_PROVENANCE_FILE),
    )
    return move_directory(
        staging_path,
        final_path,
        source=f"{SOURCE}.publish_model",
    )


def execute(plan_path: Path, journal_key_path: Path) -> dict[str, Any]:
    plan = load_and_validate_fusion_plan(
        plan_path,
        journal_key_path=journal_key_path,
        verify_full_model=True,
    )
    output = plan["output"]
    receipt_path = Path(str(output["receipt_path"]))
    if receipt_path.exists():
        receipt = validate_fusion_receipt(
            plan,
            _read_json(receipt_path),
            verify_full_model=True,
        )
        _remove_adapter_input(plan)
        return receipt

    final_path = _fuse(plan)
    provenance_path = final_path / FUSION_PROVENANCE_FILE
    validate_fusion_provenance(plan, _read_json(provenance_path))
    descriptor = build_model_artifact_descriptor(
        final_path,
        repository_id=str(output["repository_id"]),
        revision=str(output["revision"]),
    )
    descriptor_path = Path(str(output["descriptor_path"]))
    _publish_exact(
        descriptor_path,
        descriptor,
        source=f"{SOURCE}.descriptor",
    )
    body = {
        "schema": FUSION_RECEIPT_SCHEMA,
        "fusion_plan_sha256": plan["fusion_plan_sha256"],
        "generation_id": output["generation_id"],
        "model_path": str(final_path),
        "descriptor": {
            "path": str(descriptor_path.resolve(strict=True)),
            "sha256": file_sha256(descriptor_path),
            "size_bytes": descriptor_path.stat().st_size,
        },
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "provenance": {
            "path": str(provenance_path.resolve(strict=True)),
            "sha256": file_sha256(provenance_path),
            "size_bytes": provenance_path.stat().st_size,
        },
    }
    receipt = {**body, "receipt_sha256": document_sha256(body)}
    _publish_exact(receipt_path, receipt, source=f"{SOURCE}.receipt")
    validated = validate_fusion_receipt(plan, receipt, verify_full_model=False)
    _remove_adapter_input(plan)
    return validated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = execute(args.plan, args.journal_key)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
