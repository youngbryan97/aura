"""Deterministic signed migration-authority fixtures for contract tests."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from core.brain.llm.unified_recurrent_qualified_activation import ACTIVATION_SCHEMA
from core.learning.cortex_migration_authority import (
    CAA_EVALUATION_SCHEMA,
    COMPONENT_AUTHORITY_SCHEMA,
    RECURRENT_MODEL_BINDING_SCHEMA,
)
from core.learning.model_tissue_migration_inventory import FAMILIES, INVENTORY_SCHEMA


_SPECS = {
    "persona_crsm": ("fused_persona_crsm", "qualified"),
    "steering": ("caa_model_bound", "qualified"),
    "expert_adapters": ("retirement_inventory", "retired"),
    "recurrence_native": ("qualified_recurrent_activation", "qualified"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write_json(root: Path, name: str, value: dict[str, Any]) -> tuple[Path, bytes]:
    payload = _canonical(value)
    path = root / f"{name}.json"
    path.write_bytes(payload)
    os.chmod(path, 0o600)
    return path, payload


def _activation() -> dict[str, Any]:
    body = {
        "schema": ACTIVATION_SCHEMA,
        "package_id": "test-qualified-package",
        "manifest_sha256": _sha("manifest"),
        "checkpoint_sha256": _sha("checkpoint"),
        "controller_sha256": _sha("controller"),
        "pointer_sha256": _sha("pointer"),
        "lifecycle_result_sha256": _sha("lifecycle"),
        "canary_plan_sha256": _sha("canary-plan"),
        "candidate_canary_sha256": _sha("candidate-canary"),
        "qualified_canary_sha256": _sha("qualified-canary"),
        "families": ["register_trace"],
        "task_depths": [1, 2],
        "recurrence_depth": 2,
        "mode": "qualified_typed_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": True,
    }
    return {**body, "activation_sha256": _sha(body)}


def build_signed_migration_authorities(
    root: Path,
    *,
    descriptor_sha256: str,
    state_root: Path,
) -> dict[str, dict[str, Any]]:
    """Build closed-schema authorities under an isolated owner-private root."""

    custody = root / "retained-migration-evidence"
    custody.mkdir(parents=True, mode=0o700)
    os.chmod(custody, 0o700)

    key_path = state_root / "private/cortex-upgrade/migration-authority.key"
    key_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(key_path.parent, 0o700)
    if key_path.exists():
        key = key_path.read_bytes()
    else:
        key = hashlib.sha256(str(root).encode("utf-8")).digest()
        key_path.write_bytes(key)
        os.chmod(key_path, 0o600)

    plan_body = {"schema": "aura.candidate_cortex_fusion.plan.v1"}
    plan = {**plan_body, "fusion_plan_sha256": _sha(plan_body)}
    receipt_body = {
        "schema": "aura.candidate_cortex_fusion.receipt.v1",
        "fusion_plan_sha256": plan["fusion_plan_sha256"],
        "descriptor_sha256": descriptor_sha256,
    }
    receipt = {**receipt_body, "receipt_sha256": _sha(receipt_body)}
    plan_path, plan_bytes = _write_json(custody, "fusion-plan", plan)
    receipt_path, receipt_bytes = _write_json(custody, "fusion-receipt", receipt)

    extraction = {"extraction_contract_sha256": _sha("extraction-contract")}
    vector_bytes = b"test-vector-basis"
    vector_path = custody / "caa-vector.safetensors"
    vector_path.write_bytes(vector_bytes)
    os.chmod(vector_path, 0o600)
    vector_manifest = [
        {
            "name": vector_path.name,
            "size_bytes": len(vector_bytes),
            "sha256": hashlib.sha256(vector_bytes).hexdigest(),
        }
    ]
    generation_sha256 = _sha(
        {
            "extraction_contract_sha256": extraction["extraction_contract_sha256"],
            "vector_files": vector_manifest,
        }
    )
    steering_metadata = {
        "model_identity": {"model_descriptor_sha256": descriptor_sha256},
        "extraction_contract": extraction,
        "vector_files": vector_manifest,
        "generation_sha256": generation_sha256,
    }
    steering_path, steering_bytes = _write_json(
        custody, "steering-metadata", steering_metadata
    )
    evaluation_body = {
        "schema": CAA_EVALUATION_SCHEMA,
        "model_descriptor_sha256": descriptor_sha256,
        "generation_sha256": generation_sha256,
        "verdict": "PASS",
        "qualified": True,
        "sample_count": 60,
        "treatment_successes": 50,
        "matched_control_successes": 20,
        "lesion_successes": {"coefficient": 8, "wrong_state": 0},
        "no_regression": True,
        "causal_effect_positive": True,
        "independent_verifier": {
            "name": "test-independent-verifier",
            "version": "1",
            "evidence_sha256": _sha("independent-evidence"),
        },
    }
    evaluation = {**evaluation_body, "evaluation_sha256": _sha(evaluation_body)}
    evaluation_path, evaluation_bytes = _write_json(
        custody, "steering-causal-evaluation", evaluation
    )

    families = [
        {
            "family": family,
            "outcome": "retire" if family == "expert_adapters" else "qualified",
        }
        for family in FAMILIES
    ]
    inventory_body = {
        "schema": INVENTORY_SCHEMA,
        "generated_at": 1.0,
        "candidate": {"descriptor_sha256": descriptor_sha256},
        "limits": {},
        "entries": [],
        "families": families,
        "promotion_ready": True,
    }
    inventory = {**inventory_body, "inventory_sha256": _sha(inventory_body)}
    inventory_path, inventory_bytes = _write_json(custody, "inventory", inventory)

    activation = _activation()
    activation_path, activation_bytes = _write_json(custody, "activation", activation)
    binding_body = {
        "schema": RECURRENT_MODEL_BINDING_SCHEMA,
        "model_descriptor_sha256": descriptor_sha256,
        "package_id": activation["package_id"],
        "manifest_sha256": activation["manifest_sha256"],
        "activation_sha256": activation["activation_sha256"],
    }
    binding = {**binding_body, "binding_sha256": _sha(binding_body)}
    binding_path, binding_bytes = _write_json(custody, "model-binding", binding)

    component_evidence = {
        "persona_crsm": {
            "fusion_plan": (plan_path, plan_bytes),
            "fusion_receipt": (receipt_path, receipt_bytes),
        },
        "steering": {
            "metadata": (steering_path, steering_bytes),
            "causal_evaluation": (evaluation_path, evaluation_bytes),
            f"vector:{vector_path.name}": (vector_path, vector_bytes),
        },
        "expert_adapters": {
            "migration_inventory": (inventory_path, inventory_bytes),
        },
        "recurrence_native": {
            "activation": (activation_path, activation_bytes),
            "model_binding": (binding_path, binding_bytes),
        },
    }
    claims = {
        "persona_crsm": {
            "fusion_plan_sha256": plan["fusion_plan_sha256"],
            "fusion_receipt_sha256": receipt["receipt_sha256"],
        },
        "steering": {
            "generation_sha256": generation_sha256,
            "extraction_contract_sha256": extraction["extraction_contract_sha256"],
            "causal_evaluation_sha256": evaluation["evaluation_sha256"],
        },
        "expert_adapters": {"inventory_sha256": inventory["inventory_sha256"]},
        "recurrence_native": {
            "activation_sha256": activation["activation_sha256"],
            "package_id": activation["package_id"],
            "manifest_sha256": activation["manifest_sha256"],
            "model_binding_sha256": binding["binding_sha256"],
        },
    }

    authorities: dict[str, dict[str, Any]] = {}
    for component, evidence in component_evidence.items():
        kind, status = _SPECS[component]
        body = {
            "schema": COMPONENT_AUTHORITY_SCHEMA,
            "component": component,
            "authority_kind": kind,
            "status": status,
            "model_descriptor_sha256": descriptor_sha256,
            "custody_root": str(custody.resolve()),
            "evidence": {
                role: {
                    "path": str(path.resolve()),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for role, (path, payload) in evidence.items()
            },
            "claims": claims[component],
            "issued_at": 1.0,
        }
        digest = _sha(body)
        authorities[component] = {
            **body,
            "authority_sha256": digest,
            "authority_hmac_sha256": hmac.new(
                key, bytes.fromhex(digest), hashlib.sha256
            ).hexdigest(),
        }
    return authorities


__all__ = ["build_signed_migration_authorities"]
