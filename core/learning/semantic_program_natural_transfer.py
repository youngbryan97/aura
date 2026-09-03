"""Preflight a frozen semantic transducer against natural held-out schemas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.semantic_program_corpus import (
    SemanticProgramExample,
    build_semantic_program_corpus,
    build_semantic_program_fork_join_corpus,
    build_semantic_program_sequence_cataphoric_corpus,
    build_semantic_program_sequence_reserved_alias_corpus,
    build_semantic_program_sequence_role_binding_corpus,
)

NATURAL_TRANSFER_PREFLIGHT_SCHEMA: Final = "aura.semantic_program_natural_transfer_preflight.v1"

_FIT_FAMILY_BUILDERS: Final = {
    "arithmetic": build_semantic_program_corpus,
    "cataphoric": build_semantic_program_sequence_cataphoric_corpus,
    "fork_join": build_semantic_program_fork_join_corpus,
    "reserved_alias": build_semantic_program_sequence_reserved_alias_corpus,
    "role_binding": build_semantic_program_sequence_role_binding_corpus,
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _value_kind(value: Any) -> str:
    if type(value) is int:
        return "integer"
    if isinstance(value, tuple) and all(type(item) is int for item in value):
        return "integer_sequence"
    raise ValueError("natural transfer schema contains an unsupported value kind")


def procedure_schema_signature(example: SemanticProgramExample) -> str:
    """Identity of operation, dependency, and type schema, excluding wording/values."""

    body = {
        "input_types": [_value_kind(value) for value in example.inputs],
        "instructions": [
            {"op": item.instruction.op, "args": list(item.instruction.args)}
            for item in example.instructions
        ],
        "report_value": example.report_value,
    }
    return _sha(body)


def _fit_schema_inventory(families: Sequence[str]) -> set[str]:
    inventory: set[str] = set()
    for family in families:
        builder = _FIT_FAMILY_BUILDERS.get(family)
        if builder is None:
            raise ValueError(f"natural transfer cannot reconstruct fit family {family!r}")
        inventory.update(procedure_schema_signature(item) for item in builder())
    return inventory


def build_natural_request_transfer_preflight(
    *,
    examples: Sequence[SemanticProgramExample],
    transducer: Mapping[str, Any],
    source_campaign: Mapping[str, Any],
    frozen_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the cheap structural conditions before acquiring model features."""

    if not examples or any(not isinstance(item, SemanticProgramExample) for item in examples):
        raise ValueError("natural transfer preflight needs typed corpus examples")
    if any(item.split not in {"validation", "test"} for item in examples):
        raise ValueError("natural transfer corpus must remain evaluation-only")
    fit_families = tuple(source_campaign.get("fit_families", ()))
    if not fit_families or set(fit_families) != set(frozen_verification.get("fit_families", ())):
        raise ValueError("natural transfer fit-family evidence differs")
    receipt = transducer.get("training_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("natural transfer transducer has no training receipt")
    if (
        frozen_verification.get("verified") is not True
        or frozen_verification.get("serving_authority") is not False
        or frozen_verification.get("transducer_receipt_sha256") != receipt.get("receipt_sha256")
    ):
        raise ValueError("natural transfer transducer is not the frozen verified model")

    target_schemas = {procedure_schema_signature(item) for item in examples}
    source_schemas = _fit_schema_inventory(fit_families)
    overlap = target_schemas & source_schemas
    if overlap:
        raise ValueError("natural transfer target schema was present in fitting")
    target_operations = {
        item.instruction.op for example in examples for item in example.instructions
    }
    primitive_support = set(receipt.get("primitive_support", ()))
    unsupported = target_operations - primitive_support
    if unsupported:
        raise ValueError(f"natural transfer uses unsupported primitives: {sorted(unsupported)}")

    text_hashes = [
        hashlib.sha256(item.source_text.encode("utf-8")).hexdigest() for item in examples
    ]
    if len(text_hashes) != len(set(text_hashes)):
        raise ValueError("natural transfer corpus repeats source text")
    schemas_by_topology = {
        topology: sorted(
            {procedure_schema_signature(item) for item in examples if item.topology_id == topology}
        )
        for topology in sorted({item.topology_id for item in examples})
    }
    body = {
        "schema": NATURAL_TRANSFER_PREFLIGHT_SCHEMA,
        "transducer_receipt_sha256": receipt["receipt_sha256"],
        "source_campaign_report_sha256": source_campaign["report_sha256"],
        "source_verification_sha256": frozen_verification["verification_sha256"],
        "fit_families": list(fit_families),
        "example_count": len(examples),
        "splits": {
            split: sum(item.split == split for item in examples) for split in ("validation", "test")
        },
        "topologies": schemas_by_topology,
        "target_schema_count": len(target_schemas),
        "source_schema_count": len(source_schemas),
        "target_source_schema_overlap": 0,
        "target_operations": sorted(target_operations),
        "unsupported_operations": [],
        "source_text_sha256s": sorted(text_hashes),
        "fit_or_refit_calls": 0,
        "expected_answers_available_to_decode": False,
        "verifier_trace_available_to_decode": False,
        "serving_authority": False,
        "claim_boundary": (
            "evaluation-only natural-domain transfer into operation/dependency/type schemas "
            "absent from every frozen fit family; no open-domain or serving claim"
        ),
    }
    return {**body, "preflight_sha256": _sha(body)}


__all__ = [
    "NATURAL_TRANSFER_PREFLIGHT_SCHEMA",
    "build_natural_request_transfer_preflight",
    "procedure_schema_signature",
]
