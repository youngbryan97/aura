"""Qualification for a model-bound language-to-program shadow package."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_frozen_path_replication import (
    FROZEN_PATH_PREREGISTRATION_SCHEMA,
    FROZEN_PATH_RESULT_SCHEMA,
    adjudicate_frozen_path_replication,
)
from core.learning.semantic_program_ordinary_baseline import (
    ORDINARY_BASELINE_RESULT_SCHEMA,
    adjudicate_ordinary_product_bar,
    canonical_bytes,
    canonical_sha256,
    verify_embedded_receipt,
)
from core.learning.semantic_program_path_ensemble import (
    semantic_program_path_ensemble_from_dict,
)
from core.runtime.source_contract import source_contract_sha256s

COMPOSITIONAL_SEMANTIC_ACTIVATION_SCHEMA: Final = (
    "aura.compositional_semantic_activation.v1"
)
COMPOSITIONAL_SEMANTIC_PACKAGE_ID: Final = (
    "semantic-program-27b-natural-weave-a79674be5460"
)
COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS: Final = {
    "core/brain/llm/compositional_semantic_shadow.py": (
        "symbol:execute_compositional_semantic_shadow",
        "symbol:compositional_semantic_shadow_status",
    ),
    "core/learning/semantic_program_compositional_transducer.py": (
        "symbol:CompositionalSemanticProgramTransducer.decode",
        "symbol:CompositionalSemanticProgramTransducer.inference_step_limit",
        "symbol:compositional_semantic_program_transducer_from_dict",
    ),
    "core/learning/semantic_program_floor.py": (
        "symbol:compile_semantic_program_to_floor",
        "symbol:execute_semantic_floor_program",
    ),
    "core/learning/semantic_program_runtime.py": (
        "symbol:execute_compositional_semantic_observation",
    ),
    "core/learning/semantic_public_inputs.py": (
        "symbol:semantic_public_character_inputs",
        "symbol:semantic_public_token_inputs",
    ),
}
_EXPECTED_CLAIM = (
    "fresh bounded natural-language semantic-program transfer across the previously "
    "unseen six-input five-step branch-weave geometry by one frozen model-bound "
    "transducer; no selector gain, open-domain reasoning, frontier performance, or "
    "serving authority"
)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_receipt(value: Mapping[str, Any], field: str) -> str:
    verify_embedded_receipt(value, field=field)
    return str(value[field])


def _relative(root: Path, path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        raise ValueError("compositional semantic package file is outside the repository") from None


def _verify_mechanism(
    preregistration: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    *,
    transducer_receipt_sha256: str,
) -> dict[str, Any]:
    _logical_receipt(mechanism, "result_sha256")
    arms = mechanism.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("compositional semantic mechanism arms are missing")
    rows = {
        name: value.get("rows")
        for name, value in arms.items()
        if isinstance(value, Mapping)
    }
    expected = adjudicate_frozen_path_replication(
        rows,
        treatment_receipt_sha256=transducer_receipt_sha256,
        minimum_exact=int(mechanism.get("minimum_exact", -1)),
    )
    observed = {
        "treatment_answer_exact": mechanism.get("treatment_answer_exact"),
        "minimum_exact": mechanism.get("minimum_exact"),
        "paired_exact_tests": mechanism.get("paired_exact_tests"),
        "causal_treatment_receipt_contract_satisfied": mechanism.get(
            "causal_treatment_receipt_contract_satisfied"
        ),
        "mechanism_pass": mechanism.get("verdict")
        == "PASS_MECHANISM_READY_FOR_ORDINARY_BASELINE",
    }
    if (
        preregistration.get("schema") != FROZEN_PATH_PREREGISTRATION_SCHEMA
        or mechanism.get("schema") != FROZEN_PATH_RESULT_SCHEMA
        or mechanism.get("claim_boundary") != _EXPECTED_CLAIM
        or preregistration.get("claim_boundary") != _EXPECTED_CLAIM
        or mechanism.get("serving_authority") is not False
        or mechanism.get("treatment_answer_exact") != 21
        or expected != observed
        or expected["mechanism_pass"] is not True
    ):
        raise ValueError("compositional semantic mechanism evidence differs")
    return dict(arms["frozen_transducer"])


def _verify_ordinary(
    value: Mapping[str, Any],
    *,
    treatment_rows: list[dict[str, Any]],
    mechanism_result_sha256: str,
    descriptor_sha256: str,
) -> dict[str, Any]:
    _logical_receipt(value, "result_sha256")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("ordinary semantic rows are missing")
    adjudication = adjudicate_ordinary_product_bar(treatment_rows, rows)
    if (
        value.get("schema") != ORDINARY_BASELINE_RESULT_SCHEMA
        or value.get("claim_boundary") != _EXPECTED_CLAIM
        or value.get("mechanism_result_sha256") != mechanism_result_sha256
        or value.get("model_descriptor_sha256") != descriptor_sha256
        or value.get("complete") is not True
        or value.get("completed_tasks") != 48
        or value.get("verdict") != "PASS_PREREGISTERED_PRODUCT_BAR"
        or value.get("serving_authority") is not False
        or value.get("adjudication") != adjudication
        or adjudication["product_bar_pass"] is not True
        or adjudication["paired_exact_test"]["control_only"] != 0
    ):
        raise ValueError("ordinary semantic control evidence differs")
    return adjudication


def build_compositional_semantic_activation(
    *,
    repo_root: Path,
    preregistration: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    ordinary_primary: Mapping[str, Any],
    ordinary_sensitivity: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    ensemble: Mapping[str, Any],
    transducer_path: Path,
    preregistration_path: Path,
    mechanism_path: Path,
    ordinary_primary_path: Path,
    ordinary_sensitivity_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a shadow activation only after recomputing every measured decision."""

    root = repo_root.expanduser().resolve(strict=True)
    verify_embedded_receipt(descriptor, field="descriptor_sha256")
    frozen = preregistration.get("frozen_transducer")
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen transducer preregistration is missing")
    parsed_ensemble = semantic_program_path_ensemble_from_dict(ensemble)
    transducer = parsed_ensemble.challenger
    transducer_document = transducer.to_dict()
    reloaded = compositional_semantic_program_transducer_from_dict(transducer_document)
    if (
        reloaded.receipt_sha256 != transducer.receipt_sha256
        or transducer.receipt_sha256 != frozen.get("receipt_sha256")
        or transducer.training_receipt.get("coefficient_sha256")
        != frozen.get("coefficient_sha256")
        or transducer.model_basis_sha256 != frozen.get("model_basis_sha256")
    ):
        raise ValueError("frozen compositional semantic transducer differs")
    treatment = _verify_mechanism(
        preregistration,
        mechanism,
        transducer_receipt_sha256=transducer.receipt_sha256,
    )
    treatment_rows = treatment.get("rows")
    if not isinstance(treatment_rows, list):
        raise ValueError("compositional semantic treatment rows are missing")
    descriptor_sha256 = str(descriptor["descriptor_sha256"])
    primary = _verify_ordinary(
        ordinary_primary,
        treatment_rows=treatment_rows,
        mechanism_result_sha256=str(mechanism["result_sha256"]),
        descriptor_sha256=descriptor_sha256,
    )
    sensitivity = _verify_ordinary(
        ordinary_sensitivity,
        treatment_rows=treatment_rows,
        mechanism_result_sha256=str(mechanism["result_sha256"]),
        descriptor_sha256=descriptor_sha256,
    )
    primary_budget = ordinary_primary.get("decode_policy", {}).get("max_tokens")
    sensitivity_budget = ordinary_sensitivity.get("decode_policy", {}).get("max_tokens")
    representation = mechanism.get("preflight", {}).get("representation_compatibility")
    if (
        primary_budget != 768
        or sensitivity_budget != 2048
        or ordinary_primary.get("model_path") != ordinary_sensitivity.get("model_path")
        or ordinary_primary.get("preregistration_sha256")
        != canonical_sha256(preregistration)
        or ordinary_sensitivity.get("preregistration_sha256")
        != canonical_sha256(preregistration)
        or not isinstance(representation, Mapping)
        or representation.get("transducer_receipt_sha256")
        != transducer.receipt_sha256
        or representation.get("hidden_states_changed") is not False
    ):
        raise ValueError("compositional semantic robustness evidence differs")

    documents = {
        "preregistration": (preregistration_path, preregistration, None),
        "mechanism": (mechanism_path, mechanism, "result_sha256"),
        "ordinary_primary": (ordinary_primary_path, ordinary_primary, "result_sha256"),
        "ordinary_sensitivity": (
            ordinary_sensitivity_path,
            ordinary_sensitivity,
            "result_sha256",
        ),
    }
    evidence = {}
    for name, (path, document, receipt_field) in documents.items():
        evidence[name] = {
            "path": _relative(root, path),
            "file_sha256": _file_sha(path),
            "logical_sha256": (
                str(document[receipt_field])
                if receipt_field is not None
                else canonical_sha256(document)
            ),
        }
    transducer_raw = transducer_path.read_bytes()
    if json.loads(transducer_raw) != transducer_document:
        raise ValueError("materialized compositional semantic transducer differs")
    body = {
        "schema": COMPOSITIONAL_SEMANTIC_ACTIVATION_SCHEMA,
        "package_id": COMPOSITIONAL_SEMANTIC_PACKAGE_ID,
        "mode": "shadow",
        "active_by_default": True,
        "serving_authority": False,
        "transducer": {
            "path": _relative(root, transducer_path),
            "file_sha256": hashlib.sha256(transducer_raw).hexdigest(),
            "receipt_sha256": transducer.receipt_sha256,
            "coefficient_sha256": transducer.training_receipt["coefficient_sha256"],
            "model_basis_sha256": transducer.model_basis_sha256,
            "training_max_inputs": transducer.max_inputs,
            "training_max_steps": transducer.max_steps,
        },
        "model": {
            "path": str(ordinary_primary["model_path"]),
            "descriptor_sha256": descriptor_sha256,
            "tokenizer_identity_sha256": (
                transducer.input_grounding.tokenizer_identity_sha256
            ),
            "representation_basis_sha256": representation[
                "representation_basis_sha256"
            ],
        },
        "evidence": evidence,
        "measured": {
            "tasks": 48,
            "treatment_answer_exact": 21,
            "coefficient_lesion_answer_exact": int(
                mechanism["arms"]["coefficient_lesion"]["answer_exact"]
            ),
            "hidden_token_shuffle_answer_exact": int(
                mechanism["arms"]["hidden_token_shuffle"]["answer_exact"]
            ),
            "ordinary_primary_answer_exact": primary["ordinary_answer_exact"],
            "ordinary_primary_p": primary["paired_exact_test"]["one_sided_exact_p"],
            "ordinary_sensitivity_answer_exact": sensitivity["ordinary_answer_exact"],
            "ordinary_sensitivity_p": sensitivity["paired_exact_test"][
                "one_sided_exact_p"
            ],
            "ordinary_sensitivity_max_tokens": sensitivity_budget,
        },
        "source_contract_sha256s": source_contract_sha256s(
            root,
            COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS,
        ),
        "composition_policy": {
            "ordinary_response_is_immutable_incumbent": True,
            "shadow_result_is_observation_only": True,
            "replacement_requires_independent_objective_verification": True,
        },
        "claim_boundary": _EXPECTED_CLAIM,
    }
    return transducer_document, {**body, "activation_sha256": canonical_sha256(body)}


def compositional_semantic_activation_errors(
    activation: Mapping[str, Any],
    *,
    repo_root: Path,
    selected_model_path: Path | None = None,
) -> list[str]:
    """Reopen every mutable dependency of one compositional shadow activation."""

    errors: list[str] = []
    root = repo_root.expanduser().resolve(strict=True)
    body = dict(activation)
    observed_sha = body.pop("activation_sha256", None)
    if activation.get("schema") != COMPOSITIONAL_SEMANTIC_ACTIVATION_SCHEMA:
        errors.append("schema")
    if observed_sha != canonical_sha256(body):
        errors.append("activation_sha256")
    if activation.get("package_id") != COMPOSITIONAL_SEMANTIC_PACKAGE_ID:
        errors.append("package_id")
    if (
        activation.get("mode") != "shadow"
        or activation.get("active_by_default") is not True
        or activation.get("serving_authority") is not False
    ):
        errors.append("authority")
    policy = activation.get("composition_policy")
    if not isinstance(policy, Mapping) or policy != {
        "ordinary_response_is_immutable_incumbent": True,
        "shadow_result_is_observation_only": True,
        "replacement_requires_independent_objective_verification": True,
    }:
        errors.append("composition_policy")
    contracts = activation.get("source_contract_sha256s")
    try:
        current_contracts = source_contract_sha256s(
            root,
            COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS,
        )
        if contracts != current_contracts:
            errors.append("source_contract_drift")
    except (OSError, RuntimeError, SyntaxError, ValueError):
        errors.append("source_contract_unavailable")
    evidence = activation.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append("evidence")
        evidence = {}
    for name, record in evidence.items():
        try:
            if not isinstance(name, str) or not isinstance(record, Mapping):
                raise TypeError("evidence record is invalid")
            path = (root / str(record["path"])).resolve(strict=True)
            path.relative_to(root)
            if _file_sha(path) != record.get("file_sha256"):
                errors.append(f"evidence_drift:{name}")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append(f"evidence_invalid:{name}")
    transducer = activation.get("transducer")
    try:
        if not isinstance(transducer, Mapping):
            raise ValueError("transducer record is missing")
        transducer_path = (root / str(transducer["path"])).resolve(strict=True)
        transducer_path.relative_to(root)
        if _file_sha(transducer_path) != transducer.get("file_sha256"):
            errors.append("transducer_drift")
        loaded = compositional_semantic_program_transducer_from_dict(
            json.loads(transducer_path.read_text(encoding="ascii"))
        )
        if (
            loaded.receipt_sha256 != transducer.get("receipt_sha256")
            or loaded.training_receipt.get("coefficient_sha256")
            != transducer.get("coefficient_sha256")
            or loaded.model_basis_sha256 != transducer.get("model_basis_sha256")
        ):
            errors.append("transducer_identity")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        errors.append("transducer_invalid")
    model = activation.get("model")
    if not isinstance(model, Mapping):
        errors.append("model")
    elif selected_model_path is not None:
        try:
            selected = selected_model_path.expanduser().resolve(strict=True)
            expected = Path(str(model["path"])).expanduser().resolve(strict=True)
            if selected != expected:
                errors.append("active_model_mismatch")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            errors.append("active_model_invalid")
    return sorted(set(errors))


def canonical_document_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize one generated package document deterministically."""

    return canonical_bytes(value) + b"\n"


__all__ = [
    "COMPOSITIONAL_SEMANTIC_ACTIVATION_SCHEMA",
    "COMPOSITIONAL_SEMANTIC_PACKAGE_ID",
    "COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS",
    "build_compositional_semantic_activation",
    "canonical_document_bytes",
    "compositional_semantic_activation_errors",
]
