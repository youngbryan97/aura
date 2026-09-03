"""Independent recount of frozen compositional semantic replications."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.brain.llm.latent_cortex.runtime_identity import worker_representation_basis
from core.learning.semantic_program_compositional_campaign import (
    COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA,
)
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)

COMPOSITIONAL_REPLICATION_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_compositional_replication_verification.v1"
)
COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES: Final = (
    "core/learning/semantic_input_grounding.py",
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_compositional_campaign.py",
    "core/learning/semantic_program_compositional_transducer.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_shared_evaluation.py",
    "core/learning/semantic_program_shared_transducer.py",
    "core/learning/semantic_program_compositional_verification.py",
    "tools/verify_compositional_family_withheld_replication.py",
    "tools/verify_compositional_semantic_replication.py",
)
_REPORT_SCHEMA: Final = "aura.semantic_program_compositional_lesions.v1"
_SPLITS: Final = ("validation", "test")
_COUNT_FIELDS: Final = (
    "accepted",
    "program_exact",
    "operation_exact",
    "argument_exact",
    "arity_exact",
    "step_count_exact",
    "geometry_exact",
    "input_span_exact",
    "answer_emitted",
    "answer_exact",
)
_METRICS: Final = ("program_exact", "answer_exact")
_TRAINING_COMPATIBILITY_SCHEMA: Final = (
    "aura.semantic_training_representation_compatibility.v1"
)
_REPLICATION_COMPATIBILITY_SCHEMA: Final = (
    "aura.semantic_representation_compatibility.v1"
)


@dataclass(frozen=True, slots=True)
class SemanticCohortInventory:
    """Identity-only view of a strictly loaded feature bundle."""

    manifest_sha256: str
    exact_model_path: str
    tokenizer_identity_sha256: str
    example_count: int
    example_ids: tuple[str, ...]
    source_text_sha256s: tuple[str, ...]
    split_source_text_sha256s: tuple[tuple[str, tuple[str, ...]], ...]
    held_source_text_sha256s: tuple[str, ...]
    session_basis_sha256s: tuple[str, ...]
    representation_basis_sha256: str
    worker_stack_identity_gaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositionalReplicationCohort:
    """One source/fresh cohort and its frozen causal lesion."""

    family: str
    source: SemanticCohortInventory
    fresh: SemanticCohortInventory
    lesion_arm: str
    report: Mapping[str, Any]
    transfer_kind: str
    evaluation_source_commit: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def semantic_cohort_inventory(
    bundle: LoadedSemanticFeatureBundle,
) -> SemanticCohortInventory:
    """Discard hidden arrays after retaining identities needed for verification."""

    manifest = bundle.manifest
    model_bases = manifest.get("model_bases")
    if (
        manifest.get("complete") is not True
        or not _is_sha256(manifest.get("manifest_sha256"))
        or not isinstance(model_bases, list)
        or len(model_bases) != 1
        or not isinstance(model_bases[0], Mapping)
        or not isinstance(model_bases[0].get("receipt"), Mapping)
    ):
        raise ValueError("compositional feature bundle identity is incomplete")
    receipt = model_bases[0]["receipt"]
    tokenizer = manifest.get("tokenizer_identity")
    gaps = receipt.get("worker_stack_identity_gaps")
    if (
        not isinstance(tokenizer, Mapping)
        or not _is_sha256(tokenizer.get("identity_sha256"))
        or not isinstance(gaps, list)
        or any(not isinstance(value, str) for value in gaps)
        or manifest.get("exact_model_path") != receipt.get("worker_model_path")
    ):
        raise ValueError("compositional feature bundle model identity is invalid")
    ids: list[str] = []
    source_hashes: list[str] = []
    held_hashes: list[str] = []
    hashes_by_split: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for item in bundle.examples:
        example_id = item.metadata.get("example_id")
        source_hash = item.metadata.get("source_text_sha256")
        split = item.metadata.get("split")
        if not isinstance(example_id, str) or not _is_sha256(source_hash):
            raise ValueError("compositional feature example identity is invalid")
        ids.append(example_id)
        source_hashes.append(source_hash)
        if split not in hashes_by_split:
            raise ValueError("compositional feature example split is invalid")
        hashes_by_split[split].append(source_hash)
        if split in _SPLITS:
            held_hashes.append(source_hash)
    if (
        len(ids) != len(set(ids))
        or len(source_hashes) != len(set(source_hashes))
        or len(ids) != manifest.get("example_count")
    ):
        raise ValueError("compositional feature example inventory differs")
    return SemanticCohortInventory(
        manifest_sha256=manifest["manifest_sha256"],
        exact_model_path=str(manifest["exact_model_path"]),
        tokenizer_identity_sha256=str(tokenizer["identity_sha256"]),
        example_count=len(ids),
        example_ids=tuple(sorted(ids)),
        source_text_sha256s=tuple(sorted(source_hashes)),
        split_source_text_sha256s=tuple(
            (split, tuple(sorted(hashes_by_split[split])))
            for split in ("train", "validation", "test")
        ),
        held_source_text_sha256s=tuple(sorted(held_hashes)),
        session_basis_sha256s=tuple(sorted(str(item["sha256"]) for item in model_bases)),
        representation_basis_sha256=_sha(worker_representation_basis(receipt)),
        worker_stack_identity_gaps=tuple(gaps),
    )


def _recount_arm(arm: Mapping[str, Any]) -> dict[str, int]:
    rows = arm.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("compositional replication arm rows are invalid")
    identities: set[str] = set()
    counts = {field: 0 for field in _COUNT_FIELDS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("compositional replication task row is invalid")
        identity = row.get("source_text_sha256")
        if not _is_sha256(identity) or identity in identities:
            raise ValueError("compositional replication task identity is invalid")
        identities.add(identity)
        for field in _COUNT_FIELDS:
            value = row.get(field)
            if type(value) is not bool:
                raise ValueError(
                    f"compositional replication row field is invalid: {field}"
                )
            counts[field] += int(value)
    observed = {"total": len(rows), **counts}
    if any(arm.get(field) != value for field, value in observed.items()):
        raise ValueError("compositional replication arm aggregate differs from rows")
    return observed


def _pooled_rows(report: Mapping[str, Any], arm: str) -> list[Mapping[str, Any]]:
    return [row for split in _SPLITS for row in report["arms"][arm][split]["rows"]]


def _paired_exact(
    treatment_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    treatment = {
        str(row["source_text_sha256"]): row.get(metric) is True
        for row in treatment_rows
    }
    control = {
        str(row["source_text_sha256"]): row.get(metric) is True
        for row in control_rows
    }
    if treatment.keys() != control.keys():
        raise ValueError("compositional replication paired task sets differ")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    discordant = treatment_only + control_only
    numerator = (
        sum(
            math.comb(discordant, successes)
            for successes in range(treatment_only, discordant + 1)
        )
        if discordant
        else 1
    )
    denominator = 2**discordant if discordant else 1
    divisor = math.gcd(numerator, denominator)
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_exact_p_numerator": numerator // divisor,
        "one_sided_exact_p_denominator": denominator // divisor,
        "one_sided_exact_p": numerator / denominator,
    }


def _validate_report(
    cohort: CompositionalReplicationCohort,
    *,
    transducer_receipt_sha256: str,
) -> dict[str, Any]:
    report = cohort.report
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    arms = report.get("arms")
    if (
        report.get("schema") != _REPORT_SCHEMA
        or report.get("report_sha256") != _sha(body)
        or report.get("transducer_receipt_sha256") != transducer_receipt_sha256
        or report.get("fit_or_refit_calls") != 0
        or report.get("expected_answers_available_to_decode") is not False
        or report.get("serving_authority") is not False
        or not isinstance(arms, Mapping)
        or set(arms) < {"treatment", cohort.lesion_arm}
    ):
        raise ValueError(f"compositional replication report differs: {cohort.family}")
    evaluated = report.get("evaluated_arms")
    if evaluated is not None and set(evaluated) != set(arms):
        raise ValueError("compositional replication evaluated arm inventory differs")
    held_ids = set(cohort.fresh.held_source_text_sha256s)
    recounted: dict[str, dict[str, dict[str, int]]] = {}
    for arm_name in ("treatment", cohort.lesion_arm):
        recounted[arm_name] = {}
        for split in _SPLITS:
            split_arm = arms[arm_name][split]
            counts = _recount_arm(split_arm)
            row_ids = {row["source_text_sha256"] for row in split_arm["rows"]}
            if not row_ids <= held_ids:
                raise ValueError("compositional replication rows leave the fresh cohort")
            recounted[arm_name][split] = counts
    treatment_rows = _pooled_rows(report, "treatment")
    lesion_rows = _pooled_rows(report, cohort.lesion_arm)
    observed_row_ids = {row["source_text_sha256"] for row in treatment_rows}
    if observed_row_ids != held_ids or len(treatment_rows) != len(held_ids):
        raise ValueError("compositional replication held-out inventory differs")
    paired = {
        metric: _paired_exact(treatment_rows, lesion_rows, metric=metric)
        for metric in _METRICS
    }
    pooled_counts = {
        arm_name: {
            field: sum(recounted[arm_name][split][field] for split in _SPLITS)
            for field in ("total", *_COUNT_FIELDS)
        }
        for arm_name in ("treatment", cohort.lesion_arm)
    }
    if any(
        paired[metric]["treatment_only"] <= paired[metric]["control_only"]
        or paired[metric]["one_sided_exact_p"] >= 0.05
        for metric in _METRICS
    ):
        raise ValueError(f"compositional causal lesion did not replicate: {cohort.family}")
    return {
        "family": cohort.family,
        "transfer_kind": cohort.transfer_kind,
        "evaluation_source_commit": cohort.evaluation_source_commit,
        "source_feature_manifest_sha256": cohort.source.manifest_sha256,
        "fresh_feature_manifest_sha256": cohort.fresh.manifest_sha256,
        "source_example_count": cohort.source.example_count,
        "fresh_example_count": cohort.fresh.example_count,
        "source_fresh_example_overlap": 0,
        "held_out_total": len(held_ids),
        "lesion_arm": cohort.lesion_arm,
        "pooled_counts": pooled_counts,
        "paired_exact_tests": paired,
        "replication_report_sha256": report["report_sha256"],
        "fit_or_refit_calls": 0,
    }


def _validated_receipt(value: Any, *, schema: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("compositional representation receipt is absent")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("schema") != schema or value.get("receipt_sha256") != _sha(body):
        raise ValueError("compositional representation receipt identity differs")
    return value


def _source_family_recount(
    report: Mapping[str, Any],
    *,
    inventory: SemanticCohortInventory,
) -> dict[str, Any]:
    split_ids = dict(inventory.split_source_text_sha256s)
    recounted: dict[str, dict[str, int]] = {}
    for split in ("train", *_SPLITS):
        arm = report.get("splits", {}).get(split)
        if not isinstance(arm, Mapping):
            raise ValueError("held-out source family split is absent")
        recounted[split] = _recount_arm(arm)
        row_ids = {row["source_text_sha256"] for row in arm["rows"]}
        if row_ids != set(split_ids[split]):
            raise ValueError("held-out source family row inventory differs")
    held_counts = {
        field: sum(recounted[split][field] for split in _SPLITS)
        for field in ("total", *_COUNT_FIELDS)
    }
    expected = {
        "example_count": inventory.example_count,
        "held_out_program_exact": held_counts["program_exact"],
        "held_out_argument_exact": held_counts["argument_exact"],
        "held_out_answer_exact": held_counts["answer_exact"],
        "held_out_total": held_counts["total"],
    }
    if any(report.get(field) != value for field, value in expected.items()):
        raise ValueError("held-out source family aggregate differs from rows")
    return {
        "splits": recounted,
        "held_out": held_counts,
    }


def verify_compositional_family_withheld_replication(
    *,
    source_inventories: Mapping[str, SemanticCohortInventory],
    fresh_inventory: SemanticCohortInventory,
    trained_model_payload: Mapping[str, Any],
    source_report: Mapping[str, Any],
    fresh_report: Mapping[str, Any],
    lesion_arm: str,
    evaluation_source_commit: str,
    source_sha256s: Mapping[str, str],
    stored_file_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Verify that a frozen model transfers into a family absent from fitting."""

    if (
        set(source_sha256s) != set(COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
        or not stored_file_sha256s
        or any(not _is_sha256(value) for value in stored_file_sha256s.values())
        or not _is_commit(evaluation_source_commit)
    ):
        raise ValueError("family-withheld verification source identity is invalid")
    held_out_family = source_report.get("held_out_family")
    if (
        not isinstance(held_out_family, str)
        or held_out_family not in source_inventories
        or len(source_inventories) < 3
    ):
        raise ValueError("family-withheld source inventory is invalid")
    fit_families = sorted(set(source_inventories) - {held_out_family})
    source_manifest_sha256s = {
        family: inventory.manifest_sha256
        for family, inventory in sorted(source_inventories.items())
    }
    source_body = {
        key: value for key, value in source_report.items() if key != "report_sha256"
    }
    if (
        source_report.get("schema") != COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA
        or source_report.get("report_sha256") != _sha(source_body)
        or source_report.get("fit_families") != fit_families
        or source_report.get("evaluated_families") != [held_out_family]
        or source_report.get("feature_manifest_sha256s") != source_manifest_sha256s
        or source_report.get("held_out_family_was_available_to_fit") is not False
        or source_report.get("fit_example_count")
        != sum(source_inventories[family].example_count for family in fit_families)
        or source_report.get("expected_answers_available_to_training") is not False
        or source_report.get("verifier_traces_available") is not False
        or source_report.get("generated_compiler_text_available") is not False
        or source_report.get("serving_authority") is not False
    ):
        raise ValueError("family-withheld source campaign identity differs")
    model = compositional_semantic_program_transducer_from_dict(trained_model_payload)
    if (
        model.to_dict() != dict(trained_model_payload)
        or source_report.get("transducer_receipt_sha256") != model.receipt_sha256
        or source_report.get("model_basis_sha256") != model.model_basis_sha256
        or model.training_receipt.get("correctness_authority") is not False
        or model.training_receipt.get("expected_answers_available") is not False
        or model.training_receipt.get("verifier_traces_available") is not False
        or model.training_receipt.get("generated_compiler_text_available") is not False
        or model.training_receipt.get("family_router_present") is not False
    ):
        raise ValueError("family-withheld frozen model binding differs")

    fit_compatibility = _validated_receipt(
        source_report.get("representation_compatibility"),
        schema=_TRAINING_COMPATIBILITY_SCHEMA,
    )
    expected_fit_manifests = {
        family: source_inventories[family].manifest_sha256 for family in fit_families
    }
    expected_fit_sessions = {
        family: list(source_inventories[family].session_basis_sha256s)
        for family in fit_families
    }
    expected_training_basis = min(
        session
        for sessions in expected_fit_sessions.values()
        for session in sessions
    )
    representation_ids = {
        inventory.representation_basis_sha256
        for inventory in (*source_inventories.values(), fresh_inventory)
    }
    if (
        fit_compatibility.get("source_feature_manifest_sha256s")
        != expected_fit_manifests
        or fit_compatibility.get("source_session_basis_sha256s")
        != expected_fit_sessions
        or fit_compatibility.get("target_training_session_basis_sha256")
        != model.model_basis_sha256
        or model.model_basis_sha256 != expected_training_basis
        or fit_compatibility.get("hidden_states_changed") is not False
        or fit_compatibility.get("serving_authority") is not False
        or len(representation_ids) != 1
        or fit_compatibility.get("representation_basis_sha256")
        != next(iter(representation_ids))
    ):
        raise ValueError("family-withheld fit basis includes incompatible evidence")

    held_out_source = source_inventories[held_out_family]
    source_compatibility = _validated_receipt(
        source_report.get("held_out_representation_compatibility"),
        schema=_REPLICATION_COMPATIBILITY_SCHEMA,
    )
    anchor_manifest = source_compatibility.get("training_feature_manifest_sha256")
    anchor_families = [
        family
        for family, manifest_sha256 in expected_fit_manifests.items()
        if manifest_sha256 == anchor_manifest
    ]
    if (
        len(anchor_families) != 1
        or model.model_basis_sha256
        not in source_inventories[anchor_families[0]].session_basis_sha256s
        or source_compatibility.get("transducer_receipt_sha256")
        != model.receipt_sha256
        or source_compatibility.get("coefficient_sha256")
        != model.training_receipt.get("coefficient_sha256")
        or source_compatibility.get("replication_feature_manifest_sha256")
        != held_out_source.manifest_sha256
        or source_compatibility.get("training_session_basis_sha256")
        != model.model_basis_sha256
        or source_compatibility.get("replication_session_basis_sha256s")
        != list(held_out_source.session_basis_sha256s)
        or source_compatibility.get("representation_basis_sha256")
        != next(iter(representation_ids))
        or source_compatibility.get("coefficients_changed") is not False
        or source_compatibility.get("hidden_states_changed") is not False
        or source_compatibility.get("serving_authority") is not False
    ):
        raise ValueError("held-out source family was not bound after fitting")

    model_paths = {
        inventory.exact_model_path
        for inventory in (*source_inventories.values(), fresh_inventory)
    }
    tokenizer_ids = {
        inventory.tokenizer_identity_sha256
        for inventory in (*source_inventories.values(), fresh_inventory)
    }
    if (
        len(model_paths) != 1
        or len(tokenizer_ids) != 1
        or any(
            inventory.worker_stack_identity_gaps
            for inventory in (*source_inventories.values(), fresh_inventory)
        )
        or set(held_out_source.example_ids) & set(fresh_inventory.example_ids)
        or set(held_out_source.source_text_sha256s)
        & set(fresh_inventory.source_text_sha256s)
    ):
        raise ValueError("family-withheld fresh cohort is not disjoint and compatible")

    fresh_compatibility = _validated_receipt(
        fresh_report.get("representation_compatibility"),
        schema=_REPLICATION_COMPATIBILITY_SCHEMA,
    )
    if (
        fresh_compatibility.get("transducer_receipt_sha256") != model.receipt_sha256
        or fresh_compatibility.get("coefficient_sha256")
        != model.training_receipt.get("coefficient_sha256")
        or fresh_compatibility.get("training_feature_manifest_sha256") != anchor_manifest
        or fresh_compatibility.get("replication_feature_manifest_sha256")
        != fresh_inventory.manifest_sha256
        or fresh_compatibility.get("training_session_basis_sha256")
        != model.model_basis_sha256
        or fresh_compatibility.get("replication_session_basis_sha256s")
        != list(fresh_inventory.session_basis_sha256s)
        or fresh_compatibility.get("representation_basis_sha256")
        != next(iter(representation_ids))
        or fresh_compatibility.get("coefficients_changed") is not False
        or fresh_compatibility.get("hidden_states_changed") is not False
        or fresh_compatibility.get("serving_authority") is not False
    ):
        raise ValueError("fresh family lesion was not bound to the frozen model")

    family_reports = source_report.get("families")
    if not isinstance(family_reports, Mapping) or set(family_reports) != {held_out_family}:
        raise ValueError("held-out source family report inventory differs")
    source_recount = _source_family_recount(
        family_reports[held_out_family],
        inventory=held_out_source,
    )
    fresh_recount = _validate_report(
        CompositionalReplicationCohort(
            family=held_out_family,
            source=held_out_source,
            fresh=fresh_inventory,
            lesion_arm=lesion_arm,
            report=fresh_report,
            transfer_kind="whole_family_withheld_fresh_seed",
            evaluation_source_commit=evaluation_source_commit,
        ),
        transducer_receipt_sha256=model.receipt_sha256,
    )
    body = {
        "schema": "aura.semantic_program_family_withheld_verification.v1",
        "verified": True,
        "held_out_family": held_out_family,
        "fit_families": fit_families,
        "held_out_family_was_available_to_fit": False,
        "frozen_model_unchanged": True,
        "transducer_receipt_sha256": model.receipt_sha256,
        "source_campaign_report_sha256": source_report["report_sha256"],
        "source_feature_manifest_sha256s": source_manifest_sha256s,
        "source_recount": source_recount,
        "fresh_replication": fresh_recount,
        "source_fresh_example_overlap": 0,
        "source_fresh_text_overlap": 0,
        "model_path": next(iter(model_paths)),
        "tokenizer_identity_sha256": next(iter(tokenizer_ids)),
        "representation_basis_sha256": next(iter(representation_ids)),
        "fit_representation_compatibility_sha256": fit_compatibility["receipt_sha256"],
        "source_representation_compatibility_sha256": source_compatibility[
            "receipt_sha256"
        ],
        "fresh_representation_compatibility_sha256": fresh_compatibility[
            "receipt_sha256"
        ],
        "evaluation_source_commit": evaluation_source_commit,
        "stored_file_sha256s": dict(sorted(stored_file_sha256s.items())),
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_decode": False,
        "fit_or_refit_calls_during_replication": 0,
        "serving_authority": False,
        "claim_boundary": (
            "bounded resident-27B typed synthetic semantic-program transfer into a "
            "completely withheld program family and disjoint fresh seed; reusable "
            "primitives and value rules were present in other fitting families; no broad "
            "natural-language, open-domain, frontier-reasoning, or serving claim"
        ),
    }
    return {**body, "verification_sha256": _sha(body)}


def verify_compositional_semantic_replications(
    *,
    source_manifest_sha256s: Mapping[str, str],
    trained_model_payload: Mapping[str, Any],
    source_report: Mapping[str, Any],
    cohorts: Sequence[CompositionalReplicationCohort],
    source_sha256s: Mapping[str, str],
    stored_file_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Verify frozen task rows and causal effects without another decode pass."""

    if (
        set(source_sha256s) != set(COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
        or not stored_file_sha256s
        or any(not _is_sha256(value) for value in stored_file_sha256s.values())
    ):
        raise ValueError("compositional verification source identity is invalid")
    source_body = {
        key: value for key, value in source_report.items() if key != "report_sha256"
    }
    if (
        source_report.get("schema") != COMPOSITIONAL_LEAVE_FAMILY_OUT_SCHEMA
        or source_report.get("report_sha256") != _sha(source_body)
        or source_report.get("feature_manifest_sha256s")
        != dict(sorted(source_manifest_sha256s.items()))
        or source_report.get("expected_answers_available_to_training") is not False
        or source_report.get("verifier_traces_available") is not False
        or source_report.get("generated_compiler_text_available") is not False
        or source_report.get("serving_authority") is not False
    ):
        raise ValueError("compositional source campaign identity differs")
    model = compositional_semantic_program_transducer_from_dict(trained_model_payload)
    if (
        model.to_dict() != dict(trained_model_payload)
        or source_report.get("transducer_receipt_sha256") != model.receipt_sha256
        or source_report.get("model_basis_sha256") != model.model_basis_sha256
        or model.training_receipt.get("correctness_authority") is not False
        or model.training_receipt.get("expected_answers_available") is not False
        or model.training_receipt.get("verifier_traces_available") is not False
        or model.training_receipt.get("generated_compiler_text_available") is not False
        or model.training_receipt.get("family_router_present") is not False
    ):
        raise ValueError("compositional frozen model binding differs")
    if len(cohorts) < 2 or len({cohort.family for cohort in cohorts}) != len(cohorts):
        raise ValueError("compositional replication cohort inventory is invalid")
    model_paths = {
        inventory.exact_model_path
        for cohort in cohorts
        for inventory in (cohort.source, cohort.fresh)
    }
    tokenizer_ids = {
        inventory.tokenizer_identity_sha256
        for cohort in cohorts
        for inventory in (cohort.source, cohort.fresh)
    }
    representation_ids = {
        inventory.representation_basis_sha256
        for cohort in cohorts
        for inventory in (cohort.source, cohort.fresh)
    }
    if (
        len(model_paths) != 1
        or len(tokenizer_ids) != 1
        or len(representation_ids) != 1
        or any(
            inventory.worker_stack_identity_gaps
            for cohort in cohorts
            for inventory in (cohort.source, cohort.fresh)
        )
    ):
        raise ValueError("compositional replication model basis differs")
    verified_cohorts = []
    for cohort in cohorts:
        if (
            cohort.source.manifest_sha256
            != source_manifest_sha256s.get(cohort.family)
            or set(cohort.source.example_ids) & set(cohort.fresh.example_ids)
            or len(cohort.evaluation_source_commit) != 40
        ):
            raise ValueError(f"compositional cohort is not disjoint: {cohort.family}")
        verified_cohorts.append(
            _validate_report(
                cohort,
                transducer_receipt_sha256=model.receipt_sha256,
            )
        )
    body = {
        "schema": COMPOSITIONAL_REPLICATION_VERIFICATION_SCHEMA,
        "verified": True,
        "frozen_model_unchanged": True,
        "transducer_receipt_sha256": model.receipt_sha256,
        "source_campaign_report_sha256": source_report["report_sha256"],
        "source_feature_manifest_sha256s": dict(sorted(source_manifest_sha256s.items())),
        "stored_file_sha256s": dict(sorted(stored_file_sha256s.items())),
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "model_path": next(iter(model_paths)),
        "tokenizer_identity_sha256": next(iter(tokenizer_ids)),
        "cohorts": verified_cohorts,
        "expected_answers_available_to_training": False,
        "expected_answers_available_to_decode": False,
        "fit_or_refit_calls_during_replication": 0,
        "serving_authority": False,
        "claim_boundary": (
            "bounded resident-27B typed synthetic semantic-program transfer across "
            "disjoint arithmetic and fork/join cohorts; no broad natural-language, "
            "frontier-reasoning, or serving claim"
        ),
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "COMPOSITIONAL_REPLICATION_VERIFICATION_SCHEMA",
    "COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES",
    "CompositionalReplicationCohort",
    "SemanticCohortInventory",
    "semantic_cohort_inventory",
    "verify_compositional_family_withheld_replication",
    "verify_compositional_semantic_replications",
]
