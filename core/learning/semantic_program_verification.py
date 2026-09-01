"""Deterministic replay and independent recount for semantic campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.semantic_program_campaign import (
    SEMANTIC_PROGRAM_CAMPAIGN_SCHEMA,
    run_semantic_program_campaign,
)
from core.learning.semantic_program_feature_materialization import (
    CHAIN_CORPUS_KIND,
    FORK_JOIN_CORPUS_KIND,
    FORK_JOIN_FACTORIAL_CORPUS_KIND,
    FORK_JOIN_SOURCE_ORDER_CORPUS_KIND,
    SEQUENCE_BINARY_CHAIN_CORPUS_KIND,
    SEQUENCE_CHAIN_CORPUS_KIND,
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_transducer import (
    semantic_program_transducer_from_dict,
)

SEMANTIC_PROGRAM_VERIFICATION_SCHEMA: Final = "aura.semantic_program_campaign_verification.v1"
SEMANTIC_PROGRAM_VERIFICATION_SOURCES: Final = (
    "core/brain/llm/hidden_sequence_contract.py",
    "core/learning/procedure_induction.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/semantic_program_evaluation.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_transducer.py",
    "core/learning/semantic_program_verification.py",
    "tools/verify_semantic_program_campaign.py",
)
_ARMS: Final = {
    "treatment:train",
    "treatment:validation",
    "treatment:test",
    "hidden_token_shuffle:validation",
    "hidden_token_shuffle:test",
    "coefficient_lesion:validation",
    "coefficient_lesion:test",
    "label_permutation:validation",
    "label_permutation:test",
}
_COUNT_FIELDS: Final = (
    "accepted",
    "program_exact",
    "operation_exact",
    "argument_exact",
    "input_span_exact",
    "attribution_exact",
    "full_ir_exact",
    "answer_emitted",
    "answer_exact",
)
_CONTROLS: Final = (
    "hidden_token_shuffle",
    "coefficient_lesion",
    "label_permutation",
)
_ARITHMETIC_CORPUS_KINDS: Final = frozenset(
    {
        CHAIN_CORPUS_KIND,
        FORK_JOIN_CORPUS_KIND,
        FORK_JOIN_FACTORIAL_CORPUS_KIND,
        FORK_JOIN_SOURCE_ORDER_CORPUS_KIND,
    }
)


def semantic_program_claim_boundary(corpus_kind: str) -> str:
    """Name the measured family without widening its evidence authority."""

    if corpus_kind in _ARITHMETIC_CORPUS_KINDS:
        family = "construction-held-out synthetic arithmetic language"
    elif corpus_kind == SEQUENCE_CHAIN_CORPUS_KIND:
        family = (
            "construction-held-out synthetic typed sequence transformation "
            "and scalar aggregation language"
        )
    elif corpus_kind == SEQUENCE_BINARY_CHAIN_CORPUS_KIND:
        family = (
            "construction-held-out synthetic sequence lookup, occurrence counting, "
            "and scalar continuation language"
        )
    else:
        raise ValueError("semantic verification corpus family is unsupported")
    return (
        "bounded resident-27B semantic program acquisition and exact answer "
        f"execution on {family}; not broad-domain or frontier reasoning evidence"
    )


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


def _one_sided_exact_p(*, treatment_only: int, control_only: int) -> float:
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, successes) for successes in range(treatment_only, discordant + 1)
    ) / (2**discordant)


def recount_semantic_arm(arm: Mapping[str, Any]) -> dict[str, int]:
    """Recount one arm from task rows without trusting its aggregate fields."""

    rows = arm.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic verification arm rows are invalid")
    identities: set[str] = set()
    counts = {field: 0 for field in _COUNT_FIELDS}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("semantic verification task row is invalid")
        identity = row.get("source_text_sha256")
        if not isinstance(identity, str) or len(identity) != 64 or identity in identities:
            raise ValueError("semantic verification task identity is invalid")
        identities.add(identity)
        for field in _COUNT_FIELDS:
            value = row.get(field)
            if type(value) is not bool:
                raise ValueError(f"semantic verification row field is invalid: {field}")
            counts[field] += int(value)
    observed = {"total": len(rows), **counts}
    if any(arm.get(field) != value for field, value in observed.items()):
        raise ValueError("semantic verification arm aggregate differs from task rows")
    return observed


def recount_semantic_pair(
    treatment_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """Reconstruct one paired exact test directly from its two task arms."""

    if metric not in {"program_exact", "answer_exact"}:
        raise ValueError("semantic verification paired metric is unsupported")
    treatment = {str(row["source_text_sha256"]): row.get(metric) is True for row in treatment_rows}
    control = {str(row["source_text_sha256"]): row.get(metric) is True for row in control_rows}
    if treatment.keys() != control.keys():
        raise ValueError("semantic verification paired tasks differ")
    treatment_only = sum(treatment[key] and not control[key] for key in treatment)
    control_only = sum(control[key] and not treatment[key] for key in treatment)
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": treatment_only + control_only,
        "one_sided_exact_p": _one_sided_exact_p(
            treatment_only=treatment_only,
            control_only=control_only,
        ),
    }


def verify_semantic_program_campaign(
    bundle: LoadedSemanticFeatureBundle,
    *,
    stored_model_payload: Any,
    stored_report: Any,
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Replay a frozen campaign and independently reconstruct every score."""

    if (
        not isinstance(source_sha256s, Mapping)
        or set(source_sha256s) != set(SEMANTIC_PROGRAM_VERIFICATION_SOURCES)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_sha256s.values()
        )
    ):
        raise ValueError("semantic verification source identity is invalid")
    if not isinstance(stored_report, dict):
        raise ValueError("semantic verification report is invalid")
    report_body = {key: value for key, value in stored_report.items() if key != "report_sha256"}
    if (
        stored_report.get("schema") != SEMANTIC_PROGRAM_CAMPAIGN_SCHEMA
        or stored_report.get("report_sha256") != _sha(report_body)
        or set(stored_report.get("arms", {})) != _ARMS
    ):
        raise ValueError("semantic verification report envelope is invalid")
    stored_model = semantic_program_transducer_from_dict(stored_model_payload)
    replay = run_semantic_program_campaign(bundle)
    if replay.model.to_dict() != stored_model.to_dict():
        raise ValueError("semantic verification refit differs from stored model")
    if replay.report != stored_report:
        raise ValueError("semantic verification replay differs from stored report")

    recounted_arms = {
        name: recount_semantic_arm(arm) for name, arm in sorted(stored_report["arms"].items())
    }
    paired_programs: dict[str, dict[str, Any]] = {}
    paired_answers: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        treatment = stored_report["arms"][f"treatment:{split}"]["rows"]
        for control in _CONTROLS:
            key = f"{control}:{split}"
            control_rows = stored_report["arms"][key]["rows"]
            paired_programs[key] = recount_semantic_pair(
                treatment,
                control_rows,
                metric="program_exact",
            )
            paired_answers[key] = recount_semantic_pair(
                treatment,
                control_rows,
                metric="answer_exact",
            )
    if paired_programs != stored_report.get(
        "paired_program_controls"
    ) or paired_answers != stored_report.get("paired_answer_controls"):
        raise ValueError("semantic verification paired scores differ")

    held_out_programs = sum(
        recounted_arms[f"treatment:{split}"]["program_exact"] for split in ("validation", "test")
    )
    held_out_answers = sum(
        recounted_arms[f"treatment:{split}"]["answer_exact"] for split in ("validation", "test")
    )
    held_out_total = sum(
        recounted_arms[f"treatment:{split}"]["total"] for split in ("validation", "test")
    )
    if (
        held_out_programs != stored_report.get("held_out_treatment_program_exact")
        or held_out_answers != stored_report.get("held_out_treatment_answer_exact")
        or held_out_total != stored_report.get("held_out_total")
        or stored_report.get("expected_answers_available_to_training") is not False
        or stored_report.get("expected_answers_available_to_evaluation") is not True
        or stored_report.get("serving_authority") is not False
    ):
        raise ValueError("semantic verification held-out contract differs")

    body = {
        "schema": SEMANTIC_PROGRAM_VERIFICATION_SCHEMA,
        "verified": True,
        "feature_manifest_sha256": bundle.manifest["manifest_sha256"],
        "model_basis_sha256": stored_model.model_basis_sha256,
        "transducer_receipt_sha256": stored_model.receipt_sha256,
        "campaign_report_sha256": stored_report["report_sha256"],
        "stored_model_sha256": _sha(stored_model_payload),
        "stored_report_sha256": _sha(stored_report),
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "raw_feature_records_reloaded": len(bundle.examples),
        "deterministic_refit_exact": True,
        "campaign_replay_exact": True,
        "task_rows_independently_recounted": sum(
            values["total"] for values in recounted_arms.values()
        ),
        "held_out_total": held_out_total,
        "held_out_treatment_program_exact": held_out_programs,
        "held_out_treatment_answer_exact": held_out_answers,
        "paired_program_controls": paired_programs,
        "paired_answer_controls": paired_answers,
        "expected_answers_available_to_training": False,
        "serving_authority": False,
        "claim_boundary": semantic_program_claim_boundary(
            str(bundle.manifest.get("config", {}).get("corpus_kind", CHAIN_CORPUS_KIND))
        ),
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PROGRAM_VERIFICATION_SCHEMA",
    "SEMANTIC_PROGRAM_VERIFICATION_SOURCES",
    "recount_semantic_arm",
    "recount_semantic_pair",
    "semantic_program_claim_boundary",
    "verify_semantic_program_campaign",
]
