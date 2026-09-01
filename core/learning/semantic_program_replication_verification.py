"""Independent replay and recount of a frozen semantic replication."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_replication import (
    SEMANTIC_PROGRAM_REPLICATION_SCHEMA,
    SEMANTIC_PROGRAM_REPLICATION_SOURCES,
    FrozenTrainingCohort,
    evaluate_frozen_semantic_replication,
)
from core.learning.semantic_program_transducer import (
    semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_verification import recount_semantic_arm

SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_fresh_cohort_verification.v1"
)
SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES: Final = (
    *SEMANTIC_PROGRAM_REPLICATION_SOURCES,
    "core/learning/semantic_program_replication_verification.py",
    "tools/verify_semantic_program_replication.py",
)
_SPLITS: Final = ("train", "validation", "test")
_CONTROLS: Final = ("hidden_token_shuffle", "coefficient_lesion")
_METRICS: Final = ("program_exact", "answer_exact")


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
        raise ValueError("semantic replication verification paired tasks differ")
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
    numerator //= divisor
    denominator //= divisor
    return {
        "metric": metric,
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_exact_p_numerator": numerator,
        "one_sided_exact_p_denominator": denominator,
        "one_sided_exact_p": numerator / denominator,
    }


def _pooled_rows(report: Mapping[str, Any], arm: str) -> list[Mapping[str, Any]]:
    return [
        row
        for split in _SPLITS
        for row in report["arms"][f"{arm}:{split}"]["rows"]
    ]


def verify_frozen_semantic_replication(
    *,
    training_bundle: LoadedSemanticFeatureBundle,
    replication_bundle: LoadedSemanticFeatureBundle,
    trained_model_payload: Any,
    stored_report: Any,
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Replay frozen transfer and independently reconstruct its evidence."""

    if (
        not isinstance(source_sha256s, Mapping)
        or set(source_sha256s)
        != set(SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
    ):
        raise ValueError("semantic replication verification source identity is invalid")
    if not isinstance(stored_report, dict):
        raise ValueError("semantic replication verification report is invalid")
    report_body = {
        key: value for key, value in stored_report.items() if key != "report_sha256"
    }
    report_sources = stored_report.get("source_sha256s")
    if (
        stored_report.get("schema") != SEMANTIC_PROGRAM_REPLICATION_SCHEMA
        or stored_report.get("report_sha256") != _sha(report_body)
        or not isinstance(report_sources, dict)
        or set(report_sources) != set(SEMANTIC_PROGRAM_REPLICATION_SOURCES)
        or any(source_sha256s[path] != value for path, value in report_sources.items())
    ):
        raise ValueError("semantic replication report envelope or source differs")
    model = semantic_program_transducer_from_dict(trained_model_payload)
    training_cohort = FrozenTrainingCohort(
        feature_manifest_sha256=training_bundle.manifest["manifest_sha256"],
        example_ids=tuple(
            str(item.metadata["example_id"]) for item in training_bundle.examples
        ),
    )
    replay = evaluate_frozen_semantic_replication(
        replication_bundle,
        trained_model_payload=trained_model_payload,
        training_cohort=training_cohort,
        training_manifest=training_bundle.manifest,
        source_sha256s=report_sources,
    )
    if replay != stored_report:
        raise ValueError("semantic replication replay differs from stored report")

    recounted_arms = {
        name: recount_semantic_arm(arm)
        for name, arm in sorted(stored_report["arms"].items())
    }
    paired: dict[str, dict[str, Any]] = {}
    for control in _CONTROLS:
        for split in (*_SPLITS, "pooled"):
            treatment_rows = (
                _pooled_rows(stored_report, "treatment")
                if split == "pooled"
                else stored_report["arms"][f"treatment:{split}"]["rows"]
            )
            control_rows = (
                _pooled_rows(stored_report, control)
                if split == "pooled"
                else stored_report["arms"][f"{control}:{split}"]["rows"]
            )
            for metric in _METRICS:
                paired[f"{control}:{split}:{metric}"] = _paired_exact(
                    treatment_rows,
                    control_rows,
                    metric=metric,
                )
    if paired != stored_report.get("paired_exact_tests"):
        raise ValueError("semantic replication paired exact tests differ")

    held_out_treatment_answers = sum(
        recounted_arms[f"treatment:{split}"]["answer_exact"]
        for split in ("validation", "test")
    )
    held_out_shuffle_answers = sum(
        recounted_arms[f"hidden_token_shuffle:{split}"]["answer_exact"]
        for split in ("validation", "test")
    )
    held_out_lesion_answers = sum(
        recounted_arms[f"coefficient_lesion:{split}"]["answer_exact"]
        for split in ("validation", "test")
    )
    body = {
        "schema": SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SCHEMA,
        "verified": True,
        "training_feature_manifest_sha256": training_bundle.manifest[
            "manifest_sha256"
        ],
        "replication_feature_manifest_sha256": replication_bundle.manifest[
            "manifest_sha256"
        ],
        "transducer_receipt_sha256": model.receipt_sha256,
        "replication_report_sha256": stored_report["report_sha256"],
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "raw_training_records_reloaded": len(training_bundle.examples),
        "raw_replication_records_reloaded": len(replication_bundle.examples),
        "task_rows_independently_recounted": sum(
            values["total"] for values in recounted_arms.values()
        ),
        "frozen_replay_exact": True,
        "paired_tests_independently_recounted": len(paired),
        "held_out_total": sum(
            recounted_arms[f"treatment:{split}"]["total"]
            for split in ("validation", "test")
        ),
        "held_out_treatment_answer_exact": held_out_treatment_answers,
        "held_out_hidden_shuffle_answer_exact": held_out_shuffle_answers,
        "held_out_coefficient_lesion_answer_exact": held_out_lesion_answers,
        "representation_compatibility": stored_report[
            "representation_compatibility"
        ],
        "expected_answers_available_to_training": False,
        "serving_authority": False,
        "claim_boundary": stored_report["claim_boundary"],
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SCHEMA",
    "SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES",
    "verify_frozen_semantic_replication",
]
