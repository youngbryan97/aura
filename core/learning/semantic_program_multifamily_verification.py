"""Independent replay and recount for one shared multi-family campaign."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)
from core.learning.semantic_program_multifamily import (
    SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA,
    run_semantic_program_multifamily_campaign,
)
from core.learning.semantic_program_transducer import (
    semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_verification import (
    recount_semantic_arm,
    recount_semantic_pair,
)

SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_program_multifamily_verification.v1"
)
SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SOURCES: Final = (
    "core/brain/llm/hidden_sequence_contract.py",
    "core/learning/procedure_induction.py",
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/semantic_program_evaluation.py",
    "core/learning/semantic_program_execution.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_multifamily.py",
    "core/learning/semantic_program_multifamily_verification.py",
    "core/learning/semantic_program_transducer.py",
    "core/learning/semantic_program_verification.py",
    "tools/verify_semantic_program_multifamily.py",
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
_CONTROLS: Final = (
    "hidden_token_shuffle",
    "coefficient_lesion",
    "label_permutation",
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_semantic_program_multifamily_campaign(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
    *,
    stored_model_payload: Any,
    stored_report: Any,
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Refit from raw tensors and independently recount every family arm."""

    if (
        not isinstance(source_sha256s, Mapping)
        or set(source_sha256s) != set(SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
    ):
        raise ValueError("semantic multi-family verification source identity is invalid")
    if not isinstance(stored_report, dict):
        raise ValueError("semantic multi-family verification report is invalid")
    report_body = {
        key: value for key, value in stored_report.items() if key != "report_sha256"
    }
    families = stored_report.get("families")
    if (
        stored_report.get("schema") != SEMANTIC_PROGRAM_MULTIFAMILY_CAMPAIGN_SCHEMA
        or stored_report.get("report_sha256") != _sha(report_body)
        or not isinstance(families, dict)
        or set(families) != set(bundles)
        or stored_report.get("family_count") != len(bundles)
        or stored_report.get("shared_model_count") != 1
        or stored_report.get("family_router_present") is not False
        or any(set(report.get("arms", {})) != _ARMS for report in families.values())
    ):
        raise ValueError("semantic multi-family verification envelope is invalid")
    stored_model = semantic_program_transducer_from_dict(stored_model_payload)
    replay = run_semantic_program_multifamily_campaign(bundles)
    if replay.model.to_dict() != stored_model.to_dict():
        raise ValueError("semantic multi-family verification refit differs")
    if replay.report != stored_report:
        raise ValueError("semantic multi-family verification replay differs")

    recounted: dict[str, dict[str, dict[str, int]]] = {}
    paired_count = 0
    for family, family_report in sorted(families.items()):
        arms = family_report["arms"]
        recounted[family] = {
            name: recount_semantic_arm(arm) for name, arm in sorted(arms.items())
        }
        for split in ("validation", "test"):
            treatment = arms[f"treatment:{split}"]["rows"]
            for control in _CONTROLS:
                key = f"{control}:{split}"
                control_rows = arms[key]["rows"]
                if recount_semantic_pair(
                    treatment,
                    control_rows,
                    metric="program_exact",
                ) != family_report["paired_program_controls"][key]:
                    raise ValueError("semantic multi-family program pair differs")
                if recount_semantic_pair(
                    treatment,
                    control_rows,
                    metric="answer_exact",
                ) != family_report["paired_answer_controls"][key]:
                    raise ValueError("semantic multi-family answer pair differs")
                paired_count += 2
        held_out_total = sum(
            recounted[family][f"treatment:{split}"]["total"]
            for split in ("validation", "test")
        )
        held_out_program = sum(
            recounted[family][f"treatment:{split}"]["program_exact"]
            for split in ("validation", "test")
        )
        held_out_answer = sum(
            recounted[family][f"treatment:{split}"]["answer_exact"]
            for split in ("validation", "test")
        )
        if (
            family_report.get("held_out_total") != held_out_total
            or family_report.get("held_out_treatment_program_exact") != held_out_program
            or family_report.get("held_out_treatment_answer_exact") != held_out_answer
        ):
            raise ValueError("semantic multi-family held-out aggregate differs")
    body = {
        "schema": SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SCHEMA,
        "verified": True,
        "feature_manifest_sha256s": {
            family: bundle.manifest["manifest_sha256"]
            for family, bundle in sorted(bundles.items())
        },
        "transducer_receipt_sha256": stored_model.receipt_sha256,
        "campaign_report_sha256": stored_report["report_sha256"],
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "raw_records_reloaded": sum(len(bundle.examples) for bundle in bundles.values()),
        "family_count": len(bundles),
        "shared_model_count": 1,
        "family_router_present": False,
        "task_rows_independently_recounted": sum(
            arm["total"]
            for family in recounted.values()
            for arm in family.values()
        ),
        "paired_tests_independently_recounted": paired_count,
        "frozen_refit_exact": True,
        "serving_authority": False,
    }
    return {**body, "verification_sha256": _sha(body)}


__all__ = [
    "SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SCHEMA",
    "SEMANTIC_PROGRAM_MULTIFAMILY_VERIFICATION_SOURCES",
    "verify_semantic_program_multifamily_campaign",
]
