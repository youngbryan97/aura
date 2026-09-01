"""Independent source-bound replay of semantic operation transfer evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

from core.learning.semantic_operation_transfer import (
    SEMANTIC_OPERATION_TRANSFER_SCHEMA,
    run_semantic_operation_transfer,
)
from core.learning.semantic_program_feature_materialization import (
    LoadedSemanticFeatureBundle,
)

SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SCHEMA: Final = (
    "aura.semantic_operation_transfer_verification.v1"
)
SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES: Final = (
    "core/brain/llm/hidden_sequence_contract.py",
    "core/learning/semantic_operation_transfer.py",
    "core/learning/semantic_operation_transfer_verification.py",
    "core/learning/semantic_program_basis.py",
    "core/learning/semantic_program_campaign.py",
    "core/learning/semantic_program_corpus.py",
    "core/learning/semantic_program_feature_materialization.py",
    "core/learning/semantic_program_ir.py",
    "core/learning/semantic_program_transducer.py",
    "tools/run_semantic_operation_transfer.py",
    "tools/verify_semantic_operation_transfer.py",
)
_VIEWS: Final = {
    "absolute_span_mean": False,
    "counterfactual_centered_span_mean": True,
}
_DIRECTIONS: Final = {
    "arithmetic_to_fork_join",
    "arithmetic_to_sequence_binary",
    "fork_join_to_arithmetic",
    "fork_join_to_sequence_binary",
    "sequence_binary_to_arithmetic",
    "sequence_binary_to_fork_join",
}
_ARMS: Final = {
    "treatment",
    "coefficient_lesion",
    "label_permutation",
    "geometry_only",
    "token_lookup",
}


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


def verify_semantic_operation_transfer(
    bundles: Mapping[str, LoadedSemanticFeatureBundle],
    *,
    stored_report: Any,
    source_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Reload the evidence and require an exact deterministic replay."""

    if (
        not isinstance(source_sha256s, Mapping)
        or set(source_sha256s) != set(SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES)
        or any(not _is_sha256(value) for value in source_sha256s.values())
    ):
        raise ValueError("semantic operation verification source identity is invalid")
    if not isinstance(stored_report, dict):
        raise ValueError("semantic operation verification report is invalid")
    body = {key: value for key, value in stored_report.items() if key != "report_sha256"}
    views = stored_report.get("representation_views")
    manifests = stored_report.get("feature_manifest_sha256s")
    if (
        stored_report.get("schema") != SEMANTIC_OPERATION_TRANSFER_SCHEMA
        or stored_report.get("report_sha256") != _sha(body)
        or not isinstance(views, dict)
        or set(views) != set(_VIEWS)
        or not isinstance(manifests, dict)
        or manifests
        != {
            name: bundle.manifest["manifest_sha256"]
            for name, bundle in sorted(bundles.items())
        }
        or stored_report.get("serving_authority") is not False
    ):
        raise ValueError("semantic operation verification envelope is invalid")
    measured_program_rows = 0
    paired_tests = 0
    for view_name, requires_batch in _VIEWS.items():
        view = views[view_name]
        if (
            not isinstance(view, dict)
            or view.get("counterfactual_target_batch_required") is not requires_batch
            or set(view.get("directions", {})) != _DIRECTIONS
        ):
            raise ValueError("semantic operation verification view is invalid")
        for direction in view["directions"].values():
            if set(direction.get("splits", {})) != {"validation", "test"}:
                raise ValueError("semantic operation verification splits differ")
            for split in direction["splits"].values():
                arms = split.get("arms", {})
                paired = split.get("paired_program_tests", {})
                if set(arms) != _ARMS or set(paired) != _ARMS - {"treatment"}:
                    raise ValueError("semantic operation verification controls differ")
                total = split.get("program_count")
                if type(total) is not int or total < 1:
                    raise ValueError("semantic operation verification total is invalid")
                if any(
                    type(arm.get("program_exact")) is not int
                    or not 0 <= arm["program_exact"] <= total
                    for arm in arms.values()
                ):
                    raise ValueError("semantic operation verification count is invalid")
                measured_program_rows += total * len(arms)
                paired_tests += len(paired)
    replay = run_semantic_operation_transfer(bundles)
    if replay != stored_report:
        raise ValueError("semantic operation verification replay differs")
    verification_body = {
        "schema": SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SCHEMA,
        "verified": True,
        "feature_manifest_sha256s": dict(sorted(manifests.items())),
        "campaign_report_sha256": stored_report["report_sha256"],
        "source_sha256s": dict(sorted(source_sha256s.items())),
        "raw_records_reloaded": sum(len(bundle.examples) for bundle in bundles.values()),
        "representation_view_count": len(views),
        "directed_transfer_count": len(_DIRECTIONS) * len(views),
        "program_arm_rows_recounted": measured_program_rows,
        "paired_tests_recounted": paired_tests,
        "frozen_replay_exact": True,
        "serving_authority": False,
    }
    return {
        **verification_body,
        "verification_sha256": _sha(verification_body),
    }


__all__ = [
    "SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SCHEMA",
    "SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES",
    "verify_semantic_operation_transfer",
]
