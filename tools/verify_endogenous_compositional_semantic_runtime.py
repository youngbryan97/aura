#!/usr/bin/env python3
"""Verify endogenous public-input execution of a frozen semantic transducer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LESIONS = frozenset({"relation_tissue_lesion", "argument_proposal_lesion"})


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path, *, max_bytes: int) -> tuple[dict[str, Any], bytes]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    value = json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)
    if not isinstance(value, dict):
        raise ValueError("endogenous semantic artifact is not an object")
    return value, payload


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


def _cohorts(values: list[str]) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for value in values:
        family, separator, remainder = value.partition("=")
        raw_path, second_separator, lesion = remainder.rpartition("=")
        if (
            not separator
            or not second_separator
            or not _FAMILY.fullmatch(family)
            or family in result
            or lesion not in _LESIONS
        ):
            raise ValueError("cohort must be unique FAMILY=FEATURE_DIRECTORY=LESION")
        result[family] = (Path(raw_path).expanduser().resolve(strict=True), lesion)
    if len(result) < 2:
        raise ValueError("endogenous semantic verification needs at least two cohorts")
    return result


def _validate_source_evidence(
    *,
    transducer: dict[str, Any],
    transducer_raw: bytes,
    source_report: dict[str, Any],
    source_report_raw: bytes,
    source_verification: dict[str, Any],
) -> str:
    source_body = {key: value for key, value in source_report.items() if key != "report_sha256"}
    verification_body = {
        key: value for key, value in source_verification.items() if key != "verification_sha256"
    }
    compatibility = source_report.get("representation_compatibility")
    if (
        source_report.get("report_sha256") != _sha(source_body)
        or source_verification.get("verification_sha256") != _sha(verification_body)
        or source_verification.get("verified") is not True
        or source_verification.get("serving_authority") is not False
        or source_verification.get("source_campaign_report_sha256")
        != source_report.get("report_sha256")
        or source_verification.get("transducer_receipt_sha256")
        != transducer.get("training_receipt", {}).get("receipt_sha256")
        or source_verification.get("stored_file_sha256s", {}).get("model")
        != hashlib.sha256(transducer_raw).hexdigest()
        or source_verification.get("stored_file_sha256s", {}).get("source_report")
        != hashlib.sha256(source_report_raw).hexdigest()
        or not isinstance(compatibility, dict)
        or compatibility.get("hidden_states_changed") is not False
        or compatibility.get("serving_authority") is not False
        or not isinstance(compatibility.get("representation_basis_sha256"), str)
    ):
        raise ValueError("frozen compositional source evidence differs")
    return str(compatibility["representation_basis_sha256"])


def _output_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-verification", type=Path, required=True)
    parser.add_argument(
        "--cohort",
        action="append",
        required=True,
        metavar="FAMILY=FEATURE_DIRECTORY=LESION",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from mlx_lm.utils import load_tokenizer

        from core.learning.semantic_program_compositional_transducer import (
            compositional_semantic_program_transducer_from_dict,
        )
        from core.learning.semantic_program_endogenous_verification import (
            ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES,
            evaluate_endogenous_semantic_bridge,
            verify_endogenous_semantic_bridge,
        )
        from core.learning.semantic_program_feature_materialization import (
            build_semantic_program_corpus_for_config,
            load_standard_semantic_feature_bundle,
            offset_tokenizer_for_worker,
            semantic_feature_config_from_manifest,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        cohort_specs = _cohorts(args.cohort)
        transducer_payload, transducer_raw = _read_json(
            args.transducer,
            max_bytes=32 * 1024 * 1024,
        )
        source_report, source_report_raw = _read_json(
            args.source_report,
            max_bytes=32 * 1024 * 1024,
        )
        source_verification, _source_verification_raw = _read_json(
            args.source_verification,
            max_bytes=8 * 1024 * 1024,
        )
        representation_sha256 = _validate_source_evidence(
            transducer=transducer_payload,
            transducer_raw=transducer_raw,
            source_report=source_report,
            source_report_raw=source_report_raw,
            source_verification=source_verification,
        )
        model = compositional_semantic_program_transducer_from_dict(transducer_payload)
        expected_cohorts = {
            str(value["family"]): value
            for value in source_verification.get("cohorts", [])
            if isinstance(value, dict) and isinstance(value.get("family"), str)
        }
        bundles = {
            family: load_standard_semantic_feature_bundle(path)
            for family, (path, _lesion) in cohort_specs.items()
        }
        model_paths = {
            str(bundle.manifest.get("exact_model_path") or "") for bundle in bundles.values()
        }
        tokenizer_ids = {
            bundle.manifest.get("tokenizer_identity", {}).get("identity_sha256")
            for bundle in bundles.values()
        }
        if (
            len(model_paths) != 1
            or model_paths != {source_verification.get("model_path")}
            or len(tokenizer_ids) != 1
            or tokenizer_ids != {source_verification.get("tokenizer_identity_sha256")}
            or any(
                family not in expected_cohorts
                or bundle.manifest.get("manifest_sha256")
                != expected_cohorts[family].get("fresh_feature_manifest_sha256")
                or lesion != expected_cohorts[family].get("lesion_arm")
                for family, bundle in bundles.items()
                for lesion in (cohort_specs[family][1],)
            )
        ):
            raise ValueError("endogenous semantic cohort identity differs")
        tokenizer = offset_tokenizer_for_worker(
            load_tokenizer(Path(next(iter(model_paths))))
        )
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in ENDOGENOUS_SEMANTIC_VERIFICATION_SOURCES
        }
        cohort_results: dict[str, Any] = {}
        for family, bundle in bundles.items():
            lesion_name = cohort_specs[family][1]
            lesion_model = getattr(model, lesion_name)()
            corpus = build_semantic_program_corpus_for_config(
                semantic_feature_config_from_manifest(bundle.manifest)
            )
            treatment = evaluate_endogenous_semantic_bridge(
                model=model,
                bundle=bundle,
                corpus=corpus,
                tokenizer=tokenizer,
                expected_representation_basis_sha256=representation_sha256,
                arm="treatment",
            )
            print(
                json.dumps(
                    {
                        "schema": "aura.semantic_program_endogenous_progress.v1",
                        "family": family,
                        "arm": "treatment",
                        "total": treatment.total,
                        "accepted": treatment.accepted,
                        "public_input_exact": treatment.public_input_exact,
                        "program_exact": treatment.program_exact,
                        "answer_exact": treatment.answer_exact,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            lesion = evaluate_endogenous_semantic_bridge(
                model=lesion_model,
                bundle=bundle,
                corpus=corpus,
                tokenizer=tokenizer,
                expected_representation_basis_sha256=representation_sha256,
                arm=lesion_name,
            )
            verification = verify_endogenous_semantic_bridge(
                treatment=treatment,
                lesion=lesion,
                source_verification_sha256=source_verification["verification_sha256"],
                source_sha256s=source_sha256s,
            )
            print(
                json.dumps(
                    {
                        "schema": "aura.semantic_program_endogenous_progress.v1",
                        "family": family,
                        "arm": lesion_name,
                        "total": lesion.total,
                        "accepted": lesion.accepted,
                        "public_input_exact": lesion.public_input_exact,
                        "program_exact": lesion.program_exact,
                        "answer_exact": lesion.answer_exact,
                        "verified": verification["verified"],
                        "answer_exact_p": verification["paired_exact_tests"][
                            "answer_exact"
                        ]["one_sided_exact_p"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            cohort_results[family] = {
                "treatment": treatment.receipt,
                "lesion": lesion.receipt,
                "verification": verification,
            }
            if verification["verified"] is not True:
                raise RuntimeError(f"endogenous semantic cohort did not verify:{family}")
        body = {
            "schema": "aura.semantic_program_endogenous_multicohort_verification.v1",
            "verified": all(
                value["verification"]["verified"] is True
                for value in cohort_results.values()
            ),
            "source_verification_sha256": source_verification["verification_sha256"],
            "transducer_receipt_sha256": model.receipt_sha256,
            "representation_basis_sha256": representation_sha256,
            "cohorts": cohort_results,
            "source_sha256s": source_sha256s,
            "fit_or_refit_calls": 0,
            "oracle_public_values_available_to_decode": False,
            "family_available_to_decode": False,
            "expected_answer_available_to_decode": False,
            "verifier_trace_available_to_decode": False,
            "generated_text_available": False,
            "serving_authority": False,
            "claim_boundary": (
                "bounded resident-27B endogenous public-input semantic execution "
                "across two frozen cohorts; no open-domain, frontier-reasoning, "
                "or serving claim"
            ),
        }
        report = {**body, "verification_sha256": _sha(body)}
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError("endogenous semantic verification output already exists")
        if not atomic_write_bytes_if_absent(output, _output_bytes(report), mode=0o400):
            raise FileExistsError("endogenous semantic verification output raced")
    except Exception as exc:  # noqa: BLE001 - terminal verifier reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_endogenous_verifier_cli.v1",
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "schema": "aura.semantic_program_endogenous_verifier_cli.v1",
                "verified": report["verified"],
                "verification_sha256": report["verification_sha256"],
                "cohorts": {
                    family: {
                        "total": value["verification"]["total"],
                        "treatment_answer_exact": value["verification"][
                            "treatment_answer_exact"
                        ],
                        "lesion_answer_exact": value["verification"][
                            "lesion_answer_exact"
                        ],
                    }
                    for family, value in cohort_results.items()
                },
                "output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0 if report["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
