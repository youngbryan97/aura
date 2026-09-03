#!/usr/bin/env python3
"""Freeze independent evidence for compositional semantic replications."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, max_bytes: int) -> tuple[Any, str]:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return (
        json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object),
        hashlib.sha256(payload).hexdigest(),
    )


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if (
            not separator
            or not _FAMILY.fullmatch(family)
            or not raw_path
            or family in result
        ):
            raise ValueError("source bundles must be unique FAMILY=PATH values")
        result[family] = Path(raw_path).expanduser().resolve(strict=True)
    if len(result) < 2:
        raise ValueError("compositional verification needs at least two source families")
    return result


def _replication_specs(values: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    families: set[str] = set()
    for value in values:
        parts = value.split("=", 5)
        if len(parts) != 6:
            raise ValueError(
                "replication must be FAMILY=FRESH=REPORT=LESION=TRANSFER_KIND=COMMIT"
            )
        family, fresh, report, lesion, transfer_kind, commit = parts
        if (
            not _FAMILY.fullmatch(family)
            or family in families
            or not fresh
            or not report
            or not _FAMILY.fullmatch(lesion)
            or not transfer_kind
            or not _COMMIT.fullmatch(commit)
        ):
            raise ValueError("compositional replication specification is invalid")
        families.add(family)
        result.append(
            {
                "family": family,
                "fresh": Path(fresh).expanduser().resolve(strict=True),
                "report": Path(report).expanduser().resolve(strict=True),
                "lesion": lesion,
                "transfer_kind": transfer_kind,
                "commit": commit,
            }
        )
    if len(result) < 2:
        raise ValueError("compositional verification needs two fresh replications")
    return result


def _verify_commit(commit: str) -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise ValueError(f"evaluation source commit is not an ancestor: {commit}")


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
    parser.add_argument("--source-bundle", action="append", required=True)
    parser.add_argument("--replication", action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_compositional_verification import (
            COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES,
            CompositionalReplicationCohort,
            semantic_cohort_inventory,
            verify_compositional_semantic_replications,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        source_paths = _named_paths(args.source_bundle)
        specs = _replication_specs(args.replication)
        if any(spec["family"] not in source_paths for spec in specs):
            raise ValueError("fresh replication family has no source bundle")
        source_inventories = {}
        source_manifest_sha256s = {}
        for family, path in sorted(source_paths.items()):
            bundle = load_standard_semantic_feature_bundle(path)
            source_manifest_sha256s[family] = bundle.manifest["manifest_sha256"]
            if any(spec["family"] == family for spec in specs):
                source_inventories[family] = semantic_cohort_inventory(bundle)
            del bundle
            gc.collect()
        stored_model, model_file_sha256 = _load_json(
            args.model,
            max_bytes=32 * 1024 * 1024,
        )
        source_report, source_report_file_sha256 = _load_json(
            args.source_report,
            max_bytes=16 * 1024 * 1024,
        )
        stored_file_sha256s = {
            "model": model_file_sha256,
            "source_report": source_report_file_sha256,
        }
        cohorts = []
        for spec in specs:
            _verify_commit(spec["commit"])
            fresh_bundle = load_standard_semantic_feature_bundle(spec["fresh"])
            fresh_inventory = semantic_cohort_inventory(fresh_bundle)
            del fresh_bundle
            gc.collect()
            report, report_file_sha256 = _load_json(
                spec["report"],
                max_bytes=16 * 1024 * 1024,
            )
            stored_file_sha256s[f"{spec['family']}_replication_report"] = (
                report_file_sha256
            )
            cohorts.append(
                CompositionalReplicationCohort(
                    family=spec["family"],
                    source=source_inventories[spec["family"]],
                    fresh=fresh_inventory,
                    lesion_arm=spec["lesion"],
                    report=report,
                    transfer_kind=spec["transfer_kind"],
                    evaluation_source_commit=spec["commit"],
                )
            )
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES
        }
        verification = verify_compositional_semantic_replications(
            source_manifest_sha256s=source_manifest_sha256s,
            trained_model_payload=stored_model,
            source_report=source_report,
            cohorts=cohorts,
            source_sha256s=source_sha256s,
            stored_file_sha256s=stored_file_sha256s,
        )
        output = args.output.expanduser().resolve()
        if not atomic_write_bytes_if_absent(
            output,
            _output_bytes(verification),
            mode=0o400,
        ):
            raise FileExistsError("compositional verification output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_compositional_verifier_cli.v1",
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
                "schema": "aura.semantic_program_compositional_verifier_cli.v1",
                "verified": True,
                "verification_sha256": verification["verification_sha256"],
                "cohorts": {
                    cohort["family"]: {
                        "held_out_total": cohort["held_out_total"],
                        "lesion_arm": cohort["lesion_arm"],
                    }
                    for cohort in verification["cohorts"]
                },
                "output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
