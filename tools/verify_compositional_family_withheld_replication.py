#!/usr/bin/env python3
"""Independently verify whole-family-withheld semantic transfer."""

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
    if len(result) < 3:
        raise ValueError("family-withheld verification needs at least three families")
    return result


def _verify_commit(commit: str) -> None:
    if not _COMMIT.fullmatch(commit):
        raise ValueError("evaluation source commit is invalid")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise ValueError("evaluation source commit is not an ancestor")


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
    parser.add_argument("--fresh-bundle", type=Path, required=True)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--fresh-report", type=Path, required=True)
    parser.add_argument("--lesion-arm", required=True)
    parser.add_argument("--evaluation-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_compositional_verification import (
            COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES,
            semantic_cohort_inventory,
            verify_compositional_family_withheld_replication,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        _verify_commit(args.evaluation_source_commit)
        source_paths = _named_paths(args.source_bundle)
        source_inventories = {}
        for family, path in sorted(source_paths.items()):
            bundle = load_standard_semantic_feature_bundle(path)
            source_inventories[family] = semantic_cohort_inventory(bundle)
            del bundle
            gc.collect()
        fresh_bundle = load_standard_semantic_feature_bundle(
            args.fresh_bundle.expanduser().resolve(strict=True)
        )
        fresh_inventory = semantic_cohort_inventory(fresh_bundle)
        del fresh_bundle
        gc.collect()
        model, model_file_sha256 = _load_json(args.transducer, max_bytes=32 * 1024 * 1024)
        source_report, source_report_file_sha256 = _load_json(
            args.source_report,
            max_bytes=32 * 1024 * 1024,
        )
        fresh_report, fresh_report_file_sha256 = _load_json(
            args.fresh_report,
            max_bytes=32 * 1024 * 1024,
        )
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in COMPOSITIONAL_REPLICATION_VERIFICATION_SOURCES
        }
        verification = verify_compositional_family_withheld_replication(
            source_inventories=source_inventories,
            fresh_inventory=fresh_inventory,
            trained_model_payload=model,
            source_report=source_report,
            fresh_report=fresh_report,
            lesion_arm=args.lesion_arm,
            evaluation_source_commit=args.evaluation_source_commit,
            source_sha256s=source_sha256s,
            stored_file_sha256s={
                "transducer": model_file_sha256,
                "source_report": source_report_file_sha256,
                "fresh_report": fresh_report_file_sha256,
            },
        )
        output = args.output.expanduser().resolve()
        if not atomic_write_bytes_if_absent(
            output,
            _output_bytes(verification),
            mode=0o400,
        ):
            raise FileExistsError("family-withheld verification output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_family_withheld_verifier_cli.v1",
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
                "schema": "aura.semantic_program_family_withheld_verifier_cli.v1",
                "verified": True,
                "held_out_family": verification["held_out_family"],
                "verification_sha256": verification["verification_sha256"],
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
