#!/usr/bin/env python3
"""Independently replay a frozen variable-geometry semantic campaign."""

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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path, *, max_bytes: int) -> Any:
    from core.runtime.file_read_gateway import read_stable_bytes

    payload = read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=max_bytes)
    return json.loads(payload.decode("ascii"), object_pairs_hook=_strict_object)


def _bundle_arguments(values: list[str]) -> dict[str, Path]:
    bundles: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or not _FAMILY.fullmatch(family) or not raw_path:
            raise ValueError("bundle must be FAMILY=PATH with a stable lowercase family")
        if family in bundles:
            raise ValueError(f"bundle family repeats: {family}")
        bundles[family] = Path(raw_path).expanduser().resolve(strict=True)
    if len(bundles) < 2:
        raise ValueError("shared semantic verification needs at least two families")
    return bundles


def _canonical_bytes(value: object) -> bytes:
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
    parser.add_argument(
        "--bundle",
        action="append",
        required=True,
        metavar="FAMILY=PATH",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_shared_verification import (
            SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES,
            verify_shared_semantic_program_campaign,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        bundle_paths = _bundle_arguments(args.bundle)
        bundles = {
            family: load_standard_semantic_feature_bundle(path)
            for family, path in bundle_paths.items()
        }
        model = _load_json(args.model, max_bytes=32 * 1024 * 1024)
        report = _load_json(args.campaign_report, max_bytes=16 * 1024 * 1024)
        source_sha256s = {
            relative: hashlib.sha256((_REPO_ROOT / relative).read_bytes()).hexdigest()
            for relative in SEMANTIC_PROGRAM_SHARED_VERIFICATION_SOURCES
        }
        verification = verify_shared_semantic_program_campaign(
            bundles,
            stored_model_payload=model,
            stored_report=report,
            source_sha256s=source_sha256s,
        )
        output = args.verification_output.expanduser().resolve()
        if not atomic_write_bytes_if_absent(
            output,
            _canonical_bytes(verification),
            mode=0o400,
        ):
            raise FileExistsError("shared semantic verification output already exists")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_shared_verifier_cli.v1",
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
                "schema": "aura.semantic_program_shared_verifier_cli.v1",
                "verified": True,
                "verification_sha256": verification["verification_sha256"],
                "test_program_exact": verification["test_program_exact"],
                "test_answer_exact": verification["test_answer_exact"],
                "test_total": verification["test_total"],
                "verification_output": str(output),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
