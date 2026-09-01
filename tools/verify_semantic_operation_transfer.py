#!/usr/bin/env python3
"""Independently replay one semantic operation-transfer report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> Any:
    from core.runtime.file_read_gateway import read_stable_bytes

    return json.loads(
        read_stable_bytes(path.expanduser().resolve(strict=True), max_bytes=64 * 1024 * 1024),
        object_pairs_hook=_strict_object,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arithmetic", type=Path, required=True)
    parser.add_argument("--fork-join", type=Path, required=True)
    parser.add_argument("--sequence-binary", type=Path, required=True)
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--verification-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_operation_transfer_verification import (
            SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES,
            verify_semantic_operation_transfer,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.file_read_gateway import read_stable_bytes
        from core.runtime.file_write_gateway import get_file_write_gateway

        bundles = {
            "arithmetic": load_standard_semantic_feature_bundle(args.arithmetic),
            "fork_join": load_standard_semantic_feature_bundle(args.fork_join),
            "sequence_binary": load_standard_semantic_feature_bundle(
                args.sequence_binary
            ),
        }
        source_sha256s = {
            relative: hashlib.sha256(
                read_stable_bytes(_REPO_ROOT / relative, max_bytes=16 * 1024 * 1024)
            ).hexdigest()
            for relative in SEMANTIC_OPERATION_TRANSFER_VERIFICATION_SOURCES
        }
        verification = verify_semantic_operation_transfer(
            bundles,
            stored_report=_load(args.campaign_report),
            source_sha256s=source_sha256s,
        )
        with local_internal_governed_scope(
            "semantic_operation_transfer.verification",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                args.verification_output,
                json.dumps(
                    verification,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii"),
                source="semantic_operation_transfer.verification",
            )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_operation_transfer_verifier_cli.v1",
                    "verified": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "schema": "aura.semantic_operation_transfer_verifier_cli.v1",
                "verified": True,
                "verification_sha256": verification["verification_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
