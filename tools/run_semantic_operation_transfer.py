#!/usr/bin/env python3
"""Measure cross-family semantic operation transfer from verified bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arithmetic", type=Path, required=True)
    parser.add_argument("--fork-join", type=Path, required=True)
    parser.add_argument("--sequence-binary", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_operation_transfer import (
            run_semantic_operation_transfer,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        bundles = {
            "arithmetic": load_standard_semantic_feature_bundle(args.arithmetic),
            "fork_join": load_standard_semantic_feature_bundle(args.fork_join),
            "sequence_binary": load_standard_semantic_feature_bundle(args.sequence_binary),
        }
        report = run_semantic_operation_transfer(bundles)
        with local_internal_governed_scope(
            "semantic_operation_transfer.report",
            domain="file_write",
        ):
            get_file_write_gateway().write_bytes(
                args.report_output,
                json.dumps(
                    report,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii"),
                source="semantic_operation_transfer.report",
            )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_operation_transfer_cli.v2",
                    "complete": False,
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
                "schema": "aura.semantic_operation_transfer_cli.v2",
                "complete": True,
                "report_sha256": report["report_sha256"],
                "representation_views": {
                    view_name: {
                        "counterfactual_target_batch_required": view[
                            "counterfactual_target_batch_required"
                        ],
                        "directions": {
                            name: {
                                split: {
                                    "program_exact": result["arms"]["treatment"][
                                        "program_exact"
                                    ],
                                    "program_total": result["program_count"],
                                    "surface_overlap_count": result[
                                        "surface_overlap_count"
                                    ],
                                }
                                for split, result in direction["splits"].items()
                            }
                            for name, direction in view["directions"].items()
                        }
                    }
                    for view_name, view in report["representation_views"].items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
