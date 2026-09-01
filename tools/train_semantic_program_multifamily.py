#!/usr/bin/env python3
"""Train one semantic transducer from several named verified feature bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or name in result or not path:
            raise ValueError("bundles must be unique NAME=PATH values")
        result[name] = Path(path)
    if len(result) < 2:
        raise ValueError("multi-family training requires at least two bundles")
    return result


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_multifamily import (
            run_semantic_program_multifamily_campaign,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        paths = _named_paths(args.bundle)
        bundles = {
            name: load_standard_semantic_feature_bundle(path)
            for name, path in paths.items()
        }
        result = run_semantic_program_multifamily_campaign(bundles)
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "semantic_program_multifamily.outputs",
            domain="file_write",
        ):
            gateway.write_bytes(
                args.model_output,
                _bytes(result.model.to_dict()),
                source="semantic_program_multifamily.model",
            )
            gateway.write_bytes(
                args.report_output,
                _bytes(result.report),
                source="semantic_program_multifamily.report",
            )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_multifamily_cli.v1",
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
                "schema": "aura.semantic_program_multifamily_cli.v1",
                "complete": True,
                "report_sha256": result.report["report_sha256"],
                "transducer_receipt_sha256": result.model.receipt_sha256,
                "families": {
                    name: {
                        "answer_exact": report["held_out_treatment_answer_exact"],
                        "program_exact": report["held_out_treatment_program_exact"],
                        "total": report["held_out_total"],
                    }
                    for name, report in result.report["families"].items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
