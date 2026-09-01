#!/usr/bin/env python3
"""Train and adjudicate the semantic transducer from a verified feature bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.governance_context import local_internal_governed_scope
        from core.learning.semantic_program_campaign import (
            run_semantic_program_campaign,
        )
        from core.learning.semantic_program_corpus import build_semantic_program_corpus
        from core.learning.semantic_program_feature_materialization import (
            SemanticFeatureConfig,
            load_semantic_feature_bundle,
            select_bounded_semantic_examples,
        )
        from core.runtime.file_write_gateway import get_file_write_gateway

        config = SemanticFeatureConfig()
        corpus = build_semantic_program_corpus(
            seed=config.seed,
            examples_per_operation_pair=config.examples_per_operation_pair,
        )
        expected = select_bounded_semantic_examples(
            corpus,
            max_examples=config.max_examples,
        )
        bundle = load_semantic_feature_bundle(args.bundle, expected_examples=expected)
        result = run_semantic_program_campaign(bundle)
        gateway = get_file_write_gateway()
        with local_internal_governed_scope(
            "semantic_program_campaign.outputs",
            domain="file_write",
        ):
            gateway.write_bytes(
                args.model_output,
                _canonical_bytes(result.model.to_dict()),
                source="semantic_program_campaign.model",
            )
            gateway.write_bytes(
                args.report_output,
                _canonical_bytes(result.report),
                source="semantic_program_campaign.report",
            )
    except Exception as exc:  # noqa: BLE001 - CLI reports exact terminal failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_campaign_cli.v2",
                    "complete": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    summary = {
        "schema": "aura.semantic_program_campaign_cli.v2",
        "complete": True,
        "report_sha256": result.report["report_sha256"],
        "transducer_receipt_sha256": result.model.receipt_sha256,
        "held_out_treatment_program_exact": result.report[
            "held_out_treatment_program_exact"
        ],
        "held_out_treatment_answer_exact": result.report[
            "held_out_treatment_answer_exact"
        ],
        "held_out_total": result.report["held_out_total"],
        "model_output": str(args.model_output.expanduser().resolve()),
        "report_output": str(args.report_output.expanduser().resolve()),
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
