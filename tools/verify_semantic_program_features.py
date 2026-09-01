#!/usr/bin/env python3
"""Verify a semantic-program feature bundle without loading model weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    try:
        from core.learning.semantic_program_corpus import build_semantic_program_corpus
        from core.learning.semantic_program_feature_materialization import (
            SemanticFeatureConfig,
            load_semantic_feature_bundle,
            select_bounded_semantic_examples,
        )

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
        result = {
            "schema": "aura.semantic_program_feature_verification.v1",
            "verified": True,
            "manifest_sha256": bundle.manifest["manifest_sha256"],
            "example_count": len(bundle.examples),
            "split_counts": bundle.manifest["split_counts"],
            "exact_model_path": bundle.manifest["exact_model_path"],
        }
    except Exception as exc:  # noqa: BLE001 - verifier reports terminal failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_feature_verification.v1",
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
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
