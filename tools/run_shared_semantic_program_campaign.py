#!/usr/bin/env python3
"""Fit and adjudicate one variable-geometry semantic cortex across families."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FAMILY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


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
        raise ValueError("shared semantic campaign needs at least two families")
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
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from mlx_lm.utils import load_tokenizer

        from core.learning.semantic_input_grounding import (
            semantic_input_grounding_contract_from_tokenizer,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.learning.semantic_program_shared_campaign import (
            run_shared_semantic_program_campaign,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        bundle_paths = _bundle_arguments(args.bundle)
        model_path = args.model.expanduser().resolve(strict=True)
        outputs = (
            args.model_output.expanduser().resolve(),
            args.report_output.expanduser().resolve(),
        )
        if any(path.exists() for path in outputs):
            raise FileExistsError("shared semantic campaign output already exists")
        bundles = {
            family: load_standard_semantic_feature_bundle(path)
            for family, path in bundle_paths.items()
        }
        declared_models = {
            Path(bundle.manifest["exact_model_path"]).resolve(strict=True)
            for bundle in bundles.values()
        }
        if declared_models != {model_path}:
            raise ValueError("shared semantic bundles name a different checkpoint")
        tokenizer_identities = {
            bundle.manifest["tokenizer_identity"]["identity_sha256"]
            for bundle in bundles.values()
        }
        if len(tokenizer_identities) != 1:
            raise ValueError("shared semantic bundle tokenizers differ")
        tokenizer = load_tokenizer(model_path)
        input_grounding = semantic_input_grounding_contract_from_tokenizer(
            tokenizer,
            tokenizer_identity_sha256=tokenizer_identities.pop(),
        )
        result = run_shared_semantic_program_campaign(
            bundles,
            input_grounding=input_grounding,
        )
        payloads = (
            _canonical_bytes(result.model.to_dict()),
            _canonical_bytes(result.report),
        )
        for path, payload in zip(outputs, payloads, strict=True):
            if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
                raise FileExistsError(f"shared semantic campaign output raced: {path}")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_shared_campaign_cli.v1",
                    "completed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    test = result.report["arms"]["treatment:test"]
    print(
        json.dumps(
            {
                "schema": "aura.semantic_program_shared_campaign_cli.v1",
                "completed": True,
                "report_sha256": result.report["report_sha256"],
                "transducer_receipt_sha256": result.model.receipt_sha256,
                "test_program_exact": test["program_exact"],
                "test_answer_exact": test["answer_exact"],
                "test_total": test["total"],
                "model_output": str(outputs[0]),
                "report_output": str(outputs[1]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
