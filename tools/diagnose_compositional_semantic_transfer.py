#!/usr/bin/env python3
"""Fit a compositional semantic model while withholding one complete family."""

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
    if len(bundles) < 3:
        raise ValueError("compositional held-family diagnosis needs at least three bundles")
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
    parser.add_argument("--bundle", action="append", required=True, metavar="FAMILY=PATH")
    parser.add_argument("--held-out-family", required=True)
    parser.add_argument(
        "--held-out-only",
        action="store_true",
        help="Evaluate only the held-out family after fitting every source family",
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
        from core.learning.semantic_program_compositional_campaign import (
            run_compositional_leave_family_out_campaign,
        )
        from core.learning.semantic_program_feature_materialization import (
            load_standard_semantic_feature_bundle,
        )
        from core.runtime.atomic_writer import atomic_write_bytes_if_absent

        bundle_paths = _bundle_arguments(args.bundle)
        if args.held_out_family not in bundle_paths:
            raise ValueError("held-out family must name one supplied bundle")
        model_path = args.model.expanduser().resolve(strict=True)
        outputs = (
            args.model_output.expanduser().resolve(),
            args.report_output.expanduser().resolve(),
        )
        if any(path.exists() for path in outputs):
            raise FileExistsError("compositional diagnostic output already exists")
        bundles = {
            family: load_standard_semantic_feature_bundle(path)
            for family, path in bundle_paths.items()
        }
        declared_models = {
            Path(bundle.manifest["exact_model_path"]).resolve(strict=True)
            for bundle in bundles.values()
        }
        if declared_models != {model_path}:
            raise ValueError("compositional diagnostic bundles name a different checkpoint")
        tokenizer_identities = {
            bundle.manifest["tokenizer_identity"]["identity_sha256"]
            for bundle in bundles.values()
        }
        if len(tokenizer_identities) != 1:
            raise ValueError("compositional diagnostic bundle tokenizers differ")
        tokenizer = load_tokenizer(model_path)
        input_grounding = semantic_input_grounding_contract_from_tokenizer(
            tokenizer,
            tokenizer_identity_sha256=tokenizer_identities.pop(),
        )
        result = run_compositional_leave_family_out_campaign(
            bundles,
            held_out_family=args.held_out_family,
            input_grounding=input_grounding,
            evaluation_families=(args.held_out_family,) if args.held_out_only else None,
        )
        for path, payload in zip(
            outputs,
            (_canonical_bytes(result.model.to_dict()), _canonical_bytes(result.report)),
            strict=True,
        ):
            if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
                raise FileExistsError(f"compositional diagnostic output raced: {path}")
    except Exception as exc:  # noqa: BLE001 - terminal CLI reports exact failure
        print(
            json.dumps(
                {
                    "schema": "aura.semantic_program_compositional_diagnostic_cli.v1",
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
    held_out = result.report["families"][args.held_out_family]
    print(
        json.dumps(
            {
                "schema": "aura.semantic_program_compositional_diagnostic_cli.v1",
                "completed": True,
                "held_out_family": args.held_out_family,
                "held_out_program_exact": held_out["held_out_program_exact"],
                "held_out_argument_exact": held_out["held_out_argument_exact"],
                "held_out_answer_exact": held_out["held_out_answer_exact"],
                "held_out_total": held_out["held_out_total"],
                "evaluated_families": result.report["evaluated_families"],
                "report_sha256": result.report["report_sha256"],
                "transducer_receipt_sha256": result.model.receipt_sha256,
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
