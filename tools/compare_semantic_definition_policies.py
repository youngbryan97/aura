#!/usr/bin/env python3
"""Compare complete decodes under definition policies on frozen source splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _decode(model, item, *, seconds: float) -> dict:
    outcome = model.decode(
        source_token_ids=item.ir.source_token_ids,
        hidden_states=item.hidden_states,
        public_inputs=item.public_inputs,
        source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=model.model_basis_sha256,
        search_time_limit_s=seconds,
    )
    if outcome.ir is None:
        return {"decoded": False, "reason": outcome.reason,
                "program_exact": False, "public_value_correct": False}
    selected, target = outcome.ir.to_program(), item.ir.to_program()
    try:
        actual, expected = selected.run(item.public_inputs), target.run(item.public_inputs)
        correct = type(actual) is type(expected) and actual == expected
    except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
        correct = False
    return {"decoded": True, "reason": "", "program_exact": selected.sha() == target.sha(),
            "public_value_correct": correct, "program_sha256": selected.sha(),
            "program": selected.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "feature-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--construction", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--seconds", type=float, default=3.)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("search allowance must be positive")

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    source_raw = args.source_report.read_bytes()
    source_report = json.loads(source_raw)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]
               ["source_feature_manifest_sha256s"]]
    examples = load_source_examples(model, source_report, bundles)
    selected = sorted((item for item in examples
                       if item.split == args.split and item.construction_id == args.construction
                       and (not args.source or item.ir.source_text_sha256 in args.source)),
                      key=lambda item: item.ir.source_text_sha256)
    if not selected:
        raise ValueError("frozen construction has no examples")
    missing = set(args.source) - {item.ir.source_text_sha256 for item in selected}
    if missing:
        raise ValueError(f"requested source is absent from frozen split: {sorted(missing)}")
    alternate = model.with_bidirectional_input_definitions()
    rows = []
    for index, item in enumerate(selected):
        row = {"source": item.ir.source_text_sha256,
               "target_program": item.ir.to_program().to_dict(),
               "ordinary": _decode(model, item, seconds=args.seconds),
               "bidirectional": _decode(alternate, item, seconds=args.seconds)}
        rows.append(row)
        print(json.dumps({"index": index + 1, "population": len(selected),
                          "ordinary": row["ordinary"],
                          "bidirectional": row["bidirectional"]}), flush=True)
    summary = {arm: {key: sum(row[arm][key] for row in rows) for key in
                     ("decoded", "program_exact", "public_value_correct")}
               for arm in ("ordinary", "bidirectional")}
    body = {"schema": "aura.semantic_definition_policy_paired_v2",
            "serving_authority": False, "development_only": True,
            "split": args.split, "construction": args.construction,
            "search_seconds_per_arm": args.seconds,
            "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "transducer_sha256": hashlib.sha256(args.transducer.read_bytes()).hexdigest(),
            "rows": rows, "summary": summary}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    payload = json.dumps({**body, "receipt_sha256": digest}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("definition comparison output already differs")
    args.output.write_bytes(payload)
    print(json.dumps({"stage": "complete", "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
