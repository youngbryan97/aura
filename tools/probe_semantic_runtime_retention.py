#!/usr/bin/env python3
"""Run a small source-bound graph trial without exporting or promoting a model."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples


def report_progress(row):
    summary = {key: value for key, value in row.items()
               if value is None or isinstance(value, (str, int, float, bool))}
    detail = row.get("row", {})
    summary.update({key: detail[key] for key in ("source_text_sha256", "semantic_status", "status")
                    if key in detail})
    print(json.dumps(summary, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-count", type=int, default=8)
    parser.add_argument("--training-pool-count", type=int)
    parser.add_argument("--operation-retention-count", type=int)
    parser.add_argument("--validation-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-charts", type=int, default=32)
    parser.add_argument("--learn-operation-pointer", action="store_true")
    parser.add_argument("--update-rule", choices=("working_face", "minimum_change"), default="working_face")
    parser.add_argument("--boundary-policy", choices=("supervised", "retain_existing"), default="supervised")
    parser.add_argument("--objective", choices=("squared_deficit", "pairwise_logistic"), default="squared_deficit")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure_refit_environment(args.output)
    from core.learning.semantic_graph_trial import run_semantic_graph_trial
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    model = compositional_semantic_program_transducer_from_dict(json.loads(args.transducer.read_text("ascii")))
    report = json.loads(args.source_report.read_text("ascii"))
    examples = load_source_examples(model, report, args.bundle)
    result = run_semantic_graph_trial(model.with_joint_operation_argument_scores(), examples,
        training_count=args.training_count, validation_count=args.validation_count,
        training_pool_count=args.training_pool_count,
        operation_retention_count=args.operation_retention_count,
        steps=args.steps, max_charts=args.max_charts,
        objective=args.objective,
        update_rule=args.update_rule,
        boundary_policy=args.boundary_policy,
        learn_operation_pointer=args.learn_operation_pointer,
        progress=report_progress)
    if not atomic_write_bytes_if_absent(args.output,
            (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"), mode=0o400):
        raise FileExistsError(args.output)
    print(json.dumps({key: result[key] for key in ("summaries", "blockers", "receipt_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
