#!/usr/bin/env python3
"""Evaluate a frozen learned program's reuse on new public input values."""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples
    configure_refit_environment(args.output)
    if args.output.exists():
        raise FileExistsError(args.output)
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.learning.semantic_procedure_reuse_evaluation import evaluate_learned_procedure_reuse
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    model = compositional_semantic_program_transducer_from_dict(json.loads(args.transducer.read_text("ascii")))
    report = json.loads(args.source_report.read_text("ascii"))
    examples = load_source_examples(model, report, args.bundle)
    result = evaluate_learned_procedure_reuse(model, examples, split=args.split,
        probe_count=args.probe_count, seed=args.seed,
        progress=lambda row: print(json.dumps(row, sort_keys=True), flush=True))
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
    if not atomic_write_bytes_if_absent(args.output, payload, mode=0o400):
        raise FileExistsError(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
