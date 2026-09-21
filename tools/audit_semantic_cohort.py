#!/usr/bin/env python3
"""Audit the full parent-bound development cohort without fitting a candidate."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--bundle', action='append', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--observe-only', action='store_true',
                        help='measure ordinary decisions without searching diagnostic alternatives')
    parser.add_argument('--solve-time-limit-s', type=float, default=3.)
    args = parser.parse_args()
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples
    configure_refit_environment(args.directory / 'report.json')
    from core.learning.semantic_cohort_diagnosis import audit_semantic_cohort
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict as restore
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    parent = restore(json.loads(args.parent.read_text()))
    candidate = restore(json.loads(args.candidate.read_text()))
    if (candidate.model_basis_sha256 != parent.model_basis_sha256
            or candidate.input_grounding != parent.input_grounding):
        raise ValueError('audit candidate representation differs from source parent')
    examples = load_source_examples(parent, json.loads(args.source_report.read_text()), args.bundle)
    development = tuple(item for item in examples if item.split in {'train', 'validation'})
    report = audit_semantic_cohort(candidate, development, directory=args.directory / 'rows',
        diagnose_failures=not args.observe_only, solve_time_limit_s=args.solve_time_limit_s,
        progress=lambda row: print(json.dumps(row), flush=True))
    path = args.directory / 'report.json'
    payload = (json.dumps(report, sort_keys=True) + '\n').encode()
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError('published cohort audit differs')
    print(json.dumps({'receipt': report['receipt_sha256'], 'stages': report['stages']}))


if __name__ == '__main__':
    main()
