#!/usr/bin/env python3
"""Diagnose changed semantic rankings on explicit development source identities."""

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
    parser.add_argument('--source', action='append', required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--max-charts', type=int, default=2)
    parser.add_argument('--max-graphs', type=int, default=2)
    parser.add_argument('--solve-seconds', type=float, default=3.)
    args = parser.parse_args()
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples
    configure_refit_environment(args.directory / 'report.json')
    from core.learning.semantic_bank_replay import compare_semantic_candidate_banks
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict as restore
    from core.learning.semantic_validation_checkpoint import validation_identity, validation_implementation_identity
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent

    parent = restore(json.loads(args.parent.read_text()))
    candidate = restore(json.loads(args.candidate.read_text()))
    examples = load_source_examples(parent, json.loads(args.source_report.read_text()), args.bundle)
    selected = tuple(item for item in examples if item.ir.source_text_sha256 in args.source)
    if (len(args.source) != len(set(args.source)) or len(selected) != len(args.source)
            or any(item.split not in {'train', 'validation'} for item in selected)):
        raise ValueError('bank replay sources must be unique available development rows')
    implementation = validation_implementation_identity()
    options = dict(max_charts=args.max_charts, max_graphs_per_chart=args.max_graphs,
                   solve_time_limit_s=args.solve_seconds)
    identity = _sha({'observations': validation_identity({'parent': parent, 'candidate': candidate},
        selected, scoring='source_anchors_v2', implementation=implementation), 'options': options})
    rows = []
    for item in selected:
        path = args.directory / 'rows' / f'{item.ir.source_text_sha256}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if (row.get('identity') != identity
                    or row.get('receipt_sha256') != _sha({k: v for k, v in row.items() if k != 'receipt_sha256'})):
                raise ValueError('bank replay row identity differs')
        else:
            print(json.dumps({'stage': 'observation_started', 'source': item.ir.source_text_sha256}), flush=True)
            result = compare_semantic_candidate_banks(parent, candidate, item, **options,
                progress=lambda event: print(json.dumps(event), flush=True))
            body = {'identity': identity, 'result': result}
            row = {**body, 'receipt_sha256': _sha(body)}
            if not atomic_write_bytes_if_absent(path, (json.dumps(row, sort_keys=True)+'\n').encode(), mode=0o400):
                raise FileExistsError(path)
        rows.append(row)
        print(json.dumps({'stage': 'completed', 'completed': len(rows), 'total': len(selected)}), flush=True)
    if implementation != validation_implementation_identity():
        raise ValueError('bank replay implementation changed')
    body = {'schema': 'aura.semantic_bank_replay_cohort.v1', 'identity': identity,
            'implementation': implementation, 'rows': rows, 'test_examples_used': 0,
            'serving_authority': False, 'fresh_transfer_claim': False, 'learning_performed': False}
    receipt = {**body, 'receipt_sha256': _sha(body)}
    path = args.directory / 'report.json'
    payload = (json.dumps(receipt, sort_keys=True)+'\n').encode()
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400) and path.read_bytes() != payload:
        raise ValueError('published bank replay differs')
    print(json.dumps({'receipt_sha256': receipt['receipt_sha256'], 'observed': len(rows)}))


if __name__ == '__main__':
    main()
