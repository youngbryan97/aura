"""Inspect named source methods without starting a browser or executing source."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inspect(raw: bytes, targets: tuple[str, ...]) -> dict:
    from core.perception.flash_action_sources import read_flash_action_streams
    from core.perception.guarded_effects_in_flash_methods import read_flash_effect_graph
    from core.perception.symbolic_motion_in_flash_methods import bind_flash_methods, read_flash_method

    if not targets or len(targets) > 16 or any(not isinstance(name, str) or len(name) > 256 for name in targets):
        raise ValueError('one to sixteen bounded method names are required')
    started = time.monotonic()
    source = read_flash_action_streams(raw)
    bindings = bind_flash_methods(source.streams) if source.complete else None
    findings = []
    for target in targets:
        binding = bindings.method(target) if bindings is not None else None
        if binding is None:
            findings.append({'target': target, 'graph': None, 'issue': 'no unique complete source binding'})
            continue
        graph = read_flash_effect_graph(binding)
        paths = read_flash_method(binding)
        findings.append({'target': target, 'graph': asdict(graph), 'complete': graph.complete,
                         'effects_composed': graph.effects_composed,
                         'instructions': len(binding.function.body.instructions), 'nodes': len(graph.nodes),
                         'lifted_addresses': len(graph.lifted), 'reachable_addresses': len(graph.reachable),
                         'effects': dict(Counter(effect.operation for effect in graph.effects)),
                         'branches': len(graph.branches), 'exits': len(graph.exits),
                         'path_reader': {'paths': len(paths.paths), 'complete': paths.complete,
                                         'issues': paths.issues}})
    return {'schema': 'aura.guarded_source_effects.v1', 'source': source.source,
            'source_complete': source.complete, 'source_issues': [asdict(issue) for issue in source.issues],
            'binding_issues': bindings.issues if bindings is not None else (), 'findings': findings,
            'seconds': time.monotonic() - started, 'source_executed': False, 'task_outcome': None,
            'presentation_qualified': False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--method', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    result = inspect(args.source.read_bytes(), tuple(args.method))
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('source_effect_inspection'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='source_effect_inspection')
        gateway.write_text(args.out, json.dumps(result, indent=2) + '\n', source='source_effect_inspection')
    summary = {**result, 'findings': [{key: value for key, value in finding.items() if key != 'graph'}
                                     for finding in result['findings']]}
    print(json.dumps(summary, indent=2))
    return 0 if result['source_complete'] and all(finding.get('complete') for finding in result['findings']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
