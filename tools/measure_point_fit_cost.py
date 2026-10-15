"""Compare point-fit revisions on generated pixels without a browser or model.

This is a CPU measurement on a generated scene. It cannot establish live
latency, body identity or gameplay success. The baseline function comes
from a local Git revision; the rest of the observation path stays current.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def measure(baseline: str, pictures: int) -> dict:
    from core.perception import points_before_the_picture_is_reduced as source
    from core.perception.what_moves_in_the_picture import WhatMoves, prepare_play_vision
    from tests.test_source_points_keep_their_visible_detail import render

    path = 'core/perception/points_before_the_picture_is_reduced.py'
    code = subprocess.check_output(['git', 'show', f'{baseline}:{path}'], cwd=ROOT, text=True)
    tree = ast.parse(code)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_refine')
    namespace = dict(source.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), f'{baseline}:{path}', 'exec'), namespace)
    previous, current = namespace['_refine'], source._refine
    frames = [render(center=(150+20*math.sin(i*.12), 100+10*math.cos(i*.13)), angle=.12*math.sin(i*.2))
              for i in range(pictures)]
    prepare_play_vision()
    runs = []
    try:
        for mode in ('before', 'after', 'after', 'before', 'before', 'after'):
            source._refine = previous if mode == 'before' else current
            observer, durations, retained = WhatMoves(), [], 0
            for i, frame in enumerate(frames):
                began = time.perf_counter()
                observer.see(frame, 1+i*.1)
                durations.append(time.perf_counter()-began)
                retained += sum(p.pictures >= 2 for p in observer.points.points)
            runs.append({'mode': mode, 'median_s': statistics.median(durations[4:]),
                         'total_s': sum(durations), 'retained_point_observations': retained})
    finally:
        source._refine = current
    return {'scope': 'generated pixels, CPU only; no live latency or original-game proof',
            'baseline': baseline, 'pictures_per_run': pictures, 'runs': runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='local Git revision supplying only the earlier patch fit')
    parser.add_argument('--pictures', type=int, default=40)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 8 <= args.pictures <= 100:
        parser.error('pictures must be between eight and one hundred')
    if args.out.exists():
        parser.error('use a fresh output path')
    record = measure(args.baseline, args.pictures)
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope('point_fit_measurement'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='point_fit_measurement')
        gateway.write_text(args.out, json.dumps(record, indent=2), source='point_fit_measurement')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
