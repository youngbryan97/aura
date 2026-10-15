"""Measure a supplied static reference against captured RGB pixels, without a model.

This is an offline appearance probe. Its output cannot qualify a task role,
controlled identity, successful action or presentation-ready play.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _image(path: Path, *, reference: bool) -> np.ndarray:
    with Image.open(path) as image:
        limit = 512 if reference else 4096
        if min(image.size) < 16 or max(image.size) > limit or image.width*image.height > 8_000_000:
            raise ValueError('image dimensions exceed the correspondence budget')
        return np.asarray(image.convert('RGBA' if reference else 'RGB'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--picture', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--minimum-scale', type=float, default=.5)
    parser.add_argument('--maximum-scale', type=float, default=4.)
    parser.add_argument('--scale-count', type=int, default=32)
    args = parser.parse_args(argv)
    if (not all(math.isfinite(v) for v in (args.minimum_scale, args.maximum_scale))
            or not .05 <= args.minimum_scale <= args.maximum_scale <= 8 or not 1 <= args.scale_count <= 32):
        parser.error('scales need finite positive ordered bounds and at most 32 samples')
    if args.out.exists():
        parser.error('use a fresh output path')
    from core.governance_context import local_internal_governed_scope
    from core.perception.a_visual_reference_in_the_picture import reference_appearances
    from core.runtime.file_write_gateway import get_file_write_gateway

    picture, reference = _image(args.picture, reference=False), _image(args.reference, reference=True)
    scales = tuple(map(float, np.geomspace(args.minimum_scale, args.maximum_scale, args.scale_count)))
    began = time.perf_counter()
    poses = reference_appearances(picture, reference, scales=scales)
    duration = time.perf_counter()-began
    record = dict(schema='aura.supplied_visual_reference.v1',
        picture=str(args.picture.resolve()), reference=str(args.reference.resolve()),
        picture_file_sha256=hashlib.sha256(args.picture.read_bytes()).hexdigest(),
        reference_file_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256((ROOT/'core/perception/a_visual_reference_in_the_picture.py').read_bytes()).hexdigest(),
        scales=list(scales), seconds=duration, poses=[asdict(pose) for pose in poses],
        presentation_qualified=False, task_passed=None,
        evidence_scope='static reference appearance in captured pixels; no task role, temporal identity or input effect')
    with local_internal_governed_scope('supplied_visual_reference_probe'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(args.out.parent, source='supplied_visual_reference_probe')
        if not gateway.write_bytes_if_absent(args.out, json.dumps(record, indent=2).encode(),
                                            source='supplied_visual_reference_probe'):
            raise ValueError('another probe owns this output')
    print(json.dumps(dict(appearances=len(poses), points=[len(p.source_points) for p in poses],
                          seconds=round(duration, 3), task_passed=None, presentation_qualified=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
