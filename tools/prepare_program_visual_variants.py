"""Prepare missing finite sprite references with the supplied offline renderer.

Only assets already registered for this exact source are considered. One
bounded renderer runs at a time; source labels do not select goals or keys.
The output measures static reference coverage, without running the program.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def missing_variant_jobs(raw, catalog, *, limit: int = 32):
    from core.perception.visual_frames_in_a_source_declaration import read_declared_visual_frames

    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('a variant preparation needs a bounded job count')
    jobs, seen = [], set()
    for reference in catalog.references:
        key = reference.source, reference.declaration.character_id
        if key in seen:
            continue
        seen.add(key)
        timeline = read_declared_visual_frames(raw,reference.declaration)
        if not timeline.complete or timeline.source != reference.source:
            raise ValueError('variant preparation requires this current complete source timeline')
        available = {r.frame_number for r in catalog.references if (r.source,r.declaration.character_id) == key}
        for frame in range(1,timeline.frame_count+1):
            if frame not in available:
                jobs.append((reference.declaration,frame))
                if len(jobs) > limit:
                    raise ValueError('missing source variants exceed the admitted job count')
    return tuple(jobs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program',type=Path,required=True)
    parser.add_argument('--java',type=Path,required=True)
    parser.add_argument('--ffdec',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=32)
    args=parser.parse_args()
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from core.perception.visual_references_for_current_sources import acquire_visual_references, reference_variant_coverage
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.run_proof_step import main as run_step

    if not 8 <= args.program.stat().st_size <= 40*1024*1024 or args.out.exists():
        parser.error('a bounded source and fresh output directory are required')
    raw=args.program.read_bytes()
    table=read_flash_visual_assets(raw)
    catalog=acquire_visual_references((table,))
    jobs=missing_variant_jobs(raw,catalog,limit=args.limit)
    if not catalog.references:
        parser.error('no current registered asset supplies a variant family')
    with local_internal_governed_scope('prepare_visual_variants'):
        gateway=get_file_write_gateway()
        gateway.ensure_directory(args.out,source='prepare_visual_variants')
        plan=dict(source=table.source,jobs=[dict(character_id=a.character_id,frame=f) for a,f in jobs])
        if not gateway.write_bytes_if_absent(args.out/'preparation.json',json.dumps(plan).encode(),
                                             source='prepare_visual_variants'):
            raise ValueError('another preparation owns this output directory')
        completed,failed=[],[]
        for asset,frame in jobs:
            if args.program.stat().st_size != len(raw) or args.program.read_bytes() != raw:
                raise ValueError('the source changed during variant preparation')
            output=args.out/f'character-{asset.character_id}-frame-{frame}'
            status=run_step(['--name',f'visual_variant_{asset.character_id}_{frame}',
                '--timeout','150','--artifact',str(args.out/f'character-{asset.character_id}-frame-{frame}.json'),
                '--',sys.executable,str(Path(__file__).with_name('export_program_visual_reference.py')),
                '--program',str(args.program.resolve()),'--name='+asset.name,'--frame',str(frame),
                '--java',str(args.java.resolve()),'--ffdec',str(args.ffdec.resolve()),'--out',str(output.resolve()),
                '--with-registration','--publish'])
            if status:
                failed.append(dict(character_id=asset.character_id,frame=frame,status=status))
            else:
                completed.append(dict(character_id=asset.character_id,frame=frame))
        if args.program.stat().st_size != len(raw) or args.program.read_bytes() != raw:
            raise ValueError('the source changed during variant preparation')
        final=acquire_visual_references((table,))
        report=dict(source=table.source,prepared=len(completed),attempted=len(jobs),failed=failed,
            coverage=[asdict(c) for c in reference_variant_coverage(final.references)],
            unavailable=list(final.unavailable),native_play=False,demo_ready=False)
        gateway.write_text(args.out/'coverage.json',json.dumps(report,indent=2)+'\n',source='prepare_visual_variants')
    print(json.dumps(report))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
