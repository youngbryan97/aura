"""Render one declared Flash asset with an explicitly supplied external renderer.

The receipt links rendered bytes to immutable source bytes and declaration
addresses. Rendering supplies reference artwork, without a current screen,
task role, controlled identity or gameplay result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_REFERENCE_FILES = 256
MAX_RENDERED_BYTES = 64 * 1024 * 1024
_EXPORT = {'sprite': 'sprite', 'button': 'button', 'shape': 'shape', 'bitmap': 'image'}


def _sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inventory(directory: Path, character: int, *, frame: int | None = None) -> list[dict]:
    """Only bounded PNG output inside the selected character's directory qualifies."""
    files, total, found = [], 0, 0
    for path in directory.rglob('*'):
        found += 1
        if found > 2048 or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('renderer output exceeds its path budget or leaves its directory')
        if not path.is_file():
            continue
        total += path.stat().st_size
        if total > MAX_RENDERED_BYTES:
            raise ValueError('renderer output exceeds its byte budget')
        if path.suffix.lower() != '.png':
            raise ValueError('renderer produced an unexpected non-PNG file')
        relative = path.relative_to(directory)
        if not re.match(rf'Define[A-Za-z0-9]+_{character}(?:_|\.|$)', relative.parts[0]):
            raise ValueError('rendered image does not identify the selected dictionary character')
        if frame is not None and path.stem != str(frame):
            raise ValueError('rendered image does not identify the selected source frame')
        if len(files) >= MAX_REFERENCE_FILES:
            raise ValueError('renderer output exceeds its image budget')
        with Image.open(path) as image:
            if (image.format != 'PNG' or min(image.size) < 1 or max(image.size) > 4096
                    or image.width*image.height > 8_000_000):
                raise ValueError('rendered image dimensions exceed the reference budget')
            image.verify()
            size = image.size
        files.append(dict(path=str(relative), file_sha256=_sha(path), size=list(size)))
    if not files:
        raise ValueError('renderer supplied no qualifying reference images')
    return sorted(files, key=lambda item: item['path'])


def _render_command(java: Path, jar: Path, source: Path, directory: Path, home: Path, asset: object,
                    *, frame: int = 1) -> list[str]:
    kind, character = asset.kind, asset.character_id
    if type(frame) is not int or not 1 <= frame <= 4096 or kind != 'sprite' and frame != 1:
        raise ValueError('the reference frame must be a bounded supported integer')
    return [str(java), '-Xmx512m', '-XX:ActiveProcessorCount=1', '-Djava.awt.headless=true',
            '-Dsun.java2d.opengl=false', '-Duser.home='+str(home), '-jar', str(jar),
            '-onerror', 'abort', '-exportTimeout', '45', '-selectid', str(character),
            '-select', f'{character}:{frame}', '-sublength', '1', '-ignorebackground',
            '-format', f'{_EXPORT[kind]}:png', '-export', _EXPORT[kind], str(directory), str(source)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--name', required=True, help='exact exported name, treated as data')
    parser.add_argument('--frame', type=int, default=1, help='one-based source visual frame')
    parser.add_argument('--java', type=Path, required=True)
    parser.add_argument('--ffdec', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='fresh private output directory')
    parser.add_argument('--with-registration', action='store_true', help='measure source coordinates for one sprite or button frame')
    parser.add_argument('--publish', action='store_true', help='index a registered export for ordinary play under this runtime state root')
    args = parser.parse_args(argv)
    if args.publish and not args.with_registration:
        parser.error('publishing requires independent source-coordinate registration')
    from core.governance_context import local_internal_governed_scope
    from core.perception.reading_a_flash_program import MOST_UNPACKED
    from core.perception.visual_assets_in_a_flash_file import read_flash_visual_assets
    from core.perception.visual_frames_in_a_source_declaration import read_declared_visual_frames
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.run_proof_step import main as run_step

    program, java, jar, out = (path.expanduser().resolve() for path in (args.program, args.java, args.ffdec, args.out))
    if not program.is_file() or program.stat().st_size > MOST_UNPACKED:
        parser.error('program must be a bounded local Flash file')
    raw = program.read_bytes()
    table = read_flash_visual_assets(raw)
    asset = table.named(args.name)
    if asset is None or asset.kind not in _EXPORT:
        parser.error('a unique locally declared sprite, button, shape or bitmap is required: '+str(table.issues))
    timeline = read_declared_visual_frames(raw, asset)
    if not timeline.complete or not 1 <= args.frame <= timeline.frame_count:
        parser.error('the selected reference frame is not in a complete source timeline: '+str(timeline.issues))
    if not java.is_file() or not jar.is_file() or out.exists():
        parser.error('supply existing Java and FFDec files and a fresh output directory')
    source, export, home = out/'source.swf', out/'rendered', out/'renderer-home'
    command = _render_command(java, jar, source, export, home, asset, frame=args.frame)
    record = dict(schema='aura.program_visual_reference.v1', source=table.source, declaration=asdict(asset),
                  renderer_jar=str(jar), renderer_jar_sha256=_sha(jar), java=str(java), java_sha256=_sha(java),
                  implementation_sha256=_sha(Path(__file__)), command=command, frame_number=args.frame,
                  declared_frame_count=timeline.frame_count,
                  status='prepared', images=[], task_passed=None, presentation_qualified=False,
                  evidence_scope='external static artwork export; no observed task role or input effect')
    with local_internal_governed_scope('export_program_visual_reference'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(out, source='export_program_visual_reference')
        if not gateway.write_bytes_if_absent(out/'receipt.json', json.dumps(record, indent=2).encode(),
                                            source='export_program_visual_reference'):
            raise ValueError('another export owns this output')
        gateway.ensure_directory(export, source='export_program_visual_reference')
        gateway.ensure_directory(home, source='export_program_visual_reference')
        if not gateway.write_bytes_if_absent(source, raw, mode=0o400, source='export_program_visual_reference'):
            raise ValueError('another export owns the staged source')
        try:
            code = run_step(['--name', 'static_visual_reference', '--timeout', '60',
                             '--artifact', str(out/'supervisor.json'), '--', *command])
            if code:
                raise ValueError(f'external renderer failed with status {code}')
            if ('swf:'+_sha(source) != table.source or _sha(jar) != record['renderer_jar_sha256']
                    or _sha(java) != record['java_sha256']):
                raise ValueError('source or renderer changed during export')
            images = _inventory(export, asset.character_id, frame=args.frame if asset.kind == 'sprite' else None)
            if args.with_registration:
                from tools.reference_frame_registration import register_export_frame

                record['registration'] = register_export_frame(java, jar, source, out, asset, images, table.source,
                                                               frame=args.frame)
                if ('swf:'+_sha(source) != table.source or _sha(jar) != record['renderer_jar_sha256']
                        or _sha(java) != record['java_sha256']):
                    raise ValueError('source or renderer changed during coordinate acquisition')
            record.update(status='rendered', images=images)
        except (ValueError, OSError) as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        gateway.write_text(out/'receipt.json', json.dumps(record, indent=2)+'\n', source='export_program_visual_reference')
    if args.publish and record['status'] == 'rendered':
        from tools.publish_program_visual_reference import publish_reference_bundle

        publish_reference_bundle(out)
    print(json.dumps(dict(status=record['status'], images=len(record['images']), source=table.source,
                          character_id=asset.character_id, task_passed=None, presentation_qualified=False)))
    return 0 if record['status'] == 'rendered' else 1


if __name__ == '__main__':
    raise SystemExit(main())
