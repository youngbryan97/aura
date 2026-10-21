"""Read a declared sprite's finite visual frames without executing its timeline.

A frame selects reference artwork. Its label does not identify a task role,
and a matching picture does not establish the currently executing frame.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from core.perception.flash_action_sources import MAX_TAGS, _strict_body, _u16, _u32
from core.perception.reading_a_flash_program import MOST_UNPACKED
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset, read_flash_visual_assets
from core.verify import invariant

MAX_VISUAL_FRAMES = 4096


@dataclass(frozen=True)
class DeclaredVisualFrame:
    number: int
    show_offset: int
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeclaredVisualTimeline:
    source: str
    character_id: int | None
    frame_count: int | None = None
    frames: tuple[DeclaredVisualFrame, ...] = ()
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.issues and self.frame_count == len(self.frames) and bool(self.frames)


def _tag(data: bytes, pos: int, end: int) -> tuple[int, int, int]:
    header = _u16(data, pos, end)
    code, size, start = header >> 6, header & 63, pos+2
    if size == 63:
        size, start = _u32(data, start, end), start+4
    stop = start+size
    if stop > end:
        raise ValueError('visual timeline tag exceeds its container')
    return code, start, stop


def read_declared_visual_frames(raw: bytes, asset: DeclaredVisualAsset) -> DeclaredVisualTimeline:
    """Require the exact current declaration and every declared ShowFrame."""
    bounded = isinstance(raw, bytes) and 8 <= len(raw) <= MOST_UNPACKED
    source = 'swf:'+hashlib.sha256(raw).hexdigest() if bounded else ''
    character = getattr(asset, 'character_id', None)
    try:
        if (not bounded or not isinstance(asset, DeclaredVisualAsset)
                or type(asset.character_id) is not int or type(asset.definition_offset) is not int):
            raise ValueError('immutable source bytes and their declaration are required')
        table = read_flash_visual_assets(raw)
        if not table.complete or table.named(asset.name) != asset:
            raise ValueError('visual frame declaration is not in the current complete source')
        if asset.kind != 'sprite':
            if asset.kind not in ('button', 'bitmap', 'shape'):
                raise ValueError('this declaration has no supported visual reference frame')
            return DeclaredVisualTimeline(source, character, 1, (DeclaredVisualFrame(1, asset.definition_offset),))
        version, data = _strict_body(raw)
        code, start, end = _tag(data, asset.definition_offset, len(data))
        if code != 39 or _u16(data, start, end) != character:
            raise ValueError('visual frame definition differs from its dictionary address')
        count, pos = _u16(data, start+2, end), start+4
        if not 1 <= count <= MAX_VISUAL_FRAMES:
            raise ValueError('declared visual frame count exceeds its budget')
        frames, labels, ended = [], [], False
        for _ in range(MAX_TAGS):
            offset = pos
            kind, at, stop = _tag(data, pos, end)
            if kind == 0:
                if at != stop or any(data[stop:end]):
                    raise ValueError('visual timeline has invalid trailing End bytes')
                ended = True
                break
            if kind == 1:
                if at != stop or len(frames) >= count:
                    raise ValueError('visual timeline has an invalid ShowFrame')
                frames.append(DeclaredVisualFrame(len(frames)+1, offset, tuple(labels)))
                labels = []
            elif kind == 43:
                finish = data.find(b'\0', at, min(stop, at+257))
                if (finish < at or stop-finish not in (1, 2)
                        or stop-finish == 2 and (version < 6 or data[stop-1] != 1) or len(labels) >= 32):
                    raise ValueError('visual frame label exceeds its format or budget')
                labels.append(data[at:finish].decode('utf-8' if version >= 6 else 'cp1252'))
            pos = stop
        if not ended or labels or len(frames) != count:
            raise ValueError('visual timeline does not contain every declared frame')
        return DeclaredVisualTimeline(source, character, count, tuple(frames))
    except (ValueError, IndexError, UnicodeError, TypeError) as exc:
        return DeclaredVisualTimeline(source, character, issues=(str(exc),))


@invariant('perception.visual_reference_frames_require_complete_source_timelines', scope='perception',
           owner='core/perception/visual_frames_in_a_source_declaration.py', observational=False)
def _visual_frame_custody_invariant() -> tuple:
    import struct

    unknown = read_declared_visual_frames(b'not a source', DeclaredVisualAsset('arbitrary', 7, 'sprite', 0, 39, (10,)))
    assert not unknown.complete and unknown.frames == () and unknown.issues
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    stage = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)
    for count in (2, 3):
        control = struct.pack('<HH', 7, count)+b'\x40\0\x40\0\0\0'
        definition = struct.pack('<H', (39 << 6) | len(control))+control
        export = struct.pack('<H', (56 << 6) | 9)+struct.pack('<HH', 1, 7)+b'icon\0'
        body = stage+definition+export+b'\0\0'
        raw = b'FWS\x06'+struct.pack('<I', len(body)+8)+body
        asset = read_flash_visual_assets(raw).named('icon')
        measured = read_declared_visual_frames(raw, asset)
        assert measured.complete == (count == 2)
        if measured.complete:
            assert tuple(f.number for f in measured.frames) == (1, 2)
    return ()
