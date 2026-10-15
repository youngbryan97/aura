"""Keep exported visual names with their source dictionary identities.

This reads declarations, not graphics or executable state. A declared symbol
does not establish that it is on screen, what it does, or who controls it.
"""
from __future__ import annotations

import hashlib
import lzma
import struct
import zlib
from dataclasses import dataclass

from core.perception.flash_action_sources import MAX_TAGS, _strict_body, _u16, _u32
from core.perception.reading_a_flash_program import MOST_UNPACKED, is_flash
from core.verify import invariant

MAX_EXPORTS = 4096
_KINDS = {2: 'shape', 22: 'shape', 32: 'shape', 83: 'shape',
          6: 'bitmap', 20: 'bitmap', 21: 'bitmap', 35: 'bitmap', 36: 'bitmap', 90: 'bitmap',
          7: 'button', 34: 'button', 39: 'sprite', 11: 'text', 33: 'text', 37: 'text',
          46: 'morphshape', 84: 'morphshape', 10: 'font', 48: 'font', 75: 'font', 91: 'font',
          14: 'sound', 60: 'video', 87: 'binary'}
_VISUAL = frozenset({'shape', 'bitmap', 'button', 'sprite', 'text', 'morphshape'})


@dataclass(frozen=True)
class DeclaredVisualAsset:
    name: str
    character_id: int
    kind: str
    definition_offset: int
    definition_tag: int
    export_offsets: tuple[int, ...]


@dataclass(frozen=True)
class FlashVisualAssets:
    source: str
    assets: tuple[DeclaredVisualAsset, ...]
    issues: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether this dictionary-name mapping was read completely, without decoding artwork."""
        return not self.issues

    def named(self, name: str) -> DeclaredVisualAsset | None:
        candidates = [asset for asset in self.assets if asset.name == name]
        return candidates[0] if self.complete and len(candidates) == 1 else None


def _export_names(data: bytes, start: int, end: int, version: int) -> tuple[tuple[int, str], ...]:
    count = _u16(data, start, end)
    if count > MAX_EXPORTS:
        raise ValueError('export table exceeds its bound')
    pos, names = start+2, []
    for _ in range(count):
        character = _u16(data, pos, end)
        pos += 2
        stop = data.find(b'\0', pos, min(end, pos+4097))
        if stop < 0:
            raise ValueError('unterminated or oversized exported name')
        name = data[pos:stop].decode('utf-8' if version >= 6 else 'cp1252')
        if not name:
            raise ValueError('empty exported name')
        names.append((character, name))
        pos = stop+1
    if pos != end:
        raise ValueError('export table has trailing bytes')
    return tuple(names)


def read_flash_visual_assets(raw: bytes) -> FlashVisualAssets:
    """Read bounded top-level declarations with their uncompressed tag offsets.

    The original file digest namespaces every character ID. Repeated aliases
    retain every export address. Conflicting names, missing definitions,
    imports and malformed containers remain explicit issues.
    """
    if not isinstance(raw, bytes):
        raise ValueError('immutable Flash bytes are required')
    source = 'swf:'+hashlib.sha256(raw).hexdigest()
    if not 8 <= len(raw) <= MOST_UNPACKED or not is_flash(raw):
        return FlashVisualAssets(source, (), ('invalid or oversized Flash input',))
    definitions, aliases, issues = {}, {}, []
    try:
        version, data = _strict_body(raw)
        pos, tags, exports, ended = 0, 0, 0, False
        while pos < len(data):
            offset = pos
            if tags >= MAX_TAGS:
                raise ValueError('tag budget exceeded')
            tags += 1
            header = _u16(data, pos, len(data))
            code, length, pos = header >> 6, header & 63, pos+2
            if length == 63:
                length, pos = _u32(data, pos, len(data)), pos+4
            end = pos+length
            if end > len(data):
                raise ValueError('tag exceeds its container')
            if code == 0:
                if length or any(data[end:]):
                    raise ValueError('invalid trailing End tag bytes')
                ended = True
                break
            if code in _KINDS:
                character = _u16(data, pos, end)
                if character == 0 or character in definitions:
                    raise ValueError('zero or duplicate dictionary character ID')
                if code == 39 and not _u16(data, pos+2, end):
                    raise ValueError('sprite frame count must be positive')
                definitions[character] = (_KINDS[code], offset, code)
            elif code in (56, 76):
                for character, name in _export_names(data, pos, end, version):
                    exports += 1
                    if exports > MAX_EXPORTS:
                        raise ValueError('total export budget exceeded')
                    if code == 76 and character == 0:
                        continue  # The document class is not a drawable dictionary object.
                    aliases.setdefault((name, character), []).append(offset)
            elif code in (57, 71):
                issues.append('external asset imports have no local definition custody')
            pos = end
        if not ended:
            issues.append('tag stream has no End tag')
    except (IndexError, ValueError, UnicodeError, struct.error, EOFError, zlib.error, lzma.LZMAError) as exc:
        issues.append(str(exc))
    names = {}
    for name, character in aliases:
        names.setdefault(name, set()).add(character)
    if any(len(characters) > 1 for characters in names.values()):
        issues.append('one exported name binds multiple character IDs')
    assets = []
    for (name, character), offsets in aliases.items():
        definition = definitions.get(character)
        if definition is None:
            issues.append(f'export refers to missing character {character}')
            continue
        kind, offset, code = definition
        if kind in _VISUAL:
            assets.append(DeclaredVisualAsset(name, character, kind, offset, code, tuple(offsets)))
    return FlashVisualAssets(source, tuple(assets), tuple(dict.fromkeys(issues)))


@invariant('perception.visual_asset_names_keep_source_dictionary_custody', scope='perception',
           owner='core/perception/visual_assets_in_a_flash_file.py', observational=False)
def _visual_asset_custody_invariant() -> tuple:
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    stage = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)
    definition = struct.pack('<H', (39 << 6) | 6)+struct.pack('<HHH', 7, 1, 0)
    export = struct.pack('<H', (56 << 6) | 9)+struct.pack('<HH', 1, 7)+b'icon\0'
    body = stage+definition+export+b'\0\0'
    raw = b'FWS\x06'+struct.pack('<I', len(body)+8)+body
    table = read_flash_visual_assets(raw)
    asset = table.named('icon')
    assert table.complete and asset is not None and asset.character_id == 7
    assert asset.definition_offset == 0 and asset.export_offsets == (len(definition),)
    assert table.source == 'swf:'+hashlib.sha256(raw).hexdigest()
    assert not read_flash_visual_assets(raw[:-1]).complete
    return ()
