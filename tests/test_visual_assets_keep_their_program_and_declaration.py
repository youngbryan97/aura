"""Artwork declarations keep byte custody without becoming observed roles."""
from __future__ import annotations

import hashlib
import lzma
import struct
import zlib

import pytest

from core.perception.visual_assets_in_a_flash_file import (
    _visual_asset_custody_invariant,
    read_flash_visual_assets,
)


def _tag(code: int, payload: bytes) -> bytes:
    return (struct.pack('<H', code << 6 | len(payload)) if len(payload) < 63
            else struct.pack('<HI', code << 6 | 63, len(payload))) + payload


def _swf(tags: bytes) -> bytes:
    rect = '00110'+'000000'+'010100'+'000000'+'010100'
    rect += '0'*(-len(rect) % 8)
    body = int(rect, 2).to_bytes(len(rect)//8, 'big')+struct.pack('<HH', 24*256, 1)+tags+b'\0\0'
    return b'FWS\x06'+struct.pack('<I', len(body)+8)+body


def _sprite(character: int) -> bytes:
    return _tag(39, struct.pack('<HHH', character, 1, 0))


def _exports(*pairs: tuple[int, str], code: int = 56) -> bytes:
    return _tag(code, struct.pack('<H', len(pairs))+b''.join(
        struct.pack('<H', character)+name.encode()+b'\0' for character, name in pairs))


def test_declarations_keep_the_source_and_exact_tag_addresses() -> None:
    first, export = _sprite(7), _exports((7, 'icon'))
    raw = _swf(first+export+_exports((7, 'icon'), (7, 'another name')))
    got = read_flash_visual_assets(raw)
    icon = got.named('icon')
    assert got.complete and icon is not None
    assert got.source == 'swf:'+hashlib.sha256(raw).hexdigest()
    assert (icon.character_id, icon.kind, icon.definition_offset, icon.definition_tag) == (7, 'sprite', 0, 39)
    assert icon.export_offsets == (len(first), len(first)+len(export))
    assert got.named('another name').character_id == 7
    assert _visual_asset_custody_invariant() == ()


@pytest.mark.parametrize('signature', [b'CWS', b'ZWS'])
def test_compression_preserves_offsets_but_never_the_file_namespace(signature: bytes) -> None:
    raw = _swf(_sprite(7)+_exports((7, 'icon')))
    if signature == b'CWS':
        compressed = signature+raw[3:8]+zlib.compress(raw[8:])
    else:
        payload = lzma.compress(raw[8:], format=lzma.FORMAT_ALONE)
        compressed = b'ZWS\x0d'+raw[4:8]+struct.pack('<I', len(payload)-13)+payload[:5]+payload[13:]
    plain, packed = map(read_flash_visual_assets, (raw, compressed))
    assert packed.complete and packed.assets == plain.assets
    assert packed.source != plain.source


def test_document_class_is_not_a_visible_dictionary_object() -> None:
    got = read_flash_visual_assets(_swf(_sprite(7)+_exports((0, 'Main'), (7, 'Icon'), code=76)))
    assert got.complete and got.named('Main') is None
    assert got.named('Icon').character_id == 7


def test_nonvisual_exports_and_unexported_definitions_supply_no_reference() -> None:
    raw = _swf(_tag(14, b'\x08\0')+_tag(10, b'\x09\0')+_sprite(7)+_exports((8, 'alarm'), (9, 'typeface')))
    got = read_flash_visual_assets(raw)
    assert got.complete and got.assets == ()


def test_exported_names_remain_data_including_path_like_text() -> None:
    name = '../other/place; move left'
    got = read_flash_visual_assets(_swf(_sprite(7)+_exports((7, name))))
    assert got.complete and got.named(name).name == name


@pytest.mark.parametrize('tags', [
    _sprite(7)+_sprite(8)+_exports((7, 'same'), (8, 'same')),
    _sprite(7)+_sprite(7)+_exports((7, 'same')),
    _sprite(0)+_exports((0, 'zero')),
    _exports((7, 'missing')),
    _tag(39, struct.pack('<HHH', 7, 0, 0))+_exports((7, 'empty')),
    _sprite(7)+_tag(56, struct.pack('<HH', 1, 7)+b'unterminated'),
    _sprite(7)+_tag(56, struct.pack('<HH', 1, 7)+b'\xff\0'),
    _sprite(7)+_tag(56, struct.pack('<HH', 1, 7)+b'\0'),
    _sprite(7)+_tag(56, struct.pack('<HH', 1, 7)+b'name\0extra'),
    _tag(56, struct.pack('<H', 4097)),
    _tag(57, b'url\0\0\0'),
    _tag(71, b'url\0\x01\0\0\0'),
])
def test_ambiguous_missing_or_incomplete_custody_refuses_unique_lookup(tags: bytes) -> None:
    got = read_flash_visual_assets(_swf(tags))
    assert not got.complete and got.issues
    assert got.named('same') is None


@pytest.mark.parametrize('cut', [1, 2, 7, 8, 13, 22])
def test_truncated_containers_cannot_qualify(cut: int) -> None:
    raw = _swf(_sprite(7)+_exports((7, 'icon')))
    assert not read_flash_visual_assets(raw[:-cut]).complete


def test_missing_end_and_oversized_tag_remain_unread() -> None:
    raw = _swf(_sprite(7))[:-2]
    raw = raw[:4]+struct.pack('<I', len(raw))+raw[8:]
    assert not read_flash_visual_assets(raw).complete
    assert not read_flash_visual_assets(_swf(struct.pack('<HI', (39 << 6) | 63, 500))).complete
    assert not read_flash_visual_assets(b'not Flash').complete
    with pytest.raises(ValueError, match='immutable'):
        read_flash_visual_assets(bytearray(raw))


def test_regular_program_and_guide_keep_namespaces_without_giving_screen_roles() -> None:
    from core.cognition.a_guide_to_a_place import Guide
    from core.perception.reading_a_program import merged, read_program

    first = read_program(_swf(_sprite(7)+_exports((7, 'icon'))))
    other = read_program(_swf(_sprite(7)+_exports((7, 'another icon'))))
    combined = merged([first, other, first, read_program(b'const x = 1;')])
    assert len(combined.visual_assets) == 2
    assert len({table.source for table in combined.visual_assets}) == 2
    guide = Guide()
    guide.take_in_program(combined, 1.)
    guide.take_in_program(first, 2.)
    # A later acquisition is the current source scope. It cannot keep a
    # different file from the preceding visit active for runtime references.
    assert guide.program_visual_assets == first.visual_assets
    assert not guide.seen.by_kind and not guide.given
    assert 'program_visual_assets' not in guide.as_memory()
    assert Guide.from_memory(guide.as_memory()).program_visual_assets == ()
    assert read_program(b'const x = 1;').visual_assets == ()


def test_failed_asset_metadata_cannot_discard_read_controls() -> None:
    from core.cognition.a_guide_to_a_place import Guide
    from core.perception.reading_a_program import ProgramRead, merged

    bad = read_flash_visual_assets(_swf(_exports((7, 'missing'))))
    read = ProgramRead(keys={'left': ''}, visual_assets=(bad,))
    guide = Guide()
    guide.take_in_program(merged([read]), 1.)
    assert 'left' in guide.controls and not guide.program_visual_assets[0].complete
