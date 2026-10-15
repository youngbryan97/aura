"""External rendering keeps the selected source identity and refuses stray output."""
from __future__ import annotations

import hashlib
import io
from dataclasses import replace

import pytest
from PIL import Image

from core.governance_context import local_internal_governed_scope
from core.perception.visual_assets_in_a_flash_file import DeclaredVisualAsset
from core.runtime.file_write_gateway import get_file_write_gateway
from tools.export_program_visual_reference import _inventory, _render_command


def _write(path, data):
    with local_internal_governed_scope('reference_export_test'):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source='reference_export_test')
        gateway.write_bytes(path, data, source='reference_export_test')


def _png():
    encoded = io.BytesIO()
    Image.new('RGBA', (24, 20), (9, 20, 30, 255)).save(encoded, format='PNG')
    return encoded.getvalue()


def test_renderer_selects_the_dictionary_character_without_interpreting_its_alias(tmp_path):
    asset = DeclaredVisualAsset('../outside; $(something)', 17, 'sprite', 30, 39, (100,))
    args = _render_command(tmp_path/'java', tmp_path/'tool.jar', tmp_path/'source.swf',
                           tmp_path/'output', tmp_path/'home', asset)
    assert asset.name not in args and args[args.index('-selectid')+1] == '17'
    assert args[args.index('-select')+1] == '17:1'
    assert '-XX:ActiveProcessorCount=1' in args and '-Djava.awt.headless=true' in args
    assert '-Dsun.java2d.opengl=false' in args and '-Xmx512m' in args
    bitmap = _render_command(tmp_path/'java', tmp_path/'tool.jar', tmp_path/'source.swf',
                             tmp_path/'output', tmp_path/'home', replace(asset, kind='bitmap'))
    assert bitmap[bitmap.index('-format')+1] == 'image:png'


def test_reference_inventory_hashes_actual_rendered_bytes(tmp_path):
    data = _png()
    path = tmp_path/'DefineSprite_17_an_alias'/'1.png'
    _write(path, data)
    got = _inventory(tmp_path, 17)
    assert got == [dict(path='DefineSprite_17_an_alias/1.png', file_sha256=hashlib.sha256(data).hexdigest(), size=[24, 20])]
    assert not any(name in got[0] for name in ('role', 'controlled', 'task_passed'))


@pytest.mark.parametrize('name,data', [
    ('DefineSprite_170_other/1.png', _png()),
    ('DefineSprite_7_other/1.png', _png()),
    ('DefineSprite_17_alias/1.png', b'not PNG'),
    ('DefineSprite_17_alias/1.svg', b'<svg/>'),
])
def test_other_characters_and_unexpected_files_cannot_enter_the_reference_set(tmp_path, name, data):
    _write(tmp_path/name, data)
    with pytest.raises((ValueError, OSError)):
        _inventory(tmp_path, 17)


def test_symlink_output_and_missing_pixels_are_refused(tmp_path):
    with pytest.raises(ValueError, match='no qualifying'):
        _inventory(tmp_path, 17)
    target = tmp_path/'target'
    _write(target, _png())
    with local_internal_governed_scope('reference_export_test'):
        gateway = get_file_write_gateway()
        folder = tmp_path/'rendered'
        gateway.ensure_directory(folder, source='reference_export_test')
        gateway.replace_symlink(folder/'DefineSprite_17_alias.png', target, source='reference_export_test')
    with pytest.raises(ValueError, match='directory'):
        _inventory(folder, 17)


def test_exhausted_reference_budget_is_a_failure(tmp_path, monkeypatch):
    _write(tmp_path/'DefineSprite_17_alias'/'1.png', _png())
    monkeypatch.setattr('tools.export_program_visual_reference.MAX_RENDERED_BYTES', 1)
    with pytest.raises(ValueError, match='byte budget'):
        _inventory(tmp_path, 17)
