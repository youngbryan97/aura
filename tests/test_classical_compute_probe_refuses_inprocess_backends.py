"""Backend code cannot start through an ordinary import in a classical probe."""
import importlib
import sys
from types import ModuleType

import pytest

from core.runtime.classical_compute_boundary import (
    MODEL_BACKENDS,
    _classical_import_invariant,
    classical_compute_boundary,
    current_classical_boundary,
)


@pytest.fixture
def no_preloaded_models(monkeypatch):
    # Preserve the test runner's modules; no backend is imported or initialized.
    for name in tuple(sys.modules):
        if name.partition('.')[0] in MODEL_BACKENDS:
            monkeypatch.delitem(sys.modules, name)


@pytest.mark.parametrize('root', sorted(MODEL_BACKENDS))
def test_import_is_refused_before_a_backend_package_can_execute(root, tmp_path, monkeypatch, no_preloaded_models):
    package = tmp_path / root
    package.mkdir()
    (package / '__init__.py').write_text('raise AssertionError("backend package executed")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    with classical_compute_boundary() as boundary:
        with pytest.raises(ImportError, match='classical compute boundary'):
            importlib.import_module(root)
        assert root not in sys.modules
        assert boundary.report()['loaded_backends'] == []
        assert boundary.report()['refused_imports'] == [root]


def test_preloaded_backend_prevents_admission(monkeypatch, no_preloaded_models):
    original = list(sys.meta_path)
    monkeypatch.setitem(sys.modules, 'mlx', ModuleType('mlx'))
    with pytest.raises(RuntimeError, match='loaded model backends'):
        with classical_compute_boundary():
            pytest.fail('an already loaded backend was admitted')
    assert sys.meta_path == original


def test_policy_is_scoped_and_nested_owners_keep_their_own_boundary(no_preloaded_models):
    original = list(sys.meta_path)
    with classical_compute_boundary() as outer:
        assert current_classical_boundary() is outer
        with classical_compute_boundary() as inner:
            assert current_classical_boundary() is inner
            assert outer.active and inner.active
        assert current_classical_boundary() is outer and not inner.active
        importlib.import_module('fractions')
        assert outer.report()['refused_imports'] == []
    assert current_classical_boundary() is None and sys.meta_path == original


def test_unrelated_prefix_is_allowed_and_body_failure_restores_import_state(tmp_path, monkeypatch, no_preloaded_models):
    (tmp_path / 'torch_support.py').write_text('value = 7\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    original = list(sys.meta_path)
    try:
        with pytest.raises(ValueError, match='probe failed'):
            with classical_compute_boundary():
                assert importlib.import_module('torch_support').value == 7
                raise ValueError('probe failed')
        assert sys.meta_path == original
    finally:
        sys.modules.pop('torch_support', None)


def test_boundary_removal_is_detected_before_a_probe_can_report_completion(no_preloaded_models):
    original = list(sys.meta_path)
    with pytest.raises(RuntimeError, match='not installed'):
        with classical_compute_boundary() as boundary:
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not boundary]
    assert sys.meta_path == original


def test_backend_added_outside_import_is_also_visible_at_close(monkeypatch, no_preloaded_models):
    original = list(sys.meta_path)
    with pytest.raises(RuntimeError, match='already loaded'):
        with classical_compute_boundary():
            monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    assert sys.meta_path == original


@pytest.mark.asyncio
async def test_cpu_campaign_installs_the_boundary_before_asset_or_browser_work(tmp_path, monkeypatch, no_preloaded_models):
    from tools import measure_on_archived_games as runner

    before = list(sys.meta_path)
    for key in ('AURA_MODEL_WORKERS_ALLOWED', 'AURA_VISION_CPU_ONLY', 'AURA_PLAY_POINT_BACKEND', 'AURA_PLAY_POINT_DEVICE'):
        monkeypatch.setenv(key, '')

    async def campaign(*args):
        boundary = current_classical_boundary()
        assert boundary is not None
        boundary.require_intact()
        with pytest.raises(ImportError, match='classical compute boundary'):
            importlib.import_module('mlx')
        return []

    monkeypatch.setattr(runner, '_campaign', campaign)
    assert await runner.main(['--games', '18', '--cpu-only', '--out', str(tmp_path / 'pass.json')]) == []
    assert sys.meta_path == before


@pytest.mark.asyncio
async def test_cpu_case_retains_the_actual_boundary_report(tmp_path, monkeypatch, no_preloaded_models):
    from tools import measure_on_archived_games as runner

    for key in ('AURA_MODEL_WORKERS_ALLOWED', 'AURA_VISION_CPU_ONLY', 'AURA_PLAY_POINT_BACKEND', 'AURA_PLAY_POINT_DEVICE'):
        monkeypatch.setenv(key, '')
    monkeypatch.setattr(runner, 'ARCHIVED', {18: 'https://fixture.invalid'})

    async def play(*args, **kwargs):
        with pytest.raises(ImportError, match='classical compute boundary'):
            importlib.import_module('mlx')
        return {'game': 18, 'won': False}

    monkeypatch.setattr(runner, '_one', play)
    result = await runner.main(['--games', '18', '--cpu-only', '--held-out', str(tmp_path / 'no-held'),
                                '--out', str(tmp_path / 'pass.json')])
    report = result[0]['compute_boundary']
    assert report['active'] and report['scope'] == 'current_python_interpreter'
    assert report['loaded_backends'] == [] and report['refused_imports'] == ['mlx']


def test_cpu_prepare_cannot_bypass_the_boundary(tmp_path, monkeypatch):
    from tools import measure_on_archived_games as runner

    for key in ('AURA_MODEL_WORKERS_ALLOWED', 'AURA_VISION_CPU_ONLY', 'AURA_PLAY_POINT_BACKEND', 'AURA_PLAY_POINT_DEVICE'):
        monkeypatch.setenv(key, '')
    args, parser, checkpoint = runner._arguments(['--games', '18', '--cpu-only', '--out', str(tmp_path / 'pass.json')])
    with pytest.raises(RuntimeError, match='in-process compute boundary'):
        runner._prepare_campaign(args, parser, checkpoint)


def test_registered_probe_checks_backend_roots_without_changing_the_importer():
    before = list(sys.meta_path)
    assert _classical_import_invariant() == ()
    assert sys.meta_path == before
