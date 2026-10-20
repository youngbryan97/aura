"""A local cache is read as an artifact, without gaining remote or semantic authority."""
from pathlib import Path

import pytest

from core.brain.llm.a_model_in_the_local_cache import model_in_local_cache, _local_model_cache_invariant

pytestmark = pytest.mark.unit


def _snapshot(tmp_path):
    root=tmp_path/'models--example--small-model'
    snapshot=root/'snapshots'/('a'*40)
    snapshot.mkdir(parents=True)
    (root/'refs').mkdir()
    (root/'refs'/'main').write_text(snapshot.name)
    (snapshot/'config.json').write_text('{"model_type":"test"}')
    return snapshot


def test_a_cached_revision_is_returned_as_a_local_directory_without_network(tmp_path,monkeypatch):
    from huggingface_hub import constants
    import socket
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path))
    monkeypatch.setattr(socket,'create_connection',lambda *a,**kw:pytest.fail('a cache lookup attempted network'))
    snapshot=_snapshot(tmp_path)
    assert model_in_local_cache('example/small-model')==snapshot.resolve()
    assert _local_model_cache_invariant()==()


def test_the_serving_registry_uses_the_cache_after_missing_its_local_inventory(tmp_path,monkeypatch):
    from huggingface_hub import constants
    from core.brain.llm import model_registry as registry
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path))
    monkeypatch.setattr(registry,'get_models_dir',lambda:tmp_path/'inventory')
    snapshot=_snapshot(tmp_path)
    resolved=registry.get_model_path('example/small-model')
    assert resolved==str(snapshot.resolve()) and Path(resolved).is_absolute()
    assert registry.get_model_lane_role(resolved) is None


def test_an_explicit_inventory_keeps_priority_over_a_cache(tmp_path,monkeypatch):
    from huggingface_hub import constants
    from core.brain.llm import model_registry as registry
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path))
    inventory=tmp_path/'inventory';local=inventory/'example'/'small-model'
    local.mkdir(parents=True)
    monkeypatch.setattr(registry,'get_models_dir',lambda:inventory)
    _snapshot(tmp_path)
    assert registry.get_model_path('example/small-model')==str(local.resolve())


@pytest.mark.parametrize('damage',['missing','dangling','unknown_revision','negative_cache','bad_snapshot'])
def test_incomplete_cache_references_are_unavailable(tmp_path,monkeypatch,damage):
    from huggingface_hub import constants
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path))
    snapshot=_snapshot(tmp_path)
    config=snapshot/'config.json'
    if damage=='missing': config.unlink()
    elif damage=='dangling':
        config.unlink();config.symlink_to(tmp_path/'absent')
    elif damage=='unknown_revision': (snapshot.parent.parent/'refs'/'main').write_text('b'*40)
    elif damage=='negative_cache':
        config.unlink()
        negative=snapshot.parent.parent/'.no_exist'/snapshot.name
        negative.mkdir(parents=True);(negative/'config.json').touch()
    else: snapshot.rename(snapshot.parent/'bad')
    assert model_in_local_cache('example/small-model') is None


def test_a_missing_cache_never_returns_a_remote_identifier(tmp_path,monkeypatch):
    from huggingface_hub import constants
    from core.brain.llm import model_registry as registry
    monkeypatch.setattr(constants,'HF_HUB_CACHE',str(tmp_path))
    monkeypatch.setattr(registry,'get_models_dir',lambda:tmp_path/'inventory')
    assert registry.get_model_path('example/small-model')==str(tmp_path/'inventory/example/small-model')


def test_cache_lookup_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite=ValidationSuite()
    install_realtime_control_claims(suite)
    check=next(t for t in suite.tests() if t.name=='local_model_cache_cannot_supply_remote_authority')
    assert check.predict(None) is True and any(c.test==check.name for c in suite.claims())
