from __future__ import annotations

import asyncio
import json

from core.consciousness import unified_self
from core.consciousness.unified_self import UnifiedSelf


def test_unified_self_corrupt_persistence_degrades_to_default_state(monkeypatch, tmp_path):
    path = tmp_path / "unified_self.json"
    path.write_text("{not-json", encoding="utf-8")
    recorded = []
    monkeypatch.setattr(
        unified_self,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    state = UnifiedSelf(str(path)).get_state()

    assert state.name == "Aura"
    assert recorded == [("unified_self", "JSONDecodeError")]


def test_unified_self_uses_narrow_recoverable_exceptions():
    with open(unified_self.__file__, encoding="utf-8") as fh:
        text = fh.read()
    assert "except Exception" not in text
    assert "except BaseException" not in text


def test_identity_save_renews_its_grant_after_directory_setup(monkeypatch, tmp_path):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = tmp_path / "new" / "identity.json"
    state = UnifiedSelf(str(path))
    gateway = get_file_write_gateway()
    ensure_directory = gateway.ensure_directory_async
    grants = []
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None
        result = await ensure_directory(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        grants.append(token.lease_id)
        return result

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    asyncio.run(state._save_to_disk_async())
    assert grants and json.loads(path.read_text())["name"] == state.get_state().name
