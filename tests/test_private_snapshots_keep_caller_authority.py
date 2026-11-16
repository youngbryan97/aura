"""State storage uses caller authority and renews it between operations."""
from __future__ import annotations

import stat
from contextlib import nullcontext

import pytest

from core.governance_context import (
    GovernanceViolation,
    get_active_governance,
    local_internal_governed_scope,
)
from core.persistence.scoped_writes import write_private_state_async
from core.runtime.atomic_writer import read_json_envelope
from core.runtime.file_write_gateway import get_file_write_gateway


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "envelope", "append"])
async def test_snapshot_uses_a_fresh_caller_scope_after_directory_io(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "private" / "state.json"
    target.parent.mkdir()
    target.parent.chmod(0o755)
    if kind == "append":
        target.write_text("previous\n")
        target.chmod(0o600)
    gateway = get_file_write_gateway()
    ensure = gateway.ensure_directory_async
    expired, scopes = [], []

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None
        result = await ensure(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        expired.append(token.lease_id)
        return result

    def scope():
        context = local_internal_governed_scope("test.private_snapshot", domain="state_mutation")
        scopes.append(context)
        return context

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    payload = {"measurement": None} if kind == "envelope" else "kept text\n"
    options = {"schema_version": 1, "schema_name": "test.snapshot"} if kind == "envelope" else {"append": kind == "append"}
    await write_private_state_async(target, payload, source="test.private_snapshot", scope_factory=scope, **options)
    restored = read_json_envelope(target)["payload"] if kind == "envelope" else target.read_text()
    expected = "previous\n" + payload if kind == "append" else payload
    assert restored == expected and expired
    assert len(scopes) == 2 and scopes[0] is not scopes[1]
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["text", "envelope", "append"])
async def test_snapshot_helper_cannot_grant_itself_write_authority(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    target = tmp_path / "private" / "state.json"
    payload = {"measurement": None} if kind == "envelope" else "kept text\n"
    options = {"schema_version": 1, "schema_name": "test.snapshot"} if kind == "envelope" else {"append": kind == "append"}
    assert get_active_governance() is None
    with pytest.raises(GovernanceViolation):
        await write_private_state_async(
            target, payload, source="test.private_snapshot", scope_factory=nullcontext, **options,
        )
    assert not target.parent.exists()


@pytest.mark.asyncio
async def test_plain_snapshot_rejects_nontext_before_creating_a_directory(tmp_path):
    target = tmp_path / "private" / "state.json"
    with pytest.raises(TypeError, match="must be text"):
        await write_private_state_async(target, {}, source="test.private_snapshot", scope_factory=nullcontext)
    assert not target.parent.exists()


@pytest.mark.asyncio
async def test_json_envelopes_cannot_be_appended(tmp_path):
    target = tmp_path / "private" / "state.json"
    with pytest.raises(ValueError, match="cannot be appended"):
        await write_private_state_async(
            target, {}, schema_version=1, schema_name="test.snapshot", append=True,
            source="test.private_snapshot", scope_factory=nullcontext,
        )
    assert not target.parent.exists()
