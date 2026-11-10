"""R06: the runtime lease's write is authorized when it is made, not before an await the loop can stall in.

LIVE 2026-11-10, 14:58: her event loop stopped for 67 seconds. The lease writer had opened one 30-second governed
scope around both of its awaits, the grant ran out inside the first, and the second was refused: "GOVERNANCE
VIOLATION: 'file_write_gateway.write_text:lease.aura_runtime' called outside governed context".
"""
from __future__ import annotations

import asyncio

import pytest

from core.governance_context import get_active_governance, require_governance
from core.runtime import lease

pytestmark = pytest.mark.unit


class _StallingGateway:
    """Ages the grant it is called under by a minute, as a stalled loop does, then checks the next one strictly."""

    def __init__(self) -> None:
        self.written: list[str] = []

    async def ensure_directory_async(self, path, *, source):
        require_governance(f"ensure_directory:{source}", strict=True)
        get_active_governance().mono_timestamp -= 60.0

    async def write_text_async(self, path, text, *, durable, source):
        require_governance(f"write_text:{source}", strict=True)
        self.written.append(text)


def test_the_write_after_a_stalled_await_is_still_governed(tmp_path, monkeypatch):
    import core.runtime.file_write_gateway as gateways

    monkeypatch.setenv("AURA_RUNTIME_LEASE_DIR", str(tmp_path))
    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "production")
    gateway = _StallingGateway()
    monkeypatch.setattr(gateways, "get_file_write_gateway", lambda: gateway)
    elector = lease.LeaderElector("stall-probe")
    record = lease.LeaseRecord(
        name="stall-probe", identity=elector.identity, acquired_at=0.0, renewed_at=0.0,
        lease_duration_s=elector.lease_duration_s, transitions=0,
    )
    assert asyncio.run(elector._write(record)) is True
    assert gateway.written and '"stall-probe"' in gateway.written[0]
