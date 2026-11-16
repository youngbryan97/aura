"""Reality state writers keep their authority after directory I/O stalls."""
from __future__ import annotations

import pytest

from core.reality_reach.event_flow import RealityEventFlowRuntime
from core.reality_reach.live import RealityReachService
from core.reality_reach.metrology import RealityMetrologyService
from core.reality_reach.middleware import RealityMiddlewareRuntime
from core.runtime.atomic_writer import read_json_envelope
from core.runtime.audit_chain import canonical_json, sha256_hex


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["event_flow", "metrology", "middleware"])
async def test_reality_persistence_renews_its_grant_after_directory_io(tmp_path, monkeypatch, kind):
    from core.governance_context import get_active_governance
    from core.runtime.file_write_gateway import get_file_write_gateway

    monkeypatch.setenv("AURA_GOVERNANCE_MODE", "enforce")
    path = tmp_path / "private" / f"{kind}.json"
    def make_runtime():
        if kind == "event_flow":
            return RealityEventFlowRuntime(state_path=path, worker_enabled=False)
        if kind == "metrology":
            return RealityMetrologyService(RealityReachService(()), state_path=path)
        return RealityMiddlewareRuntime(RealityReachService(()), state_path=path)

    runtime = make_runtime()
    if kind == "event_flow":
        runtime._sequence = 7
    elif kind == "metrology":
        runtime._mode_generation = 4
    gateway = get_file_write_gateway()
    ensure = gateway.ensure_directory_async
    expired = []

    async def ensure_then_expire(*args, **kwargs):
        token = get_active_governance()
        assert token is not None and token.domain == "state_mutation"
        result = await ensure(*args, **kwargs)
        token.mono_timestamp -= token.ttl + 1
        expired.append(token.lease_id)
        return result

    monkeypatch.setattr(gateway, "ensure_directory_async", ensure_then_expire)
    await runtime._persist()
    payload = dict(read_json_envelope(path)["payload"])
    digest = payload.pop("state_sha256")
    assert expired and digest == str(sha256_hex(canonical_json(payload)))
    restored = make_runtime()
    if kind == "event_flow":
        assert restored._sequence == payload["sequence"] == 7
    elif kind == "metrology":
        assert restored._mode_generation == payload["mode_generation"] == 4
    else:
        assert payload["service_session_id"] == runtime._service.session_id
        assert restored._service.session_id != runtime._service.session_id
        assert restored._desired_active == runtime._desired_active
