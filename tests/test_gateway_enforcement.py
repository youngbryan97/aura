"""Tests for memory and state gateway enforcement.

Verifies that:
  - ConcreteMemoryWriteGateway writes produce receipts
  - Governance denial causes PermissionError
  - ConcreteStateGateway mutations produce receipts
  - State reads return cached and durable values
"""
from __future__ import annotations

import pytest

from core.runtime.gateways import MemoryWriteRequest, StateMutationRequest


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path


@pytest.fixture
def approve_all():
    def _approve(**_kwargs):
        return {"approved": True, "receipt_id": "rcpt-test"}

    return _approve


@pytest.mark.asyncio
async def test_memory_write_produces_receipt(tmp_root, approve_all):
    from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway

    gw = ConcreteMemoryWriteGateway(root=tmp_root / "memory", governance_decide=approve_all)
    request = MemoryWriteRequest(
        content="test memory content",
        metadata={"family": "episodic", "record_id": "test-001"},
        cause="test",
    )
    receipt = await gw.write(request)

    assert receipt.record_id == "test-001"
    assert receipt.receipt_id.startswith("memwr-")
    assert receipt.receipt_id != "rcpt-test"
    assert receipt.bytes_written > 0
    assert receipt.schema_version == 1
    # File should exist on disk
    target = tmp_root / "memory" / "episodic" / "test-001.json"
    assert target.exists()


@pytest.mark.asyncio
async def test_memory_write_without_governance_fails_closed(tmp_root):
    from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway

    gw = ConcreteMemoryWriteGateway(root=tmp_root / "memory_no_governance")

    with pytest.raises(PermissionError, match="governance denied"):
        await gw.write(
            MemoryWriteRequest(
                content="must not be written",
                metadata={"family": "episodic"},
                cause="test",
            )
        )


@pytest.mark.asyncio
async def test_memory_write_governance_denial(tmp_root):
    """When governance denies, the gateway must raise PermissionError."""

    def deny_all(**kwargs):
        return {"approved": False}

    gw_denied = __import__(
        "core.memory.memory_write_gateway", fromlist=["ConcreteMemoryWriteGateway"]
    ).ConcreteMemoryWriteGateway(root=tmp_root / "memory_denied", governance_decide=deny_all)

    request = MemoryWriteRequest(
        content="should not be written",
        metadata={"family": "episodic"},
        cause="test",
    )

    with pytest.raises(PermissionError, match="governance denied"):
        await gw_denied.write(request)


@pytest.mark.asyncio
async def test_memory_quarantine(tmp_root, approve_all):
    from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway

    gw = ConcreteMemoryWriteGateway(root=tmp_root / "memory_q", governance_decide=approve_all)
    request = MemoryWriteRequest(
        content="quarantine me",
        metadata={"family": "episodic", "record_id": "q-001"},
        cause="test",
    )
    await gw.write(request)
    assert (tmp_root / "memory_q" / "episodic" / "q-001.json").exists()

    await gw.quarantine("q-001", "test quarantine")
    assert not (tmp_root / "memory_q" / "episodic" / "q-001.json").exists()
    assert (tmp_root / "memory_q" / "_quarantine" / "episodic_q-001.json").exists()


@pytest.mark.asyncio
async def test_state_mutation_produces_receipt(tmp_root, approve_all):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state", governance_decide=approve_all)
    request = StateMutationRequest(
        key="test/value",
        new_value=42,
        cause="test_mutation",
    )
    receipt = await gw.mutate(request)

    assert receipt.key == "test/value"
    assert receipt.new_value == 42
    assert receipt.old_value is None  # first write
    assert receipt.receipt_id.startswith("statemut-")


@pytest.mark.asyncio
async def test_state_read_after_write(tmp_root, approve_all):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_rw", governance_decide=approve_all)
    request = StateMutationRequest(
        key="reading/test",
        new_value="hello",
        cause="test",
    )
    await gw.mutate(request)

    value = await gw.read("reading/test")
    assert value == "hello"


@pytest.mark.asyncio
async def test_state_snapshot(tmp_root, approve_all):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_snap", governance_decide=approve_all)
    await gw.mutate(StateMutationRequest(key="a", new_value=1, cause="test"))
    await gw.mutate(StateMutationRequest(key="b", new_value=2, cause="test"))

    snap = await gw.snapshot()
    assert snap["a"] == 1
    assert snap["b"] == 2


@pytest.mark.asyncio
async def test_state_domains_do_not_alias_same_key(tmp_root, approve_all):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_domains", governance_decide=approve_all)
    await gw.mutate(
        StateMutationRequest(
            key="mode",
            new_value="focused",
            cause="test",
            domain="cognition",
        )
    )
    await gw.mutate(
        StateMutationRequest(
            key="mode",
            new_value="resting",
            cause="test",
            domain="body",
        )
    )

    assert await gw.read("mode", domain="cognition", fresh=True) == "focused"
    assert await gw.read("mode", domain="body", fresh=True) == "resting"
    assert await gw.snapshot(domain="cognition") == {"mode": "focused"}
    assert await gw.snapshot(domain="body") == {"mode": "resting"}


@pytest.mark.asyncio
async def test_state_snapshot_reads_durable_entries_after_gateway_reopen(tmp_root, approve_all):
    from core.state.state_gateway import ConcreteStateGateway

    root = tmp_root / "state_reopened_snapshot"
    first = ConcreteStateGateway(root=root, governance_decide=approve_all)
    await first.mutate(StateMutationRequest(
        key="first", new_value={"state": "observed"}, cause="test", domain="cognition"))
    await first.mutate(StateMutationRequest(
        key="other", new_value=3, cause="test", domain="body"))
    reopened = ConcreteStateGateway(root=root)
    assert await reopened.snapshot(domain="cognition") == {"first": {"state": "observed"}}
    assert await reopened.snapshot(domain="body") == {"other": 3}


@pytest.mark.asyncio
async def test_state_receipt_failure_rolls_back_durable_and_cached_value(
    tmp_root,
    approve_all,
    monkeypatch,
):
    import core.state.state_gateway as state_module

    gw = state_module.ConcreteStateGateway(
        root=tmp_root / "state_rollback",
        governance_decide=approve_all,
    )
    await gw.mutate(
        StateMutationRequest(key="mode", new_value="before", cause="test")
    )

    class FailingReceiptStore:
        def emit(self, _receipt):
            raise OSError("receipt disk unavailable")

    monkeypatch.setattr(state_module, "get_receipt_store", lambda: FailingReceiptStore())
    with pytest.raises(RuntimeError, match="receipt_failed_rolled_back"):
        await gw.mutate(
            StateMutationRequest(key="mode", new_value="after", cause="test")
        )

    assert await gw.read("mode", fresh=True) == "before"
    assert await gw.snapshot() == {"mode": "before"}


@pytest.mark.asyncio
async def test_state_mutation_without_governance_fails_closed(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_no_governance")

    with pytest.raises(PermissionError, match="governance denied"):
        await gw.mutate(StateMutationRequest(key="denied", new_value="bad", cause="test"))


@pytest.mark.asyncio
async def test_state_governance_denial(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    def deny_all(**kwargs):
        return {"approved": False}

    gw = ConcreteStateGateway(root=tmp_root / "state_denied", governance_decide=deny_all)
    request = StateMutationRequest(key="denied", new_value="bad", cause="test")

    with pytest.raises(PermissionError, match="governance denied"):
        await gw.mutate(request)
