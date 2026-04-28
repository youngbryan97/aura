"""Tests for memory and state gateway enforcement.

Verifies that:
  - ConcreteMemoryWriteGateway writes produce receipts
  - Governance denial causes PermissionError
  - ConcreteStateGateway mutations produce receipts
  - State reads return cached and durable values
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.runtime.gateways import MemoryWriteRequest, StateMutationRequest


@pytest.fixture
def tmp_root(tmp_path):
    return tmp_path


@pytest.mark.asyncio
async def test_memory_write_produces_receipt(tmp_root):
    from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway

    gw = ConcreteMemoryWriteGateway(root=tmp_root / "memory")
    request = MemoryWriteRequest(
        content="test memory content",
        metadata={"family": "episodic", "record_id": "test-001"},
        cause="test",
    )
    receipt = await gw.write(request)

    assert receipt.record_id == "test-001"
    assert receipt.bytes_written > 0
    assert receipt.schema_version == 1
    # File should exist on disk
    target = tmp_root / "memory" / "episodic" / "test-001.json"
    assert target.exists()


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
async def test_memory_quarantine(tmp_root):
    from core.memory.memory_write_gateway import ConcreteMemoryWriteGateway

    gw = ConcreteMemoryWriteGateway(root=tmp_root / "memory_q")
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
async def test_state_mutation_produces_receipt(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state")
    request = StateMutationRequest(
        key="test/value",
        new_value=42,
        cause="test_mutation",
    )
    receipt = await gw.mutate(request)

    assert receipt.key == "test/value"
    assert receipt.new_value == 42
    assert receipt.old_value is None  # first write


@pytest.mark.asyncio
async def test_state_read_after_write(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_rw")
    request = StateMutationRequest(
        key="reading/test",
        new_value="hello",
        cause="test",
    )
    await gw.mutate(request)

    value = await gw.read("reading/test")
    assert value == "hello"


@pytest.mark.asyncio
async def test_state_snapshot(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    gw = ConcreteStateGateway(root=tmp_root / "state_snap")
    await gw.mutate(StateMutationRequest(key="a", new_value=1, cause="test"))
    await gw.mutate(StateMutationRequest(key="b", new_value=2, cause="test"))

    snap = await gw.snapshot()
    assert snap["a"] == 1
    assert snap["b"] == 2


@pytest.mark.asyncio
async def test_state_governance_denial(tmp_root):
    from core.state.state_gateway import ConcreteStateGateway

    def deny_all(**kwargs):
        return {"approved": False}

    gw = ConcreteStateGateway(root=tmp_root / "state_denied", governance_decide=deny_all)
    request = StateMutationRequest(key="denied", new_value="bad", cause="test")

    with pytest.raises(PermissionError, match="governance denied"):
        await gw.mutate(request)
