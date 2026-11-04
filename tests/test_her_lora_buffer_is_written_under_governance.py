"""Her LoRA bridge's buffer is written inside the scope her own maintenance writes are governed by.

On the day run of 1 November her disk was full, her emergency protocol flushed the bridge, and the live runtime
refused the bare write: "file_write_gateway.write_text:crsm_lora_bridge.persist_buffer called outside governed
context", escalated as a critical fault of the emergency protocol.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_the_buffer_write_passes_strict_governance(tmp_path, monkeypatch):
    from core.consciousness import crsm_lora_bridge as bridge_module
    from core.runtime import file_write_gateway as gateway_module

    target = tmp_path / "crsm_buffer.jsonl"
    monkeypatch.setattr(bridge_module, "PERSIST_PATH", target)
    import core.governance_context as governance

    monkeypatch.setattr(gateway_module, "governance_runtime_active", lambda: True)
    monkeypatch.setattr(governance, "governance_runtime_active", lambda: True)
    recorded: list[object] = []
    monkeypatch.setattr(bridge_module, "record_degradation", lambda *a, **k: recorded.append(a))
    bridge = bridge_module.CRSMLoraBridge()
    bridge._persist_buffer()
    assert target.exists()
    assert not recorded


def test_a_bare_write_is_still_refused(tmp_path, monkeypatch):
    """The scope is what lets it through: the same write without it is refused."""
    from core.governance_context import GovernanceViolationError
    from core.runtime import file_write_gateway as gateway_module

    import core.governance_context as governance

    monkeypatch.setattr(gateway_module, "governance_runtime_active", lambda: True)
    monkeypatch.setattr(governance, "governance_runtime_active", lambda: True)
    with pytest.raises(GovernanceViolationError):
        gateway_module.write_text_behind(
            tmp_path / "x.jsonl", "", source="crsm_lora_bridge.persist_buffer"
        )
