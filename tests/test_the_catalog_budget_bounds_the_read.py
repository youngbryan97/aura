"""The capability catalog's read budget bounds the read, not the gate in front of it.

The desktop inventory reads the catalog within 0.35 s. The clock started before
the memory-pressure check, whose first call imports the memory monitor: 0.19 s
on a host running a campaign on 22 September. A slow gate could spend the
budget before the first entry was read, and the reply then said the runtime
did not expose a catalog it had never looked at.
"""

from __future__ import annotations

import time

import pytest

from interface.routes import chat_desktop_repair as repair

pytestmark = pytest.mark.unit


class _Engine:
    def iter_tool_catalog(self, *, include_inactive=True):
        yield {"name": "web_search", "available": True}
        yield {"name": "memory_ops", "available": True}

    def get_catalog_health(self):
        return {"ready": True}


@pytest.fixture
def one_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repair.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: _Engine() if name == "capability_engine" else default),
    )


def test_a_slow_pressure_check_does_not_spend_the_read_budget(one_engine, monkeypatch) -> None:
    def slow_gate() -> str:
        time.sleep(repair._CAPABILITY_CATALOG_READ_BUDGET_S * 1.5)
        return ""

    monkeypatch.setattr(repair, "_capability_catalog_memory_block_reason", slow_gate)
    snapshot = repair._read_capability_catalog_snapshot()
    assert snapshot.catalog_status == "measured", snapshot.detail
    assert snapshot.available_count == 2


def test_a_pressure_block_still_skips_the_read(one_engine, monkeypatch) -> None:
    monkeypatch.setattr(repair, "_capability_catalog_memory_block_reason", lambda: "critical_memory_pressure")
    snapshot = repair._read_capability_catalog_snapshot()
    assert snapshot.catalog_status == "blocked"
    assert snapshot.available_count == 0
