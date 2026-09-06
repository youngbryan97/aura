"""What a live runtime can answer about the primitives built for it."""
from __future__ import annotations

import time

import pytest

from core.runtime.health_contract import _runtime_integrity_block

#: The eight that a running process can answer from what it has seen. The
#: three that walk the tree — clocks, deprecations, settings — are gates and
#: an inspector section, because 35 seconds is a health report nobody asks
#: for twice.
IN_THE_REPORT = (
    "trace_boundaries",
    "call_policies",
    "task_endings",
    "prompt_room",
    "memory_kinds",
    "write_drains",
    "abandoned_calls",
    "checkable_promises",
)


@pytest.fixture(scope="module")
def block():
    return _runtime_integrity_block()


@pytest.mark.parametrize("name", IN_THE_REPORT)
def test_it_is_in_the_integrity_block(block, name: str) -> None:
    assert name in block, block.get(f"{name}_error", "absent with no error")


def test_none_of_them_broke_the_report(block) -> None:
    broke = {k: v for k, v in block.items() if k.endswith("_error") and any(
        k.startswith(one) for one in IN_THE_REPORT
    )}
    assert broke == {}


def test_the_report_stays_quick_enough_to_be_asked_for(block) -> None:
    """Cached after the first call, so this measures the served cost."""
    started = time.monotonic()
    _runtime_integrity_block()
    assert time.monotonic() - started < 10.0


def test_the_slow_scans_are_reachable_somewhere_else() -> None:
    """Kept out of the report and not out of reach."""
    from tools.inspect_runtime import THE_SECTIONS

    for name in ("clocks", "deprecations"):
        assert name in THE_SECTIONS
