"""A 64GB Mac is 64GB when she reads it back.

LIVE 2026-08-18: "how much memory are you using right now?"

    The host system has around 61% utilization out of its total 69GB...

She was quoting her instruments faithfully. The instrument was wrong, which is
the worse of the two failures: a fabrication can be challenged, and a wrong
reading is repeated with total confidence by everything downstream.

A 64GB machine holds 68,719,476,736 bytes. Dividing by 1e9 and printing "GB"
gives 69 — for a machine Apple, Activity Monitor and its owner all call 64GB.

The same conversion sat in the headroom gate, where it mattered more than
wording: measured free space read about 7% larger than it is, so a campaign
was admitted with less room than the threshold demanded, and the cost of that
is memory shedding during the probe.
"""

from __future__ import annotations

import pytest

BYTES_PER_GIB = 1024**3


@pytest.fixture
def observed_total() -> int:
    from core.runtime.resource_observation import get_resource_observer

    total = int(get_resource_observer().memory(include_process_tree=False).total_bytes)
    if total <= 0:
        pytest.skip("no memory observation in this process")
    return total


def test_the_instrument_reports_the_machine_people_bought(observed_total: int) -> None:
    from core.brain.self_state_report import runtime_self_report

    expected = f"{observed_total / BYTES_PER_GIB:.0f}GB"
    report = runtime_self_report()

    assert expected in report, f"expected {expected} in the instruments block"


def test_the_decimal_reading_is_not_what_gets_reported(observed_total: int) -> None:
    """69GB is the number that made this a bug rather than a rounding choice."""
    from core.brain.self_state_report import runtime_self_report

    decimal = f"{observed_total / 1e9:.0f}GB"
    gibibyte = f"{observed_total / BYTES_PER_GIB:.0f}GB"
    if decimal == gibibyte:
        pytest.skip("the two units agree on this machine")

    assert decimal not in runtime_self_report()


def test_the_headroom_gate_measures_in_the_same_unit(monkeypatch) -> None:
    """Exactly 8 GiB free must not satisfy a threshold of 8.5.

    Measured in decimal GB the same bytes read as 8.6, which passes — the gate
    admitting a campaign with less room than it was told to require. Driving
    the gate is the check; reading its source would only prove a spelling.
    """
    from core.verify import influence_campaign

    class _Memory:
        available = True
        available_bytes = 8 * BYTES_PER_GIB

    class _Observer:
        def memory(self, **_kwargs):
            return _Memory()

    monkeypatch.setattr(
        "core.runtime.resource_observation.get_resource_observer", lambda: _Observer()
    )

    reason = influence_campaign.campaign_admission_reason(min_free_gb=8.5)

    assert reason.startswith("insufficient_memory"), reason


def test_ample_headroom_still_admits(monkeypatch) -> None:
    from core.verify import influence_campaign

    class _Memory:
        available = True
        available_bytes = 32 * BYTES_PER_GIB

    class _Observer:
        def memory(self, **_kwargs):
            return _Memory()

    monkeypatch.setattr(
        "core.runtime.resource_observation.get_resource_observer", lambda: _Observer()
    )

    assert influence_campaign.campaign_admission_reason(min_free_gb=8.5) == ""
