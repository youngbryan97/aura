"""A suspended host is not a stalled one, and they look identical.

time.monotonic() stops while macOS sleeps and time.time() does not, so
every subsystem holding "when did I last see this" wakes to an anchor
hours old and reaches the same wrong conclusion in its own vocabulary:
the heartbeat says the mind stopped, the liveness check says the worker
wedged, the staleness check says the reading is void.

core/brain/llm/mlx_client.py worked this out for the generation wait and
kept it private, so the inference lane rebased and nothing else did.
"""

from __future__ import annotations

import time

import pytest

from core.runtime import host_sleep


@pytest.fixture(autouse=True)
def _clean():
    host_sleep.reset_host_clock_for_test()
    yield
    host_sleep.reset_host_clock_for_test()


def test_an_ordinary_interval_is_all_running_time():
    host_sleep._measure_gap()
    time.sleep(0.05)
    gap = host_sleep._measure_gap()
    assert gap.running_s >= 0.04
    assert gap.slept_s < 0.05
    assert gap.unexplained_s < 0.05


def test_a_suspended_host_is_reported_as_sleep_not_as_absence(monkeypatch):
    host_sleep._measure_gap()
    # The suspend-inclusive clock advanced by an hour; the monotonic one
    # did not. That is the machine being shut.
    base = host_sleep.sleep_inclusive_monotonic()
    if base is None:
        pytest.skip("this platform has no suspend-inclusive clock")
    monkeypatch.setattr(
        host_sleep, "sleep_inclusive_monotonic", lambda: base + 3600.0
    )
    gap = host_sleep._measure_gap()
    assert gap.measured is True
    assert gap.slept_s == pytest.approx(3600.0, abs=2.0)
    assert gap.unexplained_s < 1.0, "a shut host is not an unexplained absence"


def test_a_clock_that_jumped_while_awake_is_not_called_sleep(monkeypatch):
    base = host_sleep.sleep_inclusive_monotonic()
    if base is None:
        pytest.skip("this platform has no suspend-inclusive clock")
    host_sleep._measure_gap()
    # Wall time moved; neither monotonic clock did. Somebody set the date.
    wall = time.time()
    monkeypatch.setattr(time, "time", lambda: wall + 900.0)
    gap = host_sleep._measure_gap()
    assert gap.slept_s < 1.0
    assert gap.clock_shift_s == pytest.approx(900.0, abs=2.0)
    # And so a stall during that window really was a stall.
    assert gap.unexplained_s > 800.0


def test_an_unmeasurable_platform_says_so_rather_than_guessing(monkeypatch):
    monkeypatch.setattr(host_sleep, "sleep_inclusive_monotonic", lambda: None)
    host_sleep.reset_host_clock_for_test()
    gap = host_sleep._measure_gap()
    assert gap.measured is False
    assert host_sleep.host_sleep_report()["measurable"] is False


def test_seconds_asleep_since_an_anchor(monkeypatch):
    anchor_inclusive = host_sleep.sleep_inclusive_monotonic()
    if anchor_inclusive is None:
        pytest.skip("this platform has no suspend-inclusive clock")
    anchor_monotonic = time.monotonic()
    monkeypatch.setattr(
        host_sleep, "sleep_inclusive_monotonic", lambda: anchor_inclusive + 1800.0
    )
    slept = host_sleep.seconds_asleep_since(anchor_inclusive, anchor_monotonic)
    assert slept == pytest.approx(1800.0, abs=2.0)


def test_no_anchor_is_zero_rather_than_an_invention():
    # A caller that subtracts this is no worse off than before.
    assert host_sleep.seconds_asleep_since(None, time.monotonic()) == 0.0


def test_the_report_counts_what_it_saw(monkeypatch):
    base = host_sleep.sleep_inclusive_monotonic()
    if base is None:
        pytest.skip("this platform has no suspend-inclusive clock")
    host_sleep._measure_gap()
    monkeypatch.setattr(host_sleep, "sleep_inclusive_monotonic", lambda: base + 120.0)
    host_sleep._measure_gap()
    report = host_sleep.host_sleep_report()
    assert report["sleeps"] >= 1
    assert report["slept_total_s"] >= 100.0


def test_the_inference_lane_reads_the_shared_clock():
    # Two implementations of "was the host asleep" is how one lane rebased
    # and the rest of the runtime did not.
    import inspect

    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client._sleep_inclusive_monotonic)
    assert "from core.runtime.host_sleep import sleep_inclusive_monotonic" in source


def test_the_doctor_does_not_heal_a_machine_that_was_merely_shut(monkeypatch):
    """A lid closed overnight is not eight hours of event-loop lag.

    FlagshipDoctorDaemon measures lag against the WALL clock, which counts
    every suspended hour, and triggers self-healing when it crosses a
    threshold of seconds. Unsubtracted, a laptop shut overnight comes back
    and heals itself for a stall that never happened.
    """

    import time as _time

    from core.runtime.flagship_doctor import FlagshipDoctorDaemon

    doctor = FlagshipDoctorDaemon()
    anchor = doctor._heartbeat_anchor
    if anchor[0] is None:
        pytest.skip("this platform has no suspend-inclusive clock")

    # Eight hours pass on both the wall clock and the suspend-inclusive
    # one, and none of it on the monotonic clock: the host was asleep.
    overnight = 8 * 3600.0
    stamped = doctor._last_heartbeat
    monkeypatch.setattr(_time, "time", lambda: stamped + overnight)
    monkeypatch.setattr(
        host_sleep, "sleep_inclusive_monotonic", lambda: anchor[0] + overnight
    )

    assert doctor._slept_since_heartbeat() == pytest.approx(overnight, abs=5.0)
    lag = max(0.0, (_time.time() - doctor._last_heartbeat) - doctor._slept_since_heartbeat())
    assert lag < 5.0, "a suspended host must not read as event-loop lag"


def test_a_real_stall_while_awake_still_reads_as_lag(monkeypatch):
    """The subtraction must not swallow the thing it is measuring."""

    import time as _time

    from core.runtime.flagship_doctor import FlagshipDoctorDaemon

    doctor = FlagshipDoctorDaemon()
    stamped = doctor._last_heartbeat
    # Wall time moved sixty seconds and the host never slept.
    monkeypatch.setattr(_time, "time", lambda: stamped + 60.0)

    assert doctor._slept_since_heartbeat() < 1.0
    lag = max(0.0, (_time.time() - doctor._last_heartbeat) - doctor._slept_since_heartbeat())
    assert lag == pytest.approx(60.0, abs=2.0)
