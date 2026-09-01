"""The three instruments must have populations in a runtime that actually booted.

Reporting ``NOT_MEASURED`` instead of a fake pass was half the fix. The other
half is knowing whether the subsystems run at all — because "the instrument
honestly says it measured nothing" and "the subsystem is dead" produce the
same output, and only one of them is acceptable.

They do run. Measured here rather than asserted: ``activate_foundations``
starts a 1Hz group (telemetry, diagnostics) and a 5s group (health pings),
and lockdep wraps 90 files' worth of locks that the boot exercises. Every
other test in this suite runs in a process that never boots foundations, so
all three report NOT_MEASURED there — correctly, and with the consequence
that nothing would notice if the production wiring were removed.

This is the test that would notice. It costs ~7s of real ticking, which is
the price of the claim being about the runtime rather than about a mock.
"""
from __future__ import annotations

import asyncio

import pytest

from core.organism import model_validation as mv
import logging

@pytest.fixture(scope="module")
def booted_runtime():
    """Foundations up, background work included, ticking long enough to measure.

    ``foreground_only=False`` is the default a desktop launch uses
    (``AURA_FOREGROUND_ONLY`` defaults False), so this is the real posture,
    not a special one. No model is loaded.
    """

    # Liveness has to be read INSIDE the loop the runtime booted on.
    # ``asyncio.run`` closes that loop when it returns, and the event bus
    # captured it: ``is_alive()`` checks ``loop.is_running()``, which is False
    # for a closed loop however healthy the bus was. Reading the critical-ping
    # count after the fixture therefore reported the event bus wedged on every
    # run, in a suite whose whole purpose is to refuse vacuous passes.
    #
    # Counters — rounds run, locks known, groups registered — accumulate and
    # survive the loop, so only the liveness reading is taken from inside.
    readings: dict[str, int] = {}

    async def _boot():
        from core.runtime.foundations import activate_foundations

        await activate_foundations(foreground_only=False)
        # The 5s group needs one full period to have completed a cycle; 6.5s
        # gives the 1Hz group six-plus and the 5s group one-plus.
        await asyncio.sleep(6.5)
        readings["critical_unresponsive"] = mv._critical_unresponsive()

    asyncio.run(asyncio.wait_for(_boot(), timeout=90))
    yield readings

    try:
        from core.fsw.rate_groups import get_scheduler

        asyncio.run(get_scheduler().stop_all())
    except Exception as exc:  # noqa: BLE001 — teardown may not fail the suite
        logging.getLogger(__name__).debug("scheduler stop_all failed: %s", exc)


class TestRateGroupsActuallyRun:
    def test_groups_are_registered(self, booted_runtime):
        from core.fsw.rate_groups import rate_group_report

        groups = {g["name"]: g for g in rate_group_report()["groups"]}
        assert "1hz" in groups, (
            "no 1Hz rate group after a full boot — the periodic-rate claim has "
            "no population and every scheduler measurement is vacuous"
        )
        assert "5s" in groups

    def test_groups_have_completed_real_cycles(self, booted_runtime):
        from core.fsw.rate_groups import rate_group_report

        for group in rate_group_report()["groups"]:
            assert group["cycles"] > 0, (
                f"rate group {group['name']} is registered and has never ticked"
            )

    def test_groups_carry_members(self, booted_runtime):
        """A group with no members ticks forever and measures nothing."""
        from core.fsw.rate_groups import rate_group_report

        for group in rate_group_report()["groups"]:
            assert group.get("members"), f"{group['name']} has no members"

    def test_the_declared_rate_is_the_actual_rate(self, booted_runtime):
        """The claim this backs: period is a period, not period-plus-work."""
        fraction = mv._slowest_group_fraction()
        assert fraction < 0.5, (
            f"the slowest group uses {fraction:.1%} of its period; the declared "
            "rate is drifting toward rate-plus-work-time"
        )


class TestTheHealthCheckerActuallyPings:
    def test_components_are_watched(self, booted_runtime):
        from core.fsw.health_checker import health_checker_report

        assert health_checker_report()["watched"] > 0, (
            "install_runtime_pings registered nothing; 0 unresponsive out of 0 "
            "watched is the vacuous pass this suite exists to refuse"
        )

    def test_rounds_have_run(self, booted_runtime):
        from core.fsw.health_checker import health_checker_report

        assert health_checker_report()["rounds"] > 0, (
            "components are watched and nothing has pinged them — the 5s rate "
            "group member that drives run_round is not wired"
        )

    def test_no_critical_component_is_wedged(self, booted_runtime):
        assert booted_runtime["critical_unresponsive"] == 0, (
            "a component the runtime declares critical stopped answering its ping "
            "while the runtime was up"
        )


class TestLockdepSeesRealAcquisitions:
    def test_locks_are_known(self, booted_runtime):
        from core.runtime.lockdep import lockdep_report

        assert lockdep_report()["known_locks"], (
            "lockdep knows no locks after a boot; its zero-splat claim would "
            "be measured over an empty set"
        )

    def test_acquisitions_were_observed(self, booted_runtime):
        from core.runtime.lockdep import lockdep_report

        assert lockdep_report()["acquires_checked"] > 0

    def test_the_order_is_clean(self, booted_runtime):
        assert mv._lockdep_splats() == 0


def test_all_three_instruments_measure_rather_than_abstain(booted_runtime):
    """The whole point, in one assertion.

    If any of these raises NothingMeasured in a booted runtime, the honest
    report is working and the subsystem behind it is not.
    """
    measured = {}
    for name, probe in (
        ("lockdep", mv._lockdep_splats),
        ("rate_groups", mv._slowest_group_fraction),
        ("health_checker", mv._critical_unresponsive),
    ):
        try:
            measured[name] = probe()
        except mv.NothingMeasured as exc:
            pytest.fail(
                f"{name} measured nothing in a fully booted runtime: {exc}. "
                "The instrument is honest; the subsystem is dead."
            )
    assert set(measured) == {"lockdep", "rate_groups", "health_checker"}
