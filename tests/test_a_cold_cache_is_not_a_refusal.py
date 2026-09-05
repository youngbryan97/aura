"""Every permission reading as denied because nothing had read them yet.

Settings reads never touch the filesystem. They answer from a snapshot a
worker keeps current, and until that worker has run once there is no snapshot,
so a read falls back to the fail-closed values. That is the right answer to
give and the wrong one to believe: it cannot be told apart from the person
having switched every permission off.

So anything short-lived, or anything that asks once at startup, reads its
permissions as denied however they were actually set. LIVE 2026-09-05: screen
access was on, the first read said off, and a run reported having nothing to
look at while the screen was in use the whole time.
"""

from __future__ import annotations

import asyncio
import time

import core.runtime.runtime_settings as settings
from core.runtime.runtime_settings import wait_until_settled


def test_waiting_returns_whether_it_actually_settled():
    assert wait_until_settled(2.0) is True


def test_an_already_settled_cache_costs_nothing():
    wait_until_settled(2.0)
    began = time.monotonic()
    assert wait_until_settled(5.0) is True
    assert time.monotonic() - began < 0.05


def test_it_is_bounded_by_what_it_was_given():
    """Nothing waits on this forever, whatever the worker is doing."""
    began = time.monotonic()
    settled = wait_until_settled(0.2)
    waited = time.monotonic() - began
    assert isinstance(settled, bool)
    assert waited < 1.0, f"it waited {waited:.2f}s on a 0.2s bound"


def test_a_zero_wait_answers_immediately():
    began = time.monotonic()
    assert isinstance(wait_until_settled(0.0), bool)
    assert time.monotonic() - began < 0.2


def test_a_settled_read_is_the_persisted_one_rather_than_the_fallback():
    """The fail-closed value must not be the last word once the file is read."""
    from core.runtime.runtime_settings import get_runtime_setting

    assert wait_until_settled(2.0)
    # A key nobody set falls back to what the caller asked for, rather than
    # to the fail-closed constant standing in for every answer.
    assert get_runtime_setting("nothing.has.this.key", "asked-for") == "asked-for"


def test_a_pursuit_lets_the_settings_land_before_believing_a_refusal():
    """The whole point: a cold process must not report a locked-out screen."""
    from core.skills.screen_pursuit import wait_for_a_screen_to_look_at

    began = time.monotonic()
    asyncio.run(wait_for_a_screen_to_look_at(began + 20.0))
    # It either looked, or refused for a reason it can name — never because
    # nothing had loaded the settings yet.
    import core.skills.screen_pursuit as sp

    assert sp._WHY_SHE_CANNOT_LOOK["value"] != "runtime_setting_disabled" or not (
        settings.get_runtime_setting("permissions.screen", False)
    )
