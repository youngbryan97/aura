"""A turn typed during boot must wait, not be answered with a refusal.

LIVE 2026-08-17. The first message after launch got "the live answer lane could
not finish preparing before a reasoning turn began." Ten seconds later the same
message served normally. The log line that announces a boot wait — "waiting up
to Ns for boot rather than failing the turn" — appeared ZERO times, so the turn
never waited at all.

The predicate guarding that wait short-circuited on `last_change is None`.
BootPhases sets `last_change` to a human-readable string ("organ: a -> b") only
once some organ has transitioned, so it is None during early boot: the exact
window the wait exists for. Meanwhile `last_transition_at` was validly set.

The API server accepts chat the moment the port binds — the log says so
outright, "GUI launchable while boot_phase=conversation_warming ready=False" —
so this is what a user typing immediately after launch actually hits.
"""

from __future__ import annotations

import time

from interface.routes.chat import _boot_is_still_in_progress


class _Phases:
    def __init__(self, *, ready: bool, last_change, last_transition_at: float) -> None:
        self._ready = ready
        self.last_change = last_change
        self.last_transition_at = last_transition_at

    def ready(self) -> bool:
        return self._ready


def test_early_boot_waits_even_before_any_organ_transitions() -> None:
    """The regression: last_change is None here, and this is when to wait."""
    phases = _Phases(ready=False, last_change=None, last_transition_at=time.time())

    assert _boot_is_still_in_progress(phases) is True


def test_a_transitioning_boot_still_waits() -> None:
    phases = _Phases(
        ready=False,
        last_change="cortex: starting -> warming",
        last_transition_at=time.time(),
    )

    assert _boot_is_still_in_progress(phases) is True


def test_a_ready_runtime_does_not_wait() -> None:
    phases = _Phases(ready=True, last_change=None, last_transition_at=time.time())

    assert _boot_is_still_in_progress(phases) is False


def test_a_stalled_boot_does_not_hold_the_turn() -> None:
    """Machinery that never ran must not hang the turn for its whole budget."""
    phases = _Phases(ready=False, last_change=None, last_transition_at=time.time() - 3600)

    assert _boot_is_still_in_progress(phases) is False


def test_an_unset_timestamp_does_not_wait() -> None:
    phases = _Phases(ready=False, last_change=None, last_transition_at=0.0)

    assert _boot_is_still_in_progress(phases) is False


def test_no_phases_object_does_not_wait() -> None:
    assert _boot_is_still_in_progress(None) is False


def test_a_raising_phases_object_is_safe() -> None:
    class _Broken:
        last_change = None
        last_transition_at = 0.0

        def ready(self):
            raise RuntimeError("boot machinery down")

    assert _boot_is_still_in_progress(_Broken()) is False
