"""A restore of her state leaves the process's governance as it found it.

Seed 73's campaign wrote its model-lane heartbeat inside a governed scope on a
thread of its own while the main thread restored anchors. One restore put back
an older copy of the lease registry, or moved her clock past the token's thirty
seconds, the write read as ungoverned, the lane lost its lease and the shutdown
that followed refused the run's last step. Both routes are closed by naming the
governance context as machinery, which neither the snapshot nor the clock
touches.
"""

from __future__ import annotations

from core.governance_context import is_governed, local_internal_governed_scope
from core.subject import snapshot
from core.subject.clock import ExperimentClock


def test_a_snapshot_does_not_carry_the_governance_context() -> None:
    held = [key for key in snapshot._module_holdings() if key.startswith("core.governance_context:")]
    assert held == []


def test_a_restore_inside_an_open_scope_leaves_it_governed() -> None:
    saved = snapshot._module_state()
    with local_internal_governed_scope("test.restore_inside_scope", domain="state_mutation"):
        assert is_governed()
        snapshot._restore_module_state(saved)
        assert is_governed()


def test_her_clock_moving_an_hour_leaves_an_open_scope_governed() -> None:
    clock = ExperimentClock(1.0)
    clock.install()
    try:
        with local_internal_governed_scope("test.clock_inside_scope", domain="state_mutation"):
            assert is_governed()
            clock.set(clock.now() + 3600.0)
            assert is_governed()
    finally:
        clock.uninstall()
