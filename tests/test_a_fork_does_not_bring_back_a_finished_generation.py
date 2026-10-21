"""A fork carries her, not the process's bookkeeping about its own generations.

The reports dry run of 21 October lost two of thirty-two arms to "generation
gate saturated" while the abort found no generation running. The fork's module
scan carries every module-level table in her packages, and the gate's lease
table is one: a lease taken before the snapshot came back on every restore,
with nothing behind it. A module now names its process state, and the fork
leaves it alone.
"""
from __future__ import annotations

import core.brain.llm_health_router as router
from core.subject import snapshot


def test_the_gate_tables_are_not_among_what_a_fork_carries():
    held = snapshot._module_holdings()
    for name in router.__process_state__:
        assert f"core.brain.llm_health_router:{name}" not in held


def test_a_lease_released_after_the_snapshot_stays_released_after_a_restore():
    saved = snapshot._module_holdings()
    lease = router._mark_generation_gate_acquired("system:unknown", timeout_s=30.0)
    try:
        assert lease in router._GENERATION_GATE_ACTIVE_LEASES
    finally:
        with router._GENERATION_GATE_STATE_LOCK:
            router._GENERATION_GATE_ACTIVE_LEASES.pop(lease, None)
            router._GENERATION_GATE_LEASE_DEADLINES.pop(lease, None)
    assert all(not key.startswith("core.brain.llm_health_router:_GENERATION_GATE") for key in saved)
    assert lease not in router._GENERATION_GATE_ACTIVE_LEASES


def test_her_own_module_state_is_still_carried():
    import core.agency.consequence as consequence

    consequence.get_consequence_model()
    assert "core.agency.consequence:_MODEL" in snapshot._module_holdings()
