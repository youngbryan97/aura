"""Every arm of a paired trial was reading the consequences of every other arm.

Her integration reading is computed from the consequence bus: `SystemIntegration`
takes the last two hundred consequences and asks how much her organs cause one
another. The bus lives in `core.runtime`, which a snapshot treats as the process's
own machinery and never carries. So the stream was shared by every fork in the
process, a restore rewound her and not it, and each arm read what the arms before
it had done.

MEASURED 25 October on the seed-23 anchors, forking untouched arms and rolling a
late fork of one anchor after each cut: its integration read 0.3225, then 0.3001,
0.2869, 0.2776, 0.2706, 0.2500, 0.2359 and, at the seventh cut, 0 — "federated".
The ghost flagged her integrity, a self-integrity content entered her workspace,
and her fragmentation rose by 0.0014, which is the separation the late-fork control
of the seed-23 campaign read. K was identical at every restore throughout. With the
bus history forking, the same churn read K apart by 0 at all twelve cuts.

A machinery module now names what forks with her (`__forks_with_her__`), as
`__process_state__` names what does not.
"""
from __future__ import annotations

import pytest

from core.runtime import consequence_bus as cb
from core.subject import snapshot as sn

pytestmark = pytest.mark.unit


def _event(n: int) -> cb.ConsequenceEvent:
    return cb.ConsequenceEvent(
        event_id=f"evt-{n}", timestamp=float(n), source=f"organ_{n % 3}",
        domain="tool_execution", action_content=f"act {n}",
    )


@pytest.fixture
def bus():
    cb.ConsequenceBus.reset()
    made = cb.ConsequenceBus.get()
    yield made
    cb.ConsequenceBus.reset()


def test_the_singleton_keeps_its_history_where_a_fork_can_reach_it(bus):
    assert bus._history is cb._HISTORY
    bus.publish(_event(1))
    assert cb._HISTORY[-1].action_content == "act 1"


def test_trimming_keeps_the_history_it_shares(bus):
    bus._max_history = 5
    for n in range(12):
        bus.publish(_event(n))
    assert bus._history is cb._HISTORY, "a rebinding trim leaves what a fork restores behind"
    assert [e.action_content for e in cb._HISTORY] == [f"act {n}" for n in range(7, 12)]


def test_a_bus_made_on_its_own_keeps_its_own(bus):
    alone = cb.ConsequenceBus()
    alone.publish(_event(9))
    assert alone._history is not cb._HISTORY
    assert not any(e.action_content == "act 9" for e in cb._HISTORY)


def test_the_snapshot_carries_what_a_machinery_module_names(bus):
    held = sn._module_holdings()
    assert "core.runtime.consequence_bus:_HISTORY" in held
    # And nothing else from machinery: the subscribers are wiring.
    machinery = [k for k in held if sn._in_packages(k.partition(":")[0], sn._MACHINERY_PACKAGES)]
    assert machinery == ["core.runtime.consequence_bus:_HISTORY"], machinery


def test_a_restore_rewinds_the_consequences_an_arm_published(bus):
    for n in range(3):
        bus.publish(_event(n))
    saved = sn._module_state({"core.runtime.consequence_bus:_HISTORY"})
    # The arm runs and publishes its own consequences.
    for n in range(3, 40):
        bus.publish(_event(n))
    sn._restore_module_state(saved)
    assert bus._history is cb._HISTORY, "restored in place, so the live bus sees it"
    assert [e.action_content for e in bus.recent_events(200)] == ["act 0", "act 1", "act 2"]


def test_her_integration_reads_the_same_after_a_restore(bus):
    from core.ghost.causal_integration import SystemIntegration

    for n in range(24):
        bus.publish(_event(n))
    saved = sn._module_state({"core.runtime.consequence_bus:_HISTORY"})
    first = SystemIntegration(bus=bus).report(now=100.0, force=True).phi_system
    for n in range(24, 220):
        bus.publish(cb.ConsequenceEvent(event_id=f"other-{n}", timestamp=float(n), source="another_arm",
                                        domain="tool_execution", action_content="x"))
    sn._restore_module_state(saved)
    again = SystemIntegration(bus=bus).report(now=100.0, force=True).phi_system
    assert again == first
