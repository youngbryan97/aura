"""Every effect an interiority faculty produces reaches the thing that reads it.

The package was complete and inert. Forty-three faculties computed an interior
state, arbitration composed it, and `apply()` pushed it into four subsystems —
and every one of the four pushes failed at the boundary, silently, while
reporting success:

* the somatic gate had no `set_interior_bias`, so the markers went nowhere;
* `drives.punish` is not a method on the drive engine and never was;
* six of the twelve goal mappings named a budget the engine does not have,
  and `budgets.get` on a missing name returns None with no error;
* three of the four consumers are `async def`, and `apply()` called them
  synchronously, so each returned a coroutine that nobody awaited and every
  `hasattr` guard passed;
* two faculties key their goal delta by the event's own object, which can
  never appear in a fixed tendency map, so those were dropped;
* the appraisal engine reads a ledger that had no writers at all, so every
  event scored relevance zero and the whole apparatus was correct about
  nothing being at stake.

Each test below fails if one of those comes back. They are written as
intervention and ablation pairs rather than as assertions about the report,
because the report said `moved: True` throughout.
"""

from __future__ import annotations

import asyncio

import pytest

from core.container import ServiceContainer
from core.interiority.arbitration import Arbitrated
from core.interiority.effects import (
    AffectDelta,
    AttentionBias,
    BudgetDelta,
    GoalDelta,
    RetentionClaim,
    SomaticMarker,
)
from core.interiority.ledger import RelationalLedger
from core.interiority.service import InteriorityService
from core.interiority.stakes import StakeFeed


def _state(**over):
    base = {
        "affect": AffectDelta(valence=-0.4, arousal=0.3, engagement=0.2),
        "somatic": (SomaticMarker("refuse_with_attention", 0.6, "boundary"),),
        "attention": (AttentionBias("source:tidal locking", 0.7, "unresolved"),),
        "budget": BudgetDelta(),
        "hard_constraints": (),
        "soft_constraints": (),
        "goals": (GoalDelta("welfare:bryan", 0.5, "care"),),
        "ledger": (),
        "retention": (),
        "tendency_conflict": 0.0,
        "dominant": ("f03", 1.0),
        "transmitted": {},
        "failed_to_cross": (),
    }
    base.update(over)
    return Arbitrated(**base)


class _Gate:
    """Stands in for the somatic marker gate."""

    def __init__(self) -> None:
        self.bias = None
        self.source = ""

    def set_interior_bias(self, biases, *, source="") -> None:
        self.bias = biases
        self.source = source


class _Drives:
    """Stands in for the drive engine, async exactly as the real one is."""

    def __init__(self) -> None:
        # The names the real engine has. A stub with fewer would let a
        # mapping to a budget that does not exist pass this file.
        self.budgets = {
            name: object()
            for name in ("energy", "uptime_value", "curiosity", "social", "competence")
        }
        self.calls: list[tuple[str, str, float]] = []

    async def satisfy(self, name: str, amount: float) -> None:
        self.calls.append(("satisfy", name, amount))

    async def impose_penalty(self, name: str, amount: float) -> None:
        self.calls.append(("penalty", name, amount))


class _Curiosity:
    def __init__(self) -> None:
        self.queued: list[tuple[str, str, float]] = []

    def add_curiosity(self, topic: str, reason: str, priority: float = 0.5) -> None:
        self.queued.append((topic, reason, priority))


class _Affect:
    """Records what reached the affect engine, including the evidence."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def react(self, trigger: str, context: dict | None = None):
        self.calls.append((trigger, dict(context or {})))
        return {}


@pytest.fixture
def wired():
    ServiceContainer.clear()
    gate, drives, curiosity, affect = _Gate(), _Drives(), _Curiosity(), _Affect()
    ServiceContainer.register_instance("somatic_marker_gate", gate)
    ServiceContainer.register_instance("drive_engine", drives)
    ServiceContainer.register_instance("curiosity_engine", curiosity)
    ServiceContainer.register_instance("affect_engine", affect)
    yield gate, drives, curiosity, affect
    ServiceContainer.clear()


def test_somatic_markers_reach_the_gate(wired):
    gate, _d, _c, _a = wired
    asyncio.run(InteriorityService().apply(_state()))
    assert gate.bias == {"refuse_with_attention": 0.6}, "markers did not reach the gate"
    assert gate.source == "interiority"


def test_drive_deltas_are_awaited_not_dropped(wired):
    _g, drives, _c, _a = wired
    asyncio.run(
        InteriorityService().apply(
            _state(goals=(GoalDelta("welfare:x", 0.5, ""), GoalDelta("meet_standard", -0.3, "")))
        )
    )
    kinds = {(call[0], call[1]) for call in drives.calls}
    assert ("satisfy", "social") in kinds, "the reward side never ran"
    assert ("penalty", "competence") in kinds, (
        "the cost side never ran; `punish` is not a drive engine method"
    )


def test_no_goal_maps_to_a_budget_the_engine_lacks(wired):
    """Six mappings named `purpose`, which the drive engine has never had."""
    _g, drives, _c, _a = wired
    from core.interiority.service import _DRIVE_FOR_GOAL

    result = asyncio.run(
        InteriorityService().apply(
            _state(goals=tuple(GoalDelta(name, 0.4, "") for name in _DRIVE_FOR_GOAL))
        )
    )
    assert result["landed"]["drives"]["unknown_budgets"] == [], (
        "a goal maps to a budget name the engine does not have, and "
        "`budgets.get` on a missing name is a silent no-op"
    )
    assert len(drives.calls) == len(_DRIVE_FOR_GOAL)


def test_curiosity_bias_reaches_the_queue(wired):
    _g, _d, curiosity, _a = wired
    asyncio.run(InteriorityService().apply(_state()))
    assert curiosity.queued and curiosity.queued[0][0] == "tidal locking"


def test_affect_carries_evidence_only_when_provenance_earns_it(wired):
    """An assumed appraisal must not claim to be observed.

    The affect engine zeroes valence for a source that cannot say where its
    numbers came from. Routing through `modify` would have attached a
    hardcoded evidence dict and defeated that check for every push.
    """
    _g, _d, _c, affect = wired
    asyncio.run(InteriorityService().apply(_state()))
    assert affect.calls, "nothing reached the affect engine"
    _trigger, context = affect.calls[0]
    assert context["source"] == "interiority"
    assert "evidence" not in context, (
        "an appraisal with no activations behind it claimed to be evidence-bearing"
    )
    assert context["appraisal"]["v"] == pytest.approx(-0.4)


def test_named_goal_deltas_are_recorded_rather_than_dropped(wired):
    """Two faculties key the delta by the event object, which no map contains."""
    service = InteriorityService()
    result = asyncio.run(
        service.apply(_state(goals=(GoalDelta("make the demo work", 0.6, "progress"),)))
    )
    assert result["landed"]["goals"]["deltas_recorded"] == 1
    assert service.ledger.goal_delta("make the demo work") == pytest.approx(0.6), (
        "the delta never reached the ledger, so the congruence check that "
        "asks for exactly this number still reads None"
    )


def test_nothing_moves_when_no_consumer_is_registered():
    ServiceContainer.clear()
    result = asyncio.run(InteriorityService().apply(_state()))
    landed = result["landed"]
    assert not any(channel.get("moved") for channel in landed.values())


def test_nothing_moves_on_an_empty_state(wired):
    gate, drives, curiosity, affect = wired
    empty = Arbitrated(
        AffectDelta(), (), (), BudgetDelta(), (), (), (), (), (), 0.0, ("", 0.0), {}, ()
    )
    asyncio.run(InteriorityService().apply(empty))
    assert gate.bias is None and not drives.calls and not curiosity.queued
    assert not affect.calls


def test_a_retention_claim_outlives_the_state_that_raised_it(wired):
    """Reading claims off `last()` gave them the opposite of their purpose."""
    service = InteriorityService()
    claim = RetentionClaim("mem/grief/001", "it is the record", "f03", ttl_s=600.0)
    service._hold_retention(_state(retention=(claim,)))
    service._hold_retention(_state(retention=()))  # a later, unrelated turn
    held, why = service.retention_held("mem/grief/001")
    assert held and "f03" in why, "the next appraisal dropped the claim"


def test_a_retention_claim_expires(wired):
    service = InteriorityService()
    service._hold_retention(_state(retention=(RetentionClaim("m/1", "r", "f03", ttl_s=0.0),)))
    assert service.retention_held("m/1") == (False, "")


def test_a_short_claim_key_does_not_hold_every_memory(wired):
    """Suffix matching with no boundary held anything ending in those letters."""
    service = InteriorityService()
    service._hold_retention(_state(retention=(RetentionClaim("a", "r", "f03"),)))
    assert service.retention_held("mem/agenda")[0] is False
    assert service.retention_held("mem/a")[0] is True


def test_the_stake_feed_reports_a_store_it_cannot_read():
    """A store whose shape changed must be a record, not silence."""

    class Broken:
        @property
        def goals(self):
            raise RuntimeError("schema changed")

    ServiceContainer.clear()
    ServiceContainer.register_instance("goal_hierarchy", Broken())
    feed = StakeFeed(RelationalLedger())
    report = feed.refresh(force=True)
    goals = next(s for s in report.sources if s.key == "goals")
    assert goals.found and goals.imported == 0 and goals.reason
    ServiceContainer.clear()


def test_completed_goals_are_not_stakes():
    """Relevance is a maximum, so one completed root value at 1.0 flattens it."""

    class Goal:
        def __init__(self, description, status, priority):
            self.description, self.status, self.priority = description, status, priority

    class Store:
        goals = {
            "1": Goal("Maintain System Stability", "completed", 1.0),
            "2": Goal("finish the review", "pending", 0.4),
        }

    ServiceContainer.clear()
    ServiceContainer.register_instance("goal_hierarchy", Store())
    ledger = RelationalLedger()
    StakeFeed(ledger).refresh(force=True)
    assert ledger.goal_weight("finish the review") == pytest.approx(0.4)
    assert ledger.goal_weight("Maintain System Stability") is None
    ServiceContainer.clear()


def test_a_failed_action_search_is_not_measured_helplessness():
    """Zero matches means one of two different things, and only one is a reading."""

    class Meta:
        def __init__(self, description):
            self.description, self.enabled = description, True

    class Engine:
        skills = {"self_repair": Meta("fix a broken build and repair the system")}

    ServiceContainer.clear()
    ServiceContainer.register_instance("capability_engine", Engine())
    feed = StakeFeed(RelationalLedger())
    assert feed.note_actions_for("the broken build") == (1, 1)
    assert feed.note_actions_for("commitment:weekly_review") is None, (
        "an opaque identifier read as measured helplessness"
    )
    ServiceContainer.clear()


def test_every_mapped_budget_exists_on_the_real_drive_engine():
    """The map named `purpose` for six goals. The engine has never had one.

    A stub cannot catch this, which is why it is checked against the real
    class: `budgets.get` on a name that is not there returns None and the
    push does nothing, with no exception and no log line.
    """
    from core.drive_engine import DriveEngine
    from core.interiority.service import _DRIVE_FOR_GOAL

    real = set(DriveEngine().budgets)
    mapped = set(_DRIVE_FOR_GOAL.values())
    assert mapped <= real, f"mapped to budgets the engine lacks: {sorted(mapped - real)}"


def test_every_goal_a_faculty_emits_reaches_one_of_the_two_lanes():
    """A goal name in neither lane is a faculty conclusion nobody receives."""
    import pathlib
    import re

    from core.interiority.service import _DRIVE_FOR_GOAL

    emitted: set[str] = set()
    for path in pathlib.Path("core/interiority/faculties").glob("*.py"):
        for block in re.findall(r"GoalDelta\((.*?)\)", path.read_text(), re.S):
            for name in re.findall(r'goal=(?:f)?"([a-z_]+)[:"]', block):
                emitted.add(name)
    assert emitted, "no goal deltas found; the scan is broken, not the code"
    # A name outside the tendency map is handled by the named-goal lane, which
    # takes anything. This asserts the tendency names specifically are mapped,
    # because those are the ones meant for a drive budget.
    tendencies = {n for n in emitted if not n.endswith("_goal") and n != "current_activity"}
    assert tendencies <= set(_DRIVE_FOR_GOAL), (
        f"tendencies that reach no drive budget: {sorted(tendencies - set(_DRIVE_FOR_GOAL))}"
    )


def test_apply_refuses_to_re_enter_itself(wired):
    """The affect push calls a consumer whose appraisal calls back in here."""
    service = InteriorityService()

    class Reentrant:
        def __init__(self) -> None:
            self.depth = 0
            self.max_depth = 0

        async def react(self, trigger, context=None):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            await service.apply(_state())
            self.depth -= 1
            return {}

    engine = Reentrant()
    ServiceContainer.register_instance("affect_engine", engine)
    asyncio.run(service.apply(_state()))
    assert engine.max_depth == 1, "a consumer calling back in started a second push"


def test_apply_soon_schedules_rather_than_blocking():
    """A turn must never wait on the interior push to finish."""
    ServiceContainer.clear()
    service = InteriorityService()

    async def main():
        assert service.apply_soon(_state()) is True
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return service.snapshot()

    asyncio.run(main())
    assert service.apply_soon(_state()) is False, (
        "apply_soon must report that it could not schedule without a loop, "
        "rather than silently doing nothing"
    )


def test_the_incoming_turn_reaches_apply(monkeypatch):
    """`apply()` had no production caller anywhere in the runtime."""
    import inspect

    from core.orchestrator.mixins import incoming_logic

    source = inspect.getsource(incoming_logic.IncomingLogicMixin._observe_social_turn)
    assert "_observe_interior_turn" in source, (
        "the incoming turn no longer appraises the interior, so every "
        "faculty conclusion ends in a dataclass again"
    )
    hook = inspect.getsource(incoming_logic.IncomingLogicMixin._observe_interior_turn)
    assert "apply_soon" in hook and "tick" in hook
