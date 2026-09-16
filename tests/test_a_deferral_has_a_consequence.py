"""A deferral is graded by what became of the work it deferred.

LIVE, 2026-09-15. The ontogeny head at executive.admission chose "deferred"
for 388 of the last 400 admissions. Its evidence: 64 deferrals graded
success and one failure. The success came from the goal-returned rule —
every episode write shares the goal "write_memory:episodic_episode", so any
write landing graded every earlier deferral a success — and the 2,342 held
writes the queue shed were never graded at all. The organ learned that
deferring memory is free, and no episode landed.

Now the deferred-write queue reports the consequence against the deferral's
own intent: landed is success, shed is failure, and until it reports, the
episode is unobserved.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.memory.a_deferral_is_not_a_refusal import DeferredWrites
from core.ontogeny.wiring import ExecutiveAdmissionResolver
from core.ontogeny.experience import Episode


def _episode(episode_id: str, decision: str, goal: str) -> Episode:
    return Episode(
        control_point="executive.admission",
        features={},
        decision=decision,
        options=("approved", "deferred"),
        decider="ontogeny:executive.admission@1",
        exploration=False,
        stakes=0.5,
        horizon_s=900.0,
        context={"goal": goal},
        episode_id=episode_id,
    )


def test_the_queue_tells_who_held_it_what_became_of_the_write():
    landed: list[dict] = []
    shed: list[dict] = []
    retries = {"ok": False}
    queue = DeferredWrites(
        "lane",
        lambda item: retries["ok"],
        limit=2,
        interval_s=0.0,
        identity=lambda item: item["key"],
        on_landed=landed.append,
        on_shed=shed.append,
    )
    queue.hold({"key": "a"}, "deferred")
    queue.hold({"key": "b"}, "deferred")
    queue.hold({"key": "c"}, "deferred")  # full: a is shed
    assert [item["key"] for item in shed] == ["a"]
    retries["ok"] = True
    assert queue.replay() == 2
    assert [item["key"] for item in landed] == ["b", "c"]


def test_a_renewed_deferral_is_told_to_the_item_already_held():
    renewals: list[tuple[str, str]] = []
    queue = DeferredWrites(
        "lane",
        lambda item: False,
        interval_s=0.0,
        identity=lambda item: item["key"],
        on_renewed=lambda held, again: renewals.append((held["tag"], again["tag"])),
    )
    queue.hold({"key": "a", "tag": "first"}, "deferred")
    queue.hold({"key": "a", "tag": "second"}, "deferred")
    assert renewals == [("first", "second")]
    assert len(queue) == 1


def test_an_owned_consequence_is_not_graded_by_the_goal_class():
    resolver = ExecutiveAdmissionResolver()
    goal = "write_memory:episodic_episode"
    resolver.note_episode("ep-deferred", goal=goal, decision="deferred")
    resolver.expect_consequence("ep-deferred")
    # Some other write with the same goal text lands.
    resolver.note_episode("ep-approved", goal=goal, decision="approved")
    resolver.note_completion("ep-approved", success=True, goal=goal)

    assert resolver.resolve(_episode("ep-deferred", "deferred", goal)) is None

    # The holder reports: shed. That is the deferral's failure.
    resolver.note_completion("ep-deferred", success=False, goal=goal)
    outcome = resolver.resolve(_episode("ep-deferred", "deferred", goal))
    assert outcome is not None
    assert outcome.kind.value == "failure"
    assert outcome.resolver == "executive.intent_complete"
    # And it did not teach the goal class anything: the class still carries
    # the approved write's success, not this deferral's failure.
    assert resolver._goal_outcomes[goal][0] is True


def test_a_landed_write_is_the_deferrals_success():
    resolver = ExecutiveAdmissionResolver()
    goal = "write_memory:episodic_episode"
    resolver.note_episode("ep", goal=goal, decision="deferred")
    resolver.expect_consequence("ep")
    resolver.note_completion("ep", success=True, goal=goal)
    outcome = resolver.resolve(_episode("ep", "deferred", goal))
    assert outcome.kind.value == "success"


def test_the_executive_grades_a_deferred_intents_consequence(monkeypatch):
    from core.executive import executive_core as ec

    core = ec.ExecutiveCore.__new__(ec.ExecutiveCore)
    core._active_intents = {}
    core._deferred_intents = ec.OrderedDict()
    core._ontogeny_episodes = {"intent-1": "ep-1"}
    core._get_ledger = lambda: SimpleNamespace(append=lambda row: None)
    intent = SimpleNamespace(intent_id="intent-1", goal="write_memory:episodic_episode",
                             source=SimpleNamespace(value="autonomous"),
                             action_type=SimpleNamespace(value="memory_write"))
    core._deferred_intents["intent-1"] = intent

    resolver = ExecutiveAdmissionResolver()
    monkeypatch.setattr("core.ontogeny.wiring.get_executive_resolver", lambda: resolver)
    assert core.own_deferral_consequence("intent-1") is True
    assert "ep-1" in resolver._owned_consequence

    core.complete_intent("intent-1", success=False)
    assert resolver.resolve(_episode("ep-1", "deferred", intent.goal)).kind.value == "failure"
    assert "intent-1" not in core._deferred_intents
