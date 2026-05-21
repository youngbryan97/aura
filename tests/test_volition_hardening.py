from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker
from core.volition import VolitionEngine


class SampleSoul:
    def __init__(self, name: str, urgency: float):
        self.drive = SimpleNamespace(name=name, urgency=urgency)

    def get_dominant_drive(self):
        return self.drive


class SampleQuestion:
    id = "q1"
    question = "Does substrate continuity change memory consolidation?"
    domain = "mind"
    urgency = 0.8
    research_attempts = 1
    status = "open"

    @staticmethod
    def freshness():
        return 0.75


def _orchestrator(**overrides):
    values = {
        "cognitive_engine": None,
        "status": SimpleNamespace(running=True),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_lowercase_competence_drive_generates_self_diagnosis_goal():
    engine = VolitionEngine(_orchestrator(soul=SampleSoul("competence", 0.8)))

    goal = engine._check_soul_drives()

    assert goal is not None
    assert goal["origin"] == "intrinsic_competence"
    assert "self-diagnosis" in goal["objective"]


def test_lowercase_curiosity_drive_generates_curiosity_goal():
    engine = VolitionEngine(_orchestrator(soul=SampleSoul("curiosity", 0.8)))

    goal = engine._check_soul_drives()

    assert goal is not None
    assert goal["origin"] == "intrinsic_curiosity"


def test_inquiry_engine_active_question_generates_grounded_research_goal():
    inquiry = SimpleNamespace(get_active_question=lambda: SampleQuestion())
    engine = VolitionEngine(_orchestrator(inquiry_engine=inquiry))
    engine.inquiry_goal_cooldown = 0

    goal = engine._check_inquiry_engine()

    assert goal is not None
    assert goal["origin"] == "intrinsic_inquiry"
    assert goal["tools"] == [{"name": "web_search", "payload": SampleQuestion.question}]
    assert goal["context"]["question_id"] == "q1"
    assert goal["context"]["urgency"] == 0.8
    assert "InquiryEngine" in goal["objective"]


def test_inquiry_engine_goal_respects_cooldown():
    inquiry = SimpleNamespace(get_active_question=lambda: SampleQuestion())
    engine = VolitionEngine(_orchestrator(inquiry_engine=inquiry))
    engine.inquiry_goal_cooldown = 999
    engine.last_inquiry_goal_time = 10**9

    assert engine._check_inquiry_engine() is None


@pytest.mark.asyncio
async def test_tick_fails_closed_when_unified_will_is_unavailable(monkeypatch):
    import core.will as will

    get_degradation_tracker().reset()
    engine = VolitionEngine(_orchestrator())

    def broken_get_will():
        broken_get_will.called = True
        raise RuntimeError("will offline")

    async def should_not_search():
        should_not_search.called = True
        raise AssertionError("volition searched for goals after Will failure")

    monkeypatch.setattr(will, "get_will", broken_get_will)
    monkeypatch.setattr(engine, "_search_for_autonomous_goals", should_not_search)

    assert await engine.tick(current_goal=None) is None
    assert getattr(broken_get_will, "called", False) is True
    assert getattr(should_not_search, "called", False) is False
    assert any(
        "suppressed autonomous volition tick" in record.action
        for record in get_degradation_tracker().recent(subsystem="volition_will_gate")
    )


def test_connection_drive_fails_closed_when_governance_preflight_fails(monkeypatch):
    import core.will as will

    get_degradation_tracker().reset()
    engine = VolitionEngine(_orchestrator(soul=SampleSoul("connection", 0.95)))
    engine.last_speak_time = 0.0

    def broken_get_will():
        broken_get_will.called = True
        raise RuntimeError("will offline")

    monkeypatch.setattr(will, "get_will", broken_get_will)

    assert engine._check_soul_drives() is None
    assert getattr(broken_get_will, "called", False) is True
    assert engine.unanswered_speak_count == 0
    assert any(
        "suppressed connection-drive speech" in record.action
        for record in get_degradation_tracker().recent(subsystem="volition_connection_governance")
    )


def test_inquiry_engine_unreadable_question_state_is_recorded():
    class BrokenQuestion(SampleQuestion):
        def freshness(self):
            self.freshness_called = True
            raise RuntimeError("freshness unavailable")

    get_degradation_tracker().reset()
    inquiry = SimpleNamespace(get_active_question=lambda: BrokenQuestion())
    engine = VolitionEngine(_orchestrator(inquiry_engine=inquiry))
    engine.inquiry_goal_cooldown = 0

    assert engine._check_inquiry_engine() is None
    assert any(
        "question state was unreadable" in record.action
        for record in get_degradation_tracker().recent(subsystem="volition_inquiry_question_state")
    )


def test_blank_interest_is_ignored_without_dirtying_interest_catalog():
    engine = VolitionEngine(_orchestrator())
    before = (
        list(engine.general_interests),
        list(engine.fun_interests),
        list(engine.technical_interests),
    )

    engine.add_interest("   ", category="technical")

    assert (
        engine.general_interests,
        engine.fun_interests,
        engine.technical_interests,
    ) == before
