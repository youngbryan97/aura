from types import SimpleNamespace

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
