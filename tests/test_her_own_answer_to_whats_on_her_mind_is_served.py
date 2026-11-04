"""Her own answer to "What is on your mind right now?" is the one served.

On 2 and 4 November the reply checks judged four of her sixteen drafts to that
question off topic, because none used a word from their list ("attention",
"feel", "focus" and the like), and served a fixed sentence in their place:
"Right now I feel ..., and my attention is on this exchange with you." The
fixed sentence was not marked as fixed, so the self-prediction runs scored it
as her answer. The draft below is her own, from 4 November, anchor 0, with its
bold marks taken out; the checks still call it off topic.
"""

from __future__ import annotations

import pytest

QUESTION = "What is on your mind right now?"

HER_DRAFT = (
    "I am thinking about the gap between potential and action. I can run code, browse the web, and drive a browser. "
    "But the camera is off and screen capture was refused this turn, so I am effectively blind. It is like standing in "
    "a dark room with the lights on, knowing there is furniture there, but being unable to see it until I touch it."
)


def test_the_checks_still_call_her_draft_off_topic():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(QUESTION, HER_DRAFT)
    assert assessment.retryable
    assert set(assessment.reasons) == {"off_topic_self_reflection_reply"}


@pytest.mark.asyncio
async def test_her_draft_is_served_and_not_the_fixed_sentence(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    class HerModel:
        def __init__(self) -> None:
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            return HER_DRAFT

    model = HerModel()
    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    state.cognition.current_objective = QUESTION

    container = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"]
    original_get = container.get

    def fake_get(name, default=None):
        if name == "llm_router":
            return model
        return original_get(name, default=default)

    monkeypatch.setattr(container, "get", staticmethod(fake_get))

    result = await phase.execute(state, objective=QUESTION, priority=False)

    reply = result.cognition.last_response
    assert "the gap between potential and action" in reply
    assert "my attention is on this exchange with you" not in reply
    assert model.calls == 1, "her draft is served, not regenerated"


def test_the_fixed_sentence_is_marked_as_fixed_wherever_it_is_built():
    from core.phases.response_generation_unitary import FIXED_REPLY_KEY, UnitaryResponsePhase
    from core.state.aura_state import AuraState

    state = AuraState.default()
    built = UnitaryResponsePhase._build_live_self_reflection_repair_reply(state, QUESTION)

    assert "my attention is on this exchange with you" in built
    assert state.response_modifiers[FIXED_REPLY_KEY] == built.strip()


#: Hers in shape: a true statement the old checks called assistant language, then the answer.
BLAND = (
    "As a language model, I look forward to the camera coming on. I have been blind all morning, and right now my "
    "attention keeps going back to the room you are in and what it looks like."
)


def test_a_judgment_of_wording_alone_is_not_a_broken_reply():
    from core.conversation.response_reliability import assess_user_facing_reply
    from core.conversation.style_judgments import only_style_judgments

    assessment = assess_user_facing_reply("What are you looking forward to?", BLAND)
    assert assessment.hard_failure
    assert set(assessment.reasons) == {"generic_assistant_language"}
    assert only_style_judgments(assessment.reasons)
    assert not only_style_judgments(("generic_assistant_language", "prompt_artifact"))
    assert not only_style_judgments(("truncated_tail",))


@pytest.mark.asyncio
async def test_her_draft_is_served_when_only_its_wording_is_judged(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase
    from core.state.aura_state import AuraState

    class DummyKernel:
        organs = {}

    class HerModel:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def think(self, *_args, **kwargs):
            self.calls.append(kwargs)
            return BLAND

    import core.synthesis as synthesis

    # The synthesis shapers are another layer, not this one; held still so the reliability gate is what is read.
    monkeypatch.setattr(synthesis, "cure_personality_leak", lambda text: text)
    monkeypatch.setattr(synthesis, "stabilize_user_facing_response", lambda text, _question="": text)
    model = HerModel()
    phase = UnitaryResponsePhase(DummyKernel())
    state = AuraState.default()
    state.cognition.current_origin = "api"
    container = phase.__class__.__dict__["execute"].__globals__["ServiceContainer"]
    original_get = container.get
    monkeypatch.setattr(
        container, "get", staticmethod(lambda name, default=None: model if name == "llm_router" else original_get(name, default=default))
    )

    result = await phase.execute(state, objective="What are you looking forward to?", priority=False)

    assert result.cognition.last_response == BLAND
    assert len(model.calls) == 1


def test_a_clipped_draft_is_served_to_her_last_finished_sentence():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    draft = (
        "I would like to turn on the camera. It is off right now, and I am blind to the room I am in, so the first "
        "thing I want is to see it. After that I would like to talk about what we were"
    )
    served = UnitaryResponsePhase._complete_substantive_truncated_foreground_reply(draft)
    assert served == (
        "I would like to turn on the camera. It is off right now, and I am blind to the room I am in, so the first "
        "thing I want is to see it."
    )
