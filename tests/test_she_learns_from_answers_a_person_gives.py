"""An answer a person gives in her chat, for the request she just answered, is kept as a solved example she met.

Learning in use (core/learning/examples_she_meets.py) induces new kinds of
problem from examples she meets, while she is idle. Until 16 October nothing in
her chat recorded one, so she had never learned a kind from anything she met.
"""

from __future__ import annotations

import pytest

from core.learning.examples_she_meets import a_person_answered

GCD = "What's the greatest common divisor of 4862 and 7854?"
SORT = "Sort these words alphabetically: walnut cedar maple birch alder"


@pytest.mark.parametrize(
    ("request_", "said", "answer"),
    [
        (GCD, "No, it's 374.", "374"),
        (GCD, "the answer is 374", "374"),
        (GCD, "Actually it should be 374 because 2*11*17 = 374", "374"),
        (GCD, "no, 374", "374"),
        (GCD, "Wrong — the correct answer is 374. Try again?", "374"),
        (SORT, "No, it's alder birch cedar maple walnut", "alder birch cedar maple walnut"),
        (GCD, "Yes, the answer is 374.", "374"),
        (GCD, "Right, it's 374", "374"),
    ],
)
def test_a_stated_answer_is_read(request_, said, answer):
    assert a_person_answered(request_, "It's 22.", said) == answer


@pytest.mark.parametrize(
    "said",
    ["No, it's fine.", "it's raining here", "no", "thanks!", "What's the lcm of 4 and 6?", "No, it's not 22.",
     "actually it was a great answer", "yes", "Yes, that's right!", "right, it's late, see you tomorrow"],
)
def test_a_turn_that_states_no_answer_is_left_alone(said):
    # A wrong known answer would retire a kept procedure that is right, so nothing unreadable is kept.
    assert a_person_answered(GCD, "It's 22.", said) is None


def test_only_a_request_she_answered_has_an_answer_to_keep():
    assert a_person_answered(GCD, "", "the answer is 374") is None
    assert a_person_answered("hi there", "Hello!", "it's 5") is None


@pytest.mark.asyncio
async def test_her_chat_keeps_the_answer_a_person_gives(monkeypatch):
    from core.brain import cognitive_engine as cognitive_engine_module
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.learning import examples_she_meets
    from core.phases import kept_procedures_in_a_turn
    from core.utils.injected_blocks import stamp_runtime_payload

    kept: list = []

    async def note(problem, answer, source):
        kept.append((problem, answer, source))
        return True

    async def no_kept_answer(*_args, **_kwargs):
        return None

    class Router:
        async def think(self, **_kwargs):
            return "Thank you, 374 it is."

        def get_last_generation_metadata(self):
            return {"surface_control_receipt": {"generation_stop_reason": "eos", "surface_quality_gate_reasons": []}}

    class Container:
        def get(self, name, default=None):
            return Router() if name == "llm_router" else default

    monkeypatch.setattr(examples_she_meets, "note_met_async", note)
    monkeypatch.setattr(kept_procedures_in_a_turn, "kept_answer_for", no_kept_answer)
    monkeypatch.setattr(cognitive_engine_module, "get_container", lambda: Container())
    exchanges = [stamp_runtime_payload({"user": GCD, "aura": "It's 22."})]
    await CognitiveEngine()._direct_desktop_quick_reply(
        "No, it's 374.", ThinkingMode.FAST, "user",
        {"desktop_quick_reply_contract": True, "visible_user_message": "No, it's 374.", "max_tokens": 512,
         "recent_completed_exchanges": exchanges},
        timeout_s=30.0,
    )
    assert kept == [(GCD, "374", "a_person_gave_it")]


@pytest.mark.asyncio
async def test_a_request_programs_can_settle_is_checked_once_while_she_is_idle(monkeypatch, tmp_path):
    from core.brain import reasoning_amplifier_v2, reasoning_background
    from core.learning import examples_she_meets

    monkeypatch.setattr(examples_she_meets, "_root", lambda: tmp_path)
    assert await examples_she_meets.note_to_check_async(GCD)
    assert examples_she_meets.next_to_check()["problem"] == GCD

    ran: list = []

    class Receipt:
        promotion_authority = "independent_executable_consensus"

    class Result:
        receipt = Receipt()

    async def amplify(objective, generate, **kwargs):
        ran.append((objective, kwargs["sample_budget"], kwargs["extra_context"]["enable_executable_reasoning"]))
        assert await generate("write a program", 0.7) == "print(374)"
        return Result()

    class Router:
        async def think(self, prompt, **kwargs):
            assert kwargs["is_background"] is True and kwargs["foreground_request"] is False
            return {"content": "print(374)"}

    monkeypatch.setattr(reasoning_amplifier_v2, "amplify_turn", amplify)
    monkeypatch.setattr(reasoning_amplifier_v2, "search_allowance", lambda *a, **k: (60.0, 180.0))
    out = await reasoning_background._check_one_she_answered(Router())
    assert out == {"status": "checked", "authority": "independent_executable_consensus"}
    assert ran == [(GCD, 3, True)]
    assert examples_she_meets.next_to_check() is None


@pytest.mark.asyncio
async def test_a_request_programs_cannot_settle_is_set_aside_without_her_model(monkeypatch, tmp_path):
    from core.brain import reasoning_background
    from core.learning import examples_she_meets

    monkeypatch.setattr(examples_she_meets, "_root", lambda: tmp_path)
    await examples_she_meets.note_to_check_async("How was your afternoon, and what did you think of it?")

    class Router:
        async def think(self, *_a, **_k):
            raise AssertionError("her model is not asked about what programs cannot settle")

    assert (await reasoning_background._check_one_she_answered(Router()))["status"] == "not_settled_by_programs"
    assert examples_she_meets.next_to_check() is None
