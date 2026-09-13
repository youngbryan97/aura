"""A question with one right answer is answered from a lean prompt.

Measured live on the desktop surface 2026-07-26. A 78-character arithmetic
question was sent to a healthy resident 32B as:

    Prompt plan: mode=compact_foreground_prebuilt messages=2 chars=7542
                 (scaffold=5200 request=2342 ratio=2.2x) origin=desktop_quick_user

The person's actual words were about one percent of what the model read. The
turn resolved to the `standard` profile, whose budget is ~15.6k characters, so
none of the scaffold was ever trimmed.

A prompt that is almost entirely self-description is most likely to be continued
as more self-description, and that is exactly what came back — off-topic prose,
and replies the reliability gate rejected as `runtime_boilerplate`. A control
turn in the same session ("Reply with exactly one word: PINEAPPLE") returned
"pineapple", so the words arrive and instructions are followed; it is the ratio
that buries the task.

Determinate turns therefore take the `contract` budget (~2.8k) instead.
Expressive turns are untouched and keep their full living-mind context.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate import InferenceGate

# The live desktop chat turn's context flags.
DESKTOP_CTX = {
    "desktop_quick_reply_contract": True,
    "desktop_cognitive_engine_required": True,
}


@pytest.mark.parametrize(
    "question",
    [
        "What is 17 minus 8, and then times 3? Just the number and one line of working.",
        "What is 17 minus 8?",
        "How much is 20 percent of 50?",
        "Calculate 144 divided by 12",
        "Compute the sum of 19 and 23",
    ],
)
def test_determinate_turns_take_the_lean_budget(question: str) -> None:
    assert InferenceGate._turn_is_determinate_task(question) is True
    assert InferenceGate._foreground_prompt_profile(question, DESKTOP_CTX) == "contract"


@pytest.mark.parametrize(
    "question",
    [
        "How are you feeling right now?",
        "Tell me about your day.",
        "What is love?",
        "Who wrote Dune?",
        "What do you remember about my flight?",
    ],
)
def test_expressive_turns_keep_their_context(question: str) -> None:
    assert InferenceGate._turn_is_determinate_task(question) is False
    assert InferenceGate._foreground_prompt_profile(question, DESKTOP_CTX) != "contract"


def test_a_deep_probe_still_outranks_the_lean_route() -> None:
    """Deep probes are classified before anything else and stay that way."""
    ctx = {**DESKTOP_CTX, "deep_mind_probe": True}
    assert InferenceGate._foreground_prompt_profile("What is 2 plus 2?", ctx) == "deep_probe"


def test_the_predicate_fails_closed() -> None:
    for value in (None, "", 17, object()):
        assert InferenceGate._turn_is_determinate_task(value) is False


def test_the_contract_budget_is_far_smaller_than_standard() -> None:
    """The whole point is that the lean route actually trims.

    'standard' is ~15.6k on a 16k window and never trimmed the 7,542-char turn
    that failed; 'contract' is a fixed 2,800.
    """
    import inspect

    # The prompt builder moved to `_BuildsAndFitsThePrompt` when the gate went
    # under the module ceiling, so one file no longer holds it. Read the class
    # and what it is built from, which is what the assertion is about.
    src = "".join(
        inspect.getsource(klass)
        for klass in InferenceGate.__mro__
        if klass is not object
    )
    assert 'if profile == "contract":\n            total_budget_chars = 2_800' in src
