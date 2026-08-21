"""The amplifier must not hijack imperative actions/tool commands."""
from __future__ import annotations

import pytest

from core.brain.reasoning_amplifier_v2 import is_action_request, is_amplifiable


@pytest.mark.parametrize("text", [
    "Open 3 tabs",
    "open three tabs in chrome",
    "click the submit button",
    "send the email to Sarah",
    "play the next song",
    "go to github.com",
    "set a timer for 5 minutes",
    "download the report and save it",
    "delete the temp files",
    "schedule a meeting for 3pm",
    "navigate to the settings page",
    "turn off notifications",
    "please open the file manager",
    "can you click on the link",
])
def test_actions_are_not_amplified(text):
    assert is_action_request(text) is True
    assert is_amplifiable(text) is None


@pytest.mark.parametrize("text,expected", [
    ("write a function that adds two numbers", "code"),
    ("compute the product of 12 and 12", "math"),
    ("where is the inference gate implemented", "repo_audit"),
    ("describe the architecture of the subprocess gateway", "repo_audit"),
])
def test_reasoning_questions_still_amplify(text, expected):
    assert is_action_request(text) is False
    assert is_amplifiable(text) == expected


@pytest.mark.parametrize("text", [
    "how are you today",
    "tell me a story",
    "what do you think about jazz",
])
def test_casual_chat_not_amplified(text):
    assert is_amplifiable(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Given baselines A=8, B=24, and C=60, infer the causal order from "
            "interventions A+2 -> B+6 and C+12, then predict C for A+4.",
            "logic",
        ),
        (
            "Tasks are [{'name':'A','duration':2,'deadline':5,'reward':7}, "
            "{'name':'B','duration':3,'deadline':7,'reward':9}]. Find the optimal "
            "schedule and makespan within horizon 7.",
            "planning",
        ),
        (
            "The claim says P wins. Given P score 8 and Q score 11, recompute the "
            "scores and return the actual winner.",
            "logic",
        ),
    ],
)
def test_structured_planning_and_inference_turns_amplify(text, expected):
    assert is_action_request(text) is False
    assert is_amplifiable(text) == expected


def test_plain_fact_lookup_stays_on_the_low_latency_path():
    assert is_amplifiable("what is the capital of France") is None


def test_requested_worked_trace_is_one_expository_deliverable():
    text = (
        "Explain Dijkstra's invariant and give a worked example using vertices "
        "A, B, C, and D with weighted edges."
    )
    assert is_amplifiable(text) is None


def test_concrete_shortest_path_result_remains_amplifiable():
    text = (
        "Given edges A-B=2, A-C=5, B-C=1, B-D=6, and C-D=2, compute "
        "the shortest path from A to D and return its total weight."
    )
    assert is_amplifiable(text) == "math"


@pytest.mark.parametrize(
    "text",
    [
        "Schedule these jobs to minimize makespan: "
        "[{'name':'A','duration':2}, {'name':'B','duration':3}]",
        "Can you schedule tasks A and B within horizon 7 given deadlines 4 and 6?",
    ],
)
def test_computational_scheduling_is_reasoning_not_environment_actuation(text):
    assert is_action_request(text) is False
    assert is_amplifiable(text) == "planning"


@pytest.mark.parametrize(
    "text",
    [
        "Schedule a dentist appointment tomorrow at 3 PM",
        "Can you schedule my meeting with Alex for Friday?",
    ],
)
def test_calendar_scheduling_remains_an_environment_action(text):
    assert is_action_request(text) is True
    assert is_amplifiable(text) is None


def test_action_with_number_not_treated_as_math():
    # The specific case raised: a digit in an action must not route to math.
    assert is_amplifiable("Open 3 tabs") is None
    assert is_amplifiable("click button 2 times") is None
    assert is_amplifiable("send 5 emails to the team") is None


@pytest.mark.asyncio
async def test_phase_skips_action_turn():
    import types

    from core.phases.response_generation_unitary import UnitaryResponsePhase

    class _LLM:
        calls = 0

        async def think(self, prompt, **kw):
            type(self).calls += 1
            return "x"

    llm = _LLM()
    out = await UnitaryResponsePhase._maybe_amplify_response(
        types.SimpleNamespace(_last_reasoning_receipt=None),
        objective="open 3 tabs and click the first result",
        draft="Opening the tabs now.",
        llm=llm,
        state=types.SimpleNamespace(metadata={}),
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert out == "Opening the tabs now."
    assert llm.calls == 0
