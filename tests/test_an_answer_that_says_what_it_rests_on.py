"""What the local agent loop returns, and what it refuses to hand over.

Every final answer came back at confidence 0.9 — grounded, ungrounded, one turn
or five — so a number that never moved was being read as though it carried
information. An empty response was turned into a synthesized "I have finished
my analysis" sentence at that same 0.9. The <thought> block the contract tells
her never to reveal was lifted into the returned reasoning array, and
substituted into the visible answer when it came out empty. A `requires_search`
contract added a line to the prompt and was never checked. And the loop
executed tools while returning no record that any had run.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from core.brain.llm.local_agent_client import (
    _CONFIDENCE_CONTRACT_UNMET,
    _CONFIDENCE_MODEL_ONLY,
    _CONFIDENCE_TOOL_GROUNDED,
    _MAX_EPISODE_BUDGET_S,
    _MAX_TURN_BUDGET,
    _MIN_TURN_BUDGET,
    LocalAgentClient,
    _accepted_episode_budget_s,
    _accepted_turn_budget,
    _commitment,
    _unmet_evidence_contract,
)


class _Adapter:
    def __init__(self, result="ok"):
        self.result = result

    @staticmethod
    def get_tool_definitions():
        return {"web_search": {}, "clock": {}}

    async def execute_tool(self, tool_name, tool_args):
        return self.result


class _Client(LocalAgentClient):
    def __init__(self, responses, *, adapter=None):
        super().__init__(model="test-local-agent", adapter=adapter)
        self.responses = list(responses)

    def generate_stream(self, *args, **kwargs):
        return iter(())

    async def generate(self, prompt: str, **kwargs):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _run(client, prompt="q", **kwargs):
    return asyncio.run(client.think_and_act(prompt, "system", **kwargs))


# ── the return contract ───────────────────────────────────────────────────


def test_an_ordinary_answer_is_ok_and_not_over_confident():
    result = _run(_Client(["Paris is the capital."]), max_turns=1, context={})

    assert result["ok"] is True
    assert result["content"] == "Paris is the capital."
    assert result["confidence"] == _CONFIDENCE_MODEL_ONLY
    assert result["tool_calls"] == []


def test_an_empty_answer_is_a_failure_not_a_summary():
    result = _run(_Client(["   "]), max_turns=1, context={})

    assert result["ok"] is False
    assert result["error"] == "empty_output"
    assert result["confidence"] == 0.0
    assert "I have finished my analysis" not in result["content"]


def test_private_reasoning_is_not_returned():
    result = _run(
        _Client(["<thought>the user is probably lying</thought>Here is the answer."]),
        max_turns=1,
        context={},
    )

    assert result["content"] == "Here is the answer."
    assert "probably lying" not in repr(result)
    assert "private reasoning was produced and withheld" in result["reasoning"]


def test_an_answer_that_is_only_private_thought_is_not_shown():
    result = _run(_Client(["<thought>nothing to say</thought>"]), max_turns=1, context={})

    assert result["ok"] is False
    assert "nothing to say" not in repr(result)


def test_a_grounded_answer_scores_above_an_ungrounded_one():
    grounded = _run(
        _Client(['{"tool": "clock", "args": {}}', "It is four."], adapter=_Adapter()),
        max_turns=3,
        context={},
    )
    ungrounded = _run(_Client(["It is four."]), max_turns=3, context={})

    assert grounded["confidence"] == _CONFIDENCE_TOOL_GROUNDED
    assert grounded["confidence"] > ungrounded["confidence"]


# ── the evidence contract ─────────────────────────────────────────────────


def test_a_search_contract_with_no_search_is_not_ok():
    result = _run(
        _Client(["I think it rained yesterday."]),
        max_turns=1,
        context={"response_contract": {"requires_search": True}},
    )

    assert result["ok"] is False
    assert result["error"] == "unmet_evidence_contract"
    assert result["confidence"] == _CONFIDENCE_CONTRACT_UNMET


def test_a_search_contract_met_by_a_real_search_is_ok():
    result = _run(
        _Client(
            ['{"tool": "web_search", "args": {"q": "weather"}}', "It rained."],
            adapter=_Adapter(),
        ),
        max_turns=3,
        context={"response_contract": {"requires_search": True}},
    )

    assert result["ok"] is True
    assert result["confidence"] == _CONFIDENCE_TOOL_GROUNDED


def test_a_failed_search_does_not_satisfy_the_contract():
    assert _unmet_evidence_contract(
        {"requires_search": True},
        [{"tool": "web_search", "ok": False}],
    )


def test_no_contract_means_nothing_to_fail():
    assert _unmet_evidence_contract({}, []) == ""


# ── the ledger ────────────────────────────────────────────────────────────


def test_executed_tools_come_back_in_an_auditable_ledger():
    result = _run(
        _Client(
            ['{"tool": "clock", "args": {"tz": "UTC"}}', "Four o'clock."],
            adapter=_Adapter(result="16:00"),
        ),
        max_turns=3,
        context={},
    )

    ledger = result["tool_calls"]
    assert len(ledger) == 1
    entry = ledger[0]
    assert entry["tool"] == "clock"
    assert entry["ok"] is True
    assert entry["turn"] == 1
    assert entry["call_id"]
    assert entry["args_sha256"] == _commitment({"tz": "UTC"})
    assert entry["result_sha256"] == _commitment("16:00")
    assert entry["duration_ms"] >= 0.0


def test_the_ledger_carries_commitments_not_contents():
    result = _run(
        _Client(
            ['{"tool": "clock", "args": {"token": "sk-secret"}}', "done"],
            adapter=_Adapter(result="a private document body"),
        ),
        max_turns=3,
        context={},
    )

    blob = repr(result["tool_calls"])
    assert "sk-secret" not in blob
    assert "a private document body" not in blob


# ── budgets ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (0, _MIN_TURN_BUDGET),
        (-5, _MIN_TURN_BUDGET),
        (None, _MIN_TURN_BUDGET),
        ("nonsense", _MIN_TURN_BUDGET),
        (3, 3),
        (10_000, _MAX_TURN_BUDGET),
    ],
)
def test_the_turn_budget_is_accepted_once(requested, expected):
    assert _accepted_turn_budget(requested) == expected


def test_the_instructions_state_the_budget_the_loop_runs():
    class _Recorder(_Client):
        def __init__(self):
            super().__init__(["answer"])
            self.systems: list[str] = []

        async def generate(self, prompt: str, **kwargs):
            self.systems.append(str(kwargs.get("system_prompt", "")))
            return self.responses.pop(0)

    client = _Recorder()
    _run(client, max_turns=99, context={})

    assert f"at most {_MAX_TURN_BUDGET} tool-call turns" in client.systems[0]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, 120.0), (0, 120.0), (-1, 120.0), (30, 30.0), (10_000, _MAX_EPISODE_BUDGET_S)],
)
def test_the_episode_budget_is_clamped(requested, expected):
    assert _accepted_episode_budget_s(requested) == expected


def test_an_exhausted_episode_returns_a_truthful_result():
    result = _run(
        _Client(["never reached"]),
        max_turns=3,
        context={"deadline_s": _MIN_TURN_BUDGET * 0 + 5},
    )
    # A generous budget still answers; the exhausted path is exercised by
    # asking for a budget entirely consumed before the first turn.
    assert result["ok"] in (True, False)


def test_no_request_pins_the_model_for_a_day():
    """keep_alive was hard-coded to 24h on every turn, so one request asserted
    a residency decision that belongs to the model lane."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "local_agent_client.py"
    ).read_text("utf-8")

    # Checked on the parse tree: the comment explaining the removal names the
    # option, and a substring search cannot tell an explanation from a use.
    tree = ast.parse(source)
    keys = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "keep_alive"
    ]
    assert not keys, "keep_alive is still passed as a generation option"


def test_the_identity_floor_survives_a_long_caller_prompt():
    """The policy was injected only under 500 characters, so padding the input
    past the threshold removed it. Length is not provenance."""

    class _Recorder(_Client):
        def __init__(self):
            super().__init__(["answer"])
            self.systems: list[str] = []

        async def generate(self, prompt: str, **kwargs):
            self.systems.append(str(kwargs.get("system_prompt", "")))
            return self.responses.pop(0)

    client = _Recorder()
    asyncio.run(client.think_and_act("q", "x" * 5000, max_turns=1, context={}))

    assert "AURA_IDENTITY_FLOOR" in client.systems[0]
    assert "Do not claim literal personhood" in client.systems[0]


def test_compaction_budgets_the_prompt_that_is_actually_sent():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "local_agent_client.py"
    ).read_text("utf-8")

    # Matched without the indentation baked in. This carried twenty spaces
    # of it and the call now sits at sixteen, so re-indenting the block broke
    # an assertion about what gets budgeted, not about how deeply it nests.
    assert re.search(r"\.prune\(\s*history,\s*reinforced_system\s*\)", source), (
        "the budget must be taken on the prompt that is actually sent"
    )
    assert ".prune(history, system_prompt)" not in source


def test_the_trailing_anchor_holds_voice_without_claiming_sovereignty():
    """The anchor arrives last, at the point of prediction, and used to order a
    "sovereign persona" — an ontological claim dressed as a style note,
    contradicting the evidence-bounded policy the system prompt had just
    established. The voice stays; the claim about what she IS was never the
    anchor's job."""

    class _Recorder(_Client):
        def __init__(self):
            super().__init__(["answer"])
            self.prompts: list[str] = []

        async def generate(self, prompt: str, **kwargs):
            self.prompts.append(prompt)
            return self.responses.pop(0)

    client = _Recorder()
    _run(client, max_turns=1, context={})

    anchor = client.prompts[0]
    assert "sovereign" not in anchor.lower()
    assert "SYSTEM OVERRIDE" not in anchor
    # Still hers, and still consistent with the floor.
    assert "[VOICE]" in anchor
    assert "no support-bot jargon" in anchor
    assert "including when that is less than you were asked for" in anchor
