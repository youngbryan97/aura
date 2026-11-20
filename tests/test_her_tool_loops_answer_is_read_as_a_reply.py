"""The answer her tool loop ends on is her reply, and whether it is finished is read as for any reply of hers.

LIVE 18 November 2026, "Is 2^61 - 1 prime? Check it by actually computing": she ran the code three times, wrote her
answer and ended her turn. The loop's generations carry no reading of completion, so the turn called the answer
unfinished and the person got nothing.
"""
from __future__ import annotations

import pytest

from core.brain.inference_gate_living_context import _her_loops_answer_is_read_as_any_reply_is as read

_ASKED = "Is 2^61 - 1 prime? Check it by actually computing, don't just recall it."


class _Client:
    def __init__(self):
        self.receipt = {"surface_quality_gate_enabled": False}

    def get_last_surface_control_receipt(self):
        return dict(self.receipt)

    def _set_task_surface_control_receipt(self, receipt):
        self.receipt = dict(receipt)


def test_an_answer_she_ended_herself_is_finished():
    client = _Client()
    read(client, "Yes. sympy.isprime(2**61 - 1) printed True, so 2305843009213693951 is prime.", _ASKED, "eos")
    assert client.receipt["semantic_completion_contract"] is True
    assert client.receipt["semantic_completion_satisfied"] is True
    assert client.receipt["semantic_completion_incomplete"] is False
    assert client.receipt["surface_quality_gate_enabled"] is False, "what the receipt already held is kept"


@pytest.mark.parametrize("stopped", ["deadline_exceeded", "max_tokens", ""])
def test_an_answer_cut_off_in_a_clause_is_not(stopped):
    client = _Client()
    read(client, "Yes. I ran it and sympy said that the number", _ASKED, stopped)
    assert client.receipt["semantic_completion_satisfied"] is False
    assert client.receipt["semantic_completion_incomplete"] is True


def test_nothing_is_read_where_there_is_no_answer():
    client = _Client()
    read(client, "", _ASKED, "eos")
    assert "semantic_completion_contract" not in client.receipt


def test_a_dotted_name_her_call_wrote_is_a_name_and_not_two_sentences_run_together():
    """LIVE 18 November 2026: her account of what she ran, "I ran sympy.isprime", was refused as words run together."""
    from core.conversation.response_reliability import _has_punctuation_join_artifact
    from core.conversation.surface_disposition import record_tool_receipt
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    said = "Yes, it is prime: I ran sympy.isprime on it and it came back True."
    assert _has_punctuation_join_artifact(said), "with no such call, the words read as run together"
    with bind_turn_evidence_custody(session_id="s", turn_id="t-named"):
        record_tool_receipt("code_repl", ok=True, action="execute", observed_content="True",
                            as_made={"id": "c", "arguments": {"code": "import sympy\nprint(sympy.isprime(7))"},
                                     "returned": "True"})
        assert not _has_punctuation_join_artifact(said)
        assert _has_punctuation_join_artifact("It finished.then it stopped.")


def test_the_turn_records_the_reading_of_her_loops_answer(monkeypatch):
    """LIVE 20 November 2026: the reading was written after the turn had copied the receipt, and a finished answer was
    marked "authored_answer_incomplete:nobody_checked"."""
    import asyncio

    from core.brain.inference_gate import InferenceGate

    monkeypatch.setattr("core.phases.response_contract.derive_capability_set", lambda _text, **_k: ["code_repl"])
    monkeypatch.setattr("core.brain.llm.runtime_wiring.build_agentic_tool_map",
                        lambda *a, **k: {"code_repl": {"name": "code_repl"}})

    class _Loop(_Client):
        model_path = "/models/Qwen3.8-27B-4bit"

        async def think_and_act(self, **_kwargs):
            return {"content": "It printed 2305843009213693951, which is prime.", "generation_stop_reason": "eos",
                    "tool_calls": [{"tool": "code_repl", "ok": True}]}

    gate = InferenceGate.__new__(InferenceGate)
    recorded = []
    gate._record_client_generation_metadata = lambda *a, **k: recorded.append(k["generation_metadata"])
    answer = asyncio.run(gate._tool_grounded_answer(_Loop(), visible=_ASKED, system_prompt="", timeout_s=30.0,
                                                    allow_tools=True, completed_capability_evidence=None))
    assert answer and len(recorded) == 1
    receipt = recorded[0]["live_mind_surface_control_receipt"]
    assert receipt["semantic_completion_contract"] is True
    assert receipt["semantic_completion_satisfied"] is True
