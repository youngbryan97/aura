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
