"""A reply her model ended itself is finished; only one cut off by its allowance, its deadline or a cancel continues.

LIVE 13 November 2026, her answer to "What have you been thinking about while I was testing your voice?" stopped on end
of sequence and was marked incomplete by a reading of its wording; two more generations followed in an 81 s turn.
"""
from __future__ import annotations

from core.brain.cognitive_engine import _her_model_ended_it
from core.brain.cognitive_engine_quick_reply import _AnswersTheDesktopDirectly as _QuickReply

_CLIPPED_LOOKING = "Mostly I kept coming back to how my voice sounded, and whether the"


def _incomplete(stop_reason: str, receipt: dict, reasons: tuple = ()) -> bool:
    return _QuickReply._direct_desktop_quick_reply_semantic_completion_incomplete(
        stop_reason, reasons, receipt, _CLIPPED_LOOKING)[1]


def test_her_model_ending_it_is_finished_whatever_its_wording_reads_as():
    assert _her_model_ended_it("eos") and _her_model_ended_it("STOP")
    assert not _incomplete("eos", {"semantic_completion_incomplete": True}, ("truncated_tail",))


def test_a_reply_cut_off_by_its_allowance_deadline_or_a_cancel_continues():
    for stop in ("max_tokens", "deadline_exceeded", "soft_cancelled"):
        assert not _her_model_ended_it(stop)
        assert _incomplete(stop, {})
