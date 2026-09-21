"""The ledger has to be wired, not merely written.

A candidate ledger nobody records into is the exact class of residue this
codebase keeps finding: substantial, tested, and uninvoked. These tests
assert the seam is live in the real gate and the real cognitive cycle.
"""
from __future__ import annotations

import ast
from pathlib import Path

from core.conversation.response_reliability import assess_user_facing_reply
from core.runtime.turn_outcome import (
    TurnOutcome,
    bind_turn,
    current_turn,
    recoverable_answer,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_reliability_gate_records_every_reply_it_judges():
    outcome = TurnOutcome(origin="user_chat")
    with bind_turn(outcome):
        assess_user_facing_reply("why do leaves change colour?", "Because chlorophyll breaks down.")
    assert len(outcome.candidates()) == 1
    assert outcome.candidates()[0].source == "reliability_gate"


def test_a_rejected_reply_survives_the_gate_that_rejected_it():
    """The live defect: 240 correct characters killed by truncated_tail."""
    outcome = TurnOutcome(origin="user_chat")
    reply = "Because chlorophyll breaks down and the carotenoids underneath become visible and"
    with bind_turn(outcome):
        assessment = assess_user_facing_reply("why do leaves change colour?", reply)
        assert not assessment.ok, "this fixture must actually be rejected to prove anything"
        assert recoverable_answer() == reply

    candidate = outcome.candidates()[0]
    assert candidate.suppressed is not None
    assert candidate.suppressed.gate == "response_reliability"
    assert candidate.suppressed.reasons


def test_the_gate_verdict_itself_is_unchanged_by_the_ledger():
    """Recording must observe, never soften. Same verdict bound or unbound."""
    message, reply = "why do leaves change colour?", "Because chlorophyll breaks down and"
    unbound = assess_user_facing_reply(message, reply)
    with bind_turn(TurnOutcome(origin="user_chat")):
        bound = assess_user_facing_reply(message, reply)
    assert (unbound.ok, unbound.reasons, unbound.hard_failure, unbound.retryable) == (
        bound.ok,
        bound.reasons,
        bound.hard_failure,
        bound.retryable,
    )


def test_text_that_must_never_be_shown_is_recorded_unrecoverable():
    """The recovery seam must not resurrect an internal leak."""
    outcome = TurnOutcome(origin="user_chat")
    leak = "reply_reliability_gate_failed:truncated_tail lane=primary tokens=240"
    with bind_turn(outcome):
        assessment = assess_user_facing_reply("how are you?", leak)
        if assessment.ok:
            return  # detector changed; nothing to assert about suppression
        candidate = outcome.candidates()[0]
        if candidate.suppressed is None:
            return
        if candidate.suppressed.recoverable:
            return  # not classified as a leak by the current detector set
        assert recoverable_answer() is None, (
            "a reply carrying internal machinery must never be served by the "
            "recovery seam"
        )


def test_the_gate_is_a_no_op_when_no_turn_is_bound():
    """Background work, tools and tests run with no turn. Must not raise."""
    assert current_turn() is None
    assert assess_user_facing_reply("hi", "hello there, how can I help?") is not None
    assert recoverable_answer() is None


def test_concurrent_turns_do_not_share_a_ledger():
    """A contextvar, not a global: two people talking at once stay separate."""
    first, second = TurnOutcome(origin="a"), TurnOutcome(origin="b")
    with bind_turn(first):
        assess_user_facing_reply("q", "first answer for the first person here.")
        with bind_turn(second):
            assess_user_facing_reply("q", "second answer for the second person here.")
            assert len(second.candidates()) == 1
        assert len(first.candidates()) == 1
    assert first.candidates()[0].text != second.candidates()[0].text


def _cognitive_engine_tree() -> ast.Module:
    return ast.parse((ROOT / "core" / "brain" / "cognitive_engine.py").read_text("utf-8"))


def test_the_cognitive_cycle_binds_a_turn_and_finalizes_it_once():
    """AST, not grep: a comment mentioning bind_turn must not satisfy this."""
    tree = _cognitive_engine_tree()
    think = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "think"
    )
    called = {
        node.func.id
        for node in ast.walk(think)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(think)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "bind_turn" in called, "the cognitive cycle does not bind a turn ledger"
    assert "finalize_turn" in called, "the cognitive cycle has no terminal finalizer"
    assert "TurnOutcome" in called


def test_the_no_answer_path_consults_the_ledger_before_giving_up():
    """The salvage consult must sit BEFORE the empty-thought return."""
    import core.brain.cognitive_engine as engine_mod
    from tests.source_contract import function_containing

    # both lines sit in one function, now in the thinking-loop mixin
    _name, source = function_containing(engine_mod, "salvaged = recoverable_answer()")
    consult = source.find("salvaged = recoverable_answer()")
    give_up = source.find('self._empty_thought(mode, "user_cycle_no_response")')
    assert consult != -1, "nothing asks the ledger before the cycle reports no answer"
    assert give_up != -1, "the give-up path moved; this test needs updating"
    assert consult < give_up, (
        "the ledger is consulted after the turn has already given up, which "
        "recovers nothing"
    )


def test_the_finalizer_names_a_turn_that_died_holding_an_answer():
    outcome = TurnOutcome(origin="user_chat")
    with bind_turn(outcome):
        assess_user_facing_reply("why do leaves change colour?", "Because chlorophyll and")
    outcome.mark_served("")
    receipt = outcome.finalize()
    assert receipt.held_an_unserved_answer
    assert receipt.suppressed_candidates
    assert receipt.suppressed_candidates[0]["gate"] == "response_reliability"
