"""The turn ledger is open while the reply is being rewritten, not just made.

`bind_turn` was opened inside `cognitive_engine.think()` and closed when it
returned — and the reply is not delivered when think() returns. Every honesty
gate, repair pass, shaping stage and terminal boundary runs after it, which is
the entire stretch of the turn where an answer gets lost. Anything asking
`current_turn()` from the delivery path got None, so the effect ledger and
fact custody were unreachable exactly where they matter.

The fix is one turn spanning generation and delivery, owned by the route,
joined by the engine. These tests hold that shape, because the failure mode is
silent: a no-op custody check looks identical to a turn with nothing wrong.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core.runtime.turn_outcome import TurnOutcome, bind_turn, current_turn

_CHAT = Path(__file__).resolve().parents[1] / "interface" / "routes" / "chat.py"
_ENGINE = Path(__file__).resolve().parents[1] / "core" / "brain" / "cognitive_engine.py"


def test_the_route_binds_a_turn_around_the_whole_request() -> None:
    source = _CHAT.read_text("utf-8")
    assert "_bound_http_turn(body)" in source, (
        "the HTTP turn opens no ledger, so the delivery path cannot see one"
    )
    assert "with _bound_http_turn(body) as _turn_outcome" in source
    assert "bind_turn_evidence_custody(" in source
    assert "bind_failure_ledger()," in source, (
        "the exact outcome, evidence custody, and failure ledger must surround _api_chat_turn"
    )


def test_the_engine_joins_a_bound_turn_instead_of_opening_a_second() -> None:
    """Two notions of "the current turn" is worse than the bug it would fix."""

    source = _ENGINE.read_text("utf-8")
    assert "adopted = current_turn()" in source
    assert "outcome = adopted if adopted is not None else TurnOutcome" in source


def test_the_engine_does_not_finalize_a_turn_it_does_not_own() -> None:
    """Finalizing someone else's ledger closes it before delivery."""

    source = _ENGINE.read_text("utf-8")
    tree = ast.parse(source)
    unguarded: list[int] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "finalize_turn"
        ):
            continue
        # Every finalize in think() must sit under `if owns_outcome:`.
        guarded = any(
            isinstance(parent, ast.If)
            and isinstance(parent.test, ast.Name)
            and parent.test.id == "owns_outcome"
            and any(node is inner for inner in ast.walk(parent))
            for parent in ast.walk(tree)
            if isinstance(parent, ast.If)
        )
        if not guarded:
            unguarded.append(node.lineno)
    assert not unguarded, (
        f"finalize_turn called without an ownership check at lines {unguarded}"
    )


def test_the_bound_turn_yields_exactly_once_when_the_body_raises() -> None:
    """A generator that yields twice turns any route error into a crash."""

    import interface.routes.chat as chat

    manager = chat._bound_http_turn(type("B", (), {"message": "hi"})())
    with pytest.raises(ValueError):
        with manager:
            raise ValueError("route failed")


def test_the_bound_turn_finalizes_and_releases_custody() -> None:
    import interface.routes.chat as chat
    from core.runtime.fact_custody import custody_for, hold_fact
    from core.runtime.turn_outcome import VerificationGrade

    with chat._bound_http_turn(type("B", (), {"message": "hi"})()) as outcome:
        assert outcome is not None
        assert current_turn() is outcome
        hold_fact(
            subject="s",
            predicate="p",
            value="1",
            canonical_rendering="p is 1",
            established_by="test",
            grade=VerificationGrade.OBSERVED,
        )
        turn_id = outcome.turn_id
    assert outcome.is_finalized
    assert custody_for(turn_id).facts() == ()


def test_custody_is_reachable_from_a_delivery_stage() -> None:
    """The property the whole change exists for, stated as a test."""

    from core.runtime.fact_custody import current_custody

    assert current_custody() is None
    with bind_turn(TurnOutcome("delivery", origin="test")):
        assert current_custody() is not None


def test_the_terminal_boundary_enforces_custody() -> None:
    from source_support import inlined_function_source

    # the envelope is built by a helper the size sweep cut out; read the turn
    # as it runs, with that block back at its call site, so "before" means
    # before
    source = inlined_function_source(_CHAT, "_api_chat_turn")
    restore = source.find("restore_held_facts(_final_reply)")
    envelope = source.find('"response": _final_reply,')
    assert restore != -1, "nothing enforces custody on the outgoing text"
    assert envelope != -1, "the response envelope moved; this test needs updating"
    assert restore < envelope, (
        "custody is enforced after the reply is already in the envelope, which "
        "restores nothing"
    )


def test_the_engine_still_owns_its_turn_when_nothing_bound_one() -> None:
    """A background tick has no route above it and must still be recorded."""

    import core.brain.cognitive_engine as engine

    source = inspect.getsource(engine.CognitiveEngine.think)
    assert "owns_outcome = adopted is None" in source
    assert "if owns_outcome:" in source
