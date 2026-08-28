"""At the salvage site, incompleteness is the condition, not a disqualifier.

LIVE, 2026-08-27: 948 characters of a worked derivation were produced, marked
incomplete on a deadline, denied the continuation that would have completed
them — "live desktop turns stay bounded to one foreground generation" — and
then withheld for being incomplete. The person got the apology.
"""

from __future__ import annotations

from interface.routes.chat import (
    _authored_answer_can_serve,
    _authored_answer_can_serve_unfinished,
)

_HERS_BUT_UNFINISHED = {
    # Her words. The engine ran, wrote this, and did not accept it — which is
    # the condition this site exists for, not a reason to withhold it.
    "engine_authored_the_text": True,
    "authentic_cognitive_reply": False,
    "cognitive_engine_reply_accepted": False,
    "final_requested_output_contract_proven": True,
    "answer_delivery_proven": False,
    "full_mind_missing_proofs": ("authored_answer_incomplete", "engine_reply_not_accepted"),
}


def test_the_full_test_still_refuses_an_unfinished_answer() -> None:
    assert not _authored_answer_can_serve(_HERS_BUT_UNFINISHED)


def test_the_salvage_test_admits_it() -> None:
    assert _authored_answer_can_serve_unfinished(_HERS_BUT_UNFINISHED)


def test_authorship_still_has_to_hold() -> None:
    for spoiled in (
        {"authorship_replacement_applied": True},
        {"legacy_fallback_used": True},
        {"bounded_contract_used": True},
        {"engine_authored_the_text": False},
        {"final_requested_output_contract_proven": False},
    ):
        contract = dict(_HERS_BUT_UNFINISHED)
        contract.update(spoiled)
        assert not _authored_answer_can_serve_unfinished(contract), spoiled


def test_nothing_at_all_is_still_nothing() -> None:
    assert not _authored_answer_can_serve_unfinished({})
    assert not _authored_answer_can_serve_unfinished(None)
    assert not _authored_answer_can_serve_unfinished("a string")


def test_the_salvage_site_uses_the_salvage_test() -> None:
    from pathlib import Path

    body = Path("interface/routes/chat.py").read_text()
    start = body.index("Preserved no-reply draft remained ineligible for delivery")
    window = body[start - 500 : start + 900]
    assert "_authored_answer_can_serve_unfinished(salvage_contract)" in window
    assert "unfinished authored work rather" in window


def test_authorship_and_approval_are_different_facts() -> None:
    """The proof the salvage needs is whose words these are.

    LIVE, 2026-08-28: twelve reasoning questions came back with the canned
    apology. The engine had run, written a real partial answer, judged it not
    good enough, and the salvage refused to serve it for want of a proof that
    said "the engine accepted this" — at a site reached only when it did not.
    """

    from pathlib import Path

    contract = Path("interface/routes/chat_turn_contract.py").read_text()
    start = contract.index("engine_authored_the_text = bool(")
    window = contract[start : start + 500]
    assert "engine_think_invoked" in window
    assert "engine_reply_accepted" not in window

    body = Path("interface/routes/chat.py").read_text()
    start = body.index("def _authored_answer_can_serve_unfinished(")
    # Bounded at the next function, which legitimately asks for the full proof.
    salvage = body[start : body.index("def _authored_answer_can_serve(", start)]
    assert 'contract.get("engine_authored_the_text")' in salvage
    assert 'contract.get("authentic_cognitive_reply")' not in salvage


def test_the_full_delivery_test_is_untouched() -> None:
    """Only the last-resort site changed."""

    from pathlib import Path

    body = Path("interface/routes/chat.py").read_text()
    start = body.index("def _authored_answer_can_serve(contract: Any) -> bool:")
    full = body[start : start + 600]
    assert 'contract.get("answer_delivery_proven")' in full
    assert 'contract.get("authentic_cognitive_reply")' in full
