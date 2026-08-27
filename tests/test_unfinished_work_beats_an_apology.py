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
    "authentic_cognitive_reply": True,
    "final_requested_output_contract_proven": True,
    "answer_delivery_proven": False,
    "full_mind_missing_proofs": ("authored_answer_incomplete",),
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
        {"authentic_cognitive_reply": False},
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
