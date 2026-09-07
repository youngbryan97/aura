"""An empty reply must not be presented as a completed turn.

Forty-one places in the chat route build a `"response"` field and nothing sits
between them and the person, so a guard at any one of them covers one path.

LIVE 2026-09-07: "does the file X exist, and what is in it?" came back with
`"response": ""` and `"status": "desktop_objective_completed"`. Before that,
the same turn came back as a bare "…" — the same defect wearing a character,
which is why the fix moved from the producing sites to the boundary every turn
already passes through.

It generates nothing and replaces no real answer. It changes what the turn
CLAIMS, so recovery and receipts see a failure rather than a success. Refusing
to respond at all would be the opposite failure: gating a working runtime into
silence.
"""

from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse

from interface.routes.chat_delivery import _no_empty_answer_leaves_this_boundary


def _payload(response) -> dict:
    return json.loads(bytes(response.body).decode("utf-8"))


@pytest.mark.parametrize(
    "status",
    [
        "desktop_objective_completed",
        "cognitive_engine",
        "full_mind_contract_unproven_served",
        "ok",
    ],
)
def test_an_empty_answer_under_a_success_status_is_reported(status: str) -> None:
    guarded = _no_empty_answer_leaves_this_boundary(
        JSONResponse({"response": "", "status": status})
    )
    payload = _payload(guarded)
    assert payload["response"].strip()
    assert payload["status"].startswith("empty_answer_withheld")
    assert payload["empty_answer_original_status"] == status
    assert payload["response_confidence"] == "failed"


def test_a_real_answer_is_untouched() -> None:
    original = {"response": "5050", "status": "cognitive_engine", "extra": 1}
    guarded = _no_empty_answer_leaves_this_boundary(JSONResponse(dict(original)))
    assert _payload(guarded) == original


def test_a_turn_that_already_reports_failure_is_left_alone() -> None:
    """It must not rewrite a refusal that is already honest about itself."""
    original = {"response": "", "status": "canonical_chat_no_reply"}
    guarded = _no_empty_answer_leaves_this_boundary(JSONResponse(dict(original)))
    assert _payload(guarded) == original


def test_a_whitespace_only_answer_counts_as_empty() -> None:
    guarded = _no_empty_answer_leaves_this_boundary(
        JSONResponse({"response": "   \n ", "status": "ok"})
    )
    assert _payload(guarded)["status"].startswith("empty_answer_withheld")


def test_a_response_without_the_field_is_untouched() -> None:
    original = {"status": "ok", "other": True}
    guarded = _no_empty_answer_leaves_this_boundary(JSONResponse(dict(original)))
    assert _payload(guarded) == original


def test_a_non_json_response_is_untouched() -> None:
    sentinel = object()
    assert _no_empty_answer_leaves_this_boundary(sentinel) is sentinel


def test_the_boundary_actually_calls_it() -> None:
    """A guard nobody calls is the defect it was written for."""
    from pathlib import Path

    source = Path("interface/routes/chat_delivery.py").read_text()
    start = source.index("def _paired_chat_response_boundary(")
    body = source[start:]
    assert "_no_empty_answer_leaves_this_boundary(response)" in body
