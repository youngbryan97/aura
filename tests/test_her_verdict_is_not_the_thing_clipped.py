"""What she made of the result comes first, because the reply is clipped.

LIVE 2026-09-29: she read her result, held it against what she had predicted
and said so out loud — and the reply ended mid-sentence sixteen items earlier
with nothing about the outcome, because the account is cut to a maximum length
and her verdict was last in it.

What was asked for is the verdict. Since 1 Oct each answer and its reason
reaches the person as it is made, one bubble a question, so a reply with a
verdict is the verdict; the working is the account only where there is none
(Bryan, of the rounds listed under it: "a little ugly when they come in").
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _pursuit_account

pytestmark = pytest.mark.unit


def _result(**over):
    base = {
        "concluded": "It says ESTP, which matches what I predicted.",
        "narration": [
            {"asked": f"item {n}", "chose": ["x"], "why": "because"} for n in range(40)
        ],
        "result_text": "the page's own tail",
    }
    base.update(over)
    return base


def test_her_verdict_comes_first():
    lines = _pursuit_account(_result())
    assert lines[0].startswith("It says ESTP")


def test_a_verdict_is_the_reply_on_its_own():
    lines = _pursuit_account(_result())
    assert lines == ["It says ESTP, which matches what I predicted."]


def test_a_run_with_no_verdict_still_reads():
    lines = _pursuit_account(_result(concluded=""))
    assert lines
    assert all("ESTP" not in line for line in lines)


def test_without_a_verdict_the_working_is_the_account():
    lines = _pursuit_account(_result(concluded=""))
    assert any("item 0" in line for line in lines)
    assert any("the page's own tail" in line for line in lines)
