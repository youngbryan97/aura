"""A plan written one step to a line ends on its last step, and that is not a reply cut off.

LIVE 18 November 2026: her kept procedures computed Blocksworld and Mystery Blocksworld plans, the reply gates read each
as cut off because the last step had no full stop, and the person was told no reply had formed.
"""
from __future__ import annotations

import pytest

from core.conversation.response_surface_checks import _has_truncated_tail

_PLAN = ("unstack the red block from on top of the orange block\nput down the red block\n"
         "unstack the orange block from on top of the yellow block\nput down the orange block\n"
         "pick up the yellow block\nstack the yellow block on top of the red block")
_MYSTERY = ("feast object a from object d\nsuccumb object a\nattack object d\novercome object d from object a\n"
            "attack object c\novercome object c from object d")


@pytest.mark.parametrize("plan", [_PLAN, _MYSTERY, _MYSTERY + "\nfeast object c from object a"])
def test_a_plan_of_whole_steps_is_finished(plan):
    assert not _has_truncated_tail(plan)


@pytest.mark.parametrize("cut", [
    "We need to consider each case separately and the first one is the case where both are Red",
    "Here is what I found today.\nThe first part of it went well enough for us.\nThe second part of it went badly because",
    "unstack the red block from on top of the orange block\nput down the red block\nunstack the orange block from on",
    "unstack the red block from on top of the orange block\nput down the red block\nunstack",
    "walk to the store\nbuy some milk\nthen open the",
])
def test_a_reply_really_cut_off_is_still_read_as_cut_off(cut):
    assert _has_truncated_tail(cut)
