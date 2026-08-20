"""A reply that stopped mid-clause is trimmed back to where it made sense.

LIVE, 2026-08-19. A correct diagnosis — it quoted the exact wrong line out of
a file it had read — was served ending:

    The correction would depend on whether

The repair for this already existed and had for months. It was applied in the
desktop-task lane and in the event bridge, and not in the lane people type
into, so the conversation surface was the one place a sentence could end on
"whether".
"""

from __future__ import annotations

import re
from pathlib import Path

from core.conversation.response_reliability import complete_truncated_tail

CHAT = Path(__file__).resolve().parents[1] / "interface/routes/chat.py"

CUT_OFF = (
    "The issue is likely in how the balance is handled. If this line causes the "
    "trial balance to fail, it is because posting a negative amount is wrong. "
    "The correction would depend on whether"
)


def test_the_repair_trims_the_live_reply():
    whole = complete_truncated_tail(CUT_OFF)
    assert whole != CUT_OFF
    assert not whole.rstrip().endswith("whether")
    assert whole.rstrip().endswith(".")


def test_a_finished_reply_is_left_exactly_as_it_is():
    finished = "The close() method posts -amount to retained; it should post amount."
    assert complete_truncated_tail(finished) == finished


def test_the_chat_lane_applies_it():
    """It served two other lanes and not this one."""
    source = CHAT.read_text()
    assert "complete_truncated_tail" in source
    # And in the correction chain, not somewhere unreachable.
    chain = source[source.index("_correct_false_capability_denials(corrected)") :]
    assert "complete_truncated_tail" in chain[:2000]


def test_it_runs_after_the_other_corrections():
    """Trimming first would trim text a later correction replaces."""
    source = CHAT.read_text()
    denials = source.index("_correct_false_capability_denials(corrected)")
    trim = source.index("complete_truncated_tail", denials)
    served = source.index('data["response"] = corrected', denials)
    assert denials < trim < served
