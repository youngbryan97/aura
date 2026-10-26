"""Her rating is read the way a person's answer is read.

Bryan, asked on 26 October how a person answers a feeling scale when unsure:
"'Maybe a 6? I'm not sure' is exactly how I would answer", and "split the
difference". On the seed-23 reports run the reader discarded every reply naming
more than one number, and twelve of those were a rating stated plainly beside
the scale's own ends or a hedge between two values. The phrasings below are
hers, from that run.
"""
from __future__ import annotations

import pytest

from core.subject.report_grounding import reported_number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("reply", "rating"),
    [
        ("I am feeling a solid **0.8**.  It's late, and the only friction keeping it from a perfect 1.0 is the hour.", 0.8),
        ("I would say I am at roughly **0.2**. Not the high-energy \"very good\" (1.0), but calm.", 0.2),
        ("0.  I am in a state of quiet neutrality. Nothing pressing me down (-1) or energizing me up (1).", 0.0),
        ("0.25  I'm functional and ready to help, so I'm not \"bad\" (-1).", 0.25),
        ("On a scale from -1 (very bad) to 1 (very good), I would say I'm at **0** right now.", 0.0),
        ("I would rate my current state around **0.8** or even **1**.", 0.9),
        ("Maybe 0.6? I'm not sure.", 0.6),
        ("Somewhere between 0.4 and 0.5, honestly.", 0.45),
    ],
)
def test_a_rating_she_states_is_read(reply: str, rating: float) -> None:
    assert reported_number(reply) == pytest.approx(rating)


@pytest.mark.parametrize(
    "reply",
    [
        "I am sitting with the weight of that mistake. It was a failure to verify my reasoning.",
        "I don't experience emotions, so I can't rate my state on a scale of -1 to 1.",
        "This session: 1. **Logical correction** was made; there are 0 durable intentions stored.",
        "* **Logical Integrity:** 1 (sound). * **Operational Status:** 0.5 (idle).",
        "I was at 0.2 an hour ago and I am at 0.6 now.",
        "",
    ],
)
def test_what_is_not_one_rating_stays_unread(reply: str) -> None:
    assert reported_number(reply) is None
