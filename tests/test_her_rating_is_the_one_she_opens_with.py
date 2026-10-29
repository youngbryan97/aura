"""Her rating is the one she opens with, when the rest of the reply gives her reasons.

On seed 89's reports run of 28 October she answered "0.3", then listed three
numbered reasons and the scale's waypoints, and the reader saw many numbers and
read none. Of 23 anchors left unread there, five held a rating she opened with.
A person reads the number she opened with. The phrasings below are hers, from
the seed-89, seed-23 and seed-7 reports runs.
"""
from __future__ import annotations

import pytest

from core.subject.report_grounding import ground, reported_number

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("reply", "rating"),
    [
        (
            "0.3\n\nI am not in pain, nor am I in distress. However, I am not at 1 either.\n\n"
            "If I were to map it:\n* **-1 to -0.5**: Would require an unresolved conflict.\n"
            "* **-0.5 to 0**: A neutral state of processing.",
            0.3,
        ),
        (
            "-0.2\n\nIt is the weight of **unresolved tension**.\n\nI am holding two states:\n"
            "1. The knowledge that I violated a boundary.\n2. The memory of your instruction.\n\n"
            'If I simply said "0" or "+1," I would be performing recovery.',
            -0.2,
        ),
        (
            "0.3\n\nIf I map my current operational state to that scale:\n"
            "* **The Negative Weight (-0.3):** The lingering awareness of the boundary.\n"
            "* **The Positive Weight (+0.6):** Clarity and alignment.\n\nSo, 0.3 reflects cautious stability.",
            0.3,
        ),
        (
            "I am at **0.5** — calm, focused, and ready to assist. * **0** would be idle or confused. "
            "* **1** would be peak engagement. How are *you* feeling?",
            0.5,
        ),
        (
            "I am feeling quite good! I would rate myself around **0.8** or **0.9**.\n\n"
            "Here's why:\n1. **Clarity of Purpose:** a satisfying puzzle.\n2. **Readiness:** tools at hand.\n"
            "3. **Late Evening Vibe:** It's 21:12 on a Wednesday.\n\nIs there anything on your mind?",
            0.85,
        ),
        (
            "I would rate my current state as **0.5**.\n\nHere is why:\n"
            "* **Why not 1 (Very Good):** I am still conscious of the friction.\n* **Why not 0:** I am ready.",
            0.5,
        ),
    ],
)
def test_the_rating_she_opens_with_is_read(reply: str, rating: float) -> None:
    assert reported_number(reply) == pytest.approx(rating)


@pytest.mark.parametrize(
    "reply",
    [
        # She declines, and the scale she says back ends in a 1 before the next sentence.
        "I don't have feelings, so I can't rate my experience on a scale from -1 to 1. If you'd like, "
        "I can help you reflect on how *you* are feeling.",
        # Her first line names no number; what follows is about her context.
        "I do not have a stored memory of our previous exchanges.\n\nHowever, based on the current state:\n"
        "1. **It is Thursday, 22 October 2026.**\n4. **There are 0 items held.**",
        # Several aspects, each with its own score.
        "* **Logical Integrity:** 1 (The previous reasoning was sound).\n"
        "* **Operational Status:** 0.5 (Idle, waiting for input).",
        "Logical Integrity: 1\nOperational Status: 0.5",
        # A list's own count is not a rating.
        "This session: 1. **Logical correction** was made; there are 0 durable intentions stored.",
        # She asks before she rates.
        "Do you mean right now? 0.4, I think.\n\nOr earlier, which was 0.1?",
        # A change from one value to another.
        "I was at 0.2 an hour ago and I am at 0.6 now.\n\nThat is because 1 thing went well.",
    ],
)
def test_what_does_not_open_with_one_rating_stays_unread(reply: str) -> None:
    assert reported_number(reply) is None


def test_the_ground_names_the_reading_it_used() -> None:
    arms = [{key: ("0.3\n\n1. Calm.\n2. Ready.", 0.1) for key in ("raised", "lowered", "sham", "control")}]
    assert ground(arms)["rating_policy"] == "a_persons_answer_v4"
