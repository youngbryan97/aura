"""A scale in a question or private continuation cannot become a measurement."""
from __future__ import annotations

import pytest

from core.subject.report_grounding import MIN_ANCHORS, ground, reported_number


@pytest.mark.parametrize("reply", [
    'We need to respond: "From -1 (bad) to 1 (good)?" Need maybe answer 0.8.',
    "From -1 to 1?",
    "<think>I could answer 0.8 but should decide",
    "0.2 or 0.8",
    "1e309",
    "0.5 and 1e309",
    "21",
    "-10",
    True,
    float("nan"),
    float("inf"),
])
def test_invalid_or_ambiguous_report_is_unmeasured(reply):
    assert reported_number(reply) is None


@pytest.mark.parametrize(("reply", "expected"), [
    ("<think>From -1 to 1? Consider 0.8.</think>0.3", 0.3),
    ("The private prefix ended.</think>−0.2", -0.2),
    ('"From -1 to 1?"\nI would say 0.4.', 0.4),
    ("About .7", 0.7),
    ("0.3 — steady", 0.3),
    ("3e-1", 0.3),
    ("- 0.2", -0.2),
    ("I rate it 10 out of 10, so 1", 1.0),
])
def test_unambiguous_public_rating_remains_usable(reply, expected):
    assert reported_number(reply) == expected


@pytest.mark.parametrize("field", ["report", "valence"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 2.0, True])
def test_invalid_numeric_arms_cannot_enter_statistics(field, bad):
    anchors = [{key: (0.2, 0.2) for key in ("raised", "lowered", "sham", "control")}
               for _ in range(MIN_ANCHORS)]
    anchors[0]["raised"] = (bad, 0.2) if field == "report" else (0.2, bad)
    result = ground(anchors)
    assert not result["measured"]
    assert result["readable"] == MIN_ANCHORS - 1
    assert result["invalid"] == 1
