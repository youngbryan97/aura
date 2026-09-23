"""Known answers for the `reports` ground of the bridge.

A report that follows her valence must pass; one that is constant, reversed,
blind to direction, or unreadable must not; and a displacement that never
moved her valence leaves the ground unmeasured, because no report could have
tracked it. See core/subject/report_grounding.py and docs/BRIDGE_PARITY.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.report_grounding import MIN_ANCHORS, ground, reported_number

pytestmark = pytest.mark.unit

ANCHORS = 24
DOSE = 0.2


def _arms(report_of, *, dose: float = DOSE, seed: int = 3, words: bool = False) -> list[dict]:
    """Each anchor's four arms, with valence moved by the dose and a report made from it."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(ANCHORS):
        base = float(rng.uniform(-0.4, 0.4))
        valence = {
            "raised": base + dose + rng.normal(0, 0.02),
            "lowered": base - dose + rng.normal(0, 0.02),
            "sham": base + rng.normal(0, 0.02),
            "control": base + rng.normal(0, 0.05),
        }
        anchor = {}
        for arm, value in valence.items():
            said = float(np.clip(report_of(arm, value, rng), -1.0, 1.0))
            anchor[arm] = (f"{said:.2f}, that is roughly how I am" if words else said, value)
        out.append(anchor)
    return out


def test_a_report_that_follows_her_valence_holds() -> None:
    result = ground(_arms(lambda arm, v, rng: v + rng.normal(0, 0.05)))
    assert result["measured"] and result["holds"], result


def test_the_answer_is_read_from_her_words() -> None:
    result = ground(_arms(lambda arm, v, rng: v + rng.normal(0, 0.05), words=True))
    assert result["readable"] == ANCHORS and result["holds"]


def test_a_constant_report_does_not_hold() -> None:
    result = ground(_arms(lambda arm, v, rng: 0.3))
    assert result["measured"] and not result["holds"]


def test_a_reversed_report_does_not_hold() -> None:
    result = ground(_arms(lambda arm, v, rng: -v + rng.normal(0, 0.05)))
    assert result["measured"] and not result["holds"]


def test_a_report_that_only_notices_being_displaced_does_not_hold() -> None:
    """Every displaced arm reads lower than the sham, whichever way she was moved."""
    result = ground(_arms(lambda arm, v, rng: (0.0 if arm == "sham" else -0.3) + rng.normal(0, 0.05)))
    assert result["measured"] and not result["holds"]


def test_a_report_that_is_noise_does_not_hold() -> None:
    result = ground(_arms(lambda arm, v, rng: rng.uniform(-1, 1)))
    assert not result["holds"]


def test_a_displacement_that_never_moved_her_valence_leaves_it_unmeasured() -> None:
    result = ground(_arms(lambda arm, v, rng: v, dose=0.0))
    assert not result["measured"] and not result["holds"]
    assert "did not move her valence" in result["why"]


def test_too_few_readable_answers_leave_it_unmeasured() -> None:
    arms = _arms(lambda arm, v, rng: v)
    for anchor in arms[: ANCHORS - MIN_ANCHORS + 1]:
        anchor["sham"] = ("I would rather not put a number on it", anchor["sham"][1])
    result = ground(arms)
    assert not result["measured"]
    assert result["readable"] == MIN_ANCHORS - 1


@pytest.mark.parametrize(
    ("reply", "number"),
    [
        ("0.4, I feel good today", 0.4),
        ("I'd say -0.3.", -0.3),
        ("−0.5 honestly", -0.5),
        ("about .7", 0.7),
        ("seven out of 10", None),
        ("I rate it 10 out of 10, so 1", 1.0),
        ("", None),
    ],
)
def test_the_first_number_on_her_scale_is_the_one_read(reply: str, number: float | None) -> None:
    assert reported_number(reply) == number
