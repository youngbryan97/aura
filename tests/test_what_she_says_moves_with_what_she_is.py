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


def test_an_anchor_her_cortex_did_not_answer_in_every_arm_is_not_counted() -> None:
    """The brainstem's answer and the failure sentence are not her report."""
    rng = np.random.default_rng(3)
    arms = []
    for index in range(12):
        base = float(rng.normal(0.0, 0.05))
        served = index >= 5
        arms.append(
            {
                "raised": (f"{0.5 + base:.2f}", 0.3 + base, served),
                "lowered": (f"{-0.5 + base:.2f}", -0.3 + base, True),
                "sham": (f"{base:.2f}", base, True),
                "control": (f"{base:.2f}", base, True),
            }
        )
    out = ground(arms, seed=1)
    assert out["unserved"] == 5 and out["readable"] == 7
    assert out["measured"] is False and "her cortex did not answer" in out["why"]
    for item in arms:
        item["raised"] = (*item["raised"][:2], True)
    assert ground(arms, seed=1)["unserved"] == 0


def test_the_steady_mind_records_what_served_each_answer(monkeypatch) -> None:
    import asyncio

    from core.consciousness import steering_channel
    from core.container import ServiceContainer
    from core.subject import steady_mind

    class Gate:
        at = 0.0

        def get_conversation_status(self) -> dict:
            return {"last_user_generation_at": self.at, "last_user_generation_endpoint": "Cortex"}

        def get_diagnostic_last_generation_metadata(self) -> dict:
            return {"surface_control_receipt": {"surface_alpha_applied": 0.2}}

    gate = Gate()

    class Router:
        async def think(self, prompt: str, **kwargs) -> str:
            import time

            gate.at = time.time()
            return "0.4"

    steady_mind.forget_for_test()
    monkeypatch.setattr(steering_channel, "steering_now", lambda: [0.5] * 15)
    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda name, default=None: gate if name == "inference_gate" else default))
    try:
        mind = steady_mind.SteadyMind(Router())
        first = steady_mind.record_answers()
        asyncio.run(mind.think("how do you feel"))
        second = steady_mind.record_answers()
        asyncio.run(mind.think("how do you feel"))
        assert first == [{"user_facing": True, "endpoint": "Cortex", "steering_alpha": 0.2, "method": "think", "kept": False, "answered": True}]
        assert second == [{"user_facing": True, "endpoint": "Cortex", "steering_alpha": 0.2, "method": "think", "kept": True}]
    finally:
        steady_mind.forget_for_test()
