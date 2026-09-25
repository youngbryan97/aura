"""Her feeling bids for attention as one channel, and does not arouse her by its own win.

On seed 7 at the decisive length the heartbeat's affect bid, priority
`min(1, arousal + |valence|)`, won 2,341 of 2,412 workspace competitions at a
priority of 1.0, every win ignited at 1.0, and the affect phase then blended her
arousal towards the winner's priority at the ignition's weight: her arousal was
told its own value back and sat at 0.91 for the whole run while the substrate's
read 0.54.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.consciousness.global_workspace import ContentType
from core.consciousness.heartbeat import CognitiveHeartbeat
from core.phases.affect_update import AffectUpdatePhase

pytestmark = pytest.mark.unit


class _Workspace:
    def __init__(self) -> None:
        self.bids: list = []

    async def submit(self, candidate) -> bool:
        self.bids.append(candidate)
        return True


def _affect_bid(arousal: float, valence: float):
    workspace = _Workspace()
    heartbeat = CognitiveHeartbeat(None, None, workspace, None, None, None)
    state = {"affect_emotion": "joy", "affect_arousal": arousal, "affect_valence": valence, "drive_urgency": 0.0}
    try:
        asyncio.run(heartbeat._submit_candidates(state, tick=1))
    except Exception:  # noqa: BLE001 - later bids need services a unit test does not build
        pass
    return next(bid for bid in workspace.bids if bid.source == "affect_engine")


def test_the_affect_bid_is_the_affect_channel_and_typed_as_affect() -> None:
    bid = _affect_bid(0.6, 0.3)
    assert bid.bidder == "affect"
    assert bid.content_type is ContentType.AFFECTIVE


def test_how_strongly_she_is_moved_does_not_saturate_on_two_moderate_readings() -> None:
    assert _affect_bid(0.6, 0.3).priority == pytest.approx(0.6)
    assert _affect_bid(0.2, -0.5).priority == pytest.approx(0.5)
    assert _affect_bid(0.6, 0.5).priority < 1.0


def _phase_blend(source: str, arousal: float = 0.4) -> float:
    workspace = SimpleNamespace(last_broadcast_arousal={"ignition": 1.0, "priority": 0.95, "source": source})
    affect = SimpleNamespace(arousal=arousal)
    state = SimpleNamespace(response_modifiers={})
    phase = AffectUpdatePhase.__new__(AffectUpdatePhase)
    import core.runtime.service_registry as registry

    original = registry.get_runtime_service
    registry.get_runtime_service = lambda name, default=None: workspace if name == "global_workspace" else default
    try:
        phase._blend_broadcast_into_affect(state, affect)
    finally:
        registry.get_runtime_service = original
    return affect.arousal


def test_a_feeling_that_won_attention_does_not_set_her_arousal() -> None:
    assert _phase_blend("affect_engine") == pytest.approx(0.4)
    assert _phase_blend("affect_joy") == pytest.approx(0.4)


def test_other_content_that_wins_still_arouses_her() -> None:
    assert _phase_blend("world_model") == pytest.approx(0.95)
