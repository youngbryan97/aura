"""The workspace had one winner, and three separate things made it one.

Measured over twenty-four ordinary turns of the offline organism: seven
emotions sat at exactly 1.000 from the fourth turn on, the affect bid entered
every competition at the effective ceiling, and it won twenty-three of
twenty-four. Ten sources bid and never once won — perception, deliberation,
ontogeny, self, metacognition among them. A domain whose bid cannot win is a
domain that cannot reach attention, and everything downstream of attention was
closed to it.

Three causes, each its own fix:

- writers that added a flat step and clamped at one, so a channel near its
  ceiling took the same push as one at rest;
- fatigue keyed on the bid's label while affect entered under forty-four
  labels, so the refractory period never applied to it;
- the affect bid setting its own `affect_weight`, which `priority_at` adds
  three tenths of on top of the priority — the same feeling counted twice,
  and the sum saturated.
"""

from __future__ import annotations

import pytest

from core.consciousness.global_workspace import CognitiveCandidate
from core.phases.affect_update import bump_emotion


def test_a_flat_step_no_longer_reaches_the_ceiling() -> None:
    """Repeated pushes approach the top without arriving at it."""
    emotions = {"joy": 0.0}
    for _ in range(200):
        bump_emotion(emotions, "joy", 0.3)
    assert emotions["joy"] < 1.0, "a feeling pinned at maximum is a constant"
    assert emotions["joy"] > 0.9, "the push still carries"


def test_the_room_below_is_symmetric() -> None:
    emotions = {"fear": 1.0}
    for _ in range(200):
        bump_emotion(emotions, "fear", -0.3)
    assert emotions["fear"] > 0.0
    assert emotions["fear"] < 0.1


def test_a_channel_at_rest_takes_the_full_step() -> None:
    emotions = {"hope": 0.0}
    bump_emotion(emotions, "hope", 0.4)
    assert emotions["hope"] == pytest.approx(0.4)


def test_a_missing_channel_starts_from_rest() -> None:
    emotions: dict[str, float] = {}
    bump_emotion(emotions, "wonder", 0.25)
    assert emotions["wonder"] == pytest.approx(0.25)


def test_fatigue_applies_to_the_bidder_not_the_label() -> None:
    """Affect names its bid after whichever emotion is on top."""
    joy = CognitiveCandidate(content="feeling joy", source="affect_joy",
                             priority=0.9, bidder_id="affect")
    trust = CognitiveCandidate(content="feeling trust", source="affect_trust",
                               priority=0.9, bidder_id="affect")
    memory = CognitiveCandidate(content="a recollection", source="memory", priority=0.9)
    assert joy.bidder == trust.bidder == "affect"
    assert joy.source != trust.source
    assert memory.bidder == "memory", "an undeclared bidder is its own label"


def test_the_affect_bid_does_not_pay_itself_the_affect_bonus() -> None:
    """Priority and affect weight were the same reading, counted twice."""
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    affect = state.affect
    affect.emotions = dict(affect.emotions or {})
    affect.emotions["joy"] = 0.88
    affect.mood_baselines = dict(getattr(affect, "mood_baselines", {}) or {})
    affect.mood_baselines["joy"] = 0.02

    bids = [b for b in build_candidates(state) if b.source.startswith("affect_")]
    assert bids, "affect offered nothing to the competition"
    for bid in bids:
        # The charge is still carried: the winner's affective charge is read
        # off it. What changed is that the bid no longer collects it as a
        # bonus on top of a priority that is the same reading.
        assert bid.affect_weight > 0.0
        assert bid.bidder == "affect"
        # Not inflated above its own priority. The recency factor still
        # applies and legitimately trims it, so the claim is that nothing is
        # added, not that nothing changed.
        assert bid.effective_priority <= bid.priority + 1e-9, (
            "affect is paying itself the weight that is meant for other bids"
        )
        assert bid.effective_priority > 0.9 * bid.priority
        assert bid.effective_priority < 1.0


def test_a_drive_alert_may_still_carry_an_affect_weight() -> None:
    """The weight is what affect lends to somebody else's bid."""
    bid = CognitiveCandidate(
        content="Drive alert: curiosity is depleted",
        source="drive_curiosity",
        bidder_id="drives",
        priority=0.5,
        affect_weight=0.6,
    )
    assert bid.affect_weight > 0.0
    assert bid.effective_priority > bid.priority
