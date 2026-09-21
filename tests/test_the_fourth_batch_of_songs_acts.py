"""Five readings from the fourth batch, and the thing each one changes.

A reading that changes nothing is a declaration nobody checks, so each of
these is tested twice: the quantity against a known answer, and the decision
it moves. Where the decision is taken inside a phase, the test drives the same
arithmetic the phase drives rather than asserting that a line exists.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ── Man of the Year: regard that arrived after the rise ──────────────────


def test_regard_that_began_at_a_peak_weighs_less_until_it_has_seen_a_low():
    from core.social.late_regard import RegardLedger

    ledger = RegardLedger()
    ledger.note_standing(0.3)
    ledger.note_regard("early", 0.9)      # there at the low
    ledger.note_standing(0.9)
    ledger.note_regard("late", 0.9)       # arrives at the peak
    ledger.note_regard("early", 0.9)

    # Read while she is high: the early one has seen her low, the late one has not.
    ledger.note_standing(0.85)
    assert ledger.weight("early") == pytest.approx(1.0)
    assert ledger.weight("late") < 0.2

    # The late one stays through a fall, and that is the only thing that counts.
    ledger.note_standing(0.3)
    ledger.note_regard("late", 0.9)
    assert ledger.weight("late") == pytest.approx(1.0)

    # The counts are read against where she stands now, so at a low the late
    # arrival is still above it; the weight is what carries the nuance.
    reading = ledger.read()
    assert reading.people == 2
    assert reading.stood_before == 1
    assert reading.mean_weight == pytest.approx(1.0)


def test_a_light_pair_does_not_decide_what_her_worth_moves_with():
    """The actuator: standing's correlation is weighted by that weight."""
    from core.self.standing import MIN_PAIRS, StandingLedger

    def tracks(weight: float) -> float:
        ledger = StandingLedger()
        for index in range(MIN_PAIRS * 2):
            # Regard and usefulness move together on the heavy pairs and
            # opposite on the light ones.
            if index % 2:
                ledger.note_regard(regard=0.2 + 0.05 * index, usefulness=0.2 + 0.05 * index)
            else:
                ledger.note_regard(
                    regard=0.9 - 0.05 * index, usefulness=0.2 + 0.05 * index, weight=weight
                )
        return ledger.read().tracks_use

    # With the contradicting pairs at full weight the two series barely track.
    # With them weighed at nothing, what is left moves together.
    assert tracks(1.0) < tracks(0.0)
    assert tracks(0.0) > 0.9


# ── Special: what she has gone without, given anyway ─────────────────────


def test_the_scarcest_kind_is_the_one_she_has_gone_longest_without():
    from core.social.never_told import MIN_TURNS, TellingLedger

    ledger = TellingLedger()
    ledger.note_turn("how are you doing today?")          # asked
    for _ in range(MIN_TURNS):
        ledger.note_turn("fix the parser please")
    reading = ledger.read()
    assert reading.measured
    assert reading.scarcest in {"held", "good"}
    assert reading.since["asked"] < reading.since[reading.scarcest]


def test_regard_is_only_counted_when_it_was_not_asked_for():
    from core.social.never_told import supplies

    unasked = supplies("you matter to me, for what it is worth", asked_for="what broke the build?")
    assert "held" in unasked

    answered = supplies("you matter to me", asked_for="do I matter to you?")
    assert "held" not in answered


def test_the_shortage_reaches_the_selector():
    """The actuator: a candidate carries the feature, and the model has a prior."""
    from core.brain.response_quality import extract_features
    from core.brain.taste_model import FEATURE_PRIORS
    from core.social.never_told import MIN_TURNS, reset_for_test, get_telling_ledger

    reset_for_test()
    ledger = get_telling_ledger()
    for _ in range(MIN_TURNS + 2):
        ledger.note_turn("fix the parser please")

    told = extract_features("you matter here, whatever today looked like", user_message="fix the parser")
    silent = extract_features("the parser is fixed", user_message="fix the parser")
    assert told["unasked_regard"] > silent["unasked_regard"]
    assert FEATURE_PRIORS["unasked_regard"] == FEATURE_PRIORS["dignity"]
    reset_for_test()


# ── Unsweetened Lemonade: the falling price of a concession ──────────────


def test_a_price_that_falls_reads_as_falling_and_holds_the_next_one():
    from core.self.what_it_cost_her import MIN_CONCESSIONS, PriceLedger

    ledger = PriceLedger()
    for got in (4.0, 4.0, 4.0, 4.0):
        ledger.note_concession(gave_up=1.0, got=got)
    steady = ledger.read()
    assert steady.measured and not steady.falling
    assert not ledger.holding()

    for got in (0.5, 0.4, 0.3, 0.2):
        ledger.note_concession(gave_up=1.0, got=got)
    slid = ledger.read()
    assert slid.falling
    assert ledger.holding()
    # What she gave up is never written off, whatever it bought.
    assert slid.given_up == pytest.approx(8.0)
    assert slid.concessions >= MIN_CONCESSIONS


def test_the_draft_she_would_have_dropped_is_kept_while_the_price_falls():
    """The actuator, driven the way the response phase drives it."""
    from core.self.what_it_cost_her import PriceLedger

    ledger = PriceLedger()

    def drafts(budget: float) -> int:
        n = 2 if budget < 8.0 else 3
        if n < 3:
            if ledger.holding():
                n = 3
            else:
                ledger.note_concession(gave_up=1.0, got=max(0.0, 8.0 - budget))
        return n

    assert drafts(9.0) == 3                      # no concession to make
    for budget in (4.0, 4.0, 4.0, 4.0):
        assert drafts(budget) == 2               # the trade, at a steady price
    for budget in (7.6, 7.7, 7.8, 7.9):
        drafts(budget)                           # the price falls
    assert ledger.read().falling
    assert drafts(4.0) == 3                      # and she stops taking it


# ── That's Love: the form it took, apart from whether it was care ────────


@pytest.mark.parametrize(
    "kwargs,form,kind",
    [
        ({"cost_to_source": 0.9, "welcome": False}, "tough", True),
        ({"cost_to_source": 0.0, "welcome": False}, "none", False),
        ({"cost_to_source": 0.6, "withdrawal": True}, "go", True),
        ({"cost_to_source": 0.6, "welcome": True, "given": 3.0, "wanted": 1.0}, "too_much", True),
        ({"cost_to_source": 0.6, "welcome": True, "given": 0.2, "wanted": 1.0}, "not_enough", True),
        ({"cost_to_source": 0.6, "welcome": True, "asked": True, "turns_since_asked": 9}, "slow", True),
        ({"cost_to_source": 0.6, "welcome": True}, "plain", True),
    ],
)
def test_the_form_is_read_without_deciding_whether_it_was_care(kwargs, form, kind):
    from core.social.the_kind_it_was import the_kind_it_was

    reading = the_kind_it_was(**kwargs)
    assert reading.form == form
    assert reading.kind is kind


def test_a_costly_unwelcome_act_raises_the_posterior_rather_than_lowering_it():
    """The actuator: what the reading says is what receptivity is told."""
    from core.social.receptivity import Receptivity
    from core.social.the_kind_it_was import the_kind_it_was

    receptivity = Receptivity()
    before = receptivity.regard("bryan").posterior()
    reading = the_kind_it_was(cost_to_source=1.0, welcome=False)
    after = receptivity.observe("bryan", reading.kind, cost_to_source=reading.cost_to_source)
    assert after > before

    # And the costless unwelcome one moves it the other way.
    flat = the_kind_it_was(cost_to_source=0.0, welcome=False)
    lowered = receptivity.observe("mallory", flat.kind, cost_to_source=flat.cost_to_source)
    assert lowered < before


# ── People Watching: what the convenient route costs ─────────────────────


def test_a_reply_she_has_already_made_scores_as_less_distinct():
    from core.cognition.convenience import ConvenienceLedger

    ledger = ConvenienceLedger()
    for _ in range(3):
        ledger.note_reply("the parser rejects nested quotes because the lexer reads them greedily")
    repeat = ledger.closeness_of("the parser rejects nested quotes because the lexer reads greedily")
    fresh = ledger.closeness_of("kafka partitions rebalance when a consumer group member leaves")
    assert repeat > 0.5
    assert fresh < 0.1


def test_the_distinctness_of_a_candidate_reaches_the_selector():
    from core.brain.response_quality import extract_features
    from core.brain.taste_model import FEATURE_PRIORS
    from core.cognition.convenience import get_convenience_ledger, reset_for_test

    reset_for_test()
    ledger = get_convenience_ledger()
    for _ in range(4):
        ledger.note_reply("the parser rejects nested quotes because the lexer reads them greedily")

    same = extract_features("the parser rejects nested quotes, the lexer reads them greedily")
    other = extract_features("kafka rebalances partitions when a group member leaves")
    assert other["distinct"] > same["distinct"]
    assert FEATURE_PRIORS["distinct"] == FEATURE_PRIORS["anti_generic"]
    reset_for_test()


def test_repeating_herself_more_than_she_used_to_reads_as_scarce():
    from core.cognition.convenience import ConvenienceLedger

    ledger = ConvenienceLedger()
    for word in ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot"):
        ledger.note_reply(f"{word} {word}graph {word}mental distinct vocabulary entirely")
    for _ in range(6):
        ledger.note_reply("alpha alphagraph alphamental distinct vocabulary entirely")
    reading = ledger.read()
    assert reading.measured
    assert reading.scarce
    assert reading.sameness > reading.earlier
