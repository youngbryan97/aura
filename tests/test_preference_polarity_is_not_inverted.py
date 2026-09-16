"""A stated dislike was being stored as a liking.

``_form_preferences_from`` folded every repeated position she had taken into
``prefs.encounter(subject, stance="drawn_to")`` — regardless of what the
sentence said. So:

    "I don't like brutalist architecture."

accumulated, on repetition, into an attraction to brutalist architecture.

That is worse than not forming preferences at all. A preference that fails
to form costs one missed stance; a preference that forms backwards means she
will act on, and defend, the opposite of what she said — and the mechanism
that was supposed to make her values HERS instead makes them wrong.

The correction is conservative on purpose: when polarity cannot be
established, NOTHING is recorded. Guessing in the ambiguous case is the same
defect with a smaller blast radius.
"""
from __future__ import annotations

import pytest

from core.state.aura_state import _stance_of_position


# ─────────────────────────────────────────────── the inversion is gone


@pytest.mark.parametrize(
    "sentence",
    [
        "I don't like brutalist architecture",
        "I really don't enjoy long meetings",
        "I dislike flaky tests",
        "I hate silent failures",
        "I can't stand unbounded retries",
        "I avoid magic numbers in this codebase",
    ],
)
def test_a_stated_dislike_is_never_stored_as_attraction(sentence):
    stance = _stance_of_position(sentence)

    assert stance != "drawn_to", (
        f"{sentence!r} was stored as an attraction; repeated, she would come "
        "to defend the opposite of what she said"
    )
    assert stance == "averse_to"


@pytest.mark.parametrize(
    "sentence",
    [
        "I love a small, well-named function",
        "I prefer terse answers",
        "I really enjoy a clean failure path",
        "I admire the way that verifier is written",
    ],
)
def test_a_stated_liking_is_still_attraction(sentence):
    assert _stance_of_position(sentence) == "drawn_to"


@pytest.mark.parametrize(
    "sentence",
    ["I'm curious about category theory", "I am intrigued by that failure mode"],
)
def test_interest_is_its_own_stance_not_attraction(sentence):
    """"Curious about" is a real stance and it is not liking."""
    assert _stance_of_position(sentence) == "curious_about"


def test_the_contraction_form_is_not_missed():
    """"I'm" has no space after the I. An `i\\s+` prefix silently misses it.

    Caught by driving the parser rather than reading it: the contraction is
    the common form, so missing it means missing most of the class while the
    regex still looks correct.
    """
    assert _stance_of_position("I'm curious about X") == "curious_about"
    assert _stance_of_position("I am curious about X") == "curious_about"


# ────────────────────────────────── unclear polarity records nothing


@pytest.mark.parametrize(
    "sentence",
    [
        "I think the parser is fine",
        "I believe that approach scales",
        "I disagree with that framing",
        "I don't think the cache is the problem",
        "the retry budget is set in config",
        "",
    ],
)
def test_an_unclear_position_forms_no_preference(sentence):
    """A position is not a preference.

    "I think X" and "I disagree with X" state beliefs. Treating them as
    attraction is precisely how a disagreement became a liking; they belong
    to the belief system, not this one.
    """
    assert _stance_of_position(sentence) is None


def test_a_positive_opener_with_a_later_negation_is_refused():
    """"I like the elegant solution but it is not worth the complexity."

    The opener says like. The sentence does not. Rather than pick one, the
    parser declines — a stance she cannot be shown to hold is not one she
    holds.
    """
    assert (
        _stance_of_position(
            "I like the elegant solution but it is not worth the complexity"
        )
        is None
    )


# ─────────────────────────────────── it reaches the preference store


def test_formation_uses_the_parsed_stance_end_to_end(monkeypatch):
    """The parser must be CALLED, not merely correct."""
    from core.being.individual_preferences import IndividualPreferences

    recorded: list[tuple[str, str]] = []

    class _Spy(IndividualPreferences):
        def encounter(self, subject, *, stance="drawn_to", note=""):
            recorded.append((subject, stance))
            return super().encounter(subject, stance=stance, note=note)

    # The former builds a fresh set from the ledger rather than reading the
    # stored one back, because it now runs on every commit and accumulating
    # would register one position again on each of them.
    monkeypatch.setattr(
        "core.being.individual_preferences.IndividualPreferences", _Spy
    )

    class _Entry:
        kind = "position"
        speaker = "assistant"
        text = "I don't like unbounded retries in a hot loop"
        mentions = 3

    class _Ledger:
        entries = [_Entry()]

    from core.state.aura_state import AuraState

    state = AuraState.__new__(AuraState)

    class _Identity:
        self_preferences = None

    state.identity = _Identity()
    state._form_preferences_from(_Ledger())

    assert recorded, "no preference encounter was registered at all"
    assert all(stance == "averse_to" for _subject, stance in recorded), (
        f"a stated dislike reached the store as {recorded[0][1]!r}"
    )
    assert len(recorded) == 3, "repetition count was not honoured"


def test_an_unclear_position_reaches_the_store_as_nothing(monkeypatch):
    from core.being.individual_preferences import IndividualPreferences

    recorded: list[tuple[str, str]] = []

    class _Spy(IndividualPreferences):
        def encounter(self, subject, *, stance="drawn_to", note=""):
            recorded.append((subject, stance))
            return super().encounter(subject, stance=stance, note=note)

    monkeypatch.setattr(
        "core.being.individual_preferences.IndividualPreferences.from_dict",
        classmethod(lambda cls, payload: _Spy()),
    )

    class _Entry:
        kind = "position"
        speaker = "assistant"
        text = "I think the retry budget is probably about right"
        mentions = 5

    class _Ledger:
        entries = [_Entry()]

    from core.state.aura_state import AuraState

    state = AuraState.__new__(AuraState)

    class _Identity:
        self_preferences = None

    state.identity = _Identity()
    state._form_preferences_from(_Ledger())

    assert recorded == [], (
        f"an unclear position formed a preference anyway: {recorded}"
    )


def test_averse_to_is_a_stance_the_store_accepts():
    """The vocabulary already had it. Nothing had ever used it."""
    from core.being.individual_preferences import STANCES

    assert "averse_to" in STANCES
    assert "curious_about" in STANCES
