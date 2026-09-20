"""She should know what she can do this minute, and say it herself.

The capability engine has always computed, per skill, whether it is available
and why it is not. None of it reached the part of her that speaks, so a skill
being down produced a string written months earlier:

    I can't access external data right now, but based on what I know...

Not her voice, not this moment, and — the part that matters most — it
flattened a distinction a person needs:

    "I can't search, there's no network"    true now, false in a minute
    "I don't have a way to search at all"   true until someone builds it

These assert the evidence carries that distinction and stays FACTS. The
moment this module starts emitting finished sentences, the canned reply has
just moved house, so `test_the_evidence_is_not_a_script` is as load-bearing
as the rest.
"""

import pytest

from core.conversation.capability_condition import (
    CapabilityStanding,
    capability_condition_evidence,
    condition_for,
    needed_capabilities,
)
from tests.chat_lane_support import chat_lane_source
from tests.source_contract import family_text

pytestmark = pytest.mark.unit


class _Engine:
    def __init__(self, rows):
        self._rows = rows

    def iter_tool_catalog(self, *, include_inactive: bool = True):
        return list(self._rows)


READY = _Engine([{"name": "web_search", "available": True}])
DOWN = _Engine([
    {"name": "web_search", "available": False, "availability_reason": "network_unavailable"},
])
NOT_BUILT = _Engine([{"name": "file_operation", "available": True}])


class TestNoticingWhatTheTurnNeeds:
    @pytest.mark.parametrize("message,expected", [
        ("can you look up the weather in Lisbon?", "web_search"),
        ("read the file at ~/notes.txt", "file_operation"),
        ("run this for real and show the output", "code_execution"),
    ])
    def test_a_cue_is_recognised(self, message, expected):
        assert expected in needed_capabilities(message)

    def test_an_ordinary_turn_needs_nothing(self):
        assert needed_capabilities("do you think preference is the right word?") == ()
        assert capability_condition_evidence("what do you make of entropy?") == ""


class TestTheDistinctionThatMatters:
    def test_down_right_now_is_transient(self):
        condition = condition_for("web_search", capability_engine=DOWN)
        assert condition.standing is CapabilityStanding.UNAVAILABLE_NOW
        assert condition.is_transient

    def test_never_registered_is_not_transient(self):
        condition = condition_for("web_search", capability_engine=NOT_BUILT)
        assert condition.standing is CapabilityStanding.ABSENT
        assert not condition.is_transient

    def test_available_is_available(self):
        assert condition_for("web_search", capability_engine=READY).standing is (
            CapabilityStanding.READY
        )

    def test_an_unreadable_registry_is_not_a_missing_limb(self):
        """Reporting "I can't do that at all" because a lookup failed would be
        a confident lie about herself."""
        condition = condition_for("web_search", capability_engine=None)
        assert condition.standing is not CapabilityStanding.ABSENT


class TestTheEvidenceReachesHerAsFacts:
    def test_transient_says_this_moment(self):
        block = capability_condition_evidence("look up the weather", capability_engine=DOWN)
        assert "NOT AVAILABLE THIS MOMENT" in block
        assert "no network right now" in block
        assert "may work again shortly" in block

    def test_absent_says_it_is_not_an_outage(self):
        block = capability_condition_evidence("look up the weather", capability_engine=NOT_BUILT)
        assert "NOT SOMETHING YOU HAVE" in block
        assert "not a temporary outage" in block

    def test_the_evidence_is_not_a_script(self):
        """Facts plus an instruction to speak — never a ready-made apology."""
        block = capability_condition_evidence("look up the weather", capability_engine=DOWN)
        assert "your own words" in block
        for canned in ("I can't access", "I'm sorry", "Unfortunately", "I apologize"):
            assert canned not in block


class TestItReachesTheModel:
    def test_chat_publishes_it_and_the_engine_reads_it(self):

        from core.brain import cognitive_engine

        assert '"live_capability_condition": live_capability_condition' in (
            chat_lane_source()
        )
        assert 'context.get("live_capability_condition")' in (
            family_text(cognitive_engine)
        )


class TestTheCuesAreWordsAndTheRouterIsReused:
    def test_a_substring_is_not_a_cue(self):
        """"Orca Research" contains "search" and asks nothing of the web."""
        assert "web_search" not in needed_capabilities(
            "create a folder called Orca Research on my desktop"
        )

    def test_a_desktop_objective_carries_desktop_evidence(self):
        """The cue list missed "Open the Notes app and write a new note", so
        that turn carried no evidence and she denied a capability she has.
        The desktop router already knew; reusing it means they cannot
        disagree."""
        found = needed_capabilities(
            "Open the Notes app and write a new note titled Orca Field Notes "
            "with a couple of sentences about orcas in it."
        )
        assert "desktop_task" in found
        assert "computer_use" in found

    def test_conversation_still_needs_nothing(self):
        assert needed_capabilities("what do you think about entropy?") == ()
