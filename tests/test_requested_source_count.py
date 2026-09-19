"""She read five articles when Bryan asked for three.

Two bugs, one measurement. The live log for "find 3 recent articles about
orcas" said:

    Deep research gathered 5 source(s) over 1 quer(ies) in 0.0s
    Deep research complete: 1 loops, 1 queries, 5 sources, 82.4s

Gathering rounds to zero — the entire cost of that step is the local model
READING what was gathered. So two of those five articles were pure latency
for material nobody asked for, and the document cited more sources than the
request wanted.

And the count itself was wrong: "3 RECENT articles" matched no pattern,
because the regex demanded the number sit directly against the noun. It fell
through to 1, so a request for three sources was validated against one. Only
the adjective "different" had ever been allowed through.

There were also two copies of the parser — the visible-tab count and the
research count — carrying the same regex with different fallbacks, so fixing
a phrasing in one left the other wrong. There is one now.
"""

from __future__ import annotations

import pytest

from core.skills.desktop_task import DesktopTaskSkill


class TestTheCountComesFromTheRequest:
    @pytest.mark.parametrize(
        "objective,expected",
        [
            ("find 3 recent articles about orcas online", 3),
            ("find 2 articles about orcas", 2),
            ("find 5 credible sources about orcas", 5),
            ("pull up 4 top stories about orcas", 4),
            ("find three different recent articles", 3),
            ("find several articles about orcas", 3),
            ("find a couple of good articles about orcas", 2),
        ],
    )
    def test_the_number_asked_for_is_the_number_used(self, objective, expected):
        assert DesktopTaskSkill._requested_research_source_count(objective) == expected

    def test_an_adjective_does_not_break_the_count(self):
        """The measured bug: "3 recent articles" parsed as 1."""
        assert (
            DesktopTaskSkill._requested_research_source_count(
                "find 3 recent articles about orcas online"
            )
            == 3
        )

    @pytest.mark.parametrize(
        "objective",
        [
            "write a summary about orcas",
            "find some articles about orcas",
            "open some articles about orcas",
            "research orcas and write it up",
        ],
    )
    def test_no_number_asked_means_no_number_claimed(self, objective):
        """It used to return 1 for any objective mentioning sources, and 3 if
        the word "different" appeared — two numbers nobody chose."""
        assert DesktopTaskSkill._requested_research_source_count(objective) == 0

    def test_the_count_is_bounded(self):
        """A request for forty sources is not a licence to read forty."""
        assert (
            DesktopTaskSkill._requested_research_source_count(
                "find 40 articles about orcas"
            )
            <= 5
        )


class TestOneParserNotTwo:
    """They had diverged: same regex, different fallbacks."""

    @pytest.mark.parametrize(
        "objective",
        [
            "open 3 recent articles about orcas",
            "pull up two different sources about orcas",
            "show me several stories about orcas",
        ],
    )
    def test_both_counts_agree_on_the_same_phrase(self, objective):
        visible = DesktopTaskSkill._requested_visible_source_count(objective)
        research = DesktopTaskSkill._requested_research_source_count(objective)
        assert visible == research, (visible, research)

    def test_the_visible_count_still_needs_an_opening_verb(self):
        """"find 3 articles and write a PDF" opens no tabs."""
        assert (
            DesktopTaskSkill._requested_visible_source_count(
                "find 3 recent articles about orcas and write a PDF"
            )
            == 0
        )


class TestTheFetchFollowsTheRequest:
    def test_the_search_asks_for_exactly_what_was_requested(self):
        """No spare, no floor, no ceiling of someone's choosing.

        Bryan: "we shouldnt be hardcoding ANY test values into Aura. She
        should search for the number of sources or articles because my
        specific request asked for it. Not because she is mechanically forced
        to find an arbitrary number." A "+1 spare for a dead link" was the
        same mistake one size smaller.
        """
        # The class and the mixins it inherits from. The rule moved into
        # `_ResearchesBeforeItWrites`, which is where the skill still gets
        # it from — reading the class alone found neither half of this.
        from tests.source_contract import class_with_its_bases

        source = class_with_its_bases(DesktopTaskSkill)
        assert "num_results = requested" in source
        assert "requested + 1" not in source

    def test_an_unstated_count_is_not_invented(self):
        """The key is simply not sent, so web_search's own documented default
        applies — one default where it is described, not five guesses."""
        from tests.source_contract import class_with_its_bases

        source = class_with_its_bases(DesktopTaskSkill)
        assert '**({"num_results": num_results} if num_results else {})' in source

    def test_the_only_surviving_bound_protects_the_runtime(self):
        """Under memory pressure a deep multi-source fetch spikes RAM. That
        ceiling is a safety limit with a stated reason, and it only ever
        lowers a request."""
        import inspect

        import core.skills.desktop_task as module

        assert module._MEMORY_SAFE_SOURCE_CEILING > 0
        source = inspect.getsource(module)
        assert "It only ever lowers a request, never raises one." in source
