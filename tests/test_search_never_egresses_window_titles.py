"""The titles of a person's private windows were sent to a search engine.

LIVE DEFECT, 2026-08-10, found in ~/.aura/logs. The "required search evidence"
lane in response_generation ran real web searches with these queries:

    query=The Sick Mind of EDP445 | …Documentary - YouTube
    query=License Plate Lookup & VIN Search | VehicleHistory.us
    query=Monthly Expenses
    query=t get a clear enough answer together, and I

The first three are the titles of Bryan's own windows. The fourth is a
mid-word fragment of Aura's own previous reply. Every one was logged
``origin=user``.

The cause: the query is ``contract.search_query or objective``, and
``objective`` is ``state.cognition.current_objective``, which is NOT always
the user's message — on an ambient or internally-driven turn it holds whatever
perception last put there. ``_clean_required_search_query`` strips formatting
instructions and never asks where the text came from.

Two costs. The evidence is irrelevant to the turn, which is a correctness
problem. And what a person is watching, a VIN lookup, and a document called
"Monthly Expenses" left the machine for an external service that nobody asked
to search anything — which is not reversible, and is the serious one.

So the guard fails closed: no demonstrable relation to the user's own words
means no search.
"""
from __future__ import annotations

import pytest

from core.phases.response_generation import ResponseGenerationPhase as _Phase

_comes_from_user = _Phase._query_comes_from_the_user


@pytest.mark.parametrize(
    "leaked",
    [
        "The Sick Mind of EDP445 | …Documentary - YouTube",
        "License Plate Lookup & VIN Search | VehicleHistory.us",
        "Monthly Expenses",
        "t get a clear enough answer together, and I",
    ],
)
def test_the_exact_queries_that_leaked_are_refused(leaked):
    """Each of these was really sent to a search engine."""
    context = {"visible_user_message": "how are you doing today?"}

    assert not _comes_from_user(leaked, context, leaked)


def test_a_real_question_still_searches():
    """The guard must not break the feature it protects."""
    context = {
        "visible_user_message": (
            "who won the 2026 Nobel Prize in Physics, and what for?"
        )
    }

    assert _comes_from_user("2026 Nobel Prize Physics winner", context, "")


def test_the_contract_may_rewrite_the_question():
    """Overlap, not equality: rewriting a question into a query is normal."""
    context = {"visible_user_message": "what's the latest on the Europa Clipper mission"}

    assert _comes_from_user("Europa Clipper mission status NASA", context, "")


def test_no_visible_user_message_means_no_search():
    """Ambient turns are exactly the ones with no user message.

    Unknown provenance on an egress path is a refusal, not a default-allow —
    defaulting to allow is how the window titles got out.
    """
    assert not _comes_from_user("Monthly Expenses", {}, "Monthly Expenses")


def test_a_query_sharing_only_filler_words_is_refused():
    """"the", "you", "what" are shared by any two English sentences."""
    context = {"visible_user_message": "what do you think about that?"}

    assert not _comes_from_user("What You Should Know About The Thing", context, "")


def test_a_contentless_question_still_allows_its_own_objective():
    """"why?" has nothing to match on; an unchanged objective is still honest."""
    context = {"visible_user_message": "why?"}

    assert _comes_from_user("why?", context, "why?")
    assert not _comes_from_user("Monthly Expenses", context, "Monthly Expenses")


def test_the_guard_runs_before_anything_is_sent():
    """Ordering is the whole property: a refusal after egress is not a refusal."""
    import inspect

    source = inspect.getsource(_Phase._execute_required_search_evidence)
    guard = source.index("_query_comes_from_the_user")
    execute = source.index("cap.execute")

    assert guard < execute, "provenance must be checked before the search runs"
