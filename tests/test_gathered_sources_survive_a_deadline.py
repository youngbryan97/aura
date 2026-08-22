"""Pages already fetched are not thrown away when the summary runs long.

LIVE, 2026-08-22. Asked to look up a company, the runtime searched, fetched
five pages, and then had its synthesis step run past the caller's 35-second
budget. `asyncio.wait_for` cancels the task it is waiting on, so the five pages
went with it. The header read REPLY PATH BLOCKED and the reply was "I couldn't
get to an answer I'd stand behind" — while five usable sources had been in
memory a moment earlier.

Gathering and summarising are different work with different failure modes.
"""

from __future__ import annotations

from core.search.gathered_sources import clear_gathered, record_gathered, take_gathered


def setup_function() -> None:
    clear_gathered()


def test_nothing_gathered_means_nothing_to_recover():
    assert take_gathered() is None


def test_pages_are_kept_the_moment_they_exist():
    kept = record_gathered(
        "hugging face",
        [
            {"title": "About", "url": "https://example.invalid/a", "text": "one"},
            {"title": "Docs", "url": "https://example.invalid/b", "snippet": "two"},
        ],
    )
    assert kept == 2
    held = take_gathered()
    assert held is not None
    assert held.query == "hugging face"
    assert [item.url for item in held.sources] == [
        "https://example.invalid/a",
        "https://example.invalid/b",
    ]


def test_an_entry_with_nothing_in_it_is_not_a_source():
    assert record_gathered("q", [{"title": "", "url": "", "text": ""}]) == 0
    assert take_gathered() is None


def test_objects_work_as_well_as_dicts():
    class Hit:
        title = "T"
        url = "https://example.invalid/c"
        snippet = "s"

    assert record_gathered("q", [Hit()]) == 1
    held = take_gathered()
    assert held is not None and held.sources[0].url == "https://example.invalid/c"


def test_a_stale_gather_belongs_to_a_previous_question():
    record_gathered("old", [{"url": "https://example.invalid/d", "text": "x"}])
    assert take_gathered(max_age_s=-1.0) is None


def test_a_timed_out_search_answers_from_what_it_fetched():
    from interface.routes.chat import _recovered_search_result

    record_gathered(
        "who runs hugging face",
        [
            {"title": "About", "url": "https://example.invalid/a", "text": "Founded in 2016."},
            {"title": "Team", "url": "https://example.invalid/b", "text": "Led by the founders."},
        ],
    )
    recovered = _recovered_search_result("who runs hugging face", TimeoutError())
    assert recovered["ok"] is True
    assert recovered["partial"] is True
    assert recovered["count"] == 2
    assert len(recovered["citations"]) == 2
    assert "Founded in 2016." in recovered["content"]
    assert "ran out of time" in recovered["note"]


def test_a_search_that_fetched_nothing_still_reports_failure():
    """Recovery must not invent success out of an empty hand."""
    from interface.routes.chat import _recovered_search_result

    clear_gathered()
    recovered = _recovered_search_result("q", TimeoutError())
    assert recovered["ok"] is False
    assert recovered["status"] == "required_search_failed"
