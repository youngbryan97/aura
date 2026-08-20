"""The answer is on disk, so do not search the web for it.

LIVE, 2026-08-19. "there's a python project at /private/tmp/claude-501/.../
ledger - one of its tests is failing. read the code, work out why" set
`requires_search` with the whole message as the query. The runtime searched
the WEB for a filesystem path, handed her results about /private/tmp disk
usage, and she replied:

    The search results you provided don't contain any information about a
    Python project in that directory — they're mostly about disk usage issues
    with /private/tmp/claude-501. Can you upload the code or repository URL?

Which was true about the results, and the answer was on disk the whole time.

The same line this codebase already draws between observation and actuation,
one axis over: local source of truth versus remote.
"""

from __future__ import annotations

import pytest

from core.phases.response_contract import build_response_contract
from core.state.aura_state import AuraState


def _contract(message: str):
    return build_response_contract(AuraState(), message, is_user_facing=True)


@pytest.mark.parametrize(
    "message",
    [
        "there's a python project at /private/tmp/x/ledger - one of its tests is "
        "failing. read the code, work out why",
        "read /tmp/notes.txt and tell me what it says",
        "list the files in ~/Documents and tell me the biggest",
        "check /etc/hosts and tell me if the entry is there",
    ],
)
def test_a_local_read_does_not_search_the_web(message: str):
    contract = _contract(message)
    assert contract.requires_search is False
    assert contract.required_skill != "web_search"


@pytest.mark.parametrize(
    "message",
    [
        "search the web for the latest fusion results",
        "what happened in the news today",
        "what is https://example.com about",
    ],
)
def test_a_real_lookup_still_searches(message: str):
    """The guard must not cost a single genuine search."""
    assert _contract(message).requires_search is True


def test_a_path_that_is_also_a_write_is_not_claimed_by_this_guard():
    """Only reads. A write goes to the lane that can perform it."""
    from core.runtime.desktop_objective_intent import looks_like_filesystem_observation

    assert not looks_like_filesystem_observation(
        "write a haiku to a file on my Desktop called aura_haiku.txt"
    )
