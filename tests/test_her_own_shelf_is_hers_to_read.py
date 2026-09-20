"""Reading the encyclopedia she carries needs no one's permission.

LIVE 2026-09-20, in the middle of a game she had been asked to play:
"Tool execution 'local_reference_search' blocked by Constitution:
denied_by_default: tool_execution requires validated scoped authority". The
one search lane that cannot lose the network was the one she could not make,
because no standing grant named it.
"""
from __future__ import annotations

from core.executive import standing_authority as sa


def test_the_offline_reference_is_one_of_her_own_observations():
    assert "local_reference_search" in sa.INTROSPECTION_TOOLS


def test_a_lookup_passes_the_argument_policy_of_that_grant():
    grant = next(
        g for g in sa._builtin_grants() if g.grant_id == "aura.autonomous-introspection"
    )
    ok, reason = sa.get_standing_authority_manager()._argument_policy_allows(
        grant,
        tool_name="local_reference_search",
        arguments={"query": "tides", "limit": 3},
        user_authorized=False,
    )
    assert ok, reason


def test_the_grant_that_covers_it_is_read_only_and_autonomous():
    grant = next(
        g for g in sa._builtin_grants()
        if g.grant_id == "aura.autonomous-introspection"
    )
    assert "local_reference_search" in grant.allowed_tools
    assert set(grant.allowed_effect_scopes) <= {"read_only", "status"}
    assert grant.max_risk == "low"
