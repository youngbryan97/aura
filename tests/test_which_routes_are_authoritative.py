"""A route that watches and a route that answers are different things.

An external review said several recurrent and compositional semantic routes
are shadow- or qualification-gated rather than generally authoritative for
arbitrary production cognition. That was true and it was nowhere in the code,
so the difference between a route that answers and one that watches was
something a reader had to reconstruct from imports.

Where it stands: six routes — two authoritative, three qualified, one shadow.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm.which_routes_are_authoritative import (
    THE_SEMANTIC_ROUTES,
    how_the_routes_stand,
    which_are_only_shadows,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def routes():
    return how_the_routes_stand()


def test_every_route_is_in_one_of_the_three_states():
    assert {row["state"] for row in THE_SEMANTIC_ROUTES.values()} <= {
        "authoritative", "qualified", "shadow"
    }


def test_every_route_says_why_it_is_in_that_state(routes):
    """A route listed as shadow with no reason is one somebody forgot."""
    assert routes["unexplained"] == []


@pytest.mark.parametrize("name", sorted(THE_SEMANTIC_ROUTES))
def test_every_route_points_at_a_file_that_exists(name):
    assert (ROOT / THE_SEMANTIC_ROUTES[name]["where"]).exists()


def test_the_states_add_up(routes):
    assert (
        routes["authoritative"] + routes["qualified"] + routes["shadow"]
        == routes["routes"]
    )


def test_the_induction_is_authoritative_and_the_chat_path_proves_it():
    """Not a claim in a table: the route is applied where answers are served."""
    assert THE_SEMANTIC_ROUTES["sequence_induction"]["state"] == "authoritative"
    chat = (ROOT / "interface" / "routes" / "chat.py").read_text("utf-8")
    assert "answer_sequence_question" in chat
    assert '("worked_out_sequence"' in chat


def test_the_recurrent_route_is_a_shadow_and_says_so(routes):
    """It contributes no answers however good it gets."""
    assert "unified_recurrent_shadow" in which_are_only_shadows()
    assert routes["shadow"] >= 1


def test_a_qualified_route_names_what_qualifies_it():
    said = THE_SEMANTIC_ROUTES["semantic_neural_serving"]["why"]
    assert "qualified_exact_semantic_v1" in said

    source = (ROOT / "core" / "brain" / "llm" / "semantic_neural_serving.py").read_text(
        "utf-8"
    )
    assert 'SEMANTIC_NEURAL_SERVING_MODE: Final = "qualified_exact_semantic_v1"' in source


def test_the_shadow_count_is_what_has_to_fall(routes):
    """And it falls by promotion. Deleting one would also lower it, so the
    authoritative count is tracked beside it."""
    assert routes["shadow"] <= 1
    assert routes["authoritative"] >= 2


def test_the_meaning_travels_with_the_counts(routes):
    assert "no answers however good it gets" in routes["what_this_means"]


def test_the_states_are_in_the_health_report():
    """Through the registry: core/runtime may not import core.brain."""
    import core.brain.cognitive_engine  # noqa: F401 — importing registers it
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["which_routes_are_authoritative"]
    assert set(block) >= {"routes", "authoritative", "qualified", "shadow"}


def test_health_reads_them_through_the_registry_and_not_by_importing():
    source = (ROOT / "core" / "runtime" / "health_contract.py").read_text("utf-8")
    assert "which_routes_are_authoritative import" not in source
    assert 'get_runtime_service("which_routes_are_authoritative"' in source
