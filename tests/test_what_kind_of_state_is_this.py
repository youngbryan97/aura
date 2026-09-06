"""Authorities, projections, and scratch — with the paths checked.

The conclusion the blind comparison cared about most was that Aura needs lower
causal ambiguity per unit of functionality, and it made the point by listing
eleven state holders where the peers have one. It also said the honest thing:
many of those are legitimately different, and the remaining problem is
deciding which are authorities, which are derived projections, and which are
temporary computational state.

A table of prose would be worth nothing, so these check the paths exist and
the claims are shaped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.state.what_kind_of_state_is_this import (
    THE_HOLDERS,
    AKindOfState,
    how_the_state_is_organised,
    what_is_not_classified,
)

ROOT = Path(__file__).resolve().parents[1]


def test_every_holder_is_one_of_the_three_kinds():
    for holder in THE_HOLDERS:
        assert isinstance(holder.kind, AKindOfState)


def test_nothing_is_unclassified():
    """The gate. Empty is the baseline and it only stays empty."""
    assert what_is_not_classified() == []


@pytest.mark.parametrize("holder", THE_HOLDERS, ids=lambda one: one.where)
def test_every_holder_points_at_a_file_that_exists(holder):
    """A table of prose about files that moved is worth nothing."""
    path = ROOT / holder.where.split(":", 1)[0]
    assert path.exists(), f"{holder.where} does not exist"


@pytest.mark.parametrize("holder", THE_HOLDERS, ids=lambda one: one.where)
def test_a_holder_naming_a_class_names_one_that_is_there(holder):
    if ":" not in holder.where:
        return
    path, rest = holder.where.split(":", 1)
    name = rest.split(".", 1)[0]
    source = (ROOT / path).read_text("utf-8", errors="ignore")
    assert f"class {name}" in source, f"{name} is not defined in {path}"


def test_every_projection_names_the_authority_it_comes_from():
    """A projection nobody can trace back to a source is a fork."""
    for holder in THE_HOLDERS:
        if holder.kind is AKindOfState.PROJECTION:
            assert holder.derived_from.strip(), holder.where
            assert holder.fresh.strip(), holder.where


def test_every_authority_says_what_it_decides():
    for holder in THE_HOLDERS:
        if holder.kind is AKindOfState.AUTHORITY:
            assert len(holder.holds.split()) >= 4, holder.where


def test_no_two_holders_are_the_same_place():
    wheres = [one.where for one in THE_HOLDERS]
    assert len(wheres) == len(set(wheres))


def test_the_eleven_the_review_named_are_all_here():
    """It listed them by hand; none may quietly drop out of the table."""
    covered = " ".join(one.where.lower() for one in THE_HOLDERS)
    for named in (
        "canonical_self",
        "aura_state",
        "being/runtime",
        "interiority",
        "workspace",
        "soma",
        "affect",
        "orchestrator",
        "aura_kernel",
    ):
        assert named in covered, f"{named} is not classified"


def test_the_orchestrator_holds_no_durable_fact():
    """Which is why its runtime_state goes to the kernel."""
    orchestrator = next(
        one for one in THE_HOLDERS if "orchestrator" in one.where
    )
    assert orchestrator.kind is AKindOfState.SCRATCH


def test_the_report_counts_by_kind():
    report = how_the_state_is_organised()
    assert report["holders"] == len(THE_HOLDERS)
    assert (
        report["authorities"] + report["projections"] + report["scratch"]
        == report["holders"]
    )


def test_the_report_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["what_kind_of_state_is_this"]
    assert set(block) >= {"holders", "authorities", "projections", "scratch"}
