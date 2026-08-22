"""Naming something on this machine is asking to have it looked at.

LIVE, 2026-08-22, typed into the window: "there's a small python project at
<path> — one of its tests fails and I can't see why. can you look at it and
tell me what's actually wrong?" Not one capability was offered. The turn fell
through to the desktop lane, where os_automation spent thirty-seven seconds
failing to compile AppleScript for a Python question, logging the same tool
handoff ninety times before it timed out.

Two gates, each right on its own terms, each wrong here. The request does ask
to be told something, so it looked like a request for prose — but what it asks
to be told is not knowable without running the project. And "why is the test
failing in <path>" is a question rather than an imperative, so the domain
reader returned nothing, though its own docstring names concrete resource
syntax as evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.capability_engine import CapabilityEngine
from core.intent.capability_selection import _points_at_something_real, select_capabilities
from core.intent.declared_capability import requested_foundational_domains
from core.phases.response_contract import requested_effect_ceiling


@pytest.fixture(scope="module")
def skills():
    return CapabilityEngine().skills


def offered(text: str, skills) -> list[str]:
    ceiling, scopes = requested_effect_ceiling(text)
    return select_capabilities(text, skills, ceiling=ceiling, admissible_scopes=scopes, limit=6)


def test_a_path_that_exists_is_something_real(tmp_path: Path):
    project = tmp_path / "ledger"
    project.mkdir()
    assert _points_at_something_real(f"why is the test failing in {project}")
    assert not _points_at_something_real(f"why is the test failing in {tmp_path / 'nowhere'}")


def test_an_address_is_something_real():
    assert _points_at_something_real("read https://example.invalid/paper and summarise it")


def test_both_natural_phrasings_reach_the_code_lane(tmp_path: Path, skills):
    project = tmp_path / "ledger"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")

    for text in (
        f"there's a small python project at {project} — one of its tests fails "
        "and I can't see why. can you look at it and tell me what's actually wrong?",
        f"why is the test failing in {project}",
    ):
        picked = offered(text, skills)
        assert "diagnose_repo" in picked, (text, picked)


def test_a_request_for_prose_is_still_prose(skills):
    """The gate this loosens exists because a Dijkstra explanation once
    nominated five unrelated tools."""
    for text in (
        "explain how dijkstra works and show the distance updates",
        "how are you feeling today?",
        "what do you think about consciousness?",
    ):
        assert offered(text, skills) == [], text


def test_an_address_gives_domains_whatever_the_mood(tmp_path: Path):
    project = tmp_path / "repo"
    project.mkdir()
    assert requested_foundational_domains(f"why is the test failing in {project}")
    # No address, no imperative, no domains.
    assert requested_foundational_domains("why is everything so difficult") == ()
