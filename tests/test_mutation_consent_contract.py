"""Denied(m) must mean the modification does not happen.

The growth ladder worked out the right answer and then lost it. Its inner
``propose_modification`` returns a bool; the ``submit_proposal`` wrapper every
caller uses threw that bool away and returned the proposal object, and every
object is truthy. So the natural ``if not granted:`` read as "always granted",
and Hephaestus forged skills and patched core over the top of a refusal.

These tests hold the contract at the boundary rather than at the call sites,
because the call sites are what got it wrong.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from core.self_modification.growth_ladder import (
    GrowthLadder,
    ModificationLevel,
    ModificationProposal,
)


@pytest.fixture
def ladder(tmp_path: pathlib.Path) -> GrowthLadder:
    return GrowthLadder(state_path=tmp_path / "growth_ladder.json")


def _proposal(**kw) -> ModificationProposal:
    base = dict(
        id="p",
        timestamp=0.0,
        level=ModificationLevel.OBSERVATION,
        domain="d",
        description="",
        justification="",
        diff_patch=None,
        proposed_by="aura",
    )
    base.update(kw)
    return ModificationProposal(**base)


def test_an_unadjudicated_proposal_is_not_consent():
    """Fail closed: nobody has said yes yet, so the answer is no."""
    assert _proposal().granted is False
    assert not _proposal()


def test_a_refused_proposal_is_falsy():
    """``if not granted:`` has to fire. This is the whole defect."""
    assert not _proposal(decision=False, status="rejected_user")


def test_a_granted_proposal_is_truthy():
    assert _proposal(decision=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level",
    [ModificationLevel.SKILL_CREATION, ModificationLevel.CORE_PATCH],
)
async def test_hephaestus_levels_are_refused_at_rung_zero(ladder, level):
    """The two levels Hephaestus asks for, at the rung Aura starts on.

    Before the fix, SKILL_CREATION came back truthy and CORE_PATCH did not
    exist at all, so the call raised AttributeError before reaching a single
    safety check.
    """
    proposal = await ladder.submit_proposal(level, "skill", "add one", "", None, "aura")
    assert proposal.granted is False
    assert not proposal
    assert proposal.status.startswith("rejected")


@pytest.mark.asyncio
async def test_a_refusal_before_the_record_does_not_stamp_someone_elses_proposal(
    ladder, monkeypatch
):
    """The blocked-policy path refuses before a proposal is appended.

    Reading ``_proposals[-1]`` there returns the previous caller's proposal —
    or raises IndexError when the ladder is fresh.
    """
    monkeypatch.setattr(
        "core.self_modification.growth_ladder.get_runtime_setting",
        lambda key, default=None: "blocked"
        if key == "autonomy.self_modification"
        else default,
    )
    first = await ladder.submit_proposal(
        ModificationLevel.OBSERVATION, "d", "one", "", None, "aura"
    )
    assert not first
    assert first.status == "refused_before_record"

    second = await ladder.submit_proposal(
        ModificationLevel.OBSERVATION, "d", "two", "", None, "aura"
    )
    assert second.id != first.id
    assert first.description == "one"


def test_no_level_silently_aliases_another():
    """``SKILL_CREATION = 3.5`` truncated to 3 and became BEHAVIOR."""
    names = [m.name for m in ModificationLevel]
    assert len(names) == len(set(int(m) for m in ModificationLevel)), names
    assert ModificationLevel.SKILL_CREATION != ModificationLevel.BEHAVIOR


def test_core_patch_exists():
    """Hephaestus named it; the enum did not define it."""
    assert ModificationLevel.CORE_PATCH.governing_rung is ModificationLevel.ARCHITECTURE


def test_the_two_doors_agree_on_a_level():
    """``from_string`` disagreed with the members it converts to.

    ``from_string("skill_creation")`` gave KNOWLEDGE while the member gave
    BEHAVIOR, so one modification was governed at two different tiers.
    """
    for name in ("observation", "expression", "knowledge", "behavior",
                 "architecture", "skill_creation", "core_patch"):
        assert ModificationLevel.from_string(name).name == name.upper()


def test_a_kind_is_governed_by_a_rung_not_its_own_ordinal():
    """Kinds sit above every rung, so comparing ordinals refuses forever."""
    for level in ModificationLevel:
        assert level.governing_rung.is_rung, level
        assert level.governing_rung <= ModificationLevel.ARCHITECTURE


@pytest.mark.asyncio
async def test_advancement_never_climbs_into_a_kind(ladder):
    """The ladder walks rungs. A kind is not somewhere Aura can arrive."""
    ladder._current_level = ModificationLevel.ARCHITECTURE
    assert await ladder.evaluate_advancement() is None
