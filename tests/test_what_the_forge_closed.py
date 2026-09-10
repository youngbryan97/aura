"""Where the forge's chain stops, named by the step that is empty."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agi.what_the_forge_closed import (
    THE_CHAIN,
    AForgeReading,
    how_the_forge_closes,
    where_the_forge_stops,
)


def _a_forge(*, gaps: dict[str, int], produced: dict) -> SimpleNamespace:
    return SimpleNamespace(
        _gap_counts=dict(gaps),
        what_the_forge_has_produced=lambda: produced,
    )


def _produced(**counts) -> dict:
    said = {
        "forged": 0,
        "verified": 0,
        "installed": 0,
        "taken_at_least_once": 0,
        "skills": [],
    }
    said.update(counts)
    return said


def test_a_forge_that_has_seen_nothing_stops_at_the_first_arrow():
    reading = how_the_forge_closes(_a_forge(gaps={}, produced=_produced()))
    assert reading.stops_at == "gaps seen"
    assert reading.reached == 0
    assert "failed often enough" in reading.why_it_matters


def test_forged_installed_and_never_taken_is_named_as_that():
    """The reading a forge that runs and pays nothing used to not have."""
    from core.agi.skill_synthesizer import GAP_FORGE_THRESHOLD

    reading = how_the_forge_closes(
        _a_forge(
            gaps={"a gap": GAP_FORGE_THRESHOLD},
            produced=_produced(forged=3, verified=3, installed=3),
        )
    )
    assert reading.stops_at == "skills taken"
    assert "running and paying nothing" in reading.why_it_matters
    assert reading.reached == THE_CHAIN.index("skills taken")


def test_a_chain_that_turns_over_says_so():
    from core.agi.skill_synthesizer import GAP_FORGE_THRESHOLD

    reading = how_the_forge_closes(
        _a_forge(
            gaps={"a gap": GAP_FORGE_THRESHOLD},
            produced=_produced(
                forged=1,
                verified=1,
                installed=1,
                taken_at_least_once=1,
                skills=[{"the_gap_after": {"stopped": True}}],
            ),
        )
    )
    assert reading.turns_over
    assert reading.stops_at == ""
    assert reading.reached == len(THE_CHAIN)
    assert "stopped being logged" in reading.why_it_matters


def test_only_the_first_empty_step_is_reported():
    """A chain with one broken link must not read as five."""
    reading = how_the_forge_closes(
        _a_forge(gaps={"a gap": 1}, produced=_produced(forged=0))
    )
    assert reading.stops_at == "gaps worth answering"
    # Everything after it is empty too and none of it is the finding.
    assert reading.counts["candidates drafted"] == 0
    assert reading.reached == 1


def test_a_forge_that_cannot_be_read_is_not_reported_as_a_working_one():
    class Angry:
        _gap_counts: dict = {}

        def what_the_forge_has_produced(self):
            raise RuntimeError("no")

    reading = how_the_forge_closes(Angry())
    assert not reading.turns_over
    assert reading.stops_at == THE_CHAIN[0]


def test_the_reading_carries_whether_the_target_held():
    """Handed in, not reached for: core/agi may not import core/skill_management.

    A chain that turns over on a target the drafter kept moving is not the
    same reading as one that turns over on a target that held, so the block
    travels with the chain — put there by the caller that can see both.
    """
    reading = how_the_forge_closes(
        _a_forge(gaps={}, produced=_produced()),
        targets_held={"redrafts": 2, "moved_the_target": 1},
    )
    said = reading.as_dict()
    assert said["the_target_held"]["moved_the_target"] == 1

    empty = how_the_forge_closes(_a_forge(gaps={}, produced=_produced()))
    assert empty.as_dict()["the_target_held"] == {}


def test_the_inspector_puts_the_two_blocks_side_by_side():
    """The composition lives where both packages can be seen."""
    import inspect as inspect_module

    from tools.inspect_runtime import THE_SECTIONS

    source = inspect_module.getsource(THE_SECTIONS["forge_chain"])
    assert "how_the_target_has_held" in source
    assert "targets_held=" in source


def test_the_chain_is_in_the_order_the_arrows_have_to_land():
    assert THE_CHAIN[0] == "gaps seen"
    assert THE_CHAIN[-1] == "gaps that stopped"
    assert THE_CHAIN.index("candidates verified") < THE_CHAIN.index("skills installed")
    assert THE_CHAIN.index("skills installed") < THE_CHAIN.index("skills taken")


def test_the_live_forge_can_be_read_without_arguments():
    assert where_the_forge_stops() in {*THE_CHAIN, ""}


def test_a_reading_with_an_unknown_stop_reports_no_arrows():
    made_up = AForgeReading(
        counts={}, stops_at="something else", turns_over=False, targets_held={}
    )
    assert made_up.reached == 0
    assert "something else" in made_up.why_it_matters


@pytest.mark.parametrize("step", THE_CHAIN)
def test_every_step_is_counted(step: str):
    reading = how_the_forge_closes(_a_forge(gaps={}, produced=_produced()))
    assert step in reading.counts
