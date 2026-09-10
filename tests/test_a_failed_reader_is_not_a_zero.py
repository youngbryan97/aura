"""An absent organ, a raising reader and a genuine zero were the same number.

For a runtime, defaulting a missing read is the right thing: the turn goes on.
For a measurement it is not. Every column whose source could not be read
arrived in the recording as 0.0, so a criterion could fail on a subsystem that
was never asked, and nothing in the report said which.

The reading now carries what it could not read, the recording counts it over
the run, and a criterion resting on an organ that was absent for most of the
run is marked invalid rather than failed — it says the measurement did not
happen, which is a different claim from the organism not doing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state.aura_state import AuraState
from core.subject.battery import MISSING_SHARE, assemble
from core.subject.state import Organs, read_core_state


def test_a_reading_with_no_organs_says_which_organs_it_had_none_of() -> None:
    reading = read_core_state(AuraState.default(), organs=Organs())
    absent = {source for source, why in reading.misses.items() if why == "organ absent"}
    assert absent, "no organ was present and the reading claimed nothing was missing"
    assert any(source.startswith("organ:workspace") for source in absent)


def test_a_reader_that_raises_is_recorded_as_a_miss_not_as_a_reading() -> None:
    class Angry:
        def get_status(self) -> dict:
            raise RuntimeError("no")

    reading = read_core_state(AuraState.default(), organs=Organs(workspace=Angry()))
    assert reading.misses.get("organ:workspace.get_status") == "reader raised RuntimeError"


def test_a_reading_that_worked_reports_no_miss_for_that_source() -> None:
    class Quiet:
        def get_status(self) -> dict:
            return {"tick": 3, "ignition_level": 0.4, "last_winner": "affect_joy"}

    reading = read_core_state(AuraState.default(), organs=Organs(workspace=Quiet()))
    assert "organ:workspace.get_status" not in reading.misses


def test_a_criterion_resting_on_an_absent_organ_cannot_pass() -> None:
    """The workspace was never read, so global access is not a finding about her."""
    evidence = {
        "global_access": {"consumers": 5, "kinds": ["a", "b", "c"]},
        "recording": {
            "misses": {
                "organ:workspace.get_status": {
                    "frames": 100,
                    "share": MISSING_SHARE + 0.5,
                    "reasons": {"organ absent": 100},
                }
            }
        },
    }
    verdict = assemble(evidence)
    access = next(item for item in verdict.criteria if item.key == "global_access")
    assert not access.passed
    assert "workspace" in access.detail.get("invalid", "")
    assert verdict.notes.get("organs_mostly_unread") == ["workspace"]


def test_an_organ_missing_for_a_moment_does_not_throw_the_criterion_away() -> None:
    evidence = {
        "global_access": {"consumers": 5, "kinds": ["a", "b", "c"]},
        "recording": {
            "misses": {
                "organ:workspace.get_status": {
                    "frames": 2,
                    "share": MISSING_SHARE / 4,
                    "reasons": {"organ absent": 2},
                }
            }
        },
    }
    verdict = assemble(evidence)
    access = next(item for item in verdict.criteria if item.key == "global_access")
    assert "invalid" not in (access.detail or {})
