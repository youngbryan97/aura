"""Whether a drive named for wanting to be good at things knows she is behind.

run_031 measured N's reach — perturb her developmental state and see how much
of the core moves — at 0.222, the worst of the ten domains, and
`perturbational_spread` failed at 0.5333 against a bar of 0.6. The reservoir's
reading of how far she has been moving is held by `core/self/growth.py` and
the only thing that read it was the gate on her self-modification path.
"""

from __future__ import annotations

import core.self.growth as growth_module
from core.drive_engine import DriveEngine
from core.self.growth import ORDINARY_STEP, get_growth_ledger


def _grow_at(level: float, steps: int = 6) -> None:
    ledger = get_growth_ledger()
    for _ in range(steps):
        ledger.note(level)


def _settle(drives: DriveEngine) -> None:
    """Enough readings that a middle of them exists."""
    for _ in range(3):
        drives._shortfall_shift()


def test_nothing_moves_before_her_development_has_a_reading():
    growth_module.reset_for_test()
    drives = DriveEngine()
    before = drives.get_drive_vector()["competence"]
    assert drives._shortfall_shift() == 0.0
    assert drives.get_drive_vector()["competence"] == before


def test_moving_at_an_ordinary_step_leaves_the_drive_where_it_was():
    growth_module.reset_for_test()
    _grow_at(ORDINARY_STEP)
    drives = DriveEngine()
    _settle(drives)
    assert drives.get_drive_vector()["competence"] == DriveEngine().get_drive_vector()[
        "competence"
    ]


def test_developing_below_ordinary_takes_the_competence_drive_down():
    growth_module.reset_for_test()
    _grow_at(ORDINARY_STEP)
    drives = DriveEngine()
    _settle(drives)
    keeping_up = drives.get_drive_vector()["competence"]

    _grow_at(0.05, steps=8)
    behind = drives.get_drive_vector()["competence"]
    assert behind < keeping_up, (keeping_up, behind)


def test_moving_more_than_ordinarily_is_not_a_shortfall():
    growth_module.reset_for_test()
    _grow_at(0.95)
    drives = DriveEngine()
    _settle(drives)
    assert drives._shortfall_shift() == 0.0


def test_two_readings_are_not_a_middle():
    growth_module.reset_for_test()
    _grow_at(0.05)
    drives = DriveEngine()
    assert drives._shortfall_shift() == 0.0
    assert drives._shortfall_shift() == 0.0
