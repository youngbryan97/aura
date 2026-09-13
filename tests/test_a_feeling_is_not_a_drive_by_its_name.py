"""A feeling credits the need it bears on, and never a drive named after it.

Attention used to credit `affect_joy` to a drive called joy by stripping the
prefix. No budget has ever been called joy, so thirty emotions credited
nothing, and the name of a string was standing in for a claim about
psychology. The mapping is now written out, one feeling to one of the five
budgets, and these tests hold it to that.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.consciousness.broadcast_consumers import AFFECT_DRIVES, _drive_served
from core.state.aura_state import AuraState

ROOT = Path(__file__).resolve().parents[1]


def test_every_feeling_names_a_budget_that_exists() -> None:
    budgets = set(AuraState.default().motivation.budgets)
    assert set(AFFECT_DRIVES.values()) <= budgets, set(AFFECT_DRIVES.values()) - budgets


def test_a_feeling_credits_its_need_and_not_its_own_name() -> None:
    budgets = set(AuraState.default().motivation.budgets)
    for feeling, need in AFFECT_DRIVES.items():
        assert _drive_served(f"affect_{feeling}") == need
    named_after_themselves = sorted(f for f in AFFECT_DRIVES if f in budgets and AFFECT_DRIVES[f] != f)
    assert not named_after_themselves, named_after_themselves


def test_a_feeling_the_table_does_not_name_credits_nothing() -> None:
    assert _drive_served("affect_an_unnamed_feeling") == ""


def test_no_code_turns_an_affect_name_into_a_drive_by_cutting_its_prefix() -> None:
    """Anywhere the prefix is cut, the remainder has to go through the table."""
    cut = re.compile(r"""\[len\(["']affect_["']\)\s*:\]|removeprefix\(["']affect_["']\)""")
    offenders = []
    for path in sorted((ROOT / "core").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if cut.search(line) and "AFFECT_DRIVES" not in line:
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, offenders
