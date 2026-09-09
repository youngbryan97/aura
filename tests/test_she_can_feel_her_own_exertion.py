"""The body sensed the machine's load and never her own work.

Interoception read cpu, memory and temperature from psutil. Those are facts
about the computer she runs on, and most of the load on them is not hers.
Nothing anywhere reported how hard *she* was working, so nothing she chose
could come back to her as a felt cost — and an experiment that holds the host
still to keep two arms comparable held still the only body channel she had.
"""

from __future__ import annotations

import pytest

from core.soma.effort import (
    EffortLedger,
    UNIT_COST,
    get_effort_ledger,
    note_effort,
    reset_effort_for_test,
)


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_effort_for_test()
    yield
    reset_effort_for_test()


def test_a_drain_reports_what_was_spent_and_clears_it() -> None:
    note_effort("recall", 12)
    note_effort("substrate_steps", 40)
    note_effort("recall", 3)
    ledger = get_effort_ledger()
    assert ledger.drain() == {"recall": 15.0, "substrate_steps": 40.0}
    assert ledger.drain() == {}


def test_lifetime_keeps_what_the_drain_took() -> None:
    note_effort("recall", 5)
    get_effort_ledger().drain()
    assert get_effort_ledger().lifetime()["recall"] == 5.0


def test_exertion_adds_ratios_not_counts() -> None:
    """A hundred characters and a hundred steps are not the same work."""
    one_unit_each = {kind: cost for kind, cost in UNIT_COST.items()}
    assert EffortLedger.exertion(one_unit_each) == pytest.approx(1.0)
    assert EffortLedger.exertion({}) == 0.0
    assert 0.0 < EffortLedger.exertion({"recall": UNIT_COST["recall"]}) < 1.0


def test_nonsense_is_not_effort() -> None:
    note_effort("recall", float("nan"))
    note_effort("recall", -4)
    note_effort("recall", "loads")  # type: ignore[arg-type]
    assert get_effort_ledger().peek() == {}


def test_the_subsystems_that_spend_it_report_it() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for path, kind in (
        ("core/phases/memory_retrieval.py", "recall"),
        ("core/consciousness/liquid_substrate.py", "substrate_steps"),
        ("core/world_model/learned_world_model.py", "train_steps"),
    ):
        source = (root / path).read_text()
        assert f'note_effort("{kind}"' in source, f"{path} stopped reporting {kind}"


def test_the_body_carries_it_and_the_battery_can_displace_it() -> None:
    from core.state.aura_state import AuraState
    from core.subject.state import feature_names, perturb, read_core_state

    columns = feature_names()
    assert "I.exertion" in columns
    index = columns.index("I.exertion")

    state = AuraState.default()
    before = read_core_state(state).vector()[index]
    assert perturb(state, "I", 0.15)
    after = read_core_state(state).vector()[index]
    assert after > before, "the one body channel an experiment leaves free did not move"


def test_working_hard_is_felt_as_pressure() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "phases" / "proprioceptive_loop.py"
    ).read_text()
    assert 'getattr(soma, "exertion", 0.0)' in source, "her own work no longer counts as strain"
