"""The comparator emitted and compared for one caller and told nobody.

`AgencyComparator` implements the efference-copy model: predict the outcome
before acting, compare it to what happened, and split the difference into
self-caused and world-caused. Its two write methods were wired to the executive
authority path. Every one of its read methods — the agency score, the
attribution label, the traces — was called from nowhere, so the sense of agency
it computes reached no part of her.

And an intention formed anywhere but that one path emitted nothing, so it was
compared against nothing and produced no attribution at all.
"""

from __future__ import annotations

import pytest

from core.agency.authorship import get_agency_ledger, reset_agency_ledger_for_test
from core.consciousness.agency_comparator import get_agency_comparator


@pytest.fixture(autouse=True)
def _fresh():
    reset_agency_ledger_for_test()
    yield
    reset_agency_ledger_for_test()


def _loop(tmp_path):
    from core.agency.intention_loop import IntentionLoop

    return IntentionLoop(db_path=str(tmp_path / "intentions.db"))


def test_forming_an_intention_emits_a_prediction(tmp_path):
    comparator = get_agency_comparator()
    before = comparator.get_status()["total_emissions"]
    loop = _loop(tmp_path)
    loop.intend(intention="write the notes file", drive="curiosity", expected_outcome="notes.txt exists")
    assert comparator.get_status()["total_emissions"] > before
    loop.close()


def test_observing_the_outcome_compares_it_and_the_ledger_records_the_attribution(tmp_path):
    comparator = get_agency_comparator()
    before = comparator.get_status()["total_comparisons"]
    loop = _loop(tmp_path)
    identifier = loop.intend(
        intention="write the notes file", drive="curiosity", expected_outcome="notes.txt exists"
    )
    loop.record_action(
        identifier, tool_name="write_file", args={}, result="ok", success=True, duration_ms=1.0
    )
    loop.observe(identifier, observation="notes.txt exists", actual_outcome="notes.txt exists")

    assert comparator.get_status()["total_comparisons"] > before
    ledger = get_agency_ledger()
    assert ledger.acted == 1
    assert ledger.last_event is not None
    assert "attribution" in ledger.last_event.detail
    loop.close()


def test_the_agency_score_is_readable_as_self_state():
    """It is in the subject-core schema now, which is what "read" means here."""
    from core.subject.state import feature_names

    names = feature_names("S")
    assert "S.agency_score" in names
    assert "S.agency_attribution" in names


def test_a_comparator_that_is_not_there_does_not_break_an_intention(tmp_path, monkeypatch):
    """Forming an intention must not depend on the thing that watches it."""
    import core.consciousness.agency_comparator as module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no comparator here")

    monkeypatch.setattr(module, "get_agency_comparator", _boom)
    loop = _loop(tmp_path)
    identifier = loop.intend(intention="x", drive="y", expected_outcome="z")
    assert identifier
    loop.record_action(
        identifier, tool_name="t", args={}, result="ok", success=True, duration_ms=1.0
    )
    loop.observe(identifier, observation="z", actual_outcome="z")
    assert get_agency_ledger().acted == 1
    loop.close()
