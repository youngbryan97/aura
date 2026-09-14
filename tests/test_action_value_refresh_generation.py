"""Evidence arriving during a read must remain pending for the next read."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from core.reasoning.action_value import ActionValueModel


@pytest.mark.parametrize("fail", [False, True])
def test_new_evidence_during_refresh_is_not_acknowledged(fail):
    model = ActionValueModel()

    class Ledger:
        def measured_action_stats(self, *, by_state=False):
            if not by_state:
                model.mark_stale()
                if fail:
                    raise ValueError("snapshot unavailable")
            return {}

    model.refresh(Ledger())
    assert model.is_stale()

    class CurrentLedger:
        def measured_action_stats(self, *, by_state=False):
            return {}

    model.refresh(CurrentLedger())
    assert not model.is_stale()


def test_overlapping_refreshes_cannot_publish_in_reverse_order():
    model = ActionValueModel()
    reading_old = Event()
    release_old = Event()
    started_new = Event()
    reading_new = Event()

    class Ledger:
        def __init__(self, old):
            self.old = old

        def measured_action_stats(self, *, by_state=False):
            if by_state:
                return {}
            if self.old:
                reading_old.set()
                assert release_old.wait(5)
            else:
                reading_new.set()
            return {"action": {"n": 2, "mean": 0.0 if self.old else 1.0, "m2": 0}}

    def refresh_new():
        started_new.set()
        return model.refresh(Ledger(False))

    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(model.refresh, Ledger(True))
        try:
            assert reading_old.wait(5)
            model.mark_stale()
            new = pool.submit(refresh_new)
            assert started_new.wait(5)
            assert not reading_new.wait(0.1)
        finally:
            release_old.set()
        assert old.result(timeout=5) == 1
        assert new.result(timeout=5) == 1
    assert model.value_for("action").value == 1.0
    assert not model.is_stale()


def test_real_resolution_during_snapshot_reaches_the_next_ranking(tmp_path, monkeypatch):
    import core.reasoning.action_value as values
    from core.cognition.outcome_ledger import OutcomeLedger

    model = ActionValueModel()
    ledger = OutcomeLedger(db_path=str(tmp_path / "outcomes.db"))
    monkeypatch.setattr(values, "get_action_value_model", lambda: model)
    ledger.add_resolution_observer(values.on_outcome_resolved)
    first = ledger.open("action", 0.5)
    ledger.resolve(first, 1.0)
    second = ledger.open("action", 0.5)
    read = ledger.measured_action_stats
    resolved = False

    def snapshot(*, by_state=False):
        nonlocal resolved
        result = read(by_state=by_state)
        if not by_state and not resolved:
            resolved = True
            ledger.resolve(second, 0.0)
        return result

    monkeypatch.setattr(ledger, "measured_action_stats", snapshot)
    model.refresh(ledger)
    assert model.is_stale()
    assert model.value_for("action").observations == 1
    model.refresh(ledger)
    assert not model.is_stale()
    value = model.value_for("action")
    assert value.observations == 2
    assert value.value == 0.5
