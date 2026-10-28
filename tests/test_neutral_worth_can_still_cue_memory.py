"""As-expected turns are measured, and can retrieve memories of such turns."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.affect import what_it_was_worth as worth
from core.memory import worth_at_encoding as stamps


@pytest.fixture
def ledger(monkeypatch):
    ledger = worth.WorthLedger()
    monkeypatch.setattr(worth, "get_worth_ledger", lambda: ledger)
    return ledger


def neutral(ledger):
    for _ in range(4):
        ledger.note({name: 1.0 for name in worth.CHANNELS})
    assert ledger.read().measured and not any(ledger.read().error.values())


def test_neutral_worth_is_recorded_and_a_missing_measurement_is_still_absent(ledger):
    assert stamps.worth_now() is None
    assert stamps.WORTH_KEY not in stamps.stamp({})
    neutral(ledger)
    assert np.array_equal(stamps.worth_now(), np.zeros(len(worth.CHANNELS)))
    written = stamps.stamp({})
    assert np.array_equal(stamps.decode(written[stamps.WORTH_KEY]), stamps.worth_now())


def test_neutral_worth_can_change_actual_recall_selection(ledger, monkeypatch):
    from core.consciousness import substrate_gates
    from core.phases.memory_retrieval import settle_on_what_it_was_worth

    neutral(ledger)
    monkeypatch.setattr(substrate_gates, "get_substrate_gates",
                        lambda: SimpleNamespace(multiplier=lambda key: 1.0))
    zero = stamps.stamp({})[stamps.WORTH_KEY]
    positive = stamps.encode(np.full(len(worth.CHANNELS), 2.))
    state = SimpleNamespace(cognition=SimpleNamespace(
        long_term_memory=["[memory score=0.400] unexpected gain"], memory_scores=[.4],
        recall_pool={"salience": 1., "candidates": [
            {"text": "[memory score=0.400] unexpected gain", "score": .4, "worth": positive},
            {"text": "[memory score=0.400] ordinary afternoon", "score": .4, "worth": zero},
        ]},
    ))
    assert settle_on_what_it_was_worth(state)
    assert state.cognition.long_term_memory[0].endswith("ordinary afternoon")
    assert state.cognition.memory_scores == [1.0]
    assert state.cognition.recall_pool == {}


def test_an_invalid_numeric_reading_remains_absent(ledger):
    ledger._last = worth.Worth(measured=True, error={worth.CHANNELS[0]: np.nan})
    assert stamps.worth_now() is None
    ledger._last = worth.Worth(measured=True)
    assert stamps.worth_now() is None


def test_the_registered_neutral_worth_canary_uses_a_private_ledger(ledger):
    assert stamps.neutral_worth_canary()
    assert ledger.turns == 0
    from core.verify.runtime_invariants import _neutral_worth
    assert list(_neutral_worth()) == []
    from core.organism.model_validation import ValidationSuite, _install_repair_executor_claims
    suite = ValidationSuite()
    _install_repair_executor_claims(suite)
    assert any(claim.test == "neutral_worth_remains_measured" for claim in suite.claims())
