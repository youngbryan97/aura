"""What she learns from outcomes is kept in process-wide ledgers, and a test must not inherit another's.

One test that ran the organism taught every later test a habit: a choice test
read `read_room` where the drives say `make_room`, and a workspace test saw
memory win a competition perception should. Tests run in random order here
(pytest-randomly), so which test inherited what changed from run to run. The
between-test reset (`_reset_test_scoped_runtime_services` in tests/conftest.py)
now clears the four ledgers; this teaches them and runs that reset.

The decode-rate store crosses a wider gap: every test process shares one state
root, so readings one run recorded sized the next run's deadlines. A gate test
asked for 196 seconds of Cortex and was given 443.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_the_between_test_reset_forgets_what_she_learned():
    from core.affect.what_winning_earned import get_credit_ledger
    from core.agency.consequence import get_consequence_model
    from tests.conftest import _reset_test_scoped_runtime_services

    world = np.ones(4)
    for _ in range(5):
        get_consequence_model().learn("act:read_room", world, good=True)
    get_credit_ledger().note({"memory": 3}, 1.0)
    assert get_consequence_model().chance("act:read_room", world)[1] > 0.8
    assert get_credit_ledger().earned("memory") != 0.0

    _reset_test_scoped_runtime_services()

    # Never seen: one success and one failure as the prior, so even odds.
    assert get_consequence_model().chance("act:read_room", world) == (0.5, 0.5)
    assert get_credit_ledger().earned("memory") == 0.0


def test_every_learned_ledger_is_on_the_reset_list():
    from tests.conftest import _TEST_SCOPED_RESET_FUNCTIONS

    listed = set(_TEST_SCOPED_RESET_FUNCTIONS)
    for module in ("core.agency.consequence", "core.affect.what_winning_earned",
                   "core.affect.what_it_was_worth", "core.soma.good_news",
                   "core.brain.llm.thinking_reserve"):
        assert (module, "reset_for_test") in listed, module


def test_decode_rates_another_run_left_on_disk_size_no_clock(tmp_path, monkeypatch):
    from core.brain.llm import thinking_reserve
    from tests.conftest import _reset_test_scoped_runtime_services

    store = tmp_path / "decode_measurements.json"
    monkeypatch.setattr(thinking_reserve, "_store_path", lambda: store)
    thinking_reserve.forget()
    for _ in range(12):
        thinking_reserve.record_decode_rate(generated_tokens=512, elapsed_s=51.2)
    assert thinking_reserve.save()
    assert thinking_reserve.seconds_to_decode(512) > 0.0

    _reset_test_scoped_runtime_services()

    assert store.exists()
    assert thinking_reserve.seconds_to_decode(512) == 0.0
    # Asking for the reserve consults the store for new proofs; the old file is not one.
    thinking_reserve.reserve_tokens()
    assert thinking_reserve.seconds_to_decode(512) == 0.0
    # A proof written after the reset still arrives.
    stored = json.loads(store.read_text())
    stored["proved_insufficient_by_model"] = {"": 2048}
    store.write_text(json.dumps(stored))
    assert thinking_reserve.reserve_tokens() == 2048
