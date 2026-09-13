"""The worked example for the proof-run signal guard in tests/conftest.py.

Not named ``test_*`` on purpose: a normal run must not collect it, because it
deliberately does the thing the guard exists to catch. pytest collects a file
named directly on the command line whatever it is called, which is how
``tests/test_proof_run_signal_stays_off.py`` runs it in a subprocess and checks
that the guard fires. A guard with no example that makes it fire reports green
forever.

Two leaks, because there are two places one can start. The module-level write
happens while this file is imported, which is collection, before any test runs
— the new guard covers that. The write inside a test body is covered by the
older ``_global_state_contamination_guard``, which restores every AURA_*
variable between tests; the example pins that behaviour to this property so a
later refactor of the guard cannot quietly narrow it.
"""

from __future__ import annotations

import os

os.environ["AURA_PROOF_RUN"] = "1"


def test_a_body_that_leaves_a_signal_behind_it() -> None:
    os.environ["AURA_AGI_MAX_TASKS"] = "4"
    assert os.environ["AURA_AGI_MAX_TASKS"] == "4"


def test_the_next_test_did_not_inherit_either_of_them() -> None:
    assert os.environ.get("AURA_AGI_MAX_TASKS") is None
    assert os.environ.get("AURA_PROOF_RUN") is None
