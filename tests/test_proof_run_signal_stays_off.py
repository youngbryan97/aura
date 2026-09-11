"""No test module may turn a proof-run signal on for the whole process.

``proof_run_active()`` is true for any of AURA_PROOF_RUN, AURA_AGI_MAX_TASKS or
AURA_TESTING, and a long list of subsystems reads it to defer, refuse, or take
a cheaper branch. ``tests/test_declared_channels_reach_the_cadence.py`` set
AURA_TESTING at module scope, which runs while the module is imported, which is
collection — before the first test and after the last chance anything had to
undo it. Every test in the selection after it was therefore a proof run.

Two narrative-thread tests in ``tests/test_core_affect_models.py`` were the ones
that showed it: ``NarrativeThread.start`` returns early under a proof run and
the refresh loop returns before it writes a snapshot, so one reported
``KeyError: 'evidence'`` from a file that has nothing to do with telemetry.
Green alone, red in company, cause and victim three hundred tests apart.

``_global_state_contamination_guard`` already restores every AURA_* variable
between tests, so the per-test half of this was covered; collection was not.
The guard added for it lives in ``tests/conftest.py``: collection records the
module and puts the variable back, and the first test below is where the run
goes red. The rule is that no module turns one of these on, not that the run is
never a proof run — an operator setting one deliberately is the baseline these
tests measure against.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import (
    _changed_proof_signals,
    _proof_signal_snapshot,
    _restore_proof_signals,
    proof_signal_leaks,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEAK_EXAMPLE = Path(__file__).parent / "order_dependence_leak_example.py"


def test_no_module_turned_a_proof_run_signal_on_while_it_was_imported() -> None:
    """Collection finishes before the first test runs, so this sees them all."""
    leaks = proof_signal_leaks()
    assert not leaks, (
        "these modules turned a proof-run signal on for the process while they "
        "were imported, which makes every later test in the selection a proof "
        f"run: {'; '.join(leaks)}"
    )


def test_the_detector_watches_every_variable_proof_policy_publishes() -> None:
    """The list is asked for, never repeated, so a fourth one is covered."""
    from core.runtime.proof_policy import proof_active_env_names

    assert set(_proof_signal_snapshot()) == set(proof_active_env_names())


def test_the_detector_sees_a_signal_that_was_turned_on(monkeypatch) -> None:
    from core.runtime.proof_policy import proof_active_env_names

    name = proof_active_env_names()[0]
    baseline = _proof_signal_snapshot()
    assert _changed_proof_signals(baseline) == ()

    # Not the literal "1": an operator is allowed to run the suite under one of
    # these, and a value equal to the baseline is not a change.
    monkeypatch.setenv(name, f"{baseline[name] or ''}-changed")
    assert _changed_proof_signals(baseline) == (name,)

    _restore_proof_signals(baseline)
    assert _changed_proof_signals(baseline) == ()


def test_the_guard_fires_on_a_module_that_turns_one_on() -> None:
    """The worked example, in its own process because it really does leak.

    One run proves three things: the reporting test goes red and names the
    module, the signal is put back so the rest of the selection is not a proof
    run, and a signal set inside a test body is gone by the next test.

    The child starts with every proof-run variable cleared. An operator may run
    this suite under one of them on purpose, and the example's write would then
    change nothing for the guard to find.
    """
    from core.runtime.proof_policy import proof_active_env_names

    child_env = {
        key: value
        for key, value in os.environ.items()
        if key not in proof_active_env_names()
    }
    reporter = f"tests/{Path(__file__).name}"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"tests/{_LEAK_EXAMPLE.name}",
            f"{reporter}::test_no_module_turned_a_proof_run_signal_on_while_it_was_imported",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=child_env,
        timeout=300,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, f"the guard did not fire:\n{output}"
    assert "order_dependence_leak_example.py set AURA_PROOF_RUN" in output, output
    assert "1 failed, 2 passed" in output, output
