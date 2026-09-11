"""No module may change a runtime switch for the process while it is imported.

``_global_state_contamination_guard`` restores every AURA_* variable between
tests, so a variable set inside a test body is covered. A variable set at module
scope is not: that runs while the module is imported, which is collection,
before the first test and after the last point anything had to undo it.

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

``tests/test_seal_infrastructure.py`` had the same line for AURA_TEST_MODE,
which gates the subprocess gateway, the secrets store and the experience loop.
No victim was attributed to it, which is the argument for a gate rather than
two deletions.

The guard lives in ``tests/conftest.py``: collection records the module and puts
the variable back, and the first test below is where the run goes red. The rule
is that no module changes one, not that the run never has one set — an operator
setting one deliberately is the baseline these tests measure against. Fourteen
names are let through because a production module stamps them the moment it is
imported, which is a question about the runtime rather than about a test; that
list only shrinks.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import (
    _aura_env_snapshot,
    _changed_aura_env,
    _restore_aura_env,
    import_time_env_leaks,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LEAK_EXAMPLE = Path(__file__).parent / "order_dependence_leak_example.py"


def test_no_module_changed_a_runtime_switch_while_it_was_imported() -> None:
    """Collection finishes before the first test runs, so this sees them all."""
    leaks = import_time_env_leaks()
    assert not leaks, (
        "these modules changed an AURA_* switch for the whole process while they "
        "were imported, so every test collected with them ran against a "
        f"configuration none of them asked for: {'; '.join(leaks)}"
    )


def test_the_detector_covers_every_variable_that_makes_a_proof_run() -> None:
    """The three names are asked for, never repeated, so a fourth is covered."""
    from core.runtime.proof_policy import proof_active_env_names

    from tests.conftest import _IMPORT_TIME_ENV_STAMPED_BY_RUNTIME

    for name in proof_active_env_names():
        assert name.startswith("AURA_"), name
        assert name not in _IMPORT_TIME_ENV_STAMPED_BY_RUNTIME, (
            f"{name} decides whether the run is a proof run; it cannot also be "
            "on the list of switches the guard lets through"
        )


def test_the_detector_sees_a_switch_that_was_changed(monkeypatch) -> None:
    from core.runtime.proof_policy import proof_active_env_names

    name = proof_active_env_names()[0]
    baseline = _aura_env_snapshot()
    assert _changed_aura_env(baseline) == ()

    # Not the literal "1": an operator is allowed to run the suite under one of
    # these, and a value equal to the baseline is not a change.
    monkeypatch.setenv(name, f"{baseline.get(name, '')}-changed")
    assert _changed_aura_env(baseline) == (name,)

    _restore_aura_env(baseline, (name,))
    assert _changed_aura_env(baseline) == ()


def test_the_detector_lets_through_what_the_runtime_stamps(monkeypatch) -> None:
    """The allowlist is live, so the ratchet cannot be a comment."""
    from tests.conftest import _IMPORT_TIME_ENV_STAMPED_BY_RUNTIME

    name = sorted(_IMPORT_TIME_ENV_STAMPED_BY_RUNTIME)[0]
    baseline = _aura_env_snapshot()

    monkeypatch.setenv(name, f"{baseline.get(name, '')}-changed")
    assert _changed_aura_env(baseline) == ()


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
            f"{reporter}::test_no_module_changed_a_runtime_switch_while_it_was_imported",
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
