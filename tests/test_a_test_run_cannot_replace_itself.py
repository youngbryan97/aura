"""A relaunch replays sys.argv. In a test run that argv is pytest.

On 2026-09-05 this made 5,205 chained pytest processes on one host in fifteen
minutes: each replacement reached the same code and arranged the next. The
visible symptom was not a fork bomb — it was every suite hanging after its
first test, because psutil was walking 4,855 processes inside every health
report and the load average sat above six.

Two guards, argv first. A child a test spawns has no pytest in its own
sys.modules, so it reads as a live runtime while still carrying a pytest argv,
and the profile check alone would let it through.
"""
from __future__ import annotations

import pytest

from core.runtime.runtime_relaunch import (
    _NOT_A_RUNTIME,
    _why_this_process_must_not_replace_itself,
    schedule_relaunch,
)


def test_this_test_process_refuses_to_arrange_its_own_replacement():
    receipt = schedule_relaunch(argv=["aura_main.py", "--desktop", "--port", "8000"])
    assert receipt["scheduled"] is False
    assert receipt["reason"]
    assert "waiter_pid" not in receipt


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/local/bin/pytest", "-q"],
        ["pytest", "tests/test_one.py"],
        ["py.test", "-x"],
        ["/x/python", "-m", "pytest"],
        ["/x/bin/_jb_pytest_runner.py", "tests"],
    ],
)
def test_a_pytest_argv_is_refused_whatever_the_profile_says(argv):
    why = _why_this_process_must_not_replace_itself(argv)
    assert why.startswith("argv_is_not_a_runtime"), why


def test_a_real_runtime_argv_is_refused_here_only_because_this_is_a_test():
    """The argv is fine; the process is not."""
    why = _why_this_process_must_not_replace_itself(["aura_main.py", "--desktop"])
    assert why.startswith("not_a_live_runtime"), why


def test_the_refusal_names_a_reason_rather_than_returning_a_bare_false():
    """A relaunch that silently does not happen is a runtime that stays down."""
    receipt = schedule_relaunch(argv=["aura_main.py"])
    assert isinstance(receipt.get("reason"), str)
    assert receipt["reason"].strip()


def test_the_runner_names_are_not_a_substring_trap():
    """`pytest-watcher` is a runner; a script called `pytestimony` is not.

    The check is on the basename, anchored at the start, so a real runtime
    whose script merely contains the letters cannot be refused by accident.
    """
    why = _why_this_process_must_not_replace_itself(["/x/my_pytest_helper.py"])
    assert not why.startswith("argv_is_not_a_runtime"), why
    assert "pytest" in _NOT_A_RUNTIME
