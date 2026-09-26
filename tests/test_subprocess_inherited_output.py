"""Exited direct children cannot strand either work-bound gateway on a pipe."""

import os
import subprocess
import sys
import time

import pytest

from core.runtime import subprocess_gateway as gateway

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="inherited-pipe probe requires fork")


def command():
    return [sys.executable, "-c", "import os,time; print('kept', flush=True); "
            "pid=os.fork(); time.sleep(2) if pid == 0 else None"]


@pytest.mark.parametrize("observable", [False, True])
def test_exited_parent_is_reaped_without_losing_partial_text_or_waiting_for_descendant(monkeypatch, observable):
    monkeypatch.setattr(gateway, "_child_cpu_seconds", lambda _: 0. if observable else None)
    started = time.monotonic()
    with pytest.raises(gateway.WorkBoundExpired, match="inherited output") as caught:
        gateway._run_bounded_by_its_work(command(), cwd=None, env=None, budget_s=.5,
            capture_output=True, input=None, stdin=None, stdout=None, stderr=None, text=True, check=False)
    assert caught.value.output == "kept\n"
    assert time.monotonic() - started < 1.5


def test_explicit_work_runner_reports_incomplete_transport_instead_of_success(monkeypatch):
    monkeypatch.setattr(gateway, "_child_cpu_seconds", lambda _: None)
    owner = gateway.SubprocessGateway()
    monkeypatch.setattr(owner, "spawn", lambda argv, **kwargs: subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    result = owner.run_until_its_work_is_done(command(), cpu_budget_s=.5, watch_period_s=.05)
    assert result.returncode == 124
    assert result.stdout == "kept\n"
    assert "inherited output pipes" in result.stderr
