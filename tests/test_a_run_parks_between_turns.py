"""A run held for her app parks between turns, and carries on when the hold is lifted.

Bryan has one computer, and a run measuring her yields to her app (runs.sh). The
run waits at the start of a turn, so no phase is cut in half and comes back as a
failure on the machine's clock (core/subject/parking.py).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_CHILD = r"""
import sys, time
from pathlib import Path
from core.subject.parking import offer_to_park, park_if_asked, note_turn
out = Path(sys.argv[1])
assert offer_to_park()
for turn in range(100000):
    park_if_asked()
    out.write_text(str(turn))
    note_turn(0.02)
    time.sleep(0.02)
"""


def _until(condition, seconds: float) -> bool:
    ends = time.monotonic() + seconds
    while time.monotonic() < ends:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _marker(park: Path, pid: int) -> dict:
    try:
        return json.loads((park / f"{pid}.json").read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _parks_and_carries_on(pid: int, park: Path, counter: Path) -> None:
    from core.subject.parking import HOLD

    (park / HOLD).touch()
    assert _until(lambda: _marker(park, pid).get("parked") is True, 10), "the run did not park"
    assert _marker(park, pid)["parks"] == 1
    turn = counter.read_text()
    time.sleep(1.5)
    assert counter.read_text() == turn, "a parked run took another turn"
    # A SIGCONT sent to the whole group, as zsh sends one, must not lift the hold.
    os.kill(pid, signal.SIGCONT)
    time.sleep(1.0)
    assert counter.read_text() == turn, "a stray SIGCONT lifted the hold"
    (park / HOLD).unlink()
    assert _until(lambda: counter.read_text() != turn, 10), "the run did not carry on"
    assert _until(lambda: _marker(park, pid).get("parked") is False, 10)
    assert _marker(park, pid)["parked_s"] >= 2.0


def test_nothing_is_installed_without_a_park_directory(monkeypatch):
    from core.subject import parking

    monkeypatch.delenv("AURA_PARK_DIR", raising=False)
    assert parking.offer_to_park() is False
    assert parking.park_if_asked() == 0.0


def test_a_run_parks_at_the_start_of_a_turn_and_carries_on(tmp_path):
    park, counter = tmp_path / "park", tmp_path / "turn"
    env = dict(os.environ, AURA_PARK_DIR=str(park), AURA_LOG_DIR=str(tmp_path / "logs"))
    child = subprocess.Popen([sys.executable, "-c", _CHILD, str(counter)], cwd=ROOT, env=env)
    try:
        assert _until(lambda: (park / f"{child.pid}.json").exists() and counter.exists(), 60), "the run never offered to park"
        _parks_and_carries_on(child.pid, park, counter)
    finally:
        child.send_signal(signal.SIGINT)
        child.wait(timeout=30)
    assert not (park / f"{child.pid}.json").exists(), "a run that ended left its offer to park behind"


@pytest.mark.skipif(shutil.which("zsh") is None, reason="no zsh on this host")
def test_a_park_holds_under_a_shell_that_has_more_to_run(tmp_path):
    """The decisive chain runs every stage in a zsh subshell with more after it.

    There, a process that stopped itself with SIGSTOP was continued at once and
    parked for 0 s, because zsh continues a stopped job's whole process group
    (30 October). The park is the run's own wait on a file, so it holds.
    """
    park, counter = tmp_path / "park", tmp_path / "turn"
    child_file = tmp_path / "child.py"
    child_file.write_text(_CHILD)
    env = dict(os.environ, AURA_PARK_DIR=str(park), AURA_LOG_DIR=str(tmp_path / "logs"), PYTHONPATH=str(ROOT))
    script = f'( "{sys.executable}" "{child_file}" "{counter}"; echo after ) & wait'
    shell = subprocess.Popen(["zsh", "-c", script], cwd=ROOT, env=env, start_new_session=True)
    try:
        assert _until(lambda: any(park.glob("*.json")) and counter.exists(), 60), "the run never offered to park"
        pid = int(next(p for p in park.glob("*.json")).stem)
        _parks_and_carries_on(pid, park, counter)
    finally:
        os.killpg(shell.pid, signal.SIGTERM)
        shell.wait(timeout=30)
