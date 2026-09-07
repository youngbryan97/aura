"""Exercise launch isolation with real processes, without loading Aura."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "tools/exec_detached.py"


def test_detached_exec_survives_launcher_group_cleanup(tmp_path):
    receipt = tmp_path / "child.json"
    child_code = (
        "import json,os,time,pathlib; "
        f"pathlib.Path({str(receipt)!r}).write_text(json.dumps("
        "{'pid':os.getpid(),'sid':os.getsid(0),'pgid':os.getpgrp(),"
        "'stdin':os.read(0,1).decode()})); time.sleep(30)"
    )
    launcher_code = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen(sys.argv[1:]); "
        "print(child.pid,flush=True); time.sleep(30)"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", launcher_code, sys.executable, str(HELPER),
         sys.executable, "-c", child_code],
        start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = None
    try:
        import time

        child_pid = int(launcher.stdout.readline())
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        observed = json.loads(receipt.read_text())
        assert observed == {"pid": child_pid, "sid": child_pid,
                            "pgid": child_pid, "stdin": ""}
        os.killpg(launcher.pid, signal.SIGTERM)
        launcher.wait(timeout=5)
        os.kill(child_pid, 0)
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGTERM)
            launcher.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        launcher.stdout.close()
        launcher.stdin.close()


def test_launcher_detaches_session_and_input():
    source = (ROOT / "launch_aura.sh").read_text()
    assert 'tools/exec_detached.py "$PYTHON_CMD" -u aura_main.py' in source
    assert '</dev/null >>"$ACTIVE_LAUNCH_LOG"' in source


def test_detached_exec_requires_command():
    result = subprocess.run([sys.executable, str(HELPER)], capture_output=True,
                            text=True, timeout=5)
    assert result.returncode != 0
    assert "a command is required" in result.stderr
