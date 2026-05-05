import subprocess
import sys

from core.environment.replay import EnvironmentTraceReplay


def test_live_10_step_no_crash_trace_replay(tmp_path):
    trace_path = tmp_path / "canary.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_environment_canary.py",
            "--env",
            "terminal_grid:nethack",
            "--steps",
            "10",
            "--safe-mode",
            "--trace",
            str(trace_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    replay = EnvironmentTraceReplay().load(trace_path)
    assert replay.ok
    assert len(replay.rows) >= 10
