"""Child output cannot erase a completed test's result or its failure register."""
from __future__ import annotations

import json
import sys

import pytest

from tools import run_test_chunks as runner


def _probe(tmp_path, passes):
    path = tmp_path / "test_output.py"
    path.write_text(
        "import os\n\ndef test_bytes():\n"
        "    os.write(1, b'\\xffout\\n')\n"
        "    os.write(2, b'\\xfeerr\\n')\n"
        f"    assert {passes!r}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("passes", [True, False])
def test_invalid_utf8_keeps_the_completed_chunk_verdict(tmp_path, capsys, passes):
    path = _probe(tmp_path, passes)
    ok, message, failures = runner.run_chunk(
        1, 1, [path], marker="", timeout_s=30, python=sys.executable,
        extra_args=["-s", "-p", "no:randomly"],
    )
    assert ok is passes, message
    captured = capsys.readouterr()
    assert "\ufffdout" in captured.out and "\ufffderr" in captured.err
    assert len(failures) == (0 if passes else 1)


def test_invalid_utf8_in_both_attempts_preserves_the_defect_register(tmp_path):
    path = _probe(tmp_path, False)
    register = tmp_path / "defects.json"
    assert runner.main([
        "--tests-dir", str(tmp_path), "--chunks", "1",
        "--chunk-timeout", "30", "--retry-timeout", "30",
        "--defect-register", str(register), "--progress-file", str(tmp_path / "progress.log"),
        "--", "-s", "-p", "no:randomly",
    ]) == 1
    record = json.loads(register.read_text())
    assert len(record["real_failures"]) == 1
    assert str(path.name) in record["real_failures"][0]
    assert record["order_dependent"] == []
    assert record["retry_errors"] == record["retry_refusals"] == {}
