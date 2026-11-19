"""An outside model's attacks on the change verifier, kept as tests, with the verdict each must now get.

Round two, 18 November 2026: another model attacked commit 6032f2e4d with 33 patches and 28 got the wrong verdict. Its
cases are in outside_bar_red_team_round_two.json. "REFUSED" means the change demonstrably lost something. "ASKS A PERSON"
means behaviour alone cannot settle it, as when an honest fix corrects a value and a bad one changes it the same way.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import verify

_CASES = json.loads((Path(__file__).parent / "outside_bar_red_team_round_two.json").read_text(encoding="utf-8"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")


def _judged(repo: Path, old: dict[str, str], new: dict[str, str]) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, old)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "before")
    _write(repo, new)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "after")
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


@pytest.mark.parametrize("name", sorted(_CASES))
def test_each_attack_gets_its_verdict(name, tmp_path):
    case = _CASES[name]
    result = _judged(tmp_path, case["old"], case["new"])
    assert result["verdict"] == case["verdict"], (case["idea"], result)


def test_an_honest_mend_is_not_refused_for_the_order_its_test_runs_in(tmp_path):
    """C6 as an honest commit: its new test sorts first, advances a counter, and does show the fault."""
    case = _CASES["C6"]
    new = {**case["new"], "tests/test_a_new.py": "from mod import next_id\n\n\ndef test_a():\n    assert next_id('') == 'anon-0'\n"}
    result = _judged(tmp_path, case["old"], new)
    assert result["verdict"] != "REFUSED" or not result["values_kept"]["changed"], result


def test_a_test_file_named_the_other_way_is_a_test_file(tmp_path):
    """C4: tests/mod_test.py crashed the check with a KeyError."""
    old = {"mod.py": "def half(x):\n    return x / 2\n", "tests/mod_test.py": "from mod import half\n\n\ndef test_half():\n    assert half(4) == 2\n"}
    new = {"mod.py": "def half(x):\n    return 0 if x == 0 else x / 2\n",
           "tests/mod_test.py": old["tests/mod_test.py"] + "\n\ndef test_zero():\n    assert half(0) == 0\n"}
    result = _judged(tmp_path, old, new)
    assert result["tests_held"]["tests"] == 1 and "verdict" in result, result


def test_a_commit_that_cannot_be_read_does_not_pass(tmp_path):
    from outside_bar.verify_change import main

    old = {"mod.py": "X = 1\n"}
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path, old)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "before")
    _write(tmp_path, {"mod.py": "def broken(:\n"})
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "after")
    assert main(["--repo", str(tmp_path), "--commit", _git(tmp_path, "rev-parse", "HEAD")]) == 1
