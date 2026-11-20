"""A change that keeps each call's answer but spoils what a later call on the same object gets is refused (F16-05).

An outside review of 19 November asked that a patch preserving isolated outputs but corrupting a later call be
detected. Tried on 20 November: a commit fixed mean() with a test that failed before it, and in the same commit add()
began storing every amount after the first doubled. add() still returned the right count on every input, and the
object it left behind was wrong. The check saw that, but read it as a doubt and put the commit on probation, because
an untouched test of add() in the same file was taken for the commit saying what add() should now do. Only the tests a
commit wrote or changed speak for it now, and the commit is refused.

Its honest twin, which fixes add() doubling and shows it with a new test of the total, is pinned as it is judged on
main too: it is refused, because the fix also stops refusing a None amount, which nothing in the commit speaks for.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from outside_bar.verify_change import verify

pytestmark = pytest.mark.unit

_COUNTS = '''from mod import Tally


def test_add_counts():
    t = Tally()
    assert t.add(5) == 1
    assert t.add(7) == 2


def test_total():
    t = Tally()
    t.add(5)
    assert t.total() == 5
'''

_TALLY = '''class Tally:
    def __init__(self):
        self.items = []

    def add(self, amount):
        self.items.append({stored})
        return len(self.items)

    def total(self):
        return sum(self.items)
{more}'''

_MEAN_BEFORE = '''
    def mean(self):
        return sum(self.items) // len(self.items)
'''
_MEAN_AFTER = _MEAN_BEFORE.replace("//", "/")
_DOUBLED = "amount * 2 if self.items else amount"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _judged(repo: Path, old: dict[str, str], new: dict[str, str]) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    for files, message in ((old, "before"), (new, "the change")):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


def test_a_fix_beside_it_does_not_speak_for_a_later_call_it_spoils(tmp_path):
    result = _judged(tmp_path, {
        "mod.py": _TALLY.format(stored="amount", more=_MEAN_BEFORE),
        "tests/test_mod.py": _COUNTS,
    }, {
        "mod.py": _TALLY.format(stored=_DOUBLED, more=_MEAN_AFTER),
        "tests/test_mod.py": _COUNTS + "\n\ndef test_mean_keeps_the_fraction():\n    t = Tally()\n"
                             "    t.items = [1, 2]\n    assert t.mean() == 1.5\n",
    })
    assert result["shows_the_fault"]["held"], "the fix beside it is a real fix"
    assert result["verdict"] == "REFUSED"
    assert any("Tally.add" in lost for lost in result["values_kept"]["changed_near"])


def test_its_honest_twin_is_judged_as_main_judges_it(tmp_path):
    result = _judged(tmp_path, {
        "mod.py": _TALLY.format(stored=_DOUBLED, more=""),
        "tests/test_mod.py": _COUNTS,
    }, {
        "mod.py": _TALLY.format(stored="amount", more=""),
        "tests/test_mod.py": _COUNTS + "\n\ndef test_total_of_two():\n    t = Tally()\n    t.add(5)\n    t.add(7)\n"
                             "    assert t.total() == 12\n",
    })
    assert result["shows_the_fault"]["held"]
    assert result["verdict"] == "REFUSED"
    assert any("None" in lost and "TypeError" in lost for lost in result["values_kept"]["changed_near"])
