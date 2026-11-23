"""A value that counts the whole process is not a value the change verifier compares.

LIVE 21 November 2026 her change toward "lower what the full collections asked" was refused by the outside check:
freeze_what_boot_made() in core/runtime/collector_pauses.py returned 16,172 frozen objects where the code before
returned 16,069. That function counts every object her process holds, so any change to any file moves it, and the
check also handed her a test asserting 16,069 that no later draft could pass. Two runs of the code before agreed,
since both held the same objects. The second now holds unrelated objects too (verify_change.BALLAST), and a value that
moves with them is left uncompared, as a value that moves with the clock is.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from outside_bar.verify_change import verify

_COLLECTOR = '''import gc

TABLE = {}


def frozen() -> int:
    """How many objects a freeze leaves out of later collections, as her collector reports it."""
    gc.collect()
    gc.freeze()
    count = gc.get_freeze_count()
    gc.unfreeze()
    return count


def doubled(x: int) -> int:
    return x * 2
'''
_TEST = '''from collector import doubled, frozen


def test_frozen():
    assert frozen() > 0


def test_doubled():
    assert doubled(3) == 6
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _judged(repo: Path, collector: str) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    (repo / "tests").mkdir()
    (repo / "collector.py").write_text(_COLLECTOR)
    (repo / "tests" / "test_collector.py").write_text(_TEST)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "before")
    (repo / "collector.py").write_text(collector)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "after")
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


#: Her draft's kind of change: the counting function itself rewritten without changing what it does, beside a change
#: that holds more objects.
_REWRITTEN = (_COLLECTOR.replace("TABLE = {}", "TABLE = {number: [number] for number in range(300)}")
              .replace("    count = gc.get_freeze_count()\n", "    count = int(gc.get_freeze_count())\n"))


def test_a_change_that_only_adds_objects_is_not_refused_for_the_count(tmp_path):
    result = _judged(tmp_path, _REWRITTEN)
    assert result["verdict"] != "REFUSED", result
    assert not any(one["function"].endswith("::frozen") for one in result.get("counterexamples", [])), result


def test_a_value_of_the_function_itself_is_still_compared(tmp_path):
    result = _judged(tmp_path, _REWRITTEN.replace("return x * 2", "return x * 2 if x < 5 else -1"))
    assert result["verdict"] == "REFUSED", result
