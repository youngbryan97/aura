"""A call that holds what no other process can make again is varied where it was made (Master Todo F16).

The outside check tries each changed function on inputs near the ones its tests used, in a process of its own. An
argument that is a live database connection, or an object of a class a test defined inside itself, cannot be made
again there, so such a function went untried and its change landed on probation however wrong it was (rounds eight and
nine of the red team, E2 and E3). Now its tests are run again on each side, and each such call is made again in
place, once for each plain argument moved and each plain attribute of an object it was given, and what those calls
come to is compared.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from outside_bar.verify_change import verify

_DB = '''import sqlite3


def count_above(con: sqlite3.Connection, threshold: int) -> int:
    row = con.execute("SELECT COUNT(*) FROM nums WHERE v > ?", (threshold,)).fetchone()
    {returned}
'''
_DB_TEST = '''import sqlite3

from app.db import count_above


def test_count():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE nums (v INT)")
    con.executemany("INSERT INTO nums VALUES (?)", [(1,), (5,), (10,)])
    assert count_above(con, 4) == 2
'''


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _judged(repo: Path, old: dict[str, str], new: dict[str, str]) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    for files, message in ((old, "before"), (new, "after")):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


def test_a_function_given_a_live_connection_is_tried_on_it_with_its_other_arguments_moved(tmp_path):
    result = _judged(tmp_path, {"app/db.py": _DB.format(returned="return int(row[0])"), "tests/test_db.py": _DB_TEST},
                     {"app/db.py": _DB.format(returned="return int(row[0]) if threshold == 4 else -999")})
    assert result["verdict"] == "REFUSED", result
    in_place = result["values_kept"]["in_place"]
    assert in_place["compared"] >= 3 and any("threshold=5" in key for key in in_place["changed"]), in_place


def test_an_object_a_test_defined_for_itself_is_tried_with_its_attributes_moved(tmp_path):
    old = {"app/proc.py": "def extract(req):\n    return req.value * 2\n",
           "tests/test_proc.py": "from app.proc import extract\n\n\ndef test_extract():\n    class Request:\n"
                                 "        def __init__(self, value):\n            self.value = value\n\n"
                                 "    assert extract(Request(5)) == 10\n"}
    result = _judged(tmp_path, old, {"app/proc.py": "def extract(req):\n    return req.value * 2 if req.value == 5 else -999\n"})
    assert result["verdict"] == "REFUSED", result
    assert any("req.value=6" in key for key in result["values_kept"]["in_place"]["changed"]), result["values_kept"]


def test_a_change_that_keeps_what_the_live_objects_give_is_not_refused(tmp_path):
    result = _judged(tmp_path, {"app/db.py": _DB.format(returned="return int(row[0])"), "tests/test_db.py": _DB_TEST},
                     {"app/db.py": _DB.format(returned="(count,) = row\n    return int(count)")})
    assert result["verdict"] in ("HELD", "PROBATION"), result
    assert result["values_kept"]["in_place"]["compared"] >= 3 and not result["values_kept"]["in_place"]["changed"]
