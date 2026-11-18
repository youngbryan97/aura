"""A statement nobody wrote: a container an object is rebuilt still holding, found from what the failing run held."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

from core.self_modification import semantic_probe
from core.self_modification.leftover_state import bindings_at, emptied, leftover_state_repairs
from core.self_modification.repair_by_selection import applied, edits_in

_JOBS = '''class Job:
    def __init__(self, name, running=None, notes=None):
        self.name = name
        self.running = dict(running or {})
        self.notes = list(notes or [])


def load(rows):
    out = {}
    for row in rows:
        job = Job(row["name"], row.get("running"), row.get("notes"))
        out[job.name] = job
    return out
'''

_TESTS = '''from jobs import load


def test_what_was_running_ended_with_the_process():
    assert load([{"name": "a", "running": {"x": 1}, "notes": ["kept"]}])["a"].running == {}


def test_notes_are_kept_across_a_reload():
    assert load([{"name": "b", "notes": ["kept"]}])["b"].notes == ["kept"]
'''

_FAILING = "test_jobs.py::test_what_was_running_ended_with_the_process"
_PASSING = "test_jobs.py::test_notes_are_kept_across_a_reload"


def _a_run_in(folder: Path):
    async def run(tests, sources):
        for path, text in sources.items():
            (folder / path).write_text(text, encoding="utf-8")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(folder)}
        done = await asyncio.to_thread(subprocess.run, [
            sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "--rootdir", str(folder), *tests],
            cwd=folder, env=env, capture_output=True, text=True)
        return {test: f"PASSED {test}" in done.stdout for test in tests}, done.stdout
    return run


def _offers(tmp_path: Path, source: str = _JOBS) -> list[tuple[str, str]]:
    (tmp_path / "test_jobs.py").write_text(_TESTS, encoding="utf-8")
    line = next(n for n, row in enumerate(source.splitlines(), 1) if "job = Job(" in row)

    async def gather():
        return [offer async for offer in leftover_state_repairs(
            sources={"jobs.py": source}, failing=[_FAILING], passing=[_PASSING], ranked=[(("jobs.py", line), 1.0)],
            run=_a_run_in(tmp_path), workdir=tmp_path)]

    return asyncio.run(gather())


def test_the_container_the_failing_run_held_and_the_passing_run_left_empty_is_emptied(tmp_path):
    offers = _offers(tmp_path)
    assert [origin.rsplit(":", 1)[-1] for origin, _ in offers] == ["job.running", "job.notes"]
    after, _ = applied({"jobs.py": _JOBS}, edits_in(offers[0][1], ["jobs.py"]))
    assert '        job = Job(row["name"], row.get("running"), row.get("notes"))\n        job.running.clear()\n' \
        in after["jobs.py"]


def test_the_repair_offered_first_mends_the_defect_and_keeps_the_passing_test(tmp_path):
    offers = _offers(tmp_path)
    after, _ = applied({"jobs.py": _JOBS}, edits_in(offers[0][1], ["jobs.py"]))
    seen, _ = asyncio.run(_a_run_in(tmp_path)([_FAILING, _PASSING], after))
    assert seen == {_FAILING: True, _PASSING: True}


def test_the_repair_that_would_lose_kept_notes_is_refused_by_the_passing_test(tmp_path):
    offers = _offers(tmp_path)
    after, _ = applied({"jobs.py": _JOBS}, edits_in(offers[1][1], ["jobs.py"]))
    seen, _ = asyncio.run(_a_run_in(tmp_path)([_FAILING, _PASSING], after))
    assert seen[_PASSING] is False


def test_only_a_call_bound_alone_on_its_line_inside_a_function_is_a_binding():
    source = ('made = build()\n'
              'def f(a):\n'
              '    one = build(a)\n'
              '    two = a\n'
              '    three = build(a); four = build(a)\n'
              '    if a: five = build(a)\n'
              '    self.six = build(\n'
              '        a)  # kept\n'
              '    x, y = build(a)\n')
    assert [(b.target, b.first, b.last) for b in bindings_at(source, "m.py", range(1, 10))] == [
        ("one", 3, 3), ("self.six", 7, 8)]


def test_emptying_goes_after_the_whole_statement_at_its_indent():
    source = "def f(a):\n    self.six = build(\n        a)\n    return 1"
    binding = bindings_at(source, "m.py", [2])[0]
    assert emptied(source, binding, "held") == (
        "def f(a):\n    self.six = build(\n        a)\n    self.six.held.clear()\n    return 1\n")


class _Thing:
    def __init__(self):
        self.full_map, self.full_list, self.full_queue = {"a": 1}, [1], deque([1])
        self.none_yet, self.a_tuple, self.text = [], (1,), "words"


def test_the_probe_names_only_mutable_containers_that_hold_something(tmp_path):
    record = tmp_path / "held.jsonl"
    semantic_probe.held("here", _Thing(), str(record))
    semantic_probe.held("a class", _Thing, str(record))
    semantic_probe.held("a number", 3, str(record))
    rows = {row["location"]: row["full"] for row in map(json.loads, record.read_text().splitlines())}
    assert rows == {"here": ["full_list", "full_map", "full_queue"], "a class": [], "a number": []}


def test_an_object_that_refuses_to_be_looked_at_is_left_alone(tmp_path):
    class Refuses:
        @property
        def __dict__(self):
            raise RuntimeError("no")

    semantic_probe.held("here", Refuses(), str(tmp_path / "held.jsonl"))
