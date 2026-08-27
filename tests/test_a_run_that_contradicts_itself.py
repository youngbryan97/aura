"""Find the defect by observing the run, not by recognising the defect.

`carried_state.py` reads the source for one class of bug — state that outlives
a call. It works, and it is a catalogue: it finds what somebody thought to look
for, which is the shape this codebase keeps having to undo.

The fault-localisation literature does not work that way. Delta debugging
minimises the input that triggers a failure, spectrum-based localisation ranks
statements by their appearance in failing runs, and Daikon infers invariants
from observed values and reports where they break. What they share is that the
defect is DERIVED from observation, which is also how a person debugs: hold what
should happen beside what did, and localise where they diverge.

These hold the observation half. Each fixture is a different defect class, and
none of them is named anywhere in the tracer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.diagnosis.value_trace import describe_contradictions, trace_run


def _project(tmp_path: Path, library: str, script: str) -> Path:
    (tmp_path / "lib.py").write_text(library)
    (tmp_path / "run.py").write_text(script)
    return tmp_path


def test_a_mutable_default_shows_up_as_a_contradiction(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        "def add_line(item, price, lines=[]):\n"
        "    lines.append((item, price))\n"
        "    return lines\n",
        "from lib import add_line\n"
        "add_line('consulting', 100.0)\n"
        "add_line('hosting', 25.0)\n"
        "print(add_line('consulting', 100.0))\n",
    )
    run = trace_run(project, "run.py")
    assert not run.error, run.error
    assert run.contradictions, "the same call answered differently and nothing noticed"
    first = run.contradictions[0]
    assert first.function == "add_line"
    assert first.file == "lib.py"
    assert "add_line" in describe_contradictions(run)


def test_a_cache_that_leaks_shows_up_the_same_way(tmp_path: Path) -> None:
    """A different bug the tracer was never told about."""
    project = _project(
        tmp_path,
        "_CACHE = {}\n\n\n"
        "def price_of(item):\n"
        "    _CACHE[item] = _CACHE.get(item, 0) + 1\n"
        "    return _CACHE[item]\n",
        "from lib import price_of\n"
        "print(price_of('widget'))\n"
        "print(price_of('widget'))\n",
    )
    run = trace_run(project, "run.py")
    assert run.contradictions
    assert run.contradictions[0].function == "price_of"


def test_an_ordering_dependence_shows_up_the_same_way(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        "_SEEN = []\n\n\n"
        "def rank(name):\n"
        "    _SEEN.append(name)\n"
        "    return len(_SEEN)\n",
        "from lib import rank\n"
        "print(rank('a'))\n"
        "print(rank('b'))\n"
        "print(rank('a'))\n",
    )
    run = trace_run(project, "run.py")
    assert run.contradictions
    assert run.contradictions[0].function == "rank"


def test_a_project_that_does_not_contradict_itself_reports_nothing(tmp_path: Path) -> None:
    """A clean run must stay clean, or every diagnosis is noise."""
    project = _project(
        tmp_path,
        "def add_line(item, price, lines=None):\n"
        "    lines = list(lines or [])\n"
        "    lines.append((item, price))\n"
        "    return lines\n",
        "from lib import add_line\n"
        "print(add_line('consulting', 100.0))\n"
        "print(add_line('consulting', 100.0))\n",
    )
    run = trace_run(project, "run.py")
    assert not run.error, run.error
    assert run.calls_seen >= 2, "nothing was traced at all"
    assert run.contradictions == ()
    assert describe_contradictions(run) == ""


def test_the_import_machinery_is_not_the_project(tmp_path: Path) -> None:
    """A frozen module reports a filename that is not a path.

    `Path("<frozen importlib._bootstrap>").resolve()` lands inside the working
    directory, which IS the project, so the whole import machinery came back as
    project code and the first contradictions reported were in importlib.
    """
    project = _project(tmp_path, "def f(x):\n    return x\n", "from lib import f\nprint(f(1))\n")
    run = trace_run(project, "run.py")
    assert all("importlib" not in item.file for item in run.contradictions)
    assert run.calls_seen < 100, f"traced {run.calls_seen} calls for a two-line program"


def test_a_generator_is_not_a_contradiction(tmp_path: Path) -> None:
    """A generator frame returns None after yielding, every time."""
    project = _project(
        tmp_path,
        "def total(rows):\n    return sum(value for _name, value in rows)\n",
        "from lib import total\n"
        "print(total([('a', 1.0), ('b', 2.0)]))\n"
        "print(total([('a', 1.0), ('b', 2.0)]))\n",
    )
    run = trace_run(project, "run.py")
    assert all("<" not in item.function for item in run.contradictions)


def test_the_whole_diagnosis_leads_with_what_it_observed(tmp_path: Path) -> None:
    from core.diagnosis.repository import describe_diagnosis, diagnose_repository

    (tmp_path / "README.md").write_text(
        "Every call starts a fresh invoice unless you pass one in.\n"
    )
    _project(
        tmp_path,
        "def add_line(item, price, lines=[]):\n"
        "    lines.append((item, price))\n"
        "    return lines\n",
        "from lib import add_line\n"
        "add_line('consulting', 100.0)\n"
        "print(add_line('consulting', 100.0))\n",
    )
    diagnosis = diagnose_repository(tmp_path)
    told = describe_diagnosis(diagnosis)
    assert "contradicted itself" in told
    assert "add_line" in told
    # Three independent things agree: what it printed, what it did, what it says.
    assert diagnosis.evidence_count() >= 3


@pytest.mark.parametrize("entry", ["run.py"])
def test_the_tracer_leaves_nothing_behind(tmp_path: Path, entry: str) -> None:
    """It writes into somebody else's project; it does not get to keep anything."""
    project = _project(tmp_path, "def f(x):\n    return x\n", "from lib import f\nf(1)\n")
    trace_run(project, entry)
    left = {path.name for path in project.iterdir()}
    assert not any(name.startswith("_aura_value_trace") for name in left), left
