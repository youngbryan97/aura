"""Gate 16 at runtime: the same machinery, watched, on two different families.

Gate 16 reads the source and refuses any path keyed on an evaluation, and says
of itself that the check is a grep — weak, and unarguable. It is also static.
A branch keyed on the SHAPE of a problem rather than on its name would pass it
and still be a bag of solvers.

So this watches instead: two problems from materially different families
through one entry point, every function under core that actually executes
recorded, and the difference compared. What matters is not that the sets are
identical — a search that finds different rules should run different rule
code — but that the difference stays inside the search.
"""
from __future__ import annotations

import pytest

from core.cognition.sequence_induction import answer_sequence_question
from tools.agi_gauntlet.the_same_path_twice import (
    THE_CONSEQUENCES,
    THE_SEARCH,
    score_again,
    the_same_path_twice,
    what_ran_during,
)

POSITIONAL = "If 1 2 3 becomes 3 2 1, and 4 5 6 becomes 6 5 4, what does 7 8 9 become?"
ARITHMETIC = "2 4 6 becomes 4 8 12. 1 3 5 becomes 2 6 10. what does 5 7 9 become?"


def _in_its_own_process(snippet: str) -> str:
    """Run a probe in a fresh process.

    Two runs in one process are not independent: the first teaches her
    something the second already knows, and the second then takes a shorter
    path. Every check that needs a fresh run gets a fresh process.
    """
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Its own state root as well as its own process. What she has already
    # worked out changes which path a solve takes, so a probe that inherits
    # another test's state is measuring that test.
    where = tempfile.mkdtemp(prefix="the-same-path-twice-")
    env = {**os.environ, "AURA_STATE_ROOT": where, "AURA_LOG_DIR": where}
    out = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    return out.stdout + out.stderr


def _both(first_positional: bool = True):
    families = {
        "positional": lambda: answer_sequence_question(POSITIONAL),
        "arithmetic": lambda: answer_sequence_question(ARITHMETIC),
    }
    if not first_positional:
        families = dict(reversed(list(families.items())))
    return the_same_path_twice(families)


@pytest.fixture
def watched():
    """One run, in a process where nothing has exercised her yet.

    This is not tidiness. The first solve of something new is a developmental
    event and the second is not, so a probe that runs after any other test has
    already asked her a question is comparing a learner with itself at two
    moments. In-process, that showed up as this file passing test by test and
    failing as a file.
    """
    import json

    said = _in_its_own_process(
        "import json\n"
        "from core.cognition.sequence_induction import answer_sequence_question as f\n"
        "from tools.agi_gauntlet.the_same_path_twice import the_same_path_twice\n"
        f"POS = {POSITIONAL!r}\n"
        f"ARI = {ARITHMETIC!r}\n"
        "r = the_same_path_twice({'positional': lambda: f(POS),"
        " 'arithmetic': lambda: f(ARI)})\n"
        "print('<<<' + json.dumps(r) + '>>>')\n"
    )
    assert "<<<" in said, said[-900:]
    return json.loads(said.split("<<<", 1)[1].split(">>>", 1)[0])


# ----------------------------------------------------------- the tracer


def test_the_tracer_sees_what_ran():
    seen, answer = what_ran_during(lambda: answer_sequence_question(POSITIONAL))
    assert any("sequence_induction.py" in one for one in seen)
    assert answer


def test_the_tracer_puts_the_profile_hook_back():
    import sys

    before = sys.getprofile()
    what_ran_during(lambda: None)
    assert sys.getprofile() is before


def test_module_bodies_are_import_and_not_work(watched):
    """Whichever family runs first pays for every import on the path."""
    for only in watched["what_is_outside"].values():
        assert not any(one.endswith(":<module>") for one in only)


# ---------------------------------------------------------- the finding


def test_both_families_are_actually_answered(watched):
    """A probe over two declines measures the decline path."""
    for name, answer in watched["answered"].items():
        assert answer.strip(), f"{name} was not answered"


def test_the_same_machinery_runs_for_both(watched):
    assert watched["shared"] > 50


def test_nothing_outside_the_search_runs_for_one_family_only(watched):
    assert watched["passed"], watched["what_is_outside"]
    assert all(count == 0 for count in watched["outside_the_search"].values())


def test_it_holds_whichever_family_goes_first():
    """The arithmetic family first, in its own process."""
    said = _in_its_own_process(
        "from core.cognition.sequence_induction import answer_sequence_question as f\n"
        "from tools.agi_gauntlet.the_same_path_twice import the_same_path_twice\n"
        f"POS = {POSITIONAL!r}\n"
        f"ARI = {ARITHMETIC!r}\n"
        "r = the_same_path_twice({'arithmetic': lambda: f(ARI),"
        " 'positional': lambda: f(POS)})\n"
        "print('PASSED' if r['passed'] else r['what_is_outside'])\n"
    )
    assert "PASSED" in said, said[-900:]


# ------------------------------------------------- the declared boundary


def test_the_search_is_a_package_and_not_a_list_of_modules():
    """Enumerating whatever ran would make the probe unfalsifiable.

    It stays falsifiable because the shape gate 16 refuses — a solver picked
    by the name or shape of an evaluation — would be a skill, a route, or a
    named module reached from outside, and all of those are outside this
    package.
    """
    assert THE_SEARCH == "core/cognition/"


def test_every_consequence_says_why_it_is_one():
    """None of them decides an answer, and each has to say so."""
    assert THE_CONSEQUENCES
    for where, why in THE_CONSEQUENCES.items():
        assert where.startswith("core/"), where
        assert not where.startswith(THE_SEARCH), f"{where} is inside the search"
        assert len(why.split()) >= 3, where


def test_a_solver_outside_the_search_would_be_caught(watched):
    """The probe has to be able to fail, or it is decoration.

    Scored again from the same recorded run rather than run again: running
    her twice to find out would measure what she learned in between.
    """
    narrowed = score_again(watched, ("core/cognition/sequence_induction.py",))
    assert watched["passed"]
    assert not narrowed["passed"], (
        "narrowing the declared search to one module must fail; if it does "
        "not, the probe cannot detect anything"
    )


def test_the_result_says_what_it_shows_and_not_more(watched):
    assert "outside the hypothesis search" in watched["what_this_shows"]


# ------------------------------------------------------------ the gate


def test_gate_sixteen_runs_the_probe_beside_its_scan():
    """The static half says of itself that it is a grep, and it is right."""
    said = _in_its_own_process(
        "from tools.agi_gauntlet.runnable import generality_not_a_bag_of_solvers\n"
        "r = generality_not_a_bag_of_solvers(None, {})\n"
        "print('AT_RUNTIME', r['at_runtime'].get('passed'), 'GATE', r['passed'])\n"
    )
    assert "AT_RUNTIME True GATE True" in said, said[-900:]


def test_the_gate_fails_if_either_half_fails(monkeypatch):
    """A gate that passes on one of two checks is one check."""
    import tools.agi_gauntlet.runnable as runnable

    monkeypatch.setattr(
        runnable, "_watch_two_families", lambda: {"passed": False, "why": "a solver"}
    )
    found = runnable.generality_not_a_bag_of_solvers(None, {})
    assert found["found"] == []
    assert found["passed"] is False


def test_the_gate_says_the_scan_is_not_the_whole_check():
    from tools.agi_gauntlet.gates import the_gate_called

    gate = the_gate_called("generality rather than a bag of solvers")
    assert "watched" in gate.to_dict()["control"]
