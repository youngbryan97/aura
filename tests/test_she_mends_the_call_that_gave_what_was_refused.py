"""Where a refusal was right, her loop mends the call that built what was refused (F15).

LIVE 16 November 2026 her notifications built an argument vector with an empty subtitle, her subprocess gateway passed
it on, and ``_coerce_argv`` refused it. On 26 November her loop mended the refusal instead, twice. The call that built
the vector was gone by then: only the frames below the refusal were kept.
"""
from __future__ import annotations

import asyncio
import importlib
import sys

import pytest

from core.agency import her_goal_at_work as goals
from core.agency import mending_a_fault_of_hers as mending
from core.runtime import a_call_that_failed
from core.runtime.a_call_that_failed import FailedCall, the_call_that_failed
from core.self_modification import faults_she_can_reproduce as heard
from core.self_modification import where_a_fault_lies as lies
from tests.test_her_goal_is_worked_by_her import _git, her_source  # noqa: F401
from tests.test_she_asks_her_model_where_a_fault_lies import _HerModel

pytestmark = pytest.mark.unit

PLANTED = '''def coerce(argv):
    if any(not part for part in argv):
        raise ValueError("argv entries must not be empty")
    return list(argv)


def run(argv, timeout=5):
    return coerce(argv)


def send(title, subtitle=""):
    words = [title, subtitle]
    return run(["notify", *words])
'''


@pytest.fixture
def planted(tmp_path, monkeypatch):
    package = tmp_path / "refusing_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "planted.py").write_text(PLANTED)
    monkeypatch.setattr(a_call_that_failed, "_ROOT", tmp_path.resolve())
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("refusing_pkg.planted", "refusing_pkg"):
        # Each test plants it in a directory of its own, which must be the one her source is read from.
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("refusing_pkg.planted")


def _raised(call):
    try:
        call()
    except ValueError as error:
        return error
    raise AssertionError("it did not raise")


def test_the_call_that_built_what_was_refused_is_kept_past_one_that_only_passed_it_on(planted):
    call = the_call_that_failed(_raised(lambda: planted.send("Stretch")))
    assert isinstance(call, FailedCall) and call.qualname == "coerce"
    assert call.caller is not None and call.caller.qualname == "send", "run only passed on the vector it was given"
    assert dict(call.caller.keywords) == {"title": "'Stretch'", "subtitle": "''"}
    assert call.caller.line == 13


def test_no_caller_is_kept_when_what_was_refused_came_from_outside_her_source(planted):
    call = the_call_that_failed(_raised(lambda: planted.run(["notify", ""])))
    assert call.qualname == "coerce" and call.caller is None


def test_her_state_keeps_the_caller_across_a_restart(planted):
    fault = heard.HeardFault(the_call_that_failed(_raised(lambda: planted.send("Stretch"))), "planted", 1.0, 2.0)
    again = heard._from_row(heard._as_row(fault))
    assert again == fault and again.call.caller.qualname == "send"


def test_a_refusal_her_model_reads_as_right_has_its_caller_mended_as_a_fault_of_its_own(her_source, monkeypatch):
    (her_source / "core" / "notes.py").write_text(PLANTED)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "planted", cwd=her_source)
    sender = FailedCall(module="core.notes", qualname="send", file="core/notes.py", line=13, error_type="",
                        error_message="", keywords=(("title", "'Stretch'"), ("subtitle", "''")))
    refusal = FailedCall(module="core.notes", qualname="coerce", file="core/notes.py", line=3,
                         error_type="builtins.ValueError", error_message="argv entries must not be empty",
                         keywords=(("argv", "['notify', 'Stretch', '']"),), caller=sender)
    monkeypatch.setattr(lies, "her_model", lambda schema=None: _HerModel(True, "what called it"))
    made: list[FailedCall] = []

    async def made_again(fault, workspace):
        made.append(fault.call)
        return heard.Reproduction(fault, "tests/her_faults/test_fault_send.py", False, "passed")

    monkeypatch.setattr(heard, "reproduce", made_again)
    done = asyncio.run(mending.mend(heard.HeardFault(refusal, "notes", 0.0, 0.0), proposals=4, seconds=30.0))
    assert [call.qualname for call in made] == ["send"], "the refusal is not made again; its caller is"
    assert made[0].error_type == "builtins.ValueError" and made[0].below[0] == "core/notes.py:3 coerce"
    left = goals.load(goals.slug_of(mending.goal_for(heard.HeardFault(refusal, "notes", 0.0, 0.0))))
    assert left.last_check["by_design"] and "I took up send" in left.said[-1]["words"]
    assert done["for"] == left.slug and done["slug"] != left.slug
