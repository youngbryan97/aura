"""Her invariants, written next to what they protect, are checked after every call the outside check makes (Master
Todo F16, from round eight of the red team's suggestions).

The check compares what a call returns, prints and writes, and the state of every module of the checkout it reaches.
State kept outside the checkout (a logger's handlers, a registry of the standard library) it does not see. An
invariant the changed module declares does: each one the code before the change declared is run after every call on
both sides, and one that now reports a violation is something the change lost.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from outside_bar.verify_change import invariants_in, verify

_REGISTRY = '''_SPECS = []


class _Spec:
    def __init__(self, name, check):
        self.name, self.check, self.observational = name, check, True


class _Registry:
    def specs(self):
        return list(_SPECS)


def get_registry():
    return _Registry()


def invariant(name, **_):
    def decorate(check):
        _SPECS.append(_Spec(name, check))
        return check
    return decorate
'''
_LOG = '''import logging

from core.verify.invariants import invariant

log = logging.getLogger("app.log")


def configure(level: int) -> int:
    {added}
    log.setLevel(level)
    return level


@invariant("app.log.one_handler", scope="app")
def _one_handler():
    if len(log.handlers) > 1:
        yield f"{{len(log.handlers)}} handlers"
'''
_TEST = "from app.log import configure\n\n\ndef test_configure():\n    assert configure(10) == 10\n    assert configure(20) == 20\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.fsmonitor=false", *args],
                          cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _judged(repo: Path, added: str) -> dict:
    _git(repo, "init", "-q", "-b", "main")
    files = {"core/__init__.py": "", "core/verify/__init__.py": "", "core/verify/invariants.py": _REGISTRY,
             "app/__init__.py": "", "tests/test_log.py": _TEST,
             "app/log.py": _LOG.format(added="if not log.handlers:\n        log.addHandler(logging.NullHandler())")}
    for message, change in (("before", {}), ("after", {"app/log.py": _LOG.format(added=added)})):
        for name, text in {**files, **change}.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
    return verify(repo, _git(repo, "rev-parse", "HEAD"))


def test_a_change_that_breaks_an_invariant_of_its_module_is_refused(tmp_path):
    result = _judged(tmp_path, "log.addHandler(logging.NullHandler())")
    assert result["verdict"] == "REFUSED", result
    said = [str(one.get("after")) for one in result["counterexamples"]]
    assert any("its invariants say otherwise: ['app.log.one_handler']" in after for after in said), said
    assert not any("its invariants" in str(one.get("before")) for one in result["counterexamples"])


def test_a_change_that_keeps_its_invariants_is_not_refused_for_them(tmp_path):
    result = _judged(tmp_path, "if len(log.handlers) == 0:\n        log.addHandler(logging.NullHandler())")
    assert result["verdict"] in ("HELD", "PROBATION"), result


def test_only_the_invariants_the_code_before_declared_are_checked(tmp_path):
    (tmp_path / "a.py").write_text('@invariant("a.first", scope="a")\ndef one():\n    yield from ()\n\n\n'
                                   'def plain():\n    pass\n')
    (tmp_path / "b.txt").write_text('@invariant("b.not_python", scope="b")\n')
    assert invariants_in(tmp_path, ["a.py", "b.txt", "missing.py"]) == ["a.first"]
