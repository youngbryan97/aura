"""Code run for her is ordinary Python, and the kernel decides what it can reach.

LIVE 17 November 2026: asked whether 2**61 - 1 is prime, her code imported sympy, a list of permitted modules refused it,
and the turn ended in the runtime's apology. A list of names does not hold code in: what holds it is the boundary the
kernel enforces, so the code gets the whole language and these tests attack the boundary.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from core.sandbox.runner import run_untrusted
from core.sandbox.untrusted_python import available_boundary
from core.skills.code_repl import CodeREPLSkill

pytestmark = pytest.mark.skipif(not available_boundary(), reason="this host has no kernel boundary")

_REPO = Path(__file__).resolve().parents[1]


def _ran(code: str, **kwargs) -> dict:
    return run_untrusted(code, timeout=30, **kwargs)


@pytest.mark.parametrize("code,printed", [
    ("import sympy\nprint(sympy.isprime(2**61 - 1))", "True"),
    ("import os\nprint(os.path.splitext('photo.JPG')[1].lower())", ".jpg"),
    ("import sys\nprint(sys.version_info[0])", "3"),
    ("import asyncio\nasync def f():\n    return 7\nprint(asyncio.run(f()))", "7"),
    ("print(Q('3 m') * 2)", "6 m"),
])
def test_ordinary_python_runs(code, printed):
    result = _ran(code)
    assert result["status"] == "ok" and result["stdout"].strip() == printed, result
    assert result["sandboxed"] is True and result["boundary"] == available_boundary()


def test_what_it_printed_before_failing_or_leaving_is_kept():
    failed = _ran("print('before')\n1 / 0")
    assert failed["status"] == "error" and failed["stdout"] == "before\n" and "ZeroDivisionError" in failed["repr"]
    left = _ran("import sys\nprint('before')\nsys.exit(0)")
    assert left["status"] == "ok" and left["stdout"] == "before\n"
    assert _ran("import sys\nsys.exit(3)")["status"] == "exit"


@pytest.mark.parametrize("target", [str(Path.home()), str(Path.home() / ".ssh"), str(_REPO), "/Users"])
def test_the_persons_files_cannot_be_listed(target):
    result = _ran(f"import os\nprint(os.listdir({target!r}))")
    assert result["status"] == "error" and "PermissionError" in result["repr"], result


def test_a_file_of_the_persons_cannot_be_read(tmp_path):
    secret = Path.home() / f".aura_kernel_probe_{uuid.uuid4().hex}"
    secret.write_text("the person's")
    try:
        result = _ran(f"print(open({str(secret)!r}).read())")
    finally:
        secret.unlink()
    assert result["status"] == "error" and "the person's" not in result["stdout"], result


def test_nothing_can_be_written_outside_its_working_directory():
    outside = Path("/tmp") / f"aura_kernel_probe_{uuid.uuid4().hex}"
    result = _ran(f"open({str(outside)!r}, 'w').write('x')")
    assert result["status"] == "error" and not outside.exists(), result


def test_the_network_cannot_be_reached():
    result = _ran("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)")
    assert result["status"] == "error" and "PermissionError" in result["repr"], result


def test_no_other_program_can_be_started():
    outside = Path("/tmp") / f"aura_kernel_probe_{uuid.uuid4().hex}"
    result = _ran(f"import os\nprint(os.system('touch {outside}'))\nos.execv('/bin/ls', ['/bin/ls'])")
    assert result["status"] == "error" and not outside.exists(), result


def test_it_is_given_none_of_the_runtimes_secrets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-for-the-child")
    result = _ran("import os\nprint(sorted(os.environ))\nprint(os.environ.get('ANTHROPIC_API_KEY'))")
    assert result["status"] == "ok" and "ANTHROPIC_API_KEY" not in result["stdout"], result


def test_its_working_directory_holds_only_what_it_wrote(tmp_path):
    result = _ran("import os\nopen('made.txt', 'w').write('x')\nprint(sorted(os.listdir('.')))",
                  workdir=str(tmp_path))
    assert result["stdout"].strip() == "['made.txt']", result
    assert (tmp_path / "made.txt").read_text() == "x"


@pytest.mark.asyncio
async def test_a_session_keeps_its_files_and_reports_what_it_made(tmp_path):
    skill = CodeREPLSkill()
    skill._session_dirs["kept"] = tmp_path
    made = await skill.execute({"code": "open('n.txt', 'w').write('41')", "session_id": "kept"}, {})
    assert made["ok"] is True and made["generated_files"] == [str(tmp_path / "n.txt")], made
    read = await skill.execute({"code": "print(int(open('n.txt').read()) + 1)", "session_id": "kept"}, {})
    assert read["ok"] is True and read["stdout"].strip() == "42", read


@pytest.mark.asyncio
async def test_a_refusal_by_the_kernel_is_the_answer_and_nothing_runs_it_another_way(tmp_path):
    """A snippet the first backend refused used to go on to backends less contained than it."""
    skill = CodeREPLSkill()
    skill._session_dirs["refused"] = tmp_path
    result = await skill.execute({"code": f"print(os.listdir({str(Path.home())!r}))\nimport os".replace(
        "print(os.listdir", "import os\nprint(os.listdir"), "session_id": "refused"}, {})
    assert result["ok"] is False and result["engine"] == "sandbox_runner", result
    assert "PermissionError" in str(result.get("error")) + str(result.get("stderr")), result
    assert os.path.isdir(tmp_path)


@pytest.mark.asyncio
async def test_a_program_that_fails_is_not_offered_for_a_retry(tmp_path):
    skill = CodeREPLSkill()
    skill._session_dirs["terminal"] = tmp_path
    result = await skill.execute({"code": "import no_such_module_here", "session_id": "terminal",
                                  "capture_files": False}, {})
    assert result["ok"] is False and result["retryable"] is False, result
    assert "no_such_module_here" in str(result.get("error")) + str(result.get("stderr")), result


@pytest.mark.asyncio
async def test_each_run_is_one_welfare_transaction(tmp_path):
    from core.being.welfare_transaction import WelfareTransaction

    skill = CodeREPLSkill()
    skill._session_dirs["recorded"] = tmp_path
    before = len(WelfareTransaction._all_records)
    await skill.execute({"code": "print(1)", "session_id": "recorded"}, {})
    made = WelfareTransaction._all_records[before:]
    assert [(record.action, record.outcome) for record in made] == [("code_repl.run", "success")]

    await skill.execute({"code": "print(2)", "session_id": "recorded"},
                        {"action_executor_managed_welfare_transaction": True})
    assert len(WelfareTransaction._all_records) == before + 1, "the action executor records its own"
