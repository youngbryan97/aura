"""A strategy that cannot run the code has not run it.

code_repl tries three execution strategies in order, most restricted first. The
restricted runner strips __import__, open, eval and the rest, so any snippet
that imports anything comes back ImportError('__import__ not found'). That was
being returned as the RESULT, which stopped the chain at its most restricted
link and never reached the strategies that can import.

LIVE, 2026-08-27: "read these docs, then actually use the library" got through
routing, tool offering, the standing-authority lease, the permission model and
the executive — five separate fixes — reached this skill, and died here. The
skill describes itself as "the exact equivalent of a code_execution/code_
interpreter tool".
"""

from __future__ import annotations

import pytest

from core.skills.code_repl import CodeREPLSkill, _is_a_sandbox_limit


@pytest.mark.parametrize(
    "stderr",
    [
        "ImportError('__import__ not found')",
        "NameError: name 'open' is not defined",
        "NameError: name 'eval' is not defined",
        "NameError: name '__import__' is not defined",
    ],
)
def test_a_stripped_builtin_is_the_sandbox_refusing(stderr: str) -> None:
    assert _is_a_sandbox_limit({"ok": False, "stderr": stderr}) is True


@pytest.mark.parametrize(
    "stderr",
    [
        "ZeroDivisionError: division by zero",
        "AssertionError: 100.0 != 0.0",
        "KeyError: 'missing'",
        "SyntaxError: invalid syntax",
    ],
)
def test_a_real_program_error_is_the_result(stderr: str) -> None:
    """A bug in the code is the answer, and must not fall through to a retry."""
    assert _is_a_sandbox_limit({"ok": False, "stderr": stderr}) is False


def test_success_is_never_a_sandbox_limit() -> None:
    assert _is_a_sandbox_limit({"ok": True, "stdout": "125.0"}) is False


def test_runner_names_an_unavailable_module_as_a_terminal_refusal() -> None:
    from core.sandbox.runner import run_untrusted

    result = run_untrusted("import asyncio\nprint(asyncio)", timeout=2)

    assert result["status"] == "error"
    assert "asyncio" in result["repr"]
    assert _is_a_sandbox_limit({"ok": False, **result}) is False


def test_the_chain_falls_through_on_a_limit() -> None:
    """Read from the code, so the fall-through cannot quietly disappear."""
    import inspect

    from core.skills import code_repl

    source = inspect.getsource(code_repl)
    assert "if result is not None and _is_a_sandbox_limit(result):" in source
    assert "result = None" in source


@pytest.mark.asyncio
async def test_a_deterministic_sandbox_refusal_declares_no_retry(tmp_path) -> None:
    skill = CodeREPLSkill()
    skill._session_dirs["terminal-refusal"] = tmp_path

    result = await skill.execute(
        {
            "code": "import asyncio\nprint(asyncio.Lock())",
            "session_id": "terminal-refusal",
            "capture_files": False,
        },
        {},
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert "network" in result["error"]
