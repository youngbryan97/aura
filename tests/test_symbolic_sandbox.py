"""Tests for the Symbolic Sandbox cognitive scratchpad."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.symbolic_sandbox import SymbolicSandbox, get_symbolic_sandbox

_HOME = str(Path.home())
_PROBE = Path("/tmp/aura_sandbox_escape_probe")


def _escaped() -> bool:
    return _PROBE.exists()


@pytest.mark.asyncio
async def test_runs_pure_computation():
    sbx = SymbolicSandbox(timeout=15.0)
    res = await sbx.run("print(sum(range(10)))")
    assert res.ok
    assert res.stdout.strip() == "45"
    assert res.isolation["sandboxed"] is True
    assert res.isolation["kernel_boundary"] in {"seatbelt", "bubblewrap"}
    assert res.isolation["network_denied"] is True


@pytest.mark.asyncio
async def test_os_runs_inside_the_boundary_and_reaches_only_its_scratch():
    """LIVE 17 November 2026: `import os` for os.path.splitext was refused by a list of module names, nothing ran, and
    her reply said it had. The kernel boundary is what contains a script."""
    sbx = SymbolicSandbox(timeout=15.0)
    res = await sbx.run("import os\nprint(os.path.splitext('photo.JPG')[1].lower())")
    assert res.ok and not res.refused and res.stdout.strip() == ".jpg"
    assert any("os" in w for w in res.warnings), "what the static reading noticed rides along"

    home = await sbx.run(f"import os\nprint(os.listdir({_HOME!r}))")  # its own HOME is the scratch
    assert not home.ok and not home.refused and "PermissionError" in (home.traceback + home.stderr)
    shell = await sbx.run("import os\nprint(os.system('touch /tmp/aura_sandbox_escape_probe'))")
    assert shell.stdout.strip() != "0" and not _escaped(), "no exec but the interpreter"


@pytest.mark.asyncio
async def test_refuses_syntax_error():
    sbx = SymbolicSandbox()
    res = await sbx.run("def broken(:\n  pass")
    assert not res.ok and res.refused


@pytest.mark.asyncio
async def test_captures_runtime_traceback():
    sbx = SymbolicSandbox(timeout=15.0)
    res = await sbx.run("print(1/0)")
    assert not res.ok and not res.refused
    assert "ZeroDivisionError" in (res.traceback + res.stderr)


@pytest.mark.asyncio
async def test_self_correction_loop_fixes_code():
    sbx = SymbolicSandbox(timeout=15.0)

    async def repair(code: str, traceback: str) -> str:
        # The "generator" notices the division-by-zero and fixes it.
        assert "ZeroDivisionError" in traceback
        return "print(10 // 2)"

    res = await sbx.run_with_self_correction("print(10 // 0)", repair, max_rounds=3)
    assert res.ok
    assert res.stdout.strip() == "5"
    assert res.rounds == 2


@pytest.mark.asyncio
async def test_self_correction_executes_previously_unawaited_work():
    sbx = SymbolicSandbox(timeout=15.0)
    body = "async def compute():\n    print(sum(range(10)))\n"
    broken = body + "jobs = [compute()]\n"
    repaired = "import asyncio\n" + body + "asyncio.run(compute())\n"
    attempts = []

    async def repair(code: str, diagnostic: str) -> str:
        assert code == broken.strip()
        assert "never awaited" in diagnostic
        attempts.append(diagnostic)
        return repaired

    res = await sbx.run_with_self_correction(broken, repair, max_rounds=2)
    assert len(attempts) == 1
    assert res.ok, res.to_dict()
    assert res.stdout.strip() == "45"
    assert res.final_code == repaired.strip()
    assert res.rounds == 2


@pytest.mark.asyncio
async def test_self_correction_gives_up_after_rounds():
    sbx = SymbolicSandbox(timeout=15.0)

    calls = {"n": 0}

    async def repair(code: str, traceback: str) -> str:
        calls["n"] += 1
        # Distinct-but-still-broken each time, so the loop actually retries.
        return f"x = {calls['n']}\nprint(1/0)"

    res = await sbx.run_with_self_correction("print(1/0)", repair, max_rounds=2)
    assert not res.ok
    assert res.rounds == 2


def test_singleton():
    assert get_symbolic_sandbox() is get_symbolic_sandbox()
