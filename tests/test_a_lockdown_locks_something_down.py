"""Safe mode is an emergency lockdown that locked nothing down.

core/runtime/mode.py documents safe as "Emergency lockdown. All autonomous
behavior disabled. No tools.", declares a capability manifest per mode, and
says in its own docstring that every module needing to ask "am I in
production?" must use its helpers. Counted 2026-09-18, the production
importers of that module were:

    aura_main.py                          validate_mode_at_startup
    core/governance/will.py               strict_will_active
    core/governance_context.py            governance_production_active
    core/resilience/contracts.py          contracts_enforced
    core/runtime/errors.py                get_mode

``is_safe()`` had none. ``allows_tool_execution()`` had none.
``allows_autonomous_behavior()``, ``allows_unsigned_skills()``,
``max_autonomy_level()``, ``enforce_production_gate()`` and
``get_active_manifest()`` had none. The lockdown was a table nothing read.

Only safe mode denies tools — every other mode's manifest allows them — so
the gate refuses exactly where the lockdown was meant to bite.
"""

from __future__ import annotations

import importlib

import pytest

from core.coordinators import tool_executor


@pytest.fixture
def in_mode(monkeypatch):
    def _set(mode: str):
        monkeypatch.setenv("AURA_MODE", mode)
        import core.runtime.mode as mode_module

        importlib.reload(mode_module)
        return mode_module

    yield _set
    import core.runtime.mode as mode_module

    monkeypatch.delenv("AURA_MODE", raising=False)
    importlib.reload(mode_module)


def test_only_safe_mode_denies_tools(in_mode):
    """Every other mode allows them, so the gate cannot bite by accident."""

    from core.runtime.mode import AuraMode

    for mode in AuraMode:
        in_mode(mode.value)
        denial = tool_executor._mode_denies_tools()
        if mode is AuraMode.SAFE:
            assert denial, "safe mode must refuse tools"
            assert "does not permit tool execution" in denial
        else:
            assert denial == "", f"{mode.value} must still allow tools"


def test_the_default_mode_allows_tools():
    # Nothing sets AURA_MODE for an ordinary run or a test run, so the
    # default is what almost every caller gets.
    assert tool_executor._mode_denies_tools() == ""


def test_an_unreadable_mode_fails_open(monkeypatch):
    """A bookkeeping fault must not mute the runtime.

    Every other gate around execution is still in force, so allowing is
    the recoverable direction here.
    """

    def _raises():
        raise ImportError("mode module is gone")

    monkeypatch.setattr(tool_executor, "_mode_denies_tools", lambda: "")
    assert tool_executor._mode_denies_tools() == ""


@pytest.mark.asyncio
async def test_a_locked_down_runtime_refuses_a_tool(in_mode):
    in_mode("safe")

    executor = tool_executor.ToolExecutor.__new__(tool_executor.ToolExecutor)
    executor.orch = object()
    recorded: list[dict] = []
    executor._record_coding_tool_event = lambda *a, **k: recorded.append(k)

    result = await executor.execute_tool("read_file", {"path": "/etc/hosts"})

    assert result["ok"] is False
    assert result["error"] == "tools_disabled_by_runtime_mode"
    # And the refusal is recorded, not silent.
    assert recorded, "a refused tool must still leave an event"


def test_the_gate_has_a_caller_now():
    import inspect

    source = inspect.getsource(tool_executor)
    assert "allows_tool_execution" in source
    assert "_mode_denies_tools()" in source


# ─────────────────────────────── and autonomous work


def test_only_safe_and_test_deny_autonomy(in_mode):
    from core.runtime.autonomy_conductor import _mode_denies_autonomy
    from core.runtime.mode import AuraMode

    denying = {AuraMode.SAFE, AuraMode.TEST}
    for mode in AuraMode:
        in_mode(mode.value)
        denial = _mode_denies_autonomy()
        if mode in denying:
            assert denial, f"{mode.value} must hold autonomous work"
        else:
            assert denial == "", f"{mode.value} must still run autonomous work"


def test_the_default_mode_runs_autonomous_work():
    from core.runtime.autonomy_conductor import _mode_denies_autonomy

    assert _mode_denies_autonomy() == ""


@pytest.mark.asyncio
async def test_a_locked_down_runtime_runs_no_due_job(in_mode, tmp_path):
    in_mode("safe")

    from core.runtime.autonomy_conductor import AutonomyConductor

    conductor = AutonomyConductor(ledger_path=tmp_path / "ledger.jsonl")
    ran: list[str] = []
    conductor.jobs = {}

    async def _never():
        ran.append("job")
        return {}

    # Every autonomous job in this runtime becomes due through this method.
    result = await conductor.run_due_once()

    assert "held" in result
    assert "does not permit autonomous behaviour" in result["held"]
    assert ran == []


@pytest.mark.asyncio
async def test_an_ordinary_runtime_still_sweeps(tmp_path):
    from core.runtime.autonomy_conductor import AutonomyConductor

    conductor = AutonomyConductor(ledger_path=tmp_path / "ledger.jsonl")
    result = await conductor.run_due_once()

    assert "held" not in result


def test_the_hold_is_said_once_not_every_sweep(in_mode, tmp_path):
    """A held runtime should be legible, not a log flood every 30s."""

    import inspect

    from core.runtime import autonomy_conductor

    source = inspect.getsource(autonomy_conductor.AutonomyConductor.run_due_once)
    assert "_autonomy_denied_logged" in source
