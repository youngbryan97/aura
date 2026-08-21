from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.kernel.upgrades_10x import TrueEvolutionPhase
from core.self_modification.self_modification_engine import (
    AutonomousSelfModificationEngine,
)


class _Kernel:
    def __init__(self, *, reloaded: bool = True) -> None:
        self.hot_reboots: list[tuple[str, ...]] = []
        self.auto_fix_engine = None
        self.orchestrator = None
        self.reloaded = reloaded

    async def hot_reboot(self, *, changed_files: tuple[str, ...] = ()):
        self.hot_reboots.append(changed_files)
        return {
            "ok": True,
            "reloaded": list(changed_files) if self.reloaded else [],
            "restart_required": not self.reloaded,
        }


class _Engine:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result

    async def run_refinement_cycle(self) -> dict[str, object]:
        return dict(self.result)


@pytest.mark.asyncio
async def test_successful_noop_refinement_does_not_mutate_identity_or_reload():
    kernel = _Kernel()
    phase = TrueEvolutionPhase(
        kernel,
        engine=_Engine(
            {
                "success": True,
                "refinements_applied": 0,
                "changed_files": [],
                "reload_required": False,
            }
        ),
    )
    state = SimpleNamespace(identity=SimpleNamespace(narrative_version=41))

    await phase._safe_self_modify(state)

    assert state.identity.narrative_version == 41
    assert kernel.hot_reboots == []


@pytest.mark.asyncio
async def test_applied_refinement_passes_exact_changed_files_to_refresh():
    kernel = _Kernel()
    phase = TrueEvolutionPhase(
        kernel,
        engine=_Engine(
            {
                "success": True,
                "refinements_applied": 1,
                "changed_files": ["core/phases/dialogue.py"],
                "reload_required": True,
            }
        ),
    )
    state = SimpleNamespace(identity=SimpleNamespace(narrative_version=41))

    await phase._safe_self_modify(state)

    assert state.identity.narrative_version == 42
    assert kernel.hot_reboots == [("core/phases/dialogue.py",)]


@pytest.mark.asyncio
async def test_applied_refinement_waiting_for_restart_does_not_advance_identity():
    kernel = _Kernel(reloaded=False)
    phase = TrueEvolutionPhase(
        kernel,
        engine=_Engine(
            {
                "success": True,
                "refinements_applied": 1,
                "changed_files": ["core/kernel/aura_kernel.py"],
                "reload_required": True,
            }
        ),
    )
    state = SimpleNamespace(identity=SimpleNamespace(narrative_version=41))

    await phase._safe_self_modify(state)

    assert state.identity.narrative_version == 41
    assert kernel.hot_reboots == [("core/kernel/aura_kernel.py",)]


@pytest.mark.asyncio
async def test_refinement_engine_names_noop_as_no_reload():
    engine = AutonomousSelfModificationEngine.__new__(
        AutonomousSelfModificationEngine
    )
    engine.kernel_refiner = SimpleNamespace(
        analyze_kernel_health=lambda: _async_value([])
    )

    result = await engine.run_refinement_cycle()

    assert result == {
        "success": True,
        "refinements_applied": 0,
        "changed_files": [],
        "reload_required": False,
    }


async def _async_value(value):
    return value
