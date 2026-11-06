"""Her self-maintenance and artificial-life organs are registered, and they run.

Every kernel tick, ``CognitiveIntegrationPhase`` asks the container for twelve
organs by name: the Lenia and entropy dynamics, pattern replication and
speciation, the criticality regulator, homeostatic drives, the autopoiesis
repair cycle, adaptive immunity, and six more. The dream phase asks for four
more. Their only registrar was ``init_cognitive_sensory_layer``, which nothing
has called since at least April 2026, so all sixteen read None on every tick of
her life and each runner returned at its first line. Her app logged
"Activation audit passed: 100%" over it.

It went unseen through three reviews, each of which fixed a real defect one
layer down: the mesh gained the attributes the ALife layer reads, the phase
learned to write the Lenia coupling back, and a test learned to pair the dream
phase's reads with a registrar. That last test read the source of the
uncalled initializer, so it paired them with a function nobody runs. The tests
here ask the registrar her app and the battery both call, and then run the
phase.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase
from core.providers.self_maintenance_provider import (
    ALIASES,
    SELF_MAINTENANCE_SERVICES,
    register_self_maintenance_services,
)

ROOT = Path(__file__).resolve().parents[1]
PROVIDED = {name for name, *_ in SELF_MAINTENANCE_SERVICES}


def test_every_organ_the_phase_reads_has_a_row_in_the_registrar():
    missing = sorted(name for _, name in CognitiveIntegrationPhase.SERVICES if name not in PROVIDED)
    assert not missing, f"the phase reads these and nothing registers them: {missing}"


def test_the_phase_reads_no_organ_it_does_not_declare():
    """A read added to the phase without a row in SERVICES would have no registrar either."""
    source = (ROOT / "core/phases/cognitive_integration_phase.py").read_text()
    body = source.split("def _resolve_services", 1)[1].split("async def execute", 1)[0]
    literal = set(re.findall(r'ServiceContainer\.get\(\s*"([a-z0-9_]+)"', body))
    assert not literal - {name for _, name in CognitiveIntegrationPhase.SERVICES}


def test_every_research_organ_the_dream_phase_reads_has_a_row_in_the_registrar():
    tick = (ROOT / "core/mind_tick.py").read_text()
    reads = set(re.findall(r'ServiceContainer\.get\(\s*"([a-z0-9_]+)"', tick))
    research = {"metacognitive_monitor", "intrinsic_motivation", "experience_distillery", "plasticity_governor"}
    assert research & reads, "mind_tick no longer reads any research organ"
    assert not (research & reads) - PROVIDED


def test_the_uncalled_initializer_lists_nothing_the_registrar_lacks():
    """Two lists drift. Until the old initializer is retired, its list is a subset of this one."""
    from core.orchestrator.initializers.cognitive_sensory import (
        _init_cognitive_sensory_layer_learned_cognitive_systems,
    )

    listed, _ = _init_cognitive_sensory_layer_learned_cognitive_systems()
    assert not set(listed) - PROVIDED


def test_register_all_services_registers_every_organ_and_alias(service_container):
    from core.service_registration import register_all_services

    register_all_services._full_run = False
    register_all_services()
    for name in PROVIDED | {alias for alias, _ in ALIASES}:
        assert service_container.has(name), f"{name} is not registered by register_all_services"


def test_a_name_already_registered_keeps_its_instance(service_container):
    mine = object()
    service_container.register_instance("alife_dynamics", mine, required=False)
    added = register_self_maintenance_services(service_container)
    assert "alife_dynamics" not in added
    assert service_container.get("alife_dynamics") is mine


def test_homeostasis_read_under_its_other_name_is_the_engine_with_a_vitality(service_container):
    from core.consciousness.homeostasis import HomeostasisEngine

    engine = HomeostasisEngine()
    service_container.register_instance("homeostasis", engine, required=False)
    register_self_maintenance_services(service_container)
    assert service_container.get("homeostasis_engine") is engine
    assert engine.vitality == pytest.approx(engine.compute_vitality())
    assert engine.snapshot() == engine.get_status()


def test_a_late_registration_is_seen_on_the_next_tick(service_container):
    """The phase used to read once and keep the None."""
    phase = CognitiveIntegrationPhase(SimpleNamespace(cycle_count=0))
    phase._resolve_services()
    assert phase._homeostatic_rl is None and phase._resolved is False

    marker = object()
    service_container.register_instance("homeostatic_rl", marker, required=False)
    phase._resolve_services()
    assert phase._homeostatic_rl is marker


def test_a_factory_that_raises_is_left_out_once_and_the_tick_goes_on(service_container):
    def broken():
        raise RuntimeError("this organ cannot be built")

    service_container.register("strange_loop", broken, required=False)
    phase = CognitiveIntegrationPhase(SimpleNamespace(cycle_count=0))
    phase._resolve_services()
    phase._resolve_services()
    assert phase._strange_loop is None
    assert "strange_loop" in phase._failed_services


def test_her_organs_run_in_the_phase_and_change_her_mesh(service_container):
    """The test the three reviews did not run: the registrar, then the phase, then what moved."""
    from core.consciousness.neural_mesh import NeuralMesh
    from core.state.aura_state import AuraState

    mesh = NeuralMesh()
    service_container.register_instance("neural_mesh", mesh, required=False)
    register_self_maintenance_services(service_container)

    kernel = SimpleNamespace(cycle_count=0)
    phase = CognitiveIntegrationPhase(kernel)
    state = AuraState()
    coupling_before = mesh.inter_column_weights.copy()

    async def live(ticks: int) -> AuraState:
        nonlocal state
        for tick in range(ticks):
            kernel.cycle_count = tick + 1
            mesh.inject_sensory(np.full(mesh.cfg.sensory_end * 64, 0.6, dtype=np.float32))
            mesh._tick()
            state = await phase.execute(state, objective="how are you holding up today?")
            await phase.settle_background()
        return state

    state = asyncio.run(live(3))

    assert not phase._failed_services, phase._failed_services
    held = {name for attribute, name in phase.SERVICES if getattr(phase, attribute) is not None}
    assert held == {name for _, name in phase.SERVICES}

    modifiers = state.response_modifiers
    for key in (
        "homeostatic_energy",  # homeostatic RL
        "autopoiesis_vitality",  # the repair cycle ran
        "phenomenal_weight",  # the strange loop
        "anomaly_threat_level",  # the anomaly detector
        "survival_watch",  # endogenous fitness
        "entropy",  # ALife dynamics, carried from the background task
        "criticality_score",  # the criticality regulator, likewise
    ):
        assert key in modifiers, f"{key} never reached her state"

    assert phase._alife_dynamics._tick_count >= 3
    assert modifiers.get("lenia_coupling_applied") is True
    assert not np.allclose(coupling_before, mesh.inter_column_weights), (
        "the Lenia coupling was computed and her mesh did not change"
    )


def test_no_repair_that_could_erase_her_is_wired_by_method_name():
    """`reset()` on a memory store clears it. The auto-wiring may not pick it as a repair."""
    from core.adaptation.autonomous_resilience import IntegrationAuditor

    class Store:
        def reset(self):
            raise AssertionError("an erasing method was wired as a repair")

        def rollback(self):
            raise AssertionError("an erasing method was wired as a repair")

        def restore(self):
            raise AssertionError("an erasing method was wired as a repair")

        def disable(self):
            raise AssertionError("an erasing method was wired as a repair")

        def reinitialize(self):
            raise AssertionError("an erasing method was wired as a repair")

    assert IntegrationAuditor._repair_handlers_for(Store()) == {}

    class Cache:
        def clear_cache(self):
            return True

        def restart(self):
            return True

    assert set(IntegrationAuditor._repair_handlers_for(Cache())) == {"CLEAR_CACHE", "RESTART_COMPONENT"}


def test_a_failed_memory_commit_and_a_stall_receipt_land_in_one_queue(tmp_path):
    """The memory phase calls push, stall recovery calls capture_failure; one queue answers both."""
    from core.tasks.dead_letter_queue import DeadLetterQueue

    queue = DeadLetterQueue(db_path=str(tmp_path / "dlq.db"))
    queue.push("memory_consolidation.commit_interaction", {"content": "what she said"}, "OSError: disk")
    queue.capture_failure(message="the objective", context={"recovery_attempt": 1},
                          error=RuntimeError("Cognitive Stall Detected"), source="orchestrator_stall")
    rows = queue.get_failed()
    assert {row["skill_name"] for row in rows} == {"memory_consolidation.commit_interaction", "orchestrator_stall"}
    assert any("what she said" in row["params"] for row in rows)


def test_the_initiative_arbiter_reads_the_free_energy_engine(service_container):
    """Its measured urgency into deliberation read a name nothing registered, so it was 0.0 for six weeks."""
    from core.consciousness.free_energy import FreeEnergyEngine

    engine = FreeEnergyEngine()
    service_container.register_instance("free_energy_engine", engine, required=False)
    register_self_maintenance_services(service_container)
    assert service_container.get("free_energy") is engine
    assert callable(getattr(service_container.get("free_energy"), "get_action_urgency", None))
