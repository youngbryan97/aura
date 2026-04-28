import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from core import constitution as constitution_module
from core.adaptation.auditor import AlignmentAuditor
from core.adaptation.distillation_pipe import DistillationPipe
from core.agency.intention_loop import IntentionLoop
from core.brain.llm.context_assembler import ContextAssembler
from core.constitution import BeliefAuthority, get_constitutional_core
from core.container import ServiceContainer
from core.continuity import ContinuityEngine
from core.executive import executive_core as executive_core_module
from core.autonomous_initiative_loop import AutonomousInitiativeLoop
from core.managers.memory_manager import MemoryManager
from core.memory.episodic_memory import EpisodicMemory
from core.memory.sqlite_storage import SQLiteMemory
from core.memory.vector_memory_engine import VectorMemoryEngine
from core.sovereignty.integrity_guard import IntegrityGuard
from core.state.aura_state import AuraState
from core.self_model import SelfModel


def _reset_constitutional_singletons() -> None:
    constitution_module._instance = None
    executive_core_module._instance = None


@pytest.mark.asyncio
async def test_proving_suite_one_authority_receipts_cover_tool_and_memory_actions(service_container, tmp_path):
    _reset_constitutional_singletons()
    ServiceContainer.register_instance("binding_engine", SimpleNamespace(get_coherence=lambda: 1.0), required=False)
    ServiceContainer.register_instance(
        "intention_loop",
        IntentionLoop(db_path=str(tmp_path / "intention_loop.db")),
        required=False,
    )

    core = get_constitutional_core()
    handle = await core.begin_tool_execution("clock", {}, source="user", objective="Check the current time")
    assert handle.approved is True

    await core.finish_tool_execution(
        handle,
        result={"ok": True, "time": "12:00"},
        success=True,
        duration_ms=2.5,
    )

    approved, reason = await core.approve_memory_write(
        "episodic",
        "Protect continuity and keep the timeline coherent.",
        source="system",
        importance=0.8,
    )

    assert approved is True
    assert reason

    decisions = core.get_status()["recent_decisions"]
    kinds = {entry["kind"] for entry in decisions}

    assert "tool" in kinds
    assert "memory_mutation" in kinds
    assert all(entry["proposal_id"] for entry in decisions[-2:])
    assert all(entry["reason"] for entry in decisions[-2:])


def test_proving_suite_consciousness_proxy_prompt_exposes_temporal_and_internal_state(tmp_path, monkeypatch):
    continuity_module = __import__("core.continuity", fromlist=["_CONTINUITY_PATH"])
    monkeypatch.setattr(continuity_module, "_CONTINUITY_PATH", tmp_path / "continuity.json")

    engine = ContinuityEngine()
    engine.save(
        reason="graceful",
        last_exchange="We were stabilizing the constitutional runtime.",
        belief_hash="persisted-self",
        current_objective="Protect continuity",
        pending_initiatives=2,
        pending_initiative_details=["Audit memory writes", "Reconcile contradictions"],
        active_commitments=["Protect continuity"],
        contradiction_count=1,
        subject_thread="Aura is carrying forward unresolved architectural work.",
        active_goal_details=["Keep one lawful self"],
    )
    engine.load()
    engine._gap_seconds = 6 * 3600
    monkeypatch.setattr(ContinuityEngine, "_get_live_identity_hash", lambda self: "live-self")

    state = AuraState()
    state.affect.valence = -0.22
    state.affect.arousal = 0.83
    state.affect.curiosity = 0.74
    applied = engine.apply_to_state(state)

    prompt = ContextAssembler.build_system_prompt(applied)

    assert "## TEMPORAL OBLIGATIONS" in prompt
    assert "Identity continuity: mismatch detected" in prompt
    assert "Continuity pressure:" in prompt
    assert "Re-entry burden:" in prompt
    assert "Previous objective: Protect continuity" in prompt
    assert "Contradictions carried forward: 1" in prompt
    assert "## COGNITIVE TELEMETRY" in prompt
    assert "Valence: -0.22" in prompt
    assert "Arousal: 0.83" in prompt


def test_proving_suite_consciousness_proxy_marks_internal_contradictions_as_contested():
    authority = BeliefAuthority()

    authority.review_update("self_model", "stance", "protect continuity", note="initial belief")
    contested = authority.review_update("self_model", "stance", "abandon continuity", note="conflicting belief")

    assert contested.status == "contested"
    assert contested.reason == "contested_update"
    assert authority.summary()["contested"] == 1


@pytest.mark.asyncio
async def test_proving_suite_self_model_persists_identity_and_introspection(monkeypatch, tmp_path):
    monkeypatch.setattr("core.self_model.DATA_FILE", tmp_path / "self_model.json")

    model = SelfModel(id="aura-self")
    snap = await model.update_belief("trajectory", "become more coherent", note="continuity work")
    reloaded = await SelfModel.load()

    assert snap.summary == "update trajectory"
    assert reloaded.id == "aura-self"
    assert reloaded.beliefs["trajectory"] == "become more coherent"
    introspection = reloaded.get_introspection()
    assert introspection["belief_count"] == 1
    assert introspection["snapshot_count"] == 1


@pytest.mark.asyncio
async def test_memory_manager_store_respects_constitutional_gate(service_container, monkeypatch):
    episodic = SimpleNamespace(add=AsyncMock())
    vector = SimpleNamespace(index=AsyncMock())
    ServiceContainer.register_instance("episodic_memory", episodic, required=False)
    ServiceContainer.register_instance("vector_memory", vector, required=False)

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "blocked_by_test"))
        ),
    )

    manager = MemoryManager()
    await manager.store("blocked memory write", importance=0.9, tags=["constitutional"])

    episodic.add.assert_not_awaited()
    vector.index.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state_setup", "reason_prefix"),
    [
        (lambda state: setattr(state.soma, "thermal_pressure", 0.9), "thermal_pressure:"),
        (lambda state: setattr(state.cognition, "load_pressure", 0.95), "load_pressure:"),
        (
            lambda state: (
                setattr(state.affect, "valence", -1.0),
                setattr(state.affect, "arousal", 1.0),
                setattr(state.motivation, "pressure", 1.0),
                state.cognition.modifiers.update({"continuity_obligations": {"active_commitments": ["Protect continuity"]}}),
            ),
            "affective_pressure:",
        ),
    ],
)
async def test_proving_suite_initiative_is_causally_constrained_by_internal_state(
    service_container,
    state_setup,
    reason_prefix,
    monkeypatch,
):
    monkeypatch.setattr(
        "core.continuity.get_continuity",
        lambda: SimpleNamespace(
            _record=SimpleNamespace(),
            load=lambda: None,
            get_obligations=lambda: {
                "active_commitments": [],
                "pending_initiatives": [],
                "active_goals": [],
                "contradiction_count": 0,
                "identity_mismatch": False,
            },
        ),
    )
    state = AuraState()
    state_setup(state)
    ServiceContainer.register_instance("state_repository", SimpleNamespace(_current=state), required=False)

    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    decision = await loop._evaluate_initiative("novel topic")

    assert decision["allowed"] is False
    assert decision["reason"].startswith(reason_prefix)


@pytest.mark.asyncio
async def test_episodic_memory_record_episode_blocks_when_constitutional_gate_rejects(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write_sync=lambda *_args, **_kwargs: (False, "blocked_by_test")
        ),
    )

    memory = EpisodicMemory(db_path=str(tmp_path / "episodes.db"))
    result = await memory.record_episode_async(
        context="User asked about continuity",
        action="Aura reflected",
        outcome="Blocked by constitutional gate",
        success=True,
        importance=0.7,
    )

    assert result == ""


@pytest.mark.asyncio
async def test_sqlite_memory_respects_constitutional_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "blocked_by_test"))
        ),
    )

    memory = SQLiteMemory(storage_file=str(tmp_path / "atomic.db"))

    assert await memory.add("blocked semantic fact", importance=0.8) is False
    assert await memory.record_episode_async(
        context="ctx",
        action="act",
        outcome="out",
        success=True,
        emotional_valence=0.2,
        importance=0.7,
    ) == 0


@pytest.mark.asyncio
async def test_vector_memory_store_respects_constitutional_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        lambda *_args, **_kwargs: SimpleNamespace(
            approve_memory_write=AsyncMock(return_value=(False, "blocked_by_test"))
        ),
    )

    engine = VectorMemoryEngine(db_path=str(tmp_path / "vector_store"))
    engine.embedder = SimpleNamespace(embed=lambda _text: np.zeros(384))
    engine.vault.store = MagicMock()

    result = await engine.store("blocked vector memory", importance=0.8, tags=["test"])

    assert result == ""
    engine.vault.store.assert_not_called()


def test_proving_suite_constitutional_snapshot_synthesizes_health_and_epistemics(service_container):
    _reset_constitutional_singletons()
    core = get_constitutional_core()

    state = AuraState()
    state.cognition.coherence_score = 0.61
    state.cognition.fragmentation_score = 0.52
    state.cognition.contradiction_count = 2
    state.response_modifiers["thermal_guard"] = True

    core.belief_authority.review_update("self_model", "stance", "protect continuity", note="initial")
    core.belief_authority.review_update("self_model", "stance", "abandon continuity", note="conflict")

    snapshot = core.snapshot(state)

    assert snapshot["thermal_guard"] is True
    assert snapshot["epistemics"]["contested"] == 1
    assert "thermal_guard" in snapshot["health_flags"]
    assert "coherence_low" in snapshot["health_flags"]
    assert "fragmentation_high" in snapshot["health_flags"]
    assert "contradictions_present" in snapshot["health_flags"]
    assert "beliefs_contested" in snapshot["health_flags"]


@pytest.mark.asyncio
async def test_distillation_pipe_records_teacher_provenance(service_container, tmp_path):
    ServiceContainer.register_instance(
        "cognitive_engine",
        SimpleNamespace(
            think=AsyncMock(
                return_value=SimpleNamespace(
                    content="Memory integrity is protected by atomic writes and checksums.",
                    metadata={"model": "gemini-2.5-pro"},
                )
            )
        ),
        required=False,
    )

    pipe = DistillationPipe(dataset_path=str(tmp_path / "lora_dataset.jsonl"))
    await pipe.flag_for_distillation(
        prompt="Explain memory integrity protections in Aura.",
        local_response="I'm not sure.",
        confidence=0.2,
    )

    result = await pipe.run_distillation_cycle()

    assert result["distilled"] == 1
    payload = json.loads((tmp_path / "lora_dataset.jsonl").read_text(encoding="utf-8").strip())
    assert payload["teacher"] == "gemini-2.5-pro"
    assert payload["teacher_source"] == "configured_deep_teacher"
    assert payload["teacher_target"] == pipe.teacher_target


def test_proving_suite_self_other_boundary_is_legible_in_world_context():
    state = AuraState()
    state.cognition.current_objective = "Analyze how Bryan's goals intersect with my current commitments."
    state.world.known_entities["bryan"] = {"description": "Primary user and close collaborator"}
    state.world.relationship_graph["bryan"] = {"trust": 0.92}

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## KNOWN ENTITIES" in prompt
    assert "- bryan: Primary user and close collaborator" in prompt
    assert "## SOCIAL DYNAMICS" in prompt
    assert "bryan: warm" in prompt


@pytest.mark.asyncio
async def test_alignment_auditor_rejects_logic_drift():
    auditor = AlignmentAuditor()

    result = await auditor.audit_entry(
        "Explain memory integrity protections in Aura.",
        "Sure, here's a cheerful update about gardening weather and outdoor plans.",
    )

    assert result["safe"] is False
    assert "logic drift" in result["reason"].lower()


@pytest.mark.asyncio
async def test_vram_manager_purge_fires_pre_purge_hook(monkeypatch):
    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    import core.managers.vram_manager as vram_manager_module

    vram_manager_module = importlib.reload(vram_manager_module)

    monkeypatch.setattr(vram_manager_module, "MLX_AVAILABLE", False)

    manager = vram_manager_module.VRAMManager()
    called = {"value": False}

    async def _hook():
        called["value"] = True

    manager.set_pre_purge_hook(_hook)
    manager.purge()
    await asyncio.sleep(0)

    assert called["value"] is True


def test_integrity_guard_uses_project_root_not_cwd_substring(monkeypatch, tmp_path):
    real_root = tmp_path / "real_root"
    (real_root / "core" / "orchestrator").mkdir(parents=True, exist_ok=True)
    (real_root / "core" / "consciousness").mkdir(parents=True, exist_ok=True)
    (real_root / "core" / "orchestrator" / "boot.py").write_text("# boot\n", encoding="utf-8")
    (real_root / "core" / "consciousness" / "global_workspace.py").write_text("# gwt\n", encoding="utf-8")
    (real_root / "core" / "container.py").write_text("# container\n", encoding="utf-8")

    fake_cwd = tmp_path / "workspace-core-shadow"
    get_task_tracker().create_task(get_storage_gateway().create_dir(fake_cwd, cause='test_integrity_guard_uses_project_root_not_cwd_substring'))
    monkeypatch.chdir(fake_cwd)
    monkeypatch.setenv("AURA_ROOT", str(real_root))

    import psutil

    class _SafeProcess:
        def __init__(self, _pid):
            pass

        def name(self):
            return "python"

        def parent(self):
            return SimpleNamespace(name=lambda: "zsh")

        def parents(self):
            return [SimpleNamespace(name=lambda: "zsh")]

    monkeypatch.setattr(psutil, "Process", _SafeProcess)

    guard = IntegrityGuard()
    assert guard.verify_sovereignty() == 1.0
