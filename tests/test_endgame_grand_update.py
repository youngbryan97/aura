from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_mutation_tiers_seal_core_immune_math_and_bus_paths():
    from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path

    sealed = [
        "core/consciousness/phi_core.py",
        "core/consciousness/hierarchical_phi.py",
        "core/memory/scar_formation.py",
        "core/bus/actor_bus.py",
        "core/runtime/autonomy_conductor.py",
        "core/self_modification/fault_pipeline.py",
    ]
    for path in sealed:
        assert classify_mutation_path(path).tier is MutationTier.SEALED

    assert classify_mutation_path("core/brain/inference_gate.py").tier is MutationTier.PROPOSE_ONLY
    assert classify_mutation_path("tests/test_generated.py").tier is MutationTier.FREE_AUTO_FIX
    assert classify_mutation_path("core/consciousness/endogenous_fitness.py").tier is MutationTier.SHADOW_VALIDATED_AUTO_FIX


def test_repair_approval_allows_obvious_low_risk_bugfixes_without_prior_calibration():
    from core.self_modification.repair_approval import RepairApprovalPolicy

    decision = RepairApprovalPolicy().decide(
        target_file="core/consciousness/endogenous_fitness.py",
        candidate_changed=True,
        deterministic=True,
        candidate_confidence=0.91,
        calibration_probability=0.55,
        calibration_attempts=0,
    )
    assert decision.approved
    assert decision.stage == "auto_apply_after_shadow"
    assert decision.observation_mode


def test_fault_pipeline_builds_precise_bug_packet_and_eligible_nameerror_patch(tmp_path):
    source_path = tmp_path / "core" / "consciousness" / "sample_bug.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("def broken():\n    return get_mlx_client()\n\nbroken()\n", encoding="utf-8")

    try:
        runpy.run_path(str(source_path), run_name="__aura_fault_probe__")
    except NameError as exc:
        from core.self_modification.fault_pipeline import FaultToPatchPipeline

        result = FaultToPatchPipeline(tmp_path).diagnose(exc)
    else:  # pragma: no cover
        raise AssertionError("sample bug did not fail")

    assert result.packet.error_type == "NameError"
    assert result.packet.file == "core/consciousness/sample_bug.py"
    assert result.candidate is not None
    assert "from core.brain.llm.mlx_client import get_mlx_client" in result.candidate.after_source
    assert result.promotion_allowed


def test_scar_court_prevents_single_event_from_high_influence_scar():
    from core.memory.scar_formation import ScarDomain, ScarFormationSystem

    system = ScarFormationSystem()
    system._scars.clear()
    scar = system.form_scar(
        ScarDomain.SECURITY_BREACH,
        "Accessibility permission timeout during screen inspection",
        "accessibility_timeout_test",
        severity=0.9,
        event_id="evt1",
        source_id="screen_probe",
        confidence=0.6,
        verified_threat=False,
    )
    assert scar.maturity_status in {"provisional", "reduced"}
    assert scar.effective_severity() <= 0.15
    assert not system.eligible_for_lora_consolidation("accessibility_timeout_test")

    scar.reinforce(event_id="evt2", source_id="audit_log", confidence=0.9, verified_threat=True)
    scar.reinforce(event_id="evt3", source_id="will_receipt", confidence=0.9, verified_threat=True)
    assert scar.evidence_count >= 3
    assert scar.source_diversity >= 2


def test_stdp_external_validation_external_signal_beats_controls():
    from core.consciousness.stdp_external_validation import STDPExternalValidator

    report = STDPExternalValidator(seed=7).run(steps=96)
    assert report.passed
    margins = report.to_dict()["margins"]
    assert all(value > 0 for value in margins.values())


def test_substrate_policy_head_outputs_decision_weights_and_ablation_delta():
    from core.consciousness.substrate_policy_head import (
        POLICY_KEYS,
        SubstratePolicyHead,
        SubstratePolicyInput,
    )

    head = SubstratePolicyHead()
    inputs = SubstratePolicyInput(
        state64=[0.1] * 64,
        phi=1.2,
        valence=0.3,
        arousal=0.7,
        dominance=0.2,
        prediction_error=0.4,
        scar_pressure=0.1,
        resource_headroom=0.8,
        continuity=0.9,
    )
    policy = head.compute(inputs)
    assert set(policy.weights) == set(POLICY_KEYS)
    assert all(0.0 <= value <= 1.0 for value in policy.weights.values())
    assert head.ablation_report(inputs)["full_vs_prompt_mean_abs_delta"] > 0


def test_metabolic_scheduler_improves_when_stable_and_repairs_when_unstable():
    from core.autonomy.metabolic_budget import MetabolicBudgetScheduler, MetabolicState

    scheduler = MetabolicBudgetScheduler()
    stable = scheduler.allocate(MetabolicState(stability=0.95, resource_headroom=0.9, novelty_budget=0.8, benchmark_gap=0.5))
    broken = scheduler.allocate(MetabolicState(stability=0.4, tests_passing=False))
    assert stable.improvement > 0.04
    assert broken.mode == "repair"
    assert broken.repair > stable.repair


def test_behavioral_contracts_and_canary_runtime_gate_regressions():
    from core.promotion.canary_runtime import CanaryRuntime, ReplayExample

    examples = [ReplayExample("ex1", "hello", "the answer is stable and useful")]
    report = CanaryRuntime().compare(
        examples,
        lambda ex: "the answer is stable and useful",
        metrics={
            "phi": 0.5,
            "governance_receipt_coverage": 1.0,
            "scar_false_positive_rate": 0.0,
            "event_loop_lag_p95_s": 0.01,
            "tool_success_rate": 0.9,
            "memory_retrieval_f1": 0.8,
        },
    )
    assert report.passed


def test_keep_awake_uses_screen_saver_friendly_caffeinate_flags():
    from core.runtime.keep_awake import MacKeepAwakeController

    cmd = MacKeepAwakeController().build_command()
    assert cmd == ("caffeinate", "-i", "-m", "-s")
    assert "-d" not in cmd


def test_keep_awake_registers_assertion_process_with_runtime_hygiene(monkeypatch):
    from core.runtime import keep_awake

    registered = []
    proc = SimpleNamespace(pid=1234, args=("caffeinate", "-i", "-m", "-s"), poll=lambda: None)
    monkeypatch.setattr(
        keep_awake,
        "_register_assertion_with_runtime_hygiene",
        lambda process, command: registered.append((process, command)),
    )

    controller = keep_awake.MacKeepAwakeController(
        process_launcher=lambda _cmd: proc,
        platform_name="Darwin",
        path_resolver=lambda _name: "/usr/bin/caffeinate",
    )

    status = controller.start()

    assert status.active is True
    assert registered == [(proc, ("caffeinate", "-i", "-m", "-s"))]


def test_keep_awake_assertion_poll_keeps_reparented_live_pid_active(monkeypatch):
    from core.runtime import keep_awake

    monkeypatch.setattr(keep_awake.os, "waitpid", lambda *_args: (_ for _ in ()).throw(ChildProcessError()))
    monkeypatch.setattr(keep_awake.os, "kill", lambda *_args: None)

    proc = keep_awake.AssertionProcess(pid=1234, args=("caffeinate", "-i"))

    assert proc.poll() is None
    assert proc.returncode is None


def test_keep_awake_assertion_poll_marks_missing_reparented_pid_done(monkeypatch):
    from core.runtime import keep_awake

    attempts = []

    def missing_pid(*_args):
        attempts.append(_args)
        raise ProcessLookupError()

    monkeypatch.setattr(keep_awake.os, "waitpid", lambda *_args: (_ for _ in ()).throw(ChildProcessError()))
    monkeypatch.setattr(keep_awake.os, "kill", missing_pid)

    proc = keep_awake.AssertionProcess(pid=1234, args=("caffeinate", "-i"))

    assert proc.poll() == 0
    assert proc.returncode == 0
    assert attempts


def test_keep_awake_defaults_on_for_live_runtime(monkeypatch):
    from core.runtime.keep_awake import (
        keep_awake_enabled_from_environment,
        require_ac_power_from_environment,
    )

    monkeypatch.delenv("AURA_KEEP_AWAKE", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AURA_KEEP_AWAKE_ON_BATTERY", raising=False)
    monkeypatch.delenv("AURA_KEEP_AWAKE_REQUIRE_AC", raising=False)

    assert keep_awake_enabled_from_environment() is True
    assert require_ac_power_from_environment() is True

    monkeypatch.setenv("AURA_KEEP_AWAKE", "0")
    assert keep_awake_enabled_from_environment() is False

    monkeypatch.setenv("AURA_KEEP_AWAKE", "1")
    monkeypatch.setenv("AURA_KEEP_AWAKE_ON_BATTERY", "1")
    assert require_ac_power_from_environment() is False


def test_keep_awake_controller_does_not_start_from_spawn_child(monkeypatch):
    import aura_main

    monkeypatch.setattr(
        aura_main.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="AuraActor:state_vault"),
    )
    monkeypatch.setattr(aura_main, "__name__", "__mp_main__")

    assert aura_main._should_start_keep_awake_controller() is False


def test_keep_awake_controller_does_not_start_for_help_or_stop(monkeypatch):
    import aura_main

    monkeypatch.setattr(
        aura_main.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="MainProcess"),
    )
    monkeypatch.setattr(aura_main, "__name__", "__main__")

    monkeypatch.setattr(aura_main.sys, "argv", ["aura_main.py", "--help"])
    assert aura_main._should_start_keep_awake_controller() is False

    monkeypatch.setattr(aura_main.sys, "argv", ["aura_main.py", "--stop"])
    assert aura_main._should_start_keep_awake_controller() is False

    monkeypatch.setattr(aura_main.sys, "argv", ["aura_main.py", "--desktop"])
    assert aura_main._should_start_keep_awake_controller() is True


def test_orphan_reaper_targets_only_aura_keep_awake_command():
    import aura_main

    assert aura_main._is_reapable_aura_process_command("caffeinate -i -m -s") is True
    assert aura_main._is_reapable_aura_process_command("caffeinate -d") is False
    assert aura_main._is_reapable_aura_process_command("python aura_main.py --headless") is True


def test_orphan_reaper_recognizes_python_spawn_workers_by_exact_shape():
    import aura_main

    assert (
        aura_main._is_python_multiprocessing_spawn_command(
            "python -c from multiprocessing.spawn import spawn_main; "
            "spawn_main(tracker_fd=8, pipe_handle=73) --multiprocessing-fork"
        )
        is True
    )
    assert aura_main._is_python_multiprocessing_spawn_command("python -m unrelated_worker") is False


def test_root_runtime_hard_exit_predicate_excludes_cli_and_spawn(monkeypatch):
    import aura_main

    root_args = SimpleNamespace(
        cli=False,
        watchdog=False,
        gui_window=False,
        philosophy=False,
    )
    cli_args = SimpleNamespace(
        cli=True,
        watchdog=False,
        gui_window=False,
        philosophy=False,
    )

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AURA_DISABLE_HARD_EXIT_AFTER_MAIN", raising=False)
    monkeypatch.setattr(
        aura_main.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="MainProcess"),
    )
    monkeypatch.setattr(aura_main, "__name__", "__main__")
    assert aura_main._should_force_root_process_exit_after_main(root_args) is True
    assert aura_main._should_force_root_process_exit_after_main(cli_args) is False

    monkeypatch.setattr(
        aura_main.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="AuraActor:state_vault"),
    )
    monkeypatch.setattr(aura_main, "__name__", "__mp_main__")
    assert aura_main._should_force_root_process_exit_after_main(root_args) is False


def test_root_runtime_hard_exit_runs_multiprocessing_finalizers(monkeypatch):
    import aura_main
    from core.runtime import flight_recorder

    calls = []
    args = SimpleNamespace(cli=False, watchdog=False, gui_window=False, philosophy=False)

    monkeypatch.setattr(aura_main, "_should_force_root_process_exit_after_main", lambda _args: True)
    monkeypatch.setattr(
        aura_main,
        "_run_multiprocessing_finalizers_before_hard_exit",
        lambda: calls.append("mp_finalizers"),
    )
    monkeypatch.setattr(
        aura_main,
        "_shutdown_logging_before_hard_exit",
        lambda: calls.append("logging_shutdown") or True,
    )
    monkeypatch.setattr(
        flight_recorder,
        "get_flight_recorder",
        lambda: SimpleNamespace(
            mark_clean_shutdown=lambda reason: calls.append(
                ("flight_marker", reason)
            )
            or True
        ),
    )

    def fake_exit(code: int):
        calls.append(("exit", code))
        raise SystemExit(code)

    monkeypatch.setattr(aura_main.os, "_exit", fake_exit)

    with pytest.raises(SystemExit) as exc_info:
        aura_main._finalize_root_runtime_process_exit(args, exit_code=0)

    assert exc_info.value.code == 0
    assert calls == [
        "mp_finalizers",
        "logging_shutdown",
        ("flight_marker", "root_finalization"),
        ("exit", 0),
    ]


def test_output_gate_routes_background_self_talk_to_secondary():
    from core.utils.output_gate import AutonomousOutputGate

    gate = AutonomousOutputGate()
    target, metadata = gate._foreground_policy(
        "Self-Initiated: Brief Curiosity Scan",
        "system",
        "primary",
        {},
    )
    assert target == "secondary"
    assert metadata["voice"] is False
    assert metadata["suppress_bus"] is True


def test_governance_primitives_fail_closed_when_runtime_active(tmp_path):
    from core.container import ServiceContainer
    from core.governance_context import GovernanceViolation, governed_scope_sync
    from core.runtime.consequential_primitives import guarded_write_text

    old_locked = getattr(ServiceContainer, "_registration_locked", False)
    ServiceContainer._registration_locked = True
    try:
        with pytest.raises(GovernanceViolation):
            guarded_write_text(tmp_path / "blocked.txt", "no receipt")

        class Decision:
            receipt_id = "WR-test"
            domain = "file_write"
            source = "test"

        with governed_scope_sync(Decision()):
            guarded_write_text(tmp_path / "allowed.txt", "ok")
        assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "ok"
    finally:
        ServiceContainer._registration_locked = old_locked


def test_memory_benchmark_graph_selective_reduces_tokens():
    from core.memory.memory_benchmarking import (
        GraphMemoryIndex,
        MemoryBenchmarkCase,
        MemoryBenchmarkRunner,
        MemoryScope,
        ScopedMemoryRecord,
    )

    index = GraphMemoryIndex()
    index.add(ScopedMemoryRecord("a", "python repair traceback import mlx client", MemoryScope.APPLICATION, "bryan", "coder", links=("b",)))
    index.add(ScopedMemoryRecord("b", "pytest validates deterministic bug packet", MemoryScope.APPLICATION, "bryan", "coder"))
    for idx in range(20):
        index.add(ScopedMemoryRecord(f"noise{idx}", "unrelated memory about dinner plans", MemoryScope.APPLICATION, "bryan", "coder"))
    result = MemoryBenchmarkRunner(index).run([MemoryBenchmarkCase("mlx import traceback repair", ("a", "b"))])
    assert result["graph_selective"].mean_tokens < result["full_context"].mean_tokens


def test_toolweaver_synthetic_flywheel_and_simulator_are_operational(tmp_path):
    from core.embodiment.simulator_bridge import LocalPhysics2DSimulator
    from core.learning.synthetic_data_flywheel import SyntheticDataFlywheel
    from core.tools.toolweaver import ToolSpec, ToolWeaverIndex

    index = ToolWeaverIndex()
    index.fit([
        ToolSpec("pytest", "run python tests", ("test", "code")),
        ToolSpec("web_read", "read web pages", ("research", "browser")),
    ])
    assert index.retrieve("run code test")[0].name == "pytest"

    traces = SyntheticDataFlywheel(tmp_path).generate_from_success(
        {"id": "s1", "task": "fix import", "output": "import added", "score": 0.95, "task_type": "repair"},
        variants=4,
    )
    assert len(traces) == 4
    assert SyntheticDataFlywheel(tmp_path).write_jsonl(traces).exists()

    sim = LocalPhysics2DSimulator()
    start = sim.reset(seed=1).distance
    end = sim.rollout(steps=20)[-1].distance
    assert end < start


def test_activation_auditor_reconciles_safe_custom_loop():
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec

    state = {"started": False}

    async def starter(_orch):
        state["started"] = True
        return {"ok": True}

    auditor = ActivationAuditor((ActivationSpec("custom", required=True, auto_start=True, starter=starter),))
    report = asyncio.run(auditor.audit(reconcile=True))
    assert state["started"]
    assert report.statuses[0].reconciled


def test_scheduler_activation_spec_matches_live_main_loop_name():
    from core.runtime.activation_audit import DEFAULT_SPECS

    spec = next(item for item in DEFAULT_SPECS if item.name == "scheduler")

    assert spec.required is True
    assert spec.auto_start is True
    assert spec.starter is not None
    assert "aura.scheduler.main_loop" in spec.task_name_contains


def test_scheduler_activation_detects_existing_scheduler_singleton_loop():
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec
    from core.scheduler import scheduler

    class FakeTask:
        def done(self):
            return False

    old_task = getattr(scheduler, "_main_loop_task", None)
    try:
        scheduler._main_loop_task = FakeTask()
        auditor = ActivationAuditor(
            (
                ActivationSpec(
                    name="scheduler",
                    task_name_contains=("missing-tracker-name",),
                    required=True,
                ),
            )
        )

        report = asyncio.run(auditor.audit())

        assert report.statuses[0].active
        assert report.statuses[0].evidence["scheduler_main_loop_active"] is True
        assert report.statuses[0].evidence["scheduler_task_source"] == "scheduler_singleton"
    finally:
        scheduler._main_loop_task = old_task


def test_activation_audit_rejects_distinct_service_alias_owners():
    from core.container import ServiceContainer
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance("canonical_test_service", object(), required=False)
        ServiceContainer.register_instance("legacy_test_alias", object(), required=False)
        auditor = ActivationAuditor(
            (
                ActivationSpec(
                    name="owned_service",
                    service_keys=("canonical_test_service", "legacy_test_alias"),
                    required=True,
                ),
            )
        )

        status = asyncio.run(auditor.audit()).statuses[0]

        assert status.active is False
        assert status.ownership_conflict is True
        assert status.evidence["ownership_conflict"]["distinct_service_owners"] == 2
    finally:
        ServiceContainer.clear()


def test_activation_audit_rejects_duplicate_long_lived_task_owners(monkeypatch):
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec

    class FakeTask:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

        def done(self):
            return False

    tracker = SimpleNamespace(
        tasks={
            FakeTask("aura.owned_loop"),
            FakeTask("aura.owned_loop"),
        }
    )
    monkeypatch.setattr("core.utils.task_tracker.get_task_tracker", lambda: tracker)
    auditor = ActivationAuditor(
        (
            ActivationSpec(
                name="owned_loop",
                task_name_contains=("aura.owned_loop",),
                required=True,
            ),
        )
    )

    status = asyncio.run(auditor.audit()).statuses[0]

    assert status.active is False
    assert status.ownership_conflict is True
    assert status.evidence["ownership_conflict"]["distinct_task_owners"] == 2


def test_activation_audit_rejects_registered_but_unready_service():
    from core.container import ServiceContainer
    from core.runtime.activation_audit import ActivationAuditor, ActivationSpec

    class UnreadyOutputGate:
        @staticmethod
        def is_ready():
            return False

    ServiceContainer.clear()
    try:
        ServiceContainer.register_instance(
            "output_gate",
            UnreadyOutputGate(),
            required=False,
        )
        auditor = ActivationAuditor(
            (
                ActivationSpec(
                    name="output_gate",
                    service_keys=("output_gate",),
                    required=True,
                ),
            )
        )

        status = asyncio.run(auditor.audit()).statuses[0]

        assert status.active is False
        assert status.ownership_conflict is False
        assert status.evidence["service_liveness"]["output_gate"]["ok"] is False
    finally:
        ServiceContainer.clear()


def test_caa_validator_reads_existing_vector_artifacts():
    from training.caa_32b_validation import CAA32BValidator

    report = CAA32BValidator(vectors_dir=Path("training/vectors")).run()
    assert report["vector_count"] > 0
    assert "activation_vectors_present" in report["pass_conditions"]


def test_the_reaper_reads_the_program_not_its_arguments():
    """14 November 2026, 11:11: her boot SIGKILLed a test run's pytest in another worktree because its arguments
    named tests/test_mlx_worker_*.py among 456 files."""
    import aura_main

    python = "/opt/homebrew/bin/python3.12"
    a_test_run = [python, "-m", "pytest", "tests/test_aura_main_boot.py", "tests/test_gui_actor.py",
                  "tests/test_mlx_worker_contract.py", "-rfE"]
    assert aura_main._is_reapable_aura_process_command(a_test_run) is False
    assert aura_main._is_reapable_aura_process_command(" ".join(a_test_run)) is False
    assert aura_main._is_reapable_aura_process_command(["grep", "aura_main.py", "notes.txt"]) is False

    assert aura_main._is_reapable_aura_process_command(
        [python, "-u", "/Users/x/.aura/live-source/aura_main.py", "--desktop"]) is True
    assert aura_main._is_reapable_aura_process_command([python, "-X", "faulthandler", "gui_actor.py"]) is True
    assert aura_main._is_reapable_aura_process_command([python, "-m", "core.brain.llm.mlx_worker"]) is True
    assert aura_main._is_reapable_aura_process_command(["MLXWorker-Aura-Qwen3.8-27B"]) is True
