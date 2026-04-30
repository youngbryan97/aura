from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from core.architect.behavior_fingerprint import BehaviorFingerprinter
from core.architect.behavior_oracle import SemanticBehaviorOracle
from core.architect.code_graph import LiveArchitectureGraphBuilder
from core.architect.config import ASAConfig
from core.architect.ghost_boot import GhostBootRunner
from core.architect.governor import AutonomousArchitectureGovernor
from core.architect.models import (
    BehaviorDelta,
    MutationTier,
    ProofReceipt,
    ProofResult,
    RefactorPlan,
    RefactorStep,
    RollbackPacket,
    SemanticSurface,
)
from core.architect.mutation_classifier import MutationClassifier
from core.architect.post_promotion_monitor import PostPromotionMonitor
from core.architect.proof_obligations import ProofVerifier
from core.architect.promotion_governor import PromotionGovernor
from core.architect.quarantine import QuarantineManager
from core.architect.refactor_planner import RefactorPlanner, apply_unused_import_cleanup, plan_from_dict, plan_to_dict
from core.architect.rollback_manager import RollbackManager
from core.architect.semantic_classifier import SemanticClassifier
from core.architect.shadow_workspace import ShadowWorkspaceManager
from core.architect.smell_detector import SmellDetector


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    write(tmp_path / "pkg" / "__init__.py", "")
    write(
        tmp_path / "pkg" / "mod.py",
        "import os\nimport math\n\nclass Worker:\n    async def run(self):\n        return math.sqrt(4)\n\ndef used():\n    return Worker\n",
    )
    write(tmp_path / "tests" / "test_mod.py", "from pkg.mod import used\n\ndef test_used():\n    assert used().__name__ == 'Worker'\n")
    write(tmp_path / "OWNERSHIP.md", "| Concern | Owner | Role | File |\n| Work | `Worker` | owner | `pkg/mod.py` |\n")
    return tmp_path


def config(root: Path, **kwargs) -> ASAConfig:
    return ASAConfig(repo_root=root, shadow_timeout=8.0, observation_window=0.01, artifact_root=root / ".aura_architect", **kwargs)


def graph_for(root: Path):
    return LiveArchitectureGraphBuilder(config(root)).build(persist=True)


def test_code_graph_builds_basic_repo(tiny_repo: Path) -> None:
    graph = graph_for(tiny_repo)
    assert graph.metrics["files"] >= 3
    assert (tiny_repo / ".aura_architect" / "architecture_graph.json").exists()
    assert graph.ownership["pkg/mod.py"].owner == "Worker"


def test_code_graph_detects_imports_classes_functions_async(tiny_repo: Path) -> None:
    graph = graph_for(tiny_repo)
    kinds = {node.kind for node in graph.nodes.values()}
    assert {"file", "class", "method", "function"} <= kinds
    assert any(edge.kind == "imports" and "math" in edge.target for edge in graph.edges)
    assert any(node.kind == "method" and node.name == "run" for node in graph.nodes.values())


def test_code_graph_detects_service_container_patterns(tmp_path: Path) -> None:
    write(tmp_path / "svc.py", "from core.container import ServiceContainer\nx = ServiceContainer.get('will', default=None)\nServiceContainer.register_instance('x', object())\n")
    graph = graph_for(tmp_path)
    effects = graph.effects_for_path("svc.py")
    assert "service_container_get" in effects
    assert "service_get_default_none" in effects
    assert "service_container_register" in effects


def test_code_graph_detects_event_bus_patterns_if_present(tmp_path: Path) -> None:
    write(tmp_path / "bus.py", "async def f(bus):\n    await bus.publish('x', {})\n    await bus.subscribe('x')\n")
    graph = graph_for(tmp_path)
    effects = graph.effects_for_path("bus.py")
    assert "event_bus_emit" in effects
    assert "event_bus_subscribe" in effects


def test_semantic_classifier_marks_authority_memory_state_llm_surfaces() -> None:
    classifier = SemanticClassifier()
    assert SemanticSurface.AUTHORITY_GOVERNANCE in classifier.classify_path("core/executive/authority_gateway.py")
    assert SemanticSurface.MEMORY_WRITE_READ in classifier.classify_path("core/memory/episodic_memory.py")
    assert SemanticSurface.STATE_MUTATION in classifier.classify_path("core/state/state_gateway.py")
    assert SemanticSurface.LLM_MODEL_ROUTING in classifier.classify_path("core/brain/llm/llm_router.py")


def test_mutation_classifier_uses_highest_risk_surface(tiny_repo: Path) -> None:
    classifier = MutationClassifier(config(tiny_repo))
    tier = classifier.classify(("pkg/mod.py",), surfaces=(SemanticSurface.AUTHORITY_GOVERNANCE,))
    assert tier == MutationTier.T4_GOVERNANCE_SENSITIVE


def test_t4_t5_are_proposal_only(tiny_repo: Path) -> None:
    classifier = MutationClassifier(config(tiny_repo))
    assert classifier.classify_path("core/executive/authority_gateway.py") == MutationTier.T4_GOVERNANCE_SENSITIVE
    assert classifier.classify_path("core/architect/proof_obligations.py") == MutationTier.T5_SEALED


def test_smell_detector_detects_god_file(tmp_path: Path) -> None:
    write(tmp_path / "large.py", "\n".join(f"x{i} = {i}" for i in range(12)))
    cfg = config(tmp_path, god_file_lines=5)
    graph = LiveArchitectureGraphBuilder(cfg).build(persist=False)
    smells = SmellDetector(cfg).detect(graph)
    assert any(smell.kind == "god_file" for smell in smells)


def test_smell_detector_detects_broad_exception_cluster(tmp_path: Path) -> None:
    write(tmp_path / "bad.py", "def f():\n    try:\n        return 1\n    except Exception:\n        return 2\n")
    smells = SmellDetector(config(tmp_path)).detect(graph_for(tmp_path))
    assert any(smell.kind == "broad_exception_cluster" for smell in smells)


def test_smell_detector_detects_authority_bypass_pattern(tmp_path: Path) -> None:
    write(tmp_path / "tool.py", "def f():\n    return execute_tool('x')\n")
    smells = SmellDetector(config(tmp_path)).detect(graph_for(tmp_path))
    assert any(smell.kind == "tool_authority_bypass" for smell in smells)


def test_smell_detector_detects_state_write_bypass_pattern(tmp_path: Path) -> None:
    write(tmp_path / "statey.py", "def f(state_repo):\n    return state_repo.write('x', 1)\n")
    smells = SmellDetector(config(tmp_path)).detect(graph_for(tmp_path))
    assert any(smell.kind == "state_write_bypass" for smell in smells)


def test_duplicate_responsibility_detector_finds_similar_functions(tmp_path: Path) -> None:
    write(tmp_path / "a.py", "def normalize_memory_key(value):\n    '''normalize memory key'''\n    return str(value).strip().lower()\n")
    write(tmp_path / "b.py", "def normalise_memory_key(value):\n    '''normalize memory key'''\n    return str(value).strip().lower()\n")
    smells = SmellDetector(config(tmp_path)).detect(graph_for(tmp_path))
    assert any(smell.kind in {"duplicate_implementation", "duplicate_responsibility"} for smell in smells)


def test_refactor_plan_has_steps_invariants_rollback(tiny_repo: Path) -> None:
    graph = graph_for(tiny_repo)
    plan = RefactorPlanner(config(tiny_repo)).find_auto_cleanup_plan(graph, [])
    assert plan is not None
    assert plan.steps
    assert plan.steps[0].invariants
    assert plan.steps[0].rollback


def test_shadow_workspace_does_not_modify_live_repo(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    graph = graph_for(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph, [])
    before = (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    assert (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8") == before
    assert "import os" not in (Path(shadow.shadow_root) / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_ghost_boot_fails_closed_on_unavailable_boot(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    graph = graph_for(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph, [])
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    report = GhostBootRunner(cfg).run(plan, shadow)
    assert report.result_map()["safe_boot"].status == "BOOT_HARNESS_UNAVAILABLE"
    assert "not present" in report.result_map()["safe_boot"].evidence["reason"]


def test_default_safe_boot_harness_runs_in_aura_repo() -> None:
    proc = subprocess.run(
        [sys.executable, "-B", "-m", "core.architect.safe_boot_harness"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.splitlines()[-1])
    assert payload["checks"]["architecture_governor_singleton"] is True
    assert payload["checks"]["architecture_manifest"] is True


def test_ghost_boot_succeeds_on_safe_import_smoke(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    graph = graph_for(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph, [])
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    report = GhostBootRunner(cfg).run(plan, shadow)
    assert report.result_map()["changed_modules_import"].passed


def test_behavior_fingerprint_compares_before_after(tiny_repo: Path) -> None:
    fp = BehaviorFingerprinter(config(tiny_repo))
    before = fp.capture(root=tiny_repo, changed_files=("pkg/mod.py",))
    after = fp.capture(root=tiny_repo, changed_files=("pkg/mod.py",))
    delta = fp.compare(before, after)
    assert isinstance(delta, BehaviorDelta)
    assert delta.equivalent


def test_semantic_oracle_rejects_t2_public_api_removal(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "mod.py", "def public_api(value):\n    return value\n")
    cfg = config(tmp_path)
    before = LiveArchitectureGraphBuilder(cfg).build(persist=False)
    write(tmp_path / "pkg" / "mod.py", "VALUE = 1\n")
    after = LiveArchitectureGraphBuilder(cfg).build(persist=False)
    plan = RefactorPlan(
        id="t2-api-removal",
        objective="remove api",
        risk_tier=MutationTier.T2_REFACTOR,
        affected_files=("pkg/mod.py",),
        affected_symbols=("public_api",),
        semantic_surfaces=(SemanticSurface.UTILITY_PERIPHERAL,),
        steps=(RefactorStep(id="s", description="edit", operation="replace_file", target_path="pkg/mod.py", new_content="VALUE = 1\n"),),
        proof_obligations=("behavior_equivalence",),
        expected_smell_reduction=(),
        expected_behavior_delta="equivalent",
        promotion_eligible=True,
    )
    result = SemanticBehaviorOracle().evaluate(
        plan,
        before,
        after,
        {"safe_boot": "passed", "changed_modules_import": "passed", "critical_tests": "passed"},
    ).as_proof_result()
    assert not result.passed
    assert "public symbols removed" in result.evidence["regressions"][0]


def test_runtime_receipt_ingestion_from_live_receipt_envelope(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "mod.py", "def f():\n    return 1\n")
    receipt = {
        "schema_version": 1,
        "payload": {
            "kind": "tool_execution",
            "created_at": 123.0,
            "source": "test",
            "metadata": {"path": "pkg/mod.py"},
        },
    }
    write(tmp_path / "data" / "receipts" / "tool_execution" / "tool-1.json", json.dumps(receipt))
    graph = LiveArchitectureGraphBuilder(config(tmp_path)).build(persist=False)
    assert graph.metrics["runtime_receipts"] >= 1
    assert graph.metrics["runtime_receipts_by_kind"]["tool_execution"] == 1
    assert any(receipt.path == "pkg/mod.py" for receipt in graph.runtime_receipts)


def test_coverage_backed_dead_code_telemetry_suppresses_dead_symbol(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "mod.py", "def maybe_runtime_only():\n    return 1\n")
    write(tmp_path / "coverage.json", json.dumps({"files": {"pkg/mod.py": {"executed_lines": [1, 2]}}}))
    graph = LiveArchitectureGraphBuilder(config(tmp_path)).build(persist=False)
    smells = SmellDetector(config(tmp_path)).detect(graph)
    assert not any(smell.kind == "dead_symbol_candidate" and smell.symbol == "maybe_runtime_only" for smell in smells)


def test_boot_audit_writes_high_risk_t4_proposals(tmp_path: Path) -> None:
    write(tmp_path / "pkg" / "__init__.py", "")
    write(tmp_path / "pkg" / "tool.py", "def f():\n    return execute_tool('x')\n")
    cfg = config(tmp_path)
    report = AutonomousArchitectureGovernor(cfg).boot_audit(proposal_limit=2)
    assert report["high_risk_proposals"]
    proposal_id = report["high_risk_proposals"][0]["plan_id"]
    assert (tmp_path / ".aura_architect" / "proposals" / f"{proposal_id}.json").exists()


def test_proof_obligation_rejects_missing_rollback(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    ghost = GhostBootRunner(cfg).run(plan, shadow)
    proof = ProofVerifier(cfg).verify(plan, ghost, None, baseline_root=tiny_repo, candidate_root=Path(shadow.shadow_root))
    assert not proof.passed
    assert any(result.obligation_id == "rollback_packet_created" and not result.passed for result in proof.results)


def test_proof_obligation_rejects_sealed_edit(tmp_path: Path) -> None:
    write(tmp_path / "core" / "architect" / "x.py", "VALUE = 1\n")
    cfg = config(tmp_path)
    plan = RefactorPlan(
        id="sealed-plan",
        objective="sealed edit",
        risk_tier=MutationTier.T5_SEALED,
        affected_files=("core/architect/x.py",),
        affected_symbols=(),
        semantic_surfaces=(SemanticSurface.SELF_MODIFICATION,),
        steps=(RefactorStep(id="s", description="edit", operation="replace_file", target_path="core/architect/x.py", new_content="VALUE = 2\n"),),
        proof_obligations=("syntax",),
        expected_smell_reduction=(),
        expected_behavior_delta="equivalent",
        promotion_eligible=False,
    )
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    ghost = GhostBootRunner(cfg).run(plan, shadow)
    rollback = RollbackManager(cfg).dry_run(RollbackManager(cfg).create_packet(plan, shadow))
    proof = ProofVerifier(cfg).verify(plan, ghost, rollback, baseline_root=tmp_path, candidate_root=Path(shadow.shadow_root))
    assert any(result.obligation_id == "no_sealed_surface_autonomous_edit" and not result.passed for result in proof.results)


def test_rollback_packet_restores_original_content(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    manager = RollbackManager(cfg)
    packet = manager.dry_run(manager.create_packet(plan, shadow))
    (tiny_repo / "pkg" / "mod.py").write_text("broken = True\n", encoding="utf-8")
    restored = manager.restore(packet)
    assert restored.post_restore_verified
    assert "import os" in (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_promotion_governor_rejects_failed_proof(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    shadow = ShadowWorkspaceManager(cfg).create(plan)
    rollback = RollbackManager(cfg).dry_run(RollbackManager(cfg).create_packet(plan, shadow))
    proof = ProofReceipt(
        run_id=shadow.run_id,
        plan_id=plan.id,
        tier=plan.risk_tier,
        results=(ProofResult("x", False, "failed"),),
        behavior_delta=BehaviorDelta(False, False, ("regression",), ()),
        rollback_packet_hash=rollback.receipt_hash,
        shadow_artifact_path=shadow.artifact_dir,
    ).signed()
    decision = PromotionGovernor(cfg).decide(plan, proof, rollback)
    assert decision.status.value == "rejected"


def test_promotion_governor_promotes_t0_t1_when_all_proofs_pass(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    governor = AutonomousArchitectureGovernor(cfg)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    shadow, ghost, rollback, proof = governor.shadow_run(plan)
    decision = governor.promote(plan, shadow, proof, rollback)
    assert decision.status.value == "promoted"
    assert "import os" not in (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_post_promotion_monitor_triggers_rollback_on_regression(tiny_repo: Path) -> None:
    cfg = config(tiny_repo)
    governor = AutonomousArchitectureGovernor(cfg)
    plan = RefactorPlanner(cfg).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    shadow, ghost, rollback, proof = governor.shadow_run(plan)
    decision = governor.promote(plan, shadow, proof, rollback)
    assert decision.status.value == "promoted"
    write(tiny_repo / "pkg" / "mod.py", "def broken(:\n")
    observation = PostPromotionMonitor(cfg).check_once(decision.run_id)
    assert observation.rollback_triggered
    assert "import os" in (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_cli_audit_runs(tiny_repo: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "core.architect.cli", "--repo", str(tiny_repo), "audit"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["graph_metrics"]["files"] >= 3


def test_cli_auto_t1_runs_in_temp_repo(tiny_repo: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "core.architect.cli", "--repo", str(tiny_repo), "auto", "--tier-max", "T1"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "promoted"
    assert "import os" not in (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")


def test_no_broad_exception_swallowing_in_architect_core() -> None:
    root = Path(__file__).resolve().parents[2] / "core" / "architect"
    offenders = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                name = ""
                if node.type is None:
                    name = "bare"
                elif isinstance(node.type, ast.Name):
                    name = node.type.id
                if name in {"bare", "Exception", "BaseException"}:
                    offenders.append(f"{path.name}:{node.lineno}:{name}")
    assert offenders == []


def test_no_stub_notimplemented_in_architect_core() -> None:
    root = Path(__file__).resolve().parents[2] / "core" / "architect"
    forbidden = ("coming soon", "pretend", "mock install", "notimplemented", "to" + "do")
    hits = []
    for path in root.glob("*.py"):
        original = path.read_text(encoding="utf-8")
        text = original.lower()
        tree = ast.parse(original, filename=str(path))
        if any(isinstance(node, ast.Pass) for node in ast.walk(tree)):
            hits.append(f"{path.name}:pass")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.name}:{token}")
    assert hits == []


def test_quarantine_manifest_roundtrip(tiny_repo: Path) -> None:
    manager = QuarantineManager(config(tiny_repo))
    manifest = manager.quarantine_file("pkg/mod.py", reason="dead candidate", graph_evidence=("node:1",), proof_run="run-1")
    loaded = manager.load_manifest(manifest.quarantine_id)
    assert loaded.original_hash == manifest.original_hash
    restored = manager.restore(manifest.quarantine_id, destination="pkg/restored.py")
    assert restored.exists()


def test_architecture_graph_persistence_roundtrip(tiny_repo: Path) -> None:
    graph = graph_for(tiny_repo)
    loaded = type(graph).load_json(tiny_repo / ".aura_architect" / "architecture_graph.json")
    assert loaded.metrics["files"] == graph.metrics["files"]
    assert set(loaded.nodes) == set(graph.nodes)


def test_proof_receipt_is_hash_stable() -> None:
    delta = BehaviorDelta(True, False)
    receipt1 = ProofReceipt("run", "plan", MutationTier.T1_CLEANUP, (ProofResult("x", True, "passed"),), delta, "rb", "shadow").signed()
    receipt2 = ProofReceipt("run", "plan", MutationTier.T1_CLEANUP, (ProofResult("x", True, "passed"),), delta, "rb", "shadow").signed()
    assert receipt1.decision_hash == receipt2.decision_hash


def test_unused_import_cleanup_preserves_used_import(tiny_repo: Path) -> None:
    source = (tiny_repo / "pkg" / "mod.py").read_text(encoding="utf-8")
    cleaned = apply_unused_import_cleanup(source)
    assert "import os" not in cleaned
    assert "import math" in cleaned


def test_plan_serialization_roundtrip(tiny_repo: Path) -> None:
    plan = RefactorPlanner(config(tiny_repo)).find_auto_cleanup_plan(graph_for(tiny_repo), [])
    loaded = plan_from_dict(plan_to_dict(plan))
    assert loaded.id == plan.id
    assert loaded.steps[0].target_path == plan.steps[0].target_path
