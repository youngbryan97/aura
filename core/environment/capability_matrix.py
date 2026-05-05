"""Executable capability audit for general environment autonomy.

This module turns the "could this architecture survive a NetHack-scale
environment?" question into runnable checks over the live kernel. The checks
are deliberately environment-agnostic: they verify observation, belief, policy,
action, learning, trace, lifecycle, and benchmark organs rather than any game
mechanic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CapabilityRequirement:
    key: str
    name: str
    rationale: str


@dataclass
class CapabilityFinding:
    key: str
    passed: bool
    detail: str = ""


@dataclass
class CapabilityAuditReport:
    total: int
    passed: int
    findings: list[CapabilityFinding] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 1.0

    @property
    def missing(self) -> list[CapabilityFinding]:
        return [finding for finding in self.findings if not finding.passed]

    def require_clean(self) -> None:
        if self.missing:
            missing = ", ".join(finding.key for finding in self.missing)
            raise RuntimeError(f"environment_capability_audit_failed:{missing}")


REQUIREMENTS: tuple[CapabilityRequirement, ...] = (
    CapabilityRequirement("adapter_contract", "Bounded Adapter Contract", "Every environment must expose observe/execute/close and typed capabilities."),
    CapabilityRequirement("state_compiler", "Observation Normalization", "Raw sensory data must compile into a shared ParsedState ontology."),
    CapabilityRequirement("belief_graph", "Persistent Belief Graph", "Long-horizon tasks require durable entities, contexts, frontiers, hazards, and contradictions."),
    CapabilityRequirement("canonical_spatial", "Canonical Spatial/Topological Memory", "The system needs one writable map source to avoid split-brain control."),
    CapabilityRequirement("homeostasis", "Resource Homeostasis", "Survival and service health depend on resource extraction, trends, and emergency bias."),
    CapabilityRequirement("policy_stack", "Policy Stack", "Candidate generation, simulation, ranking, and strategic planning must be connected."),
    CapabilityRequirement("command_compiler", "Semantic Command Compilation", "Policies emit intents; adapters execute compiled commands, never raw prose."),
    CapabilityRequirement("action_semantics", "Closed-Loop Action Semantics", "Compiled commands need preconditions, predicted effects, reversibility, and uncertainty checks."),
    CapabilityRequirement("action_gateway", "Effect Gate", "All actions pass through legality, modal, risk, uncertainty, and loop checks."),
    CapabilityRequirement("action_budget", "Action Budget", "Long runs need bounded unknown, irreversible, repeated-failure, modal, and resource-cost accounting."),
    CapabilityRequirement("modal_manager", "Modal State Machine", "Blocking prompts, dialogs, menus, and confirmations must suspend normal policy."),
    CapabilityRequirement("semantic_diff", "Outcome Diffing", "The kernel must compare pre/post state semantically rather than by raw pixels/text alone."),
    CapabilityRequirement("outcome_learning", "Outcome Learning", "Action outcomes must feed durable ledger/procedural/competence stores."),
    CapabilityRequirement("replay_learning", "Hindsight Replay Learning", "Failures must become reusable causal policy rules instead of isolated memories."),
    CapabilityRequirement("abstraction_discovery", "Abstraction Discovery", "Repeated surprise patterns must induce transferable categories without hand-coded domain labels."),
    CapabilityRequirement("curriculum_engine", "Open-Ended Curriculum", "Bottlenecks should create self-generated practice tasks, not only reactive fixes."),
    CapabilityRequirement("trace_replay", "Black-Box Trace", "Every control step needs replayable observation->belief->intent->command->outcome receipts."),
    CapabilityRequirement("run_lifecycle", "Run Lifecycle", "Runs must start, terminate, postmortem, and preserve cross-run learning."),
    CapabilityRequirement("governance_bridge", "Governance Bridge", "Risky effects must connect to will/authority receipts."),
    CapabilityRequirement("benchmark_integrity", "Benchmark Integrity", "Strict-real, simulated, and fixture runs must stay distinguishable."),
    CapabilityRequirement("external_task_proof", "External Task Proof Gate", "Placeholder scaffolds and canaries must not be counted as broad task wins."),
)


class EnvironmentCapabilityMatrix:
    """Evaluate whether a live kernel has the required general organs."""

    requirements = REQUIREMENTS

    def audit(self, kernel: Any) -> CapabilityAuditReport:
        findings = [self._check(kernel, requirement) for requirement in self.requirements]
        return CapabilityAuditReport(
            total=len(findings),
            passed=sum(1 for finding in findings if finding.passed),
            findings=findings,
        )

    def _check(self, kernel: Any, requirement: CapabilityRequirement) -> CapabilityFinding:
        checker = getattr(self, f"_check_{requirement.key}")
        try:
            ok, detail = checker(kernel)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        return CapabilityFinding(requirement.key, bool(ok), detail)

    @staticmethod
    def _has_methods(obj: Any, names: tuple[str, ...]) -> bool:
        return all(callable(getattr(obj, name, None)) for name in names)

    def _check_adapter_contract(self, kernel: Any) -> tuple[bool, str]:
        adapter = getattr(kernel, "adapter", None)
        ok = adapter is not None and self._has_methods(adapter, ("start", "observe", "execute", "close", "is_alive"))
        return ok, adapter.__class__.__name__ if adapter is not None else "missing adapter"

    def _check_state_compiler(self, kernel: Any) -> tuple[bool, str]:
        compiler = getattr(kernel, "state_compiler", None)
        return compiler is not None and callable(getattr(compiler, "compile", None)), compiler.__class__.__name__ if compiler else "missing"

    def _check_belief_graph(self, kernel: Any) -> tuple[bool, str]:
        belief = getattr(kernel, "belief", None)
        ok = belief is not None and all(hasattr(belief, attr) for attr in ("nodes", "edges", "frontiers", "hazards", "contradictions"))
        return ok, f"nodes={len(getattr(belief, 'nodes', {}))}" if belief else "missing"

    def _check_canonical_spatial(self, kernel: Any) -> tuple[bool, str]:
        belief = getattr(kernel, "belief", None)
        ok = belief is not None and hasattr(belief, "spatial") and callable(getattr(belief, "upsert_spatial", None))
        return ok, f"cells={len(getattr(belief, 'spatial', {}))}" if belief else "missing"

    def _check_homeostasis(self, kernel: Any) -> tuple[bool, str]:
        homeostasis = getattr(kernel, "homeostasis", None)
        ok = homeostasis is not None and self._has_methods(homeostasis, ("extract", "assess"))
        return ok, homeostasis.__class__.__name__ if homeostasis else "missing"

    def _check_policy_stack(self, kernel: Any) -> tuple[bool, str]:
        policy = getattr(kernel, "policy", None)
        ok = policy is not None and all(hasattr(policy, attr) for attr in ("candidate_generator", "action_ranker", "strategic_policy", "tactical_policy", "simulator"))
        return ok, policy.__class__.__name__ if policy else "missing"

    def _check_command_compiler(self, kernel: Any) -> tuple[bool, str]:
        compiler = getattr(kernel, "command_compiler", None)
        ok = compiler is not None and callable(getattr(compiler, "compile", None)) and bool(getattr(compiler, "_handlers", {}))
        return ok, f"handlers={len(getattr(compiler, '_handlers', {}))}" if compiler else "missing"

    def _check_action_semantics(self, kernel: Any) -> tuple[bool, str]:
        semantics = getattr(kernel, "action_semantics", None)
        ok = semantics is not None and callable(getattr(semantics, "validate", None))
        return ok, semantics.__class__.__name__ if semantics else "missing"

    def _check_action_gateway(self, kernel: Any) -> tuple[bool, str]:
        gateway = getattr(kernel, "gateway", None)
        ok = gateway is not None and self._has_methods(gateway, ("approve", "record_failure"))
        return ok, gateway.__class__.__name__ if gateway else "missing"

    def _check_action_budget(self, kernel: Any) -> tuple[bool, str]:
        budget = getattr(kernel, "action_budget", None)
        ok = budget is not None and self._has_methods(budget, ("record", "exhausted_reasons"))
        return ok, f"steps={getattr(budget, 'used_total_steps', 0)}" if budget else "missing"

    def _check_modal_manager(self, kernel: Any) -> tuple[bool, str]:
        modal = getattr(kernel, "modal_manager", None)
        ok = modal is not None and self._has_methods(modal, ("resolve", "should_block_normal_policy"))
        return ok, modal.__class__.__name__ if modal else "missing"

    def _check_semantic_diff(self, kernel: Any) -> tuple[bool, str]:
        diff = getattr(kernel, "semantic_diff", None)
        ok = diff is not None and self._has_methods(diff, ("compute_diff", "learn_from_transition"))
        return ok, diff.__class__.__name__ if diff else "missing"

    def _check_outcome_learning(self, kernel: Any) -> tuple[bool, str]:
        ok = all(hasattr(kernel, attr) for attr in ("outcomes", "outcome_ledger", "procedural_store", "competence_tracker"))
        return ok, "ledger+procedural+competence" if ok else "missing learning store"

    def _check_replay_learning(self, kernel: Any) -> tuple[bool, str]:
        replay = getattr(kernel, "replay_buffer", None)
        ok = replay is not None and self._has_methods(replay, ("add_transition", "applicable_rules")) and hasattr(replay, "rules")
        return ok, f"rules={len(getattr(replay, 'rules', {}) or {})}" if replay else "missing"

    def _check_abstraction_discovery(self, kernel: Any) -> tuple[bool, str]:
        discovery = getattr(kernel, "abstraction_discovery", None)
        ok = discovery is not None and callable(getattr(discovery, "observe_transition", None)) and hasattr(discovery, "abstractions")
        return ok, f"abstractions={len(getattr(discovery, 'abstractions', {}) or {})}" if discovery else "missing"

    def _check_curriculum_engine(self, kernel: Any) -> tuple[bool, str]:
        curriculum = getattr(kernel, "curriculum", None)
        ok = curriculum is not None and self._has_methods(curriculum, ("record_result", "propose_next_task"))
        return ok, curriculum.__class__.__name__ if curriculum else "missing"

    def _check_trace_replay(self, kernel: Any) -> tuple[bool, str]:
        blackbox = getattr(kernel, "blackbox", None)
        ok = blackbox is not None and callable(getattr(blackbox, "record", None))
        return ok, blackbox.__class__.__name__ if blackbox else "missing"

    def _check_run_lifecycle(self, kernel: Any) -> tuple[bool, str]:
        ok = hasattr(kernel, "episode") and self._has_methods(getattr(kernel, "run_manager", None), ("start_run", "record_step", "end_run", "detect_death"))
        return ok, getattr(getattr(kernel, "run_manager", None), "__class__", type("", (), {})).__name__

    def _check_governance_bridge(self, kernel: Any) -> tuple[bool, str]:
        bridge = getattr(kernel, "governance_bridge", None)
        ok = bridge is not None and callable(getattr(bridge, "decide_action", None))
        return ok, bridge.__class__.__name__ if bridge else "missing"

    def _check_benchmark_integrity(self, kernel: Any) -> tuple[bool, str]:
        guard = getattr(kernel, "boundary_guard", None)
        manager = getattr(kernel, "run_manager", None)
        ok = guard is not None and manager is not None and hasattr(manager, "mode")
        return ok, f"mode={getattr(manager, 'mode', 'unknown')}" if manager else "missing"

    def _check_external_task_proof(self, kernel: Any) -> tuple[bool, str]:
        gate = getattr(kernel, "external_proof_gate", None)
        ok = gate is not None and callable(getattr(gate, "evaluate_kernel", None))
        return ok, gate.__class__.__name__ if gate else "missing"


__all__ = [
    "CapabilityAuditReport",
    "CapabilityFinding",
    "CapabilityRequirement",
    "EnvironmentCapabilityMatrix",
    "REQUIREMENTS",
]
