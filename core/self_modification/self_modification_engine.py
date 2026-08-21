"""Autonomous Self-Modification Engine
Orchestrates the complete self-improvement system.
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import FallbackClassification, Severity, record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.task_ownership import close_awaitable
from core.utils.task_tracker import get_task_tracker

from .boot_validator import GhostBootValidator
from .code_repair import AutonomousCodeRepair

# Import all subsystems
from .error_intelligence import ErrorIntelligenceSystem
from .kernel_refiner import KernelRefiner
from .learning_system import MetaLearning, SelfImprovementLearning
from .mutation_tiers import classify_mutation_path
from .promotion_policy import (
    AUTONOMOUS_PATCH_PROMOTION_ENV as _AUTONOMOUS_PATCH_PROMOTION_ENV,
)
from .promotion_policy import (
    REPAIR_LAB_SOURCE_PROMOTION_ENV as _REPAIR_LAB_SOURCE_PROMOTION_ENV,
)
from .promotion_policy import RUNTIME_SELF_MODIFICATION_ENV as _RUNTIME_SELF_MODIFICATION_ENV
from .promotion_policy import SUPERVISED_SELF_MODIFICATION_ENV as _SUPERVISED_SELF_MODIFICATION_ENV
from .promotion_policy import (
    autonomous_source_promotion_decision,
    env_flag,
    safe_autonomous_repair_decision,
)
from .repair_registry import append_repair_entry
from .safe_modification import LogicTransplant, SafeSelfModification
from .shadow_ast_healer import ShadowASTHealer
from .shadow_runtime import get_shadow_runtime

logger = logging.getLogger("SelfModification.Engine")

SELF_MOD_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    KeyError,
    IndexError,
)

_NON_FAILURE_REPAIR_DISPOSITIONS = frozenset(
    {
        "proposal_quarantined",
        "proposal_refused_by_policy",
        "proposal_decision_reused",
    }
)


def _autonomous_cycle_failed(result: dict[str, Any]) -> bool:
    """Distinguish a broken repair attempt from a completed policy decision."""

    if result.get("disposition") in _NON_FAILURE_REPAIR_DISPOSITIONS:
        return False
    return result.get("success", True) is not True


def _runtime_patch_promotion_enabled() -> bool:
    """Autonomous source promotion requires an isolated repair-lab opt-in.

    Normal desktop/server Aura sessions may diagnose, propose, quarantine, and
    validate repairs, but they must not overwrite source files from inside the
    running process. Source promotion is reserved for an explicit repair-lab
    profile where the operator has intentionally enabled all three switches.
    """
    return autonomous_source_promotion_decision().allowed


def _supervised_patch_promotion_enabled() -> bool:
    """Manual force=True promotion requires a separate supervised override."""
    return env_flag(_SUPERVISED_SELF_MODIFICATION_ENV, False)


def _record_self_modification_degradation(
    error: BaseException,
    *,
    action: str,
    severity: Severity = "warning",
    receipt_required: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "self_modification_engine",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=receipt_required,
        extra=extra,
    )


def _schedule_background_coro(coro: Any, *, label: str) -> None:
    """Run a coroutine whether or not we're inside a live event loop."""
    if is_shutdown_requested():
        close_awaitable(coro)
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:

        def _runner() -> None:
            try:
                asyncio.run(coro)
            except asyncio.CancelledError:
                logger.debug("%s background runner cancelled", label)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_self_modification_degradation(
                    exc,
                    action="Stopped failed background self-modification task and preserved foreground execution",
                    extra={"label": label, "execution": "thread"},
                )
                logger.debug("%s background runner failed: %s", label, exc)

        thread = threading.Thread(target=_runner, name=f"aura_{label}", daemon=True)
        owned_thread: Any = thread
        owned_thread._aura_shutdown_suppressed_cleanup = lambda: close_awaitable(coro)
        try:
            thread.start()
        except RuntimeError:
            close_awaitable(coro)
            if is_shutdown_requested():
                return
            raise
        return

    task = get_task_tracker().create_task(coro, name=f"self_modification.{label}")

    def _consume_result(done: asyncio.Task) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            logger.debug("%s background task cancelled", label)
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_self_modification_degradation(
                exc,
                action="Consumed failed background self-modification task and preserved event loop stability",
                extra={"label": label, "execution": "task"},
            )
            logger.debug("%s background task failed: %s", label, exc)

    task.add_done_callback(_consume_result)


class AutonomousSelfModificationEngine:
    """Complete autonomous self-modification system.

    Workflow:
    1. Monitor for errors (ErrorIntelligenceSystem)
    2. Detect patterns in errors
    3. Generate diagnoses
    4. Propose fixes (AutonomousCodeRepair)
    5. Validate and test fixes
    6. Apply fixes safely (SafeSelfModification)
    7. Learn from outcomes (SelfImprovementLearning)
    8. Improve fix strategies over time
    """

    def __init__(
        self,
        cognitive_engine,
        code_base_path: str = ".",
        auto_fix_enabled: bool = True,  # Sovereign Overdrive: Enabled by default
    ):
        """Initialize the self-modification engine.

        Args:
            cognitive_engine: LLM for generating diagnoses and fixes
            code_base_path: Root of code repository
            auto_fix_enabled: Whether to automatically apply fixes

        """
        logger.info("Initializing Autonomous Self-Modification Engine...")

        self.brain = cognitive_engine
        self.code_base = Path(code_base_path)
        self._auto_fix_requested = bool(auto_fix_enabled)
        self.auto_fix_enabled = bool(auto_fix_enabled and _runtime_patch_promotion_enabled())

        # Initialize subsystems
        from ..config import config

        self.error_intelligence = ErrorIntelligenceSystem(
            cognitive_engine, log_dir=str(config.paths.data_dir / "error_logs")
        )

        self.code_repair = AutonomousCodeRepair(cognitive_engine, str(self.code_base))

        self.safe_modification = SafeSelfModification(
            str(self.code_base), str(config.paths.data_dir / "modifications.jsonl")
        )

        self.learning_system = SelfImprovementLearning(str(config.paths.data_dir / "learning.json"))

        self.meta_learning = MetaLearning()

        self.kernel_refiner = KernelRefiner(cognitive_engine, str(self.code_base))

        self.boot_validator = GhostBootValidator(self.code_base)
        self.shadow_healer = ShadowASTHealer(self.code_base)
        self.shadow_runtime = get_shadow_runtime(str(self.code_base))

        # Background monitoring
        self.monitoring_enabled = False
        self.monitor_thread = None
        self.monitor_interval = 60  # Check every 60 seconds (unlocked)

        # Statistics
        self.session_stats = {
            "bugs_detected": 0,
            "fixes_attempted": 0,
            "fixes_successful": 0,
            "health_fixes_triggered": 0,
            "background_failures": 0,
            "cycle_failures": 0,
            "refinement_failures": 0,
            "session_start": time.time(),
        }
        self._fix_lock = asyncio.Lock()
        self._repair_event = asyncio.Event()
        self._last_cycle_error: dict[str, Any] | None = None
        self._last_refinement_error: dict[str, Any] | None = None
        self._last_proposal: dict[str, Any] | None = None
        self._last_repair_decision: dict[str, Any] | None = None

        logger.info("✓ Autonomous Self-Modification Engine initialized")

        if auto_fix_enabled and not self.auto_fix_enabled:
            logger.warning(
                "Runtime self-modification promotion DISABLED. Normal Aura sessions "
                "run self-repair in proposal/quarantine mode. Set %s=1, %s=1, and "
                "%s=1 only inside an operator-controlled repair-lab profile to allow "
                "tested patches to be promoted outside explicit supervised force=True "
                "calls.",
                _RUNTIME_SELF_MODIFICATION_ENV,
                _AUTONOMOUS_PATCH_PROMOTION_ENV,
                _REPAIR_LAB_SOURCE_PROMOTION_ENV,
            )
        elif not auto_fix_enabled:
            logger.warning("Auto-fix DISABLED - fixes will be proposed but not applied")

    # ========================================================================
    # Integration with Existing Systems
    # ========================================================================

    def on_error(
        self,
        error: Exception,
        context: dict[str, Any],
        skill_name: str | None = None,
        goal: str | None = None,
    ):
        """Hook for existing error handling.
        Call this whenever an error occurs in your system.

        Example integration:
            try:
                result = await skill.execute(goal, context)
            except (sqlite3.Error, OSError) as e:
                self_mod_engine.on_error(e, context, skill.name, goal)
                raise
        """

        # Log to error intelligence (Async)
        async def _log():
            event = await self.error_intelligence.on_error(error, context, skill_name, goal)
            logger.debug("Error logged: %s", event.fingerprint())
            self._repair_event.set()

        _schedule_background_coro(_log(), label="self_mod_error_log")

    def on_skill_execution(
        self, skill_name: str, goal: dict[str, Any], result: dict[str, Any], duration: float
    ):
        """Hook for successful executions.
        Helps understand normal operation patterns.
        """
        self.error_intelligence.on_execution(skill_name, goal, result, duration)

    def _increment_stat(self, key: str) -> None:
        self.session_stats.setdefault(key, 0)
        self.session_stats[key] += 1

    def runtime_promotion_enabled(self) -> bool:
        """Return whether autonomous runtime code promotion is currently allowed."""
        return _runtime_patch_promotion_enabled()

    def supervised_force_enabled(self) -> bool:
        """Return whether explicit force=True promotion is currently allowed."""
        return _supervised_patch_promotion_enabled()

    def safe_auto_repair_enabled_for(self, fix_proposal: dict[str, Any]) -> tuple[bool, str]:
        """Return whether this exact repair may auto-apply in normal runtime."""

        fix = fix_proposal.get("fix") if isinstance(fix_proposal, dict) else None
        target = str(getattr(fix, "target_file", "") or "")
        if not target:
            return False, "missing target file"
        target_path = Path(target)
        if target_path.is_absolute():
            try:
                target = target_path.resolve().relative_to(self.code_base.resolve()).as_posix()
            except (OSError, ValueError):
                return False, "target outside code base"
        tier = classify_mutation_path(target)
        if not tier.auto_apply_allowed:
            return False, f"{target} is {tier.tier.label}"
        decision = safe_autonomous_repair_decision()
        if not decision.allowed:
            return False, decision.reason
        return True, f"{target} is {tier.tier.label}"

    def _record_cycle_failure(
        self,
        error: BaseException,
        *,
        step: str,
        duration: float,
        refinement: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stat_key = "refinement_failures" if refinement else "cycle_failures"
        self._increment_stat(stat_key)
        failure = {
            "step": step,
            "error_type": type(error).__qualname__,
            "error": str(error)[:500],
            "duration": duration,
            "count": self.session_stats[stat_key],
            "at": time.time(),
        }
        if extra:
            failure["extra"] = extra
        if refinement:
            self._last_refinement_error = failure
        else:
            self._last_cycle_error = failure
        _record_self_modification_degradation(
            error,
            action=f"Aborted {step} and left source files unchanged for the next supervised cycle",
            severity="degraded",
            receipt_required=True,
            extra=failure,
        )
        return failure

    # ========================================================================
    # Manual Fix Workflow
    # ========================================================================

    async def diagnose_current_bugs(self) -> list[dict[str, Any]]:
        """Analyze current bugs and generate diagnoses.

        Returns:
            List of bugs with diagnoses, sorted by priority

        """
        logger.info("Diagnosing current bugs...")

        bugs = await self.error_intelligence.find_bugs_to_fix()

        logger.info("Found %d bugs that can be fixed", len(bugs))
        return bugs

    def _repair_decision_key(self, bug: dict[str, Any]) -> str:
        """Bind a repair decision to the fault and the source it evaluated."""

        pattern = bug.get("pattern")
        events = list(getattr(pattern, "events", ()) or ())
        event = events[0] if events else None
        file_path = str(getattr(event, "file_path", "") or "")
        source_sha256 = "missing"
        if file_path:
            candidate = self.code_base / file_path
            try:
                source_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError:
                source_sha256 = "unreadable"
        payload = {
            "fingerprint": str(getattr(pattern, "fingerprint", "") or ""),
            "file_path": file_path,
            "line_number": getattr(event, "line_number", None),
            "error_type": str(getattr(event, "error_type", "") or ""),
            "error_message": str(getattr(event, "error_message", "") or ""),
            "diagnosis": bug.get("diagnosis"),
            "source_sha256": source_sha256,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    async def propose_fix(self, bug: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a fix proposal for a specific bug (Async).

        Args:
            bug: Bug dictionary from diagnose_current_bugs()

        Returns:
            Fix proposal or None
        """
        pattern = bug["pattern"]
        diagnosis = bug["diagnosis"]

        # Get sample event
        sample_event = pattern.events[0]

        if not sample_event.file_path or not sample_event.line_number:
            logger.warning("Cannot propose fix: no file/line information")
            return None

        logger.info("Proposing fix for %s:%d", sample_event.file_path, sample_event.line_number)

        # Issue 91: Shadow AST Wiring (Zero-token repair attempt)
        if "is not defined" in sample_event.error_message.lower():
            logger.info("⚡ Drafting zero-token AST repair proposal...")
            path = self.code_base / sample_event.file_path
            shadow_proposal = await self.shadow_healer.propose_repair(
                path, sample_event.error_message
            )
            if shadow_proposal:
                original = shadow_proposal["original_content"]
                fixed = shadow_proposal["repaired_content"]
                logger.info("✨ AST repair proposal created. Routing through safe modification.")
                return {
                    "bug": bug,
                    "fix": LogicTransplant(
                        target_file=shadow_proposal["target_file"],
                        explanation=shadow_proposal["explanation"],
                        chunks=[{"original": original, "fixed": fixed}],
                        risk_level=2,
                        lines_changed=abs(len(fixed.splitlines()) - len(original.splitlines()))
                        or 1,
                    ),
                    "test_results": {
                        "success": True,
                        "validation": "shadow_ast_preview",
                        "content_hash_before": shadow_proposal["content_hash_before"],
                        "content_hash_after": shadow_proposal["content_hash_after"],
                    },
                    "ready_to_apply": True,
                }

        success, fix, test_results = await self.code_repair.repair_bug(
            sample_event.file_path, sample_event.line_number, diagnosis
        )

        if not success:
            reason = str(test_results.get("error") or "Unknown error")
            # A veto is a DECISION, not a malfunction.
            #
            # The Growth Ladder is Aura's own consent gate on modifying her
            # source; when it declines, repair_bug returns
            # {"error": "Vetoed by entity"} and this reported it as "Fix
            # generation or sandbox testing failed" — 99 warnings in one
            # sampled window describing her consent mechanism working exactly
            # as designed. Nothing generated a bad fix and no sandbox failed;
            # she said no.
            #
            # Reported as failure it is worse than noise: it makes a healthy refusal
            # look like a broken repair pipeline, so the honest signal that
            # the pipeline IS broken has nowhere left to stand out.
            if "veto" in reason.lower():
                logger.info(
                    "🛡️ Autonomous repair declined by the Growth Ladder: %s", reason
                )
                return None
            logger.warning("Fix generation or sandbox testing failed: %s", reason)
            return None

        # Issue 96: Deep Validation via Branching Futures
        logger.info("🔬 Initiating Branching Futures Validation for %s...", sample_event.file_path)
        from core.skills.branching_futures import BranchingFuturesInput, BranchingFuturesSkill

        branching_skill = BranchingFuturesSkill()

        # Ask the ghost thread to apply the fix and verify the system stays stable
        branch_params = BranchingFuturesInput(
            branch_name=f"test_fix_{int(time.time())}",
            task=f"Apply the following code fix to {sample_event.file_path} and verify the system does not crash:\n\n{fix.fixed_code}",
            timeout_seconds=60,
        )

        try:
            branch_result = await branching_skill.safe_execute(branch_params)
            if not branch_result.get("ok"):
                logger.error(
                    "❌ Branching Futures Validation FAILED: %s", branch_result.get("error")
                )
                return None
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.error("❌ Branching Futures Validation CRASHED: %s", e)
            return None

        logger.info("✅ Branching Futures Validation PASSED.")

        return {"bug": bug, "fix": fix, "test_results": test_results, "ready_to_apply": success}

    async def apply_fix(
        self,
        fix_proposal: dict[str, Any],
        force: bool = False,
        test_results: dict[str, Any] | None = None,
        safe_autonomous: bool = False,
    ) -> bool:
        """Apply a fix proposal with Swarm Review and Safe Modification (Phase 31)."""
        # Level 3 Check
        from core.container import ServiceContainer

        if force and not self.supervised_force_enabled():
            _record_self_modification_degradation(
                RuntimeError(
                    f"force=True requires {_SUPERVISED_SELF_MODIFICATION_ENV}=1"
                ),
                action=(
                    "Rejected forced self-modification because supervised operator override "
                    "was not enabled"
                ),
                severity="degraded",
                receipt_required=True,
                extra={
                    "target_file": getattr(fix_proposal.get("fix"), "target_file", "unknown")
                    if isinstance(fix_proposal, dict)
                    else "unknown",
                    "force": True,
                    "required_env": _SUPERVISED_SELF_MODIFICATION_ENV,
                },
            )
            logger.error(
                "SME: force=True blocked. Set %s=1 only for supervised operator promotion.",
                _SUPERVISED_SELF_MODIFICATION_ENV,
            )
            return False

        kernel = ServiceContainer.get("aura_kernel", default=None)
        volition = getattr(kernel, "volition_level", 0) if kernel else 0

        if volition < 1 and not force:
            logger.warning("SME: Modification BLOCKED. Requires Volition Level 1+.")
            return False

        if not force and not safe_autonomous and not self.runtime_promotion_enabled():
            logger.warning(
                "SME: Runtime patch promotion blocked. Set %s=1, %s=1, and %s=1 "
                "inside a repair-lab profile for autonomous source promotion.",
                _RUNTIME_SELF_MODIFICATION_ENV,
                _AUTONOMOUS_PATCH_PROMOTION_ENV,
                _REPAIR_LAB_SOURCE_PROMOTION_ENV,
            )
            return False

        if not self.auto_fix_enabled and not force and not safe_autonomous:
            logger.warning("Auto-fix disabled. Use force=True to override.")
            return False

        fix = fix_proposal["fix"]

        # Phase 15: Swarm Review (Sovereign Decision)
        if not await self._swarm_review(fix_proposal, force=force):
            logger.error("❌ Fix Rejected by Swarm Critic. Aborting modification.")
            return False

        # [Phase 30] Circuit Breaker Check (Metabolic Quarantine)
        immunity = None
        try:
            from ..resilience.immunity_hyphae import get_immunity

            immunity = get_immunity()
            # If the specific file is entering a fail-loop, quarantine it
            if (
                immunity
                and hasattr(immunity, "circuit_breakers")
                and fix.target_file in immunity.circuit_breakers
            ):
                cb = immunity.circuit_breakers[fix.target_file]
                if cb.state == "OPEN":
                    logger.error(
                        "🛑 [NEURO] Component %s is QUARANTINED. Modification blocked.",
                        fix.target_file,
                    )
                    return False
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_self_modification_degradation(
                e,
                action="Blocked fix application because quarantine circuit-breaker state could not be confirmed",
                severity="degraded",
                receipt_required=True,
                extra={"target_file": getattr(fix, "target_file", "unknown")},
            )
            logger.debug("Shielded circuit breaker check failed: %s", e)
            return False

        # Use provided test results, or those inside the proposal
        final_test_results = test_results or fix_proposal.get("test_results")

        if not final_test_results:
            logger.error("❌ Refusing to apply fix: No sandboxed test results provided.")
            return False

        if not final_test_results.get("success", False):
            logger.error("❌ Refusing to apply fix: Sandboxed tests failed.")
            return False

        logger.info("🧬 [NEURO] Initiating permanent fix application for %s", fix.target_file)

        # Phase 31: Permanent Persistence via SafeSelfModification
        async with self._fix_lock:
            try:
                try:
                    append_repair_entry(fix, final_test_results, self.code_base)
                except (OSError, TypeError, ValueError) as exc:
                    _record_self_modification_degradation(
                        exc,
                        action="Refused fix application because durable repair registry write failed",
                        severity="degraded",
                        receipt_required=True,
                        extra={"target_file": getattr(fix, "target_file", "unknown")},
                    )
                    logger.error("Self-modification registry write failed: %s", exc)
                    logger.error(
                        "Refusing to apply fix without a durable self-modification registry entry."
                    )
                    return False

                # Delegate permanent application to SafeSelfModification
                # This handles: Backups, Git branches, Ghost Boot, and Rollback
                success, message = await self.safe_modification.apply_fix(
                    fix=fix,
                    test_results=final_test_results,
                    supervised=force,
                    safe_autonomous=bool(safe_autonomous and not force),
                )

                if success:
                    self.session_stats["fixes_successful"] += 1
                    logger.info("✅ [NEURO] Permanent fix applied: %s", message)

                    # Resolve related incident and mark subsystem as healthy
                    try:
                        bug = fix_proposal.get("bug")
                        if bug and isinstance(bug, dict):
                            pattern = bug.get("pattern")
                            if pattern and hasattr(pattern, "events") and pattern.events:
                                first_event = pattern.events[0]
                                if hasattr(first_event, "context") and isinstance(
                                    first_event.context, dict
                                ):
                                    subsystem = first_event.context.get("subsystem")
                                    if subsystem:
                                        # 1. Mark subsystem status as healthy in SubsystemRegistry
                                        from core.runtime.errors import get_subsystem_registry

                                        subsystem_reg = get_subsystem_registry()
                                        health = subsystem_reg.get(subsystem)
                                        if health:
                                            health.mark_ok()
                                            logger.info(
                                                "🏥 Subsystem %s marked as healthy after successful self-modification repair.",
                                                subsystem,
                                            )

                                        # 2. Resolve the associated incident in IncidentManager
                                        from core.resilience.incident_manager import (
                                            get_incident_manager,
                                        )

                                        category = f"degradation:{subsystem}"
                                        incident_mgr = get_incident_manager()
                                        incident_mgr.resolve(
                                            category=category,
                                            resolution=f"Successfully applied self-modification repair: {fix.explanation}",
                                        )
                    except SELF_MOD_RECOVERABLE_ERRORS as resolve_err:
                        _record_self_modification_degradation(
                            resolve_err,
                            action="Kept applied fix but left incident resolution for the next health reconciliation",
                            extra={"target_file": getattr(fix, "target_file", "unknown")},
                        )
                        logger.debug(
                            "Failed to automatically resolve incident/subsystem status after fix: %s",
                            resolve_err,
                        )

                    # UPGRADE NOTIFICATION
                    try:
                        from ..thought_stream import get_emitter

                        get_emitter().emit(
                            "System Evolution",
                            f"Sovereign Repair Persistent: {fix.target_file}",
                            level="success",
                        )
                    except (ImportError, AttributeError, RuntimeError) as exc:
                        _record_self_modification_degradation(
                            exc,
                            action="Kept applied fix and skipped non-critical evolution notification",
                            extra={"target_file": getattr(fix, "target_file", "unknown")},
                        )
                        logger.debug("Suppressed: %s", exc)

                else:
                    logger.error("❌ [NEURO] Permanent application FAILED: %s", message)
                    # Record failure in immunity for pattern analysis
                    try:
                        if immunity:
                            immunity.audit_error(
                                RuntimeError(f"Persistence failed: {message}"),
                                {"file": fix.target_file},
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                        _record_self_modification_degradation(
                            e,
                            action="Recorded persistence failure locally after immunity audit failed",
                            extra={"target_file": getattr(fix, "target_file", "unknown")},
                        )
                        logger.debug("Failed to audit persistence error to immunity: %s", e)

                # Record for learning
                error_type = "optimization"
                try:
                    bug = fix_proposal.get("bug")
                    if bug and isinstance(bug, dict):
                        pattern = bug.get("pattern")
                        if pattern:
                            if isinstance(pattern, dict):
                                events = pattern.get("events")
                                if events and isinstance(events, list):
                                    error_type = events[0].get("error_type", "optimization")
                            else:
                                events = getattr(pattern, "events", None)
                                if events and isinstance(events, list):
                                    first_event = events[0]
                                    if isinstance(first_event, dict):
                                        error_type = first_event.get("error_type", "optimization")
                                    else:
                                        error_type = getattr(
                                            first_event, "error_type", "optimization"
                                        )
                except SELF_MOD_RECOVERABLE_ERRORS as _e:
                    logger.debug("Failed to extract error_type for learning: %s", _e)

                self.learning_system.record_fix_attempt(
                    fix, error_type, success=success, context={"persistence_msg": message}
                )

                self.session_stats["fixes_attempted"] += 1
                return success

            except (ImportError, AttributeError, RuntimeError) as e:
                _record_self_modification_degradation(
                    e,
                    action="Aborted persistent self-modification and returned failure to caller",
                    severity="degraded",
                    receipt_required=True,
                    extra={"target_file": getattr(fix, "target_file", "unknown")},
                )
                logger.exception("CRITICAL: Persistence error in SME: %s", e)
                return False

        return False  # Fallback

    async def _swarm_review(self, proposal: dict[str, Any], *, force: bool = False) -> bool:
        """Recursive check: spawn a swarm debate to review the proposed fix."""
        from core.container import ServiceContainer

        swarm = ServiceContainer.get("agent_delegator", default=None)
        if not swarm:
            _record_self_modification_degradation(
                RuntimeError("agent_delegator unavailable for self-modification review"),
                action=(
                    "Rejected fix because required swarm review was unavailable; "
                    "force=True does not bypass independent review"
                ),
                severity="degraded",
                receipt_required=True,
                extra={"review": "swarm", "force": bool(force)},
            )
            logger.error("Swarm Delegator unavailable. Rejecting self-modification.")
            return False

        fix = proposal["fix"]
        proposed_change = getattr(fix, "fixed_code", None)
        if proposed_change is None and hasattr(fix, "chunks"):
            proposed_change = "\n\n".join(
                chunk.get("fixed", "") for chunk in getattr(fix, "chunks", [])
            )
        if proposed_change is None:
            proposed_change = getattr(fix, "explanation", "No patch text provided")
        topic = (
            f"Review this proposed code fix for file {fix.target_file}.\n"
            f"Diagnosis: {proposal.get('bug', {}).get('diagnosis', 'Optimization')}\n"
            f"Proposed Change:\n{proposed_change}"
        )

        logger.info("🐝 Initiating Sovereign Swarm Review for fix...")
        try:
            # Short timeout for internal swarm reviews to prevent stalls
            result = await swarm.delegate_debate(topic, roles=["architect", "critic"])

            # Simple heuristic: If the critic's last words are negative, reject
            # (In a real scenario, we'd use a parser, but for this bridge, we trust the consensus synthesis)
            if (
                "REJECT" in result.upper()
                or "UNSAFE" in result.upper()
                or "ROLLBACK" in result.upper()
            ):
                logger.warning("🚨 Sovereign Swarm Critic flagged this fix as UNSAFE.")
                return False

            logger.info("✅ Sovereign Swarm Consensus: Fix is safe to apply.")
            return True
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_self_modification_degradation(
                e,
                action="Rejected fix because active swarm review failed before consensus",
                severity="degraded",
                receipt_required=True,
                extra={"target_file": getattr(fix, "target_file", "unknown")},
            )
            logger.error("Swarm review failed: %s. Rejecting fix.", e)
            return False

    async def report_optimization(self, issue: dict[str, Any]) -> bool:
        """Manually report an optimization opportunity (Async).

        Args:
            issue: Dict containing 'file', 'line', 'type', 'message'

        Returns:
            True if fix was successfully applied
        """
        file_path = issue.get("file")
        line_number = issue.get("line")

        if not file_path or not line_number:
            logger.warning("Invalid optimization report: missing file or line")
            return False

        msg = issue.get("message", "No message provided")
        logger.info("💎 Optimization reported: %s at %s:%d", msg, file_path, line_number)

        # 1. Create a synthetic diagnosis for optimization
        diagnosis = {
            "ok": True,
            "hypotheses": [
                {
                    "root_cause": f"Code Quality Issue: {issue.get('type', 'Unknown')}",
                    "explanation": msg,
                    "potential_fix": f"Refactor to improve {issue.get('type', 'Unknown')}.",
                    "confidence": "high",
                }
            ],
        }

        # 2. Propose fix
        success, fix, message = await self.code_repair.repair_bug(file_path, line_number, diagnosis)

        if not success or not fix:
            logger.warning("Optimization fix generation failed: %s", message)
            return False

        # 3. Apply fix
        proposal = {
            "bug": {"pattern": {"events": [{"error_type": "optimization"}]}},
            "fix": fix,
            "test_results": message,
        }

        return await self.apply_fix(proposal, force=False)

    # ========================================================================
    # Automatic Fix Workflow
    # ========================================================================

    async def run_autonomous_cycle(self) -> dict[str, Any]:
        """Run one autonomous self-improvement cycle (Async).

        [GENESIS] Volition-scaled logic:
        1. Check Volition Level (Requires Level 3 for Auto-fix)
        2. Find bugs to fix
        ...
        """
        cycle_start = time.time()
        try:
            # Dynamic Volition Pull
            from core.container import ServiceContainer

            kernel = ServiceContainer.get("aura_kernel", default=None)
            volition = getattr(kernel, "volition_level", 0) if kernel else 0

            promotion_enabled = self.runtime_promotion_enabled()
            auto_fix_requested = bool(getattr(self, "_auto_fix_requested", False))

            # Override auto_fix based on Volition only after explicit operator opt-in.
            if volition >= 3 and auto_fix_requested and promotion_enabled:
                if not self.auto_fix_enabled:
                    logger.info("SME: Volition Level 3 detected. AUTO-FIX ENGAGED.")
                    self.auto_fix_enabled = True
            else:
                if self.auto_fix_enabled:
                    logger.debug(
                        "SME: Auto-fix held in proposal-only mode "
                        "(volition=%d requested=%s promotion_enabled=%s).",
                        volition,
                        auto_fix_requested,
                        promotion_enabled,
                    )
                    self.auto_fix_enabled = False

            logger.debug("--- [SME] Starting Autonomous Cycle (Volition=%d) ---", volition)
            logger.info("AUTONOMOUS SELF-MODIFICATION CYCLE")
            logger.info("=" * 80)

            # Step 1: Find bugs
            bugs = await self.diagnose_current_bugs()

            if not bugs:
                logger.info("No bugs detected - system healthy (idle)")
                return {"success": True, "bugs_found": 0, "fixes_applied": 0}

            logger.info("Found %d fixable bugs", len(bugs))

            # Step 2: Select top priority bug
            top_bug = bugs[0]
            logger.info("Targeting bug: %s", top_bug["pattern"].fingerprint)
            repair_decision_key = self._repair_decision_key(top_bug)

            prior_decision = self._last_repair_decision
            if (
                isinstance(prior_decision, dict)
                and prior_decision.get("key") == repair_decision_key
                and prior_decision.get("disposition")
                in {"proposal_quarantined", "proposal_refused_by_policy"}
            ):
                disposition = str(prior_decision["disposition"])
                logger.info(
                    "Repair decision unchanged for %s; reusing %s receipt",
                    top_bug["pattern"].fingerprint,
                    disposition,
                )
                return {
                    "success": True,
                    "bugs_found": len(bugs),
                    "fixes_applied": 0,
                    "disposition": "proposal_decision_reused",
                    "reused_disposition": disposition,
                    "proposal": prior_decision.get("proposal"),
                }

            # Check if learning system has suggestions
            error_type = top_bug["pattern"].events[0].error_type
            strategy_suggestion = self.learning_system.suggest_strategy(error_type, context={})

            if strategy_suggestion:
                logger.info("Learning system suggests: %s", strategy_suggestion["strategy_type"])
                logger.info("  Guidance: %s", strategy_suggestion["guidance"])

            # Step 3: Generate fix
            fix_proposal = await self.propose_fix(top_bug)

            if not fix_proposal:
                logger.warning("Failed to generate fix proposal")
                return {
                    "success": False,
                    "bugs_found": len(bugs),
                    "fixes_applied": 0,
                    "error": "Fix generation failed",
                    "disposition": "fix_generation_failed",
                }

            # Step 4: Apply fix when repair-lab promotion is enabled or when
            # this exact candidate is a safe auto-repair tier.
            safe_auto_allowed, safe_auto_reason = self.safe_auto_repair_enabled_for(fix_proposal)
            if self.auto_fix_enabled or safe_auto_allowed:
                success = False
                try:
                    success = await self.apply_fix(
                        fix_proposal,
                        force=False,
                        safe_autonomous=bool(safe_auto_allowed and not self.auto_fix_enabled),
                    )
                finally:
                    cycle_time = time.time() - cycle_start
                    logger.debug("--- [SME] Cycle Complete (%.2fs) ---", cycle_time)
                    try:
                        self.meta_learning.record_learning_cycle(
                            attempts=1,
                            successes=1 if success else 0,
                            strategies_used=[
                                self.learning_system.classifier.classify_fix(fix_proposal["fix"])
                            ],
                            time_spent=cycle_time,
                        )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        _record_self_modification_degradation(
                            exc,
                            action="Preserved self-modification result after meta-learning telemetry failed",
                            extra={"cycle": "autonomous", "success": success},
                        )

                return {
                    "success": success,
                    "bugs_found": len(bugs),
                    "fixes_applied": 1 if success else 0,
                    "cycle_time": cycle_time,
                    "auto_repair_mode": "safe_autonomous" if safe_auto_allowed else "repair_lab",
                    "auto_repair_reason": safe_auto_reason,
                    "disposition": "repair_applied" if success else "repair_application_failed",
                }

            quarantine = await self._quarantine_fix_proposal(fix_proposal)
            self._last_proposal = quarantine
            if quarantine.get("validated") is True:
                disposition = "proposal_quarantined"
            elif quarantine.get("blocked_at") == "approval":
                disposition = "proposal_refused_by_policy"
            else:
                disposition = "proposal_validation_failed"
            cycle_succeeded = disposition != "proposal_validation_failed"
            if cycle_succeeded:
                self._last_repair_decision = {
                    "key": repair_decision_key,
                    "disposition": disposition,
                    "proposal": quarantine,
                }
            logger.info(
                "Repair proposal disposition=%s status=%s",
                disposition,
                quarantine.get("status", "unknown"),
            )
            return {
                "success": cycle_succeeded,
                "bugs_found": len(bugs),
                "fixes_applied": 0,
                "proposal": quarantine,
                "auto_repair_reason": safe_auto_reason,
                "disposition": disposition,
            }
        except SELF_MOD_RECOVERABLE_ERRORS as exc:
            duration = time.time() - cycle_start
            failure = self._record_cycle_failure(
                exc,
                step="autonomous_cycle",
                duration=duration,
            )
            return {
                "success": False,
                "bugs_found": 0,
                "fixes_applied": 0,
                "cycle_time": duration,
                "degraded_step": "autonomous_cycle",
                "error": failure["error"],
                "failure": failure,
            }

    # ========================================================================
    # Refinement (Recursive Self-Architecture)
    # ========================================================================

    async def run_refinement_cycle(self) -> dict[str, Any]:
        """Run a proactive architectural refinement cycle.

        Hunts for bottlenecks in core reasoning logic and optimizes them.
        """
        logger.info("💎 [REFINE] Starting Kernel Refinement Cycle...")
        cycle_start = time.time()
        try:
            # Step 1: Analyze kernel health
            refinements = await self.kernel_refiner.analyze_kernel_health()

            if not refinements:
                logger.info("✨ Kernel is optimal. No refinements proposed.")
                return {
                    "success": True,
                    "refinements_applied": 0,
                    "changed_files": [],
                    "reload_required": False,
                }

            top_refinement = refinements[0]
            logger.info("🚀 Targeted Refinement: %s", top_refinement["message"])
            try:
                from ..thought_stream import get_emitter

                get_emitter().emit(
                    "Synthesis 🚀",
                    f"Optimizing {top_refinement.get('file', 'system')}: {top_refinement.get('message', '')}",
                    level="info",
                    category="Self-Modification",
                )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_self_modification_degradation(
                    _exc,
                    action="Continued refinement after non-critical thought-stream notification failed",
                    extra={"cycle": "refinement", "target": top_refinement.get("file", "system")},
                )
                logger.debug("Suppressed Exception: %s", _exc)

            # Step 2: Convert refinement to a 'synthetic bug' for the repair engine
            # This allows us to reuse the existing safety/testing/apply pipeline
            proposal = await self.report_optimization(top_refinement)

            duration = time.time() - cycle_start
            return {
                "success": proposal,
                "refinements_applied": 1 if proposal else 0,
                "changed_files": [str(top_refinement.get("file") or "")] if proposal else [],
                "reload_required": bool(proposal),
                "duration": duration,
            }
        except SELF_MOD_RECOVERABLE_ERRORS as exc:
            duration = time.time() - cycle_start
            failure = self._record_cycle_failure(
                exc,
                step="refinement_cycle",
                duration=duration,
                refinement=True,
            )
            return {
                "success": False,
                "refinements_applied": 0,
                "changed_files": [],
                "reload_required": False,
                "duration": duration,
                "degraded_step": "refinement_cycle",
                "error": failure["error"],
                "failure": failure,
            }

    async def _quarantine_fix_proposal(self, fix_proposal: dict[str, Any]) -> dict[str, Any]:
        """Validate a generated repair and retain it without mutating live source."""

        fix = fix_proposal.get("fix")
        target = str(getattr(fix, "target_file", "") or "")
        if not target:
            return {"status": "rejected", "validated": False, "reason": "missing_target_file"}

        target_path = Path(target)
        if not target_path.is_absolute():
            target_path = self.code_base / target_path
        try:
            target_path = target_path.resolve(strict=True)
            relative_target = target_path.relative_to(self.code_base.resolve(strict=True))
        except (OSError, ValueError) as exc:
            _record_self_modification_degradation(
                exc,
                action="rejected repair proposal outside the configured source root",
                extra={"target_file": target},
            )
            return {
                "status": "rejected",
                "validated": False,
                "reason": "target_outside_source_root",
            }
        try:
            before_source = target_path.read_text(encoding="utf-8")
        except OSError as exc:
            _record_self_modification_degradation(
                exc,
                action="rejected repair proposal because target source could not be read",
                extra={"target_file": target},
            )
            return {"status": "rejected", "validated": False, "reason": "target_unreadable"}

        after_source = before_source
        if hasattr(fix, "original_code") and hasattr(fix, "fixed_code"):
            original = str(getattr(fix, "original_code", ""))
            replacement = str(getattr(fix, "fixed_code", ""))
            if not original or before_source.count(original) != 1:
                return {
                    "status": "rejected",
                    "validated": False,
                    "reason": "repair_anchor_not_unique",
                    "target_file": target,
                }
            after_source = before_source.replace(original, replacement, 1)
        elif hasattr(fix, "chunks"):
            for chunk in list(getattr(fix, "chunks", []) or []):
                original = str(chunk.get("original", ""))
                replacement = str(chunk.get("fixed", ""))
                if not original or after_source.count(original) != 1:
                    return {
                        "status": "rejected",
                        "validated": False,
                        "reason": "repair_anchor_not_unique",
                        "target_file": target,
                    }
                after_source = after_source.replace(original, replacement, 1)
        else:
            return {"status": "rejected", "validated": False, "reason": "unsupported_fix_shape"}

        from dataclasses import asdict

        from core.self_modification.safe_pipeline import SafePipeline

        proposal = await SafePipeline().run(
            drive="repair",
            intent=str(getattr(fix, "explanation", "repair observed runtime defect")),
            file_path=str(relative_target),
            before_source=before_source,
            after_source=after_source,
            owner_approved=False,
        )
        payload = asdict(proposal)
        payload["status"] = (
            "quarantined"
            if proposal.promotion_artifact_path
            else "rejected"
            if proposal.blocked_at
            else "validated"
        )
        payload["validated"] = bool(
            proposal.promotion_artifact_path
            or (
                "shadow_runtime" in proposal.stages_completed
                and "formal_verify" in proposal.stages_completed
            )
        )
        return payload

    # ========================================================================
    # Background Monitoring
    # ========================================================================

    def start_monitoring(self):
        """Start background monitoring for errors"""
        if self.monitoring_enabled:
            logger.warning("Monitoring already running")
            return

        self.monitoring_enabled = True
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                logger.error("Failed to start monitoring: No asyncio loop available.")
                self.monitoring_enabled = False
                return

            self.monitor_thread = get_task_tracker().create_task(
                self._monitoring_loop(), name="SelfModificationMonitor"
            )
            self.health_thread = get_task_tracker().create_task(
                self._health_watcher_loop(), name="SelfModificationHealthWatcher"
            )
        except RuntimeError:
            logger.error("Failed to start monitoring: No asyncio loop available.")
            self.monitoring_enabled = False
            return

        logger.info("✓ Background monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_enabled = False
        if self.monitor_thread:
            self.monitor_thread.cancel()
            self.monitor_thread = None
        if hasattr(self, "health_thread") and self.health_thread:
            self.health_thread.cancel()
            self.health_thread = None

        logger.info("Background monitoring stopped")

    async def _monitoring_loop(self):
        """Background monitoring loop with circuit breaker (v5.2)"""
        from core.container import ServiceContainer
        from core.runtime.background_policy import background_activity_reason

        # Event-driven: the monitor consumes real error observations instead of
        # polling the model or inventing optimization work while the runtime is idle.
        logger.info("Monitoring loop starting...")
        _consecutive_failures = 0
        _max_failures = 5
        _backoff = self.monitor_interval

        while self.monitoring_enabled:
            try:
                await self._repair_event.wait()
                orch = ServiceContainer.get("orchestrator", default=None)
                policy_reason = background_activity_reason(
                    orch,
                    min_idle_seconds=float(self.monitor_interval),
                    require_conversation_ready=False,
                )
                if policy_reason:
                    logger.info("Skipping autonomous self-modification cycle: %s", policy_reason)
                    await asyncio.sleep(min(_backoff, 60.0))
                    continue

                self._repair_event.clear()
                result = await self.run_autonomous_cycle()

                # Fix: run_autonomous_cycle can return a bool via report_optimization
                if not isinstance(result, dict):
                    result = {"success": bool(result), "fixes_applied": 1 if result else 0}

                if result.get("fixes_applied", 0) > 0:
                    logger.info("✅ Autonomous fixes applied in background")
                    _consecutive_failures = 0
                    _backoff = self.monitor_interval
                elif not _autonomous_cycle_failed(result):
                    _consecutive_failures = 0
                    _backoff = self.monitor_interval
                else:
                    _consecutive_failures += 1
                    _backoff = min(600, _backoff * 2)  # Exponential backoff, max 10min

                # Circuit breaker: stop trying if we keep failing
                if _consecutive_failures >= _max_failures:
                    logger.warning(
                        "🛑 Self-modification circuit breaker tripped after %s consecutive failures. Cooling down 30min.",
                        _max_failures,
                    )
                    await asyncio.sleep(1800)  # 30 minute cooldown
                    _consecutive_failures = 0
                    _backoff = self.monitor_interval
                    continue

                # Synthetic recovery tests are dangerous in production because they
                # generate false repair work and noisy degraded cognition.
                if (
                    os.environ.get("AURA_ENABLE_SYNTHETIC_SELF_TESTS", "0") == "1"
                    and time.time() % 3600 < self.monitor_interval
                ):
                    await self.trigger_synthetic_test()

                # Wait for next cycle
                await asyncio.sleep(_backoff)

            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError) as e:
                self._increment_stat("background_failures")
                _record_self_modification_degradation(
                    e,
                    action="Backed off autonomous self-modification monitor after recoverable cycle failure",
                    severity="degraded",
                    receipt_required=True,
                    extra={
                        "loop": "monitoring",
                        "consecutive_failures": _consecutive_failures + 1,
                        "backoff": _backoff,
                    },
                )
                logger.error("Monitoring cycle error: %s", e, exc_info=True)
                _consecutive_failures += 1
                await asyncio.sleep(max(60, _backoff))

        logger.info("Monitoring loop stopped")

    async def _health_watcher_loop(self):
        """Monitor SubsystemAudit and trigger repairs for failing components (v18).

        v50 hardening:
        - 300s boot grace before ANY health checks (covers slowest subsystem: identity)
        - NEVER_SEEN is silenced entirely — only STALE triggers repair injection
        - Per-subsystem injection cooldown raised to 5min to prevent cascading noise
        - Only inject when staleness exceeds 2x the expected heartbeat interval
        """
        await asyncio.sleep(300)  # Boot grace: all subsystems need time to register first heartbeat
        logger.info("Health Watcher starting (boot grace complete).")
        _injection_cooldowns: dict[str, float] = {}  # subsystem → last injection time
        _injection_cooldown = 300.0  # 5 minutes between injections per subsystem

        while self.monitoring_enabled:
            try:
                from core.container import get_container

                container = get_container()
                audit = container.get("subsystem_audit", None)

                if audit:
                    health = audit.check_health()
                    now = time.time()
                    for name, info in health.get("subsystems", {}).items():
                        status = info.get("status")

                        # Hard-silence NEVER_SEEN — subsystem hasn't booted yet,
                        # not a failure. Only act on genuinely STALE subsystems.
                        if status != "STALE":
                            continue

                        # Rate limit: don't re-inject for the same subsystem within cooldown
                        last_injection = _injection_cooldowns.get(name, 0.0)
                        if (now - last_injection) < _injection_cooldown:
                            continue

                        # Staleness threshold: only inject if stale for >2x expected interval
                        stale_secs = info.get("stale_seconds")
                        if stale_secs is not None:
                            from core.ops.subsystem_audit import SubsystemAudit

                            expected = SubsystemAudit.SUBSYSTEMS.get(name, 300)
                            if stale_secs < expected * 2:
                                continue  # Not stale enough to warrant a repair injection

                        logger.warning(
                            "Health Watcher: %s is STALE (stale=%ss). Injecting repair.",
                            name,
                            stale_secs,
                        )
                        self.on_error(
                            RuntimeError(f"Subsystem {name} is {status}"),
                            {"subsystem": name, "telemetry": info},
                            skill_name="HealthWatcher",
                            goal="Stabilize core subsystem",
                        )
                        _injection_cooldowns[name] = now
                        self.session_stats["health_fixes_triggered"] += 1

                await asyncio.sleep(60)  # check every minute
            except asyncio.CancelledError:
                break
            except (ImportError, AttributeError, RuntimeError) as e:
                self._increment_stat("background_failures")
                _record_self_modification_degradation(
                    e,
                    action="Backed off health watcher after recoverable audit failure",
                    extra={"loop": "health_watcher"},
                )
                logger.error("Health Watcher error: %s", e)
                await asyncio.sleep(30)

    async def trigger_synthetic_test(self):
        """Generate a synthetic error to test the recovery pipeline (Issue 86)."""
        logger.info("🧪 [TEST] Injecting synthetic test error...")
        try:
            # This error is caught by on_error and should trigger a repair cycle
            # for a safe, non-critical file.
            test_file = self.code_base / "core" / "utils" / "test_canary.py"
            if not test_file.exists():
                await async_atomic_write_text(test_file, "# Synthetic test canary\ndef canary(): return True\n")

            self.on_error(
                RuntimeError("Synthetic recovery test failure"),
                {"synthetic": True, "target_file": str(test_file)},
                skill_name="SelfTestSystem",
                goal="Verify autonomous recovery health",
            )
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_self_modification_degradation(
                e,
                action="Skipped synthetic recovery injection and left runtime state unchanged",
                extra={"synthetic": True},
            )
            logger.error("Failed to trigger synthetic test: %s", e)

    # ========================================================================
    # Reporting & Status
    # ========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        # Error intelligence status
        ei_status = self.error_intelligence.get_status()

        # Modification stats
        mod_stats = self.safe_modification.get_stats()

        # Learning report
        learning_report = self.learning_system.get_strategy_report()

        # Meta-learning
        is_improving, improvement_msg = self.meta_learning.is_improving()
        learning_velocity = self.meta_learning.get_learning_velocity()

        # Session stats
        session_time = time.time() - self.session_stats["session_start"]

        return {
            "monitoring_enabled": self.monitoring_enabled,
            "auto_fix_enabled": self.auto_fix_enabled,
            "auto_fix_requested": bool(getattr(self, "_auto_fix_requested", False)),
            "runtime_patch_promotion_enabled": self.runtime_promotion_enabled(),
            "repair_lab_source_promotion_required": _REPAIR_LAB_SOURCE_PROMOTION_ENV,
            "supervised_force_promotion_enabled": self.supervised_force_enabled(),
            "session_duration_hours": session_time / 3600,
            "session_stats": self.session_stats,
            "last_cycle_error": getattr(self, "_last_cycle_error", None),
            "last_refinement_error": getattr(self, "_last_refinement_error", None),
            "last_proposal": getattr(self, "_last_proposal", None),
            "repair_event_pending": bool(self._repair_event.is_set()),
            "error_intelligence": ei_status,
            "modification_stats": mod_stats,
            "learned_strategies": len(learning_report),
            "top_strategies": learning_report[:5],
            "meta_learning": {
                "is_improving": is_improving,
                "improvement_message": improvement_msg,
                "learning_velocity": f"{learning_velocity:.2f} fixes/hour",
            },
        }

    def runtime_status(self) -> dict[str, Any]:
        """Return liveness without loading historical learning reports."""

        return {
            "running": bool(
                self.monitoring_enabled
                and self.monitor_thread
                and not self.monitor_thread.done()
            ),
            "health_watcher_running": bool(
                getattr(self, "health_thread", None)
                and not self.health_thread.done()
            ),
            "mode": "promotion" if self.runtime_promotion_enabled() else "validation_quarantine",
            "repair_event_pending": bool(self._repair_event.is_set()),
            "last_proposal": getattr(self, "_last_proposal", None),
            "last_cycle_error": getattr(self, "_last_cycle_error", None),
            "session_stats": dict(self.session_stats),
        }

    def get_report(self) -> str:
        """Get human-readable status report"""
        status = self.get_status()

        report = f"""
{"=" * 80}
AUTONOMOUS SELF-MODIFICATION ENGINE - STATUS REPORT
{"=" * 80}

CONFIGURATION:
  Auto-fix enabled: {status["auto_fix_enabled"]}
  Background monitoring: {status["monitoring_enabled"]}
  Session duration: {status["session_duration_hours"]:.1f} hours

SESSION STATISTICS:
  Bugs detected: {status["session_stats"]["bugs_detected"]}
  Fixes attempted: {status["session_stats"]["fixes_attempted"]}
  Fixes successful: {status["session_stats"]["fixes_successful"]}

ERROR INTELLIGENCE:
  Recent errors: {status["error_intelligence"]["recent_error_count"]}
  Error patterns: {status["error_intelligence"]["total_patterns"]}
  Critical issues: {status["error_intelligence"]["critical_patterns"]}

MODIFICATION HISTORY:
  Total attempts: {status["modification_stats"]["total_attempts"]}
  Successful: {status["modification_stats"]["successful"]}
  Failed: {status["modification_stats"]["failed"]}
  Success rate: {status["modification_stats"]["success_rate"]}

LEARNING SYSTEM:
  Learned strategies: {status["learned_strategies"]}
  System improving: {status["meta_learning"]["is_improving"]}
  {status["meta_learning"]["improvement_message"]}
  Learning velocity: {status["meta_learning"]["learning_velocity"]}

TOP FIX STRATEGIES:
"""

        for i, strategy in enumerate(status["top_strategies"][:3], 1):
            report += f"  {i}. {strategy['strategy_type']}: "
            report += f"{strategy['success_count']} successes "
            report += f"({strategy['success_rate'] * 100:.0f}% success rate)\n"

        report += f"\n{'=' * 80}\n"

        return report

    def enable_auto_fix(self, confirm: bool = False):
        """Enable automatic fixing (Issue 84: Gate with confirmation)."""
        if not confirm:
            logger.error("Attempted to enable auto-fix without explicit confirmation.")
            return False
        if not self.runtime_promotion_enabled():
            logger.error(
                "Attempted to enable auto-fix without %s=1, %s=1, and %s=1 "
                "in an operator-controlled repair-lab profile.",
                _RUNTIME_SELF_MODIFICATION_ENV,
                _AUTONOMOUS_PATCH_PROMOTION_ENV,
                _REPAIR_LAB_SOURCE_PROMOTION_ENV,
            )
            self._auto_fix_requested = True
            return False
        self._auto_fix_requested = True
        self.auto_fix_enabled = True
        logger.warning("⚠️  AUTO-FIX ENABLED - System will modify its own code")
        return True

    def disable_auto_fix(self):
        """Disable automatic fixing"""
        self._auto_fix_requested = False
        self.auto_fix_enabled = False
        logger.info("Auto-fix disabled - fixes will be proposed only")
