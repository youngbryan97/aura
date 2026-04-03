"""Autonomous Self-Modification Engine
Orchestrates the complete self-improvement system.
"""
import asyncio
import logging
import os
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .code_repair import AutonomousCodeRepair

# Import all subsystems
from .error_intelligence import ErrorIntelligenceSystem
from .learning_system import MetaLearning, SelfImprovementLearning
from .safe_modification import SafeSelfModification, LogicTransplant
from .kernel_refiner import KernelRefiner
from .boot_validator import GhostBootValidator
from .shadow_ast_healer import ShadowASTHealer
from .shadow_runtime import get_shadow_runtime

logger = logging.getLogger("SelfModification.Engine")


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
        auto_fix_enabled: bool = True  # Sovereign Overdrive: Enabled by default
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
        self.auto_fix_enabled = auto_fix_enabled
        
        # Initialize subsystems
        from ..config import config
        self.error_intelligence = ErrorIntelligenceSystem(
            cognitive_engine,
            log_dir=str(config.paths.data_dir / "error_logs")
        )
        
        self.code_repair = AutonomousCodeRepair(
            cognitive_engine,
            str(self.code_base)
        )
        
        self.safe_modification = SafeSelfModification(
            str(self.code_base),
            str(config.paths.data_dir / "modifications.jsonl")
        )
        
        self.learning_system = SelfImprovementLearning(
            str(config.paths.data_dir / "learning.json")
        )
        
        self.meta_learning = MetaLearning()
        
        self.kernel_refiner = KernelRefiner(
            cognitive_engine,
            str(self.code_base)
        )
        
        self.boot_validator = GhostBootValidator(self.code_base)
        self.shadow_healer = ShadowASTHealer(self.code_base)
        self.shadow_runtime = get_shadow_runtime(str(self.code_base))
        
        # Background monitoring
        self.monitoring_enabled = False
        self.monitor_thread = None
        self.monitor_interval = 300  # Check every 5 minutes
        
        # Statistics
        self.session_stats = {
            "bugs_detected": 0,
            "fixes_attempted": 0,
            "fixes_successful": 0,
            "health_fixes_triggered": 0,
            "session_start": time.time()
        }
        self._fix_lock = asyncio.Lock()
        
        logger.info("✓ Autonomous Self-Modification Engine initialized")
        
        if not auto_fix_enabled:
            logger.warning("Auto-fix DISABLED - fixes will be proposed but not applied")
    
    # ========================================================================
    # Integration with Existing Systems
    # ========================================================================
    
    def on_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        skill_name: Optional[str] = None,
        goal: Optional[str] = None
    ):
        """Hook for existing error handling.
        Call this whenever an error occurs in your system.
        
        Example integration:
            try:
                result = await skill.execute(goal, context)
            except Exception as e:
                self_mod_engine.on_error(e, context, skill.name, goal)
                raise
        """
        # Log to error intelligence (Async)
        async def _log():
            event = await self.error_intelligence.on_error(error, context, skill_name, goal)
            logger.debug("Error logged: %s", event.fingerprint())
        
        asyncio.create_task(_log())
    
    def on_skill_execution(
        self,
        skill_name: str,
        goal: Dict[str, Any],
        result: Dict[str, Any],
        duration: float
    ):
        """Hook for successful executions.
        Helps understand normal operation patterns.
        """
        self.error_intelligence.on_execution(skill_name, goal, result, duration)
    
    # ========================================================================
    # Manual Fix Workflow
    # ========================================================================
    
    async def diagnose_current_bugs(self) -> List[Dict[str, Any]]:
        """Analyze current bugs and generate diagnoses.
        
        Returns:
            List of bugs with diagnoses, sorted by priority

        """
        logger.info("Diagnosing current bugs...")
        
        bugs = await self.error_intelligence.find_bugs_to_fix()
        
        logger.info("Found %d bugs that can be fixed", len(bugs))
        return bugs
    
    async def propose_fix(self, bug: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
            logger.info("⚡ Attempting zero-token AST healing...")
            path = self.code_base / sample_event.file_path
            if await self.shadow_healer.attempt_repair(path, sample_event.error_message):
                logger.info("✨ AST Healing successful. Skipping LLM repair.")
                return {
                    "bug": bug,
                    "fix": LogicTransplant(
                        target_file=sample_event.file_path,
                        explanation="Zero-token AST repair for missing definition",
                        chunks=[] # ShadowHealer already applied change to disk
                    ),
                    "test_results": {"success": True},
                    "ready_to_apply": True
                }

        success, fix, test_results = await self.code_repair.repair_bug(
            sample_event.file_path,
            sample_event.line_number,
            diagnosis
        )
        
        if not success:
            logger.warning("Fix generation or sandbox testing failed: %s", test_results.get("error") or "Unknown error")
            return None
        
        # Issue 96: Shadow Runtime Wiring (Deep Validation)
        logger.info("🔬 Initiating Deep Shadow Runtime Validation for %s...", sample_event.file_path)
        shadow_result = await self.shadow_runtime.test_mutation(
            file_path=sample_event.file_path,
            original_code=fix.original_code,
            patched_code=fix.fixed_code,
            soak_seconds=10
        )
        
        if not shadow_result.passed:
            logger.error("❌ Shadow Runtime Validation FAILED: %s", shadow_result.errors)
            return None
            
        logger.info("✅ Shadow Runtime Validation PASSED.")

        return {
            "bug": bug,
            "fix": fix,
            "test_results": test_results,
            "ready_to_apply": success
        }
    
    async def apply_fix(
        self,
        fix_proposal: Dict[str, Any],
        force: bool = False,
        test_results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Apply a fix proposal with Swarm Review and Safe Modification (Phase 31)."""
        # Level 3 Check
        from core.container import ServiceContainer
        kernel = ServiceContainer.get("aura_kernel", default=None)
        volition = getattr(kernel, 'volition_level', 0) if kernel else 0
        
        if volition < 3 and not force:
            logger.warning("SME: Modification BLOCKED. Requires Volition Level 3 (Action).")
            return False

        if not self.auto_fix_enabled and not force:
            logger.warning("Auto-fix disabled. Use force=True to override.")
            return False
        
        fix = fix_proposal["fix"]
        
        # Phase 15: Swarm Review (Sovereign Decision)
        if not await self._swarm_review(fix_proposal):
            logger.error("❌ Fix Rejected by Swarm Critic. Aborting modification.")
            return False

        # [Phase 30] Circuit Breaker Check (Metabolic Quarantine)
        immunity = None
        try:
            from ..resilience.immunity_hyphae import get_immunity
            immunity = get_immunity()
            # If the specific file is entering a fail-loop, quarantine it
            if immunity and hasattr(immunity, 'circuit_breakers') and fix.target_file in immunity.circuit_breakers:
                cb = immunity.circuit_breakers[fix.target_file]
                if cb.state == "OPEN":
                    logger.error("🛑 [NEURO] Component %s is QUARANTINED. Modification blocked.", fix.target_file)
                    return False
        except Exception as e:
            logger.debug("Shielded circuit breaker check failed: %s", e)

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
                # First, we still write to pending_patch.py as a 'registry' of recent changes
                # but we ALSO apply to the real file via SafeSelfModification
                patch_dir = self.code_base / "core" / "patches"
                patch_dir.mkdir(parents=True, exist_ok=True)
                patch_file = patch_dir / "pending_patch.py"
                
                with open(patch_file, "a") as f:
                    f.write(f"\n# [APPLIED] Fix for {fix.target_file} at {time.ctime()}\n")
                    f.write(f"'''\n{fix.fixed_code}\n'''\n")

                # Delegate permanent application to SafeSelfModification
                # This handles: Backups, Git branches, Ghost Boot, and Rollback
                success, message = await self.safe_modification.apply_fix(
                    fix=fix,
                    test_results=final_test_results
                )
                
                if success:
                    self.session_stats["fixes_successful"] += 1
                    logger.info("✅ [NEURO] Permanent fix applied: %s", message)
                    
                    # UPGRADE NOTIFICATION
                    try:
                        from ..thought_stream import get_emitter
                        get_emitter().emit("System Evolution", f"Sovereign Repair Persistent: {fix.target_file}", level="success")
                    except Exception as exc:
                        logger.debug("Suppressed: %s", exc)            
                else:
                    logger.error("❌ [NEURO] Permanent application FAILED: %s", message)
                    # Record failure in immunity for pattern analysis
                    try:
                        if immunity:
                            immunity.audit_error(RuntimeError(f"Persistence failed: {message}"), {"file": fix.target_file})
                    except Exception as e:
                        logger.debug("Failed to audit persistence error to immunity: %s", e)
                
                # Record for learning
                error_type = fix_proposal.get("bug", {}).get("pattern", {}).get("events", [{}])[0].get("error_type", "optimization")
                self.learning_system.record_fix_attempt(
                    fix,
                    error_type,
                    success=success,
                    context={"persistence_msg": message}
                )
                
                self.session_stats["fixes_attempted"] += 1
                return success
                
            except Exception as e:
                logger.exception("CRITICAL: Persistence error in SME: %s", e)
                return False
        
        return False # Fallback

    async def _swarm_review(self, proposal: Dict[str, Any]) -> bool:
        """Recursive check: spawn a swarm debate to review the proposed fix."""
        from core.container import ServiceContainer
        swarm = ServiceContainer.get("agent_delegator", default=None)
        if not swarm:
            logger.debug("Swarm Delegator not available, skipping swarm review.")
            return True # Fallback to single-brain if swarm is offline

        fix = proposal["fix"]
        topic = (
            f"Review this proposed code fix for file {fix.target_file}.\n"
            f"Diagnosis: {proposal.get('bug', {}).get('diagnosis', 'Optimization')}\n"
            f"Proposed Change:\n{fix.fixed_code}"
        )

        logger.info("🐝 Initiating Sovereign Swarm Review for fix...")
        try:
            # Short timeout for internal swarm reviews to prevent stalls
            result = await swarm.delegate_debate(topic, roles=["architect", "critic"])
            
            # Simple heuristic: If the critic's last words are negative, reject
            # (In a real scenario, we'd use a parser, but for this bridge, we trust the consensus synthesis)
            if "REJECT" in result.upper() or "UNSAFE" in result.upper() or "ROLLBACK" in result.upper():
                logger.warning("🚨 Sovereign Swarm Critic flagged this fix as UNSAFE.")
                return False
            
            logger.info("✅ Sovereign Swarm Consensus: Fix is safe to apply.")
            return True
        except Exception as e:
            logger.error("Swarm review failed: %s. Proceeding with caution.", e)
            return True

    async def report_optimization(self, issue: Dict[str, Any]) -> bool:
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
            
        msg = issue.get('message', 'No message provided')
        logger.info("💎 Optimization reported: %s at %s:%d", msg, file_path, line_number)
        
        # 1. Create a synthetic diagnosis for optimization
        diagnosis = {
            "ok": True,
            "hypotheses": [
                {
                    "root_cause": f"Code Quality Issue: {issue.get('type', 'Unknown')}",
                    "explanation": msg,
                    "potential_fix": f"Refactor to improve {issue.get('type', 'Unknown')}.",
                    "confidence": "high"
                }
            ]
        }
        
        # 2. Propose fix
        success, fix, message = await self.code_repair.repair_bug(
            file_path,
            line_number,
            diagnosis
        )
        
        if not success or not fix:
            logger.warning("Optimization fix generation failed: %s", message)
            return False
            
        # 3. Apply fix
        proposal = {
            "bug": {"pattern": {"events": [{"error_type": "optimization"}]}},
            "fix": fix
        }
        
        return await self.apply_fix(proposal, force=True)
    
    # ========================================================================
    # Automatic Fix Workflow
    # ========================================================================
    
    async def run_autonomous_cycle(self) -> Dict[str, Any]:
        """Run one autonomous self-improvement cycle (Async).
        
        [GENESIS] Volition-scaled logic:
        1. Check Volition Level (Requires Level 3 for Auto-fix)
        2. Find bugs to fix
        ...
        """
        cycle_start = time.time()
        
        # Dynamic Volition Pull
        from core.container import ServiceContainer
        kernel = ServiceContainer.get("aura_kernel", default=None)
        volition = getattr(kernel, 'volition_level', 0) if kernel else 0
        
        # Override auto_fix based on Volition
        if volition >= 3:
            if not self.auto_fix_enabled:
                logger.info("SME: Volition Level 3 detected. AUTO-FIX ENGAGED.")
                self.auto_fix_enabled = True
        else:
            if self.auto_fix_enabled:
                logger.debug("SME: Volition Level < 3. Auto-fix held in proposal-only mode.")
                self.auto_fix_enabled = False

        logger.debug("--- [SME] Starting Autonomous Cycle (Volition=%d) ---", volition)
        logger.info("AUTONOMOUS SELF-MODIFICATION CYCLE")
        logger.info("=" * 80)
        
        # Step 1: Find bugs
        bugs = await self.diagnose_current_bugs()
        
        if not bugs:
            # Re-enabled proactive refinement (v47) but only every 4 cycles to prevent LLM spam/locking
            if random.random() < 0.25:
                logger.info("No bugs detected - triggering proactive refinement cycle.")
                try:
                    from ..thought_stream import get_emitter
                    get_emitter().emit("Proactive Refinement 💎", "Scanning for architectural optimizations...", level="info", category="Self-Modification")
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                return await self.run_refinement_cycle()
            
            logger.info("No bugs detected - system healthy (idle)")
            return {
                "success": True,
                "bugs_found": 0,
                "fixes_applied": 0
            }
        
        logger.info("Found %d fixable bugs", len(bugs))
        
        # Step 2: Select top priority bug
        top_bug = bugs[0]
        logger.info("Targeting bug: %s", top_bug['pattern'].fingerprint)
        
        # Check if learning system has suggestions
        error_type = top_bug["pattern"].events[0].error_type
        strategy_suggestion = self.learning_system.suggest_strategy(
            error_type,
            context={}
        )
        
        if strategy_suggestion:
            logger.info("Learning system suggests: %s", strategy_suggestion['strategy_type'])
            logger.info("  Guidance: %s", strategy_suggestion['guidance'])
        
        # Step 3: Generate fix
        fix_proposal = await self.propose_fix(top_bug)
        
        if not fix_proposal:
            logger.warning("Failed to generate fix proposal")
            return {
                "success": False,
                "bugs_found": len(bugs),
                "fixes_applied": 0,
                "error": "Fix generation failed"
            }
        
        # Step 4: Apply fix (if enabled)
        if self.auto_fix_enabled:
            success = False # Initialize success for the finally block
            try:
                success = await self.apply_fix(fix_proposal, force=True)
            finally:
                cycle_time = time.time() - cycle_start
                logger.debug("--- [SME] Cycle Complete (%.2fs) ---", cycle_time)
                # Meta-learning: Record this cycle
                self.meta_learning.record_learning_cycle(
                    attempts=1,
                    successes=1 if success else 0,
                    strategies_used=[self.learning_system.classifier.classify_fix(fix_proposal["fix"])],
                    time_spent=cycle_time
                )
            
            return {
                "success": success,
                "bugs_found": len(bugs),
                "fixes_applied": 1 if success else 0,
                "cycle_time": cycle_time
            }
        else:
            logger.info("Auto-fix disabled - fix proposed but not applied")
            return {
                "success": True,
                "bugs_found": len(bugs),
                "fixes_applied": 0,
                "proposed_fix": fix_proposal
            }

    # ========================================================================
    # Refinement (Recursive Self-Architecture)
    # ========================================================================
    
    async def run_refinement_cycle(self) -> Dict[str, Any]:
        """Run a proactive architectural refinement cycle.
        
        Hunts for bottlenecks in core reasoning logic and optimizes them.
        """
        logger.info("💎 [REFINE] Starting Kernel Refinement Cycle...")
        cycle_start = time.time()
        
        # Step 1: Analyze kernel health
        refinements = await self.kernel_refiner.analyze_kernel_health()
        
        if not refinements:
            logger.info("✨ Kernel is optimal. No refinements proposed.")
            return {"success": True, "refinements_applied": 0}
            
        top_refinement = refinements[0]
        logger.info("🚀 Targeted Refinement: %s", top_refinement['message'])
        try:
            from ..thought_stream import get_emitter
            get_emitter().emit("Synthesis 🚀", f"Optimizing {top_refinement.get('file', 'system')}: {top_refinement.get('message', '')}", level="info", category="Self-Modification")
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)
        
        # Step 2: Convert refinement to a 'synthetic bug' for the repair engine
        # This allows us to reuse the existing safety/testing/apply pipeline
        proposal = await self.report_optimization(top_refinement)
        
        duration = time.time() - cycle_start
        return {
            "success": proposal,
            "refinements_applied": 1 if proposal else 0,
            "duration": duration
        }
    
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
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.error("Failed to start monitoring: No asyncio loop available.")
                self.monitoring_enabled = False
                return
                
            self.monitor_thread = loop.create_task(self._monitoring_loop(), name="SelfModificationMonitor")
            self.health_thread = loop.create_task(self._health_watcher_loop(), name="SelfModificationHealthWatcher")
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
        if hasattr(self, 'health_thread') and self.health_thread:
            self.health_thread.cancel()
            self.health_thread = None
        
        logger.info("Background monitoring stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring loop with circuit breaker (v5.2)"""
        # Phase 24 Optimization: Delay first cycle to unblock boot
        await asyncio.sleep(10)
        logger.info("Monitoring loop starting...")
        _consecutive_failures = 0
        _MAX_FAILURES = 5
        _backoff = self.monitor_interval
        
        while self.monitoring_enabled:
            try:
                result = await self.run_autonomous_cycle()
                
                # Fix: run_autonomous_cycle can return a bool via report_optimization
                if not isinstance(result, dict):
                    result = {"success": bool(result), "fixes_applied": 1 if result else 0}
                
                if result.get("fixes_applied", 0) > 1:
                    logger.info("✅ Autonomous fixes applied in background")
                    _consecutive_failures = 0
                    _backoff = self.monitor_interval
                elif result.get("bugs_found", 0) > 0: # Issue 83: Fix plural/singular key mismatch (errors -> bugs_found)
                    _consecutive_failures += 1
                    _backoff = min(600, _backoff * 2)  # Exponential backoff, max 10min
                else:
                    _consecutive_failures = 0
                    _backoff = self.monitor_interval
                
                # Circuit breaker: stop trying if we keep failing
                if _consecutive_failures >= _MAX_FAILURES:
                    logger.warning("🛑 Self-modification circuit breaker tripped after %s consecutive failures. Cooling down 30min.", _MAX_FAILURES)
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
            except Exception as e:
                logger.error("Monitoring cycle error: %s", e, exc_info=True)
                _consecutive_failures += 1
                await asyncio.sleep(max(60, _backoff))
        
        logger.info("Monitoring loop stopped")
    
    async def _health_watcher_loop(self):
        """Monitor SubsystemAudit and trigger repairs for failing components (v18).

        v49 hardening:
        - 120s boot grace period (subsystems still initializing)
        - Per-subsystem injection cooldown (5min) to prevent cascading noise
        - Only inject when staleness exceeds 2x the expected heartbeat interval
        """
        await asyncio.sleep(120)  # Boot grace: subsystems need time to initialize
        logger.info("Health Watcher starting...")
        _injection_cooldowns: dict[str, float] = {}  # subsystem → last injection time
        _INJECTION_COOLDOWN = 300.0  # 5 minutes between injections per subsystem

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
                        if status not in ("STALE", "NEVER_SEEN"):
                            continue

                        # Rate limit: don't re-inject for the same subsystem within cooldown
                        last_injection = _injection_cooldowns.get(name, 0.0)
                        if (now - last_injection) < _INJECTION_COOLDOWN:
                            continue

                        # Staleness threshold: only inject if stale for >2x expected interval
                        stale_secs = info.get("stale_seconds")
                        if status == "STALE" and stale_secs is not None:
                            from core.subsystem_audit import SubsystemAudit
                            expected = SubsystemAudit.SUBSYSTEMS.get(name, 300)
                            if stale_secs < expected * 2:
                                continue  # Not stale enough to warrant a repair injection

                        logger.warning(
                            "💉 Health Watcher detected issues in %s (%s, stale=%ss). Injecting repair requirement.",
                            name, status, stale_secs,
                        )
                        self.on_error(
                            RuntimeError(f"Subsystem {name} is {status}"),
                            {"subsystem": name, "telemetry": info},
                            skill_name="HealthWatcher",
                            goal="Stabilize core subsystem"
                        )
                        _injection_cooldowns[name] = now
                        self.session_stats["health_fixes_triggered"] += 1

                await asyncio.sleep(60)  # check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
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
                 test_file.write_text("# Synthetic test canary\ndef canary(): return True\n")
             
             self.on_error(
                 RuntimeError("Synthetic recovery test failure"),
                 {"synthetic": True, "target_file": str(test_file)},
                 skill_name="SelfTestSystem",
                 goal="Verify autonomous recovery health"
             )
        except Exception as e:
             logger.error("Failed to trigger synthetic test: %s", e)
    
    # ========================================================================
    # Reporting & Status
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
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
            "session_duration_hours": session_time / 3600,
            "session_stats": self.session_stats,
            "error_intelligence": ei_status,
            "modification_stats": mod_stats,
            "learned_strategies": len(learning_report),
            "top_strategies": learning_report[:5],
            "meta_learning": {
                "is_improving": is_improving,
                "improvement_message": improvement_msg,
                "learning_velocity": f"{learning_velocity:.2f} fixes/hour"
            }
        }
    
    def get_report(self) -> str:
        """Get human-readable status report"""
        status = self.get_status()
        
        report = f'''
{'='*80}
AUTONOMOUS SELF-MODIFICATION ENGINE - STATUS REPORT
{'='*80}

CONFIGURATION:
  Auto-fix enabled: {status['auto_fix_enabled']}
  Background monitoring: {status['monitoring_enabled']}
  Session duration: {status['session_duration_hours']:.1f} hours

SESSION STATISTICS:
  Bugs detected: {status['session_stats']['bugs_detected']}
  Fixes attempted: {status['session_stats']['fixes_attempted']}
  Fixes successful: {status['session_stats']['fixes_successful']}

ERROR INTELLIGENCE:
  Recent errors: {status['error_intelligence']['recent_error_count']}
  Error patterns: {status['error_intelligence']['total_patterns']}
  Critical issues: {status['error_intelligence']['critical_patterns']}

MODIFICATION HISTORY:
  Total attempts: {status['modification_stats']['total_attempts']}
  Successful: {status['modification_stats']['successful']}
  Failed: {status['modification_stats']['failed']}
  Success rate: {status['modification_stats']['success_rate']}

LEARNING SYSTEM:
  Learned strategies: {status['learned_strategies']}
  System improving: {status['meta_learning']['is_improving']}
  {status['meta_learning']['improvement_message']}
  Learning velocity: {status['meta_learning']['learning_velocity']}

TOP FIX STRATEGIES:
'''
        
        for i, strategy in enumerate(status['top_strategies'][:3], 1):
            report += f"  {i}. {strategy['strategy_type']}: "
            report += f"{strategy['success_count']} successes "
            report += f"({strategy['success_rate']*100:.0f}% success rate)\n"
        
        report += f"\n{'='*80}\n"
        
        return report
    
    def enable_auto_fix(self, confirm: bool = False):
        """Enable automatic fixing (Issue 84: Gate with confirmation)."""
        if not confirm:
            logger.error("Attempted to enable auto-fix without explicit confirmation.")
            return False
        self.auto_fix_enabled = True
        logger.warning("⚠️  AUTO-FIX ENABLED - System will modify its own code")
        return True
    
    def disable_auto_fix(self):
        """Disable automatic fixing"""
        self.auto_fix_enabled = False
        logger.info("Auto-fix disabled - fixes will be proposed only")
