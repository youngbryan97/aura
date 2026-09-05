"""Meta-Cognition Engine for Aura.

Orchestrates the 'Self-Evolution Loop' by coordinating audit, 
patch generation, and safe application of core logic improvements.
"""
from core.runtime.errors import record_degradation
import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from core.runtime.base_module import AuraBaseModule
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Cognition.Meta")

#: How many queued optimization requests are retained. Beyond this the
#: oldest is dropped, loudly — the loop consumes one per cycle, so a backlog
#: this deep is already many cycles of stale intent.
PENDING_CURIOSITY_LIMIT = 50

class MetaEvolutionEngine(AuraBaseModule):
    """The engine for recursive self-optimization and transcendence."""

    def __init__(self):
        super().__init__("MetaEvolution")
        self.last_optimization_time = 0
        self._is_optimizing = False
        # CP126 (core/curiosity_engine.py): "Curiosity mutates another
        # subsystem's private queue. When no public queue exists, the engine
        # creates and appends to meta_evo._pending_curiosity directly. The
        # list is unsynchronized, unbounded, undurable, and outside any
        # governed write contract."
        #
        # The queue is now bounded by construction (a deque drops its own
        # oldest rather than the owner rebinding a slice, which was racy: a
        # reader holding the old list saw a snapshot that silently stopped
        # being the queue), guarded by a ranked lock, and reachable only
        # through queue_optimization().
        self._pending_curiosity: Deque[Dict[str, Any]] = deque(maxlen=PENDING_CURIOSITY_LIMIT)
        self._pending_lock = checked_lock("meta_evolution.pending_curiosity", rank=LockRank.LEAF)
        self._dropped_curiosity = 0
        logger.info("⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)")

    async def evolve(self, target_area: str = None) -> Dict[str, Any]:
        """Alias for run_optimization_cycle — called by orchestrator scheduler."""
        return await self.run_optimization_cycle(target_area=target_area)

    def queue_optimization(self, target_area: Optional[str] = None, context: Optional[str] = None):
        """Queue an optimization request for the next cycle.

        The only supported way in. Called by the Curiosity Engine and
        internal monitors; anything that reaches past this into the queue
        itself writes a record the consumer cannot read (it looks for
        ``context`` and ``target_area``, and a foreign schema is dropped in
        silence).
        """
        record = {
            "target_area": target_area,
            "context": context,
            "timestamp": time.time(),
        }
        with self._pending_lock:
            if len(self._pending_curiosity) == self._pending_curiosity.maxlen:
                # Say when work is being discarded. A queue that silently
                # forgets its oldest item under load looks identical to one
                # that is keeping up.
                self._dropped_curiosity += 1
                logger.warning(
                    "Pending-curiosity queue full (%d); dropping the oldest "
                    "request. %d dropped since boot.",
                    PENDING_CURIOSITY_LIMIT,
                    self._dropped_curiosity,
                )
            self._pending_curiosity.append(record)

        logger.info("📋 Queued autonomous optimization: %s", (context or "No context")[:100])

    def take_pending_optimization(self) -> Optional[Dict[str, Any]]:
        """Pop the oldest queued request, or None. The only supported way out."""
        with self._pending_lock:
            if not self._pending_curiosity:
                return None
            return self._pending_curiosity.popleft()

    def pending_optimization_count(self) -> int:
        with self._pending_lock:
            return len(self._pending_curiosity)

    async def run_optimization_cycle(self, target_area: Optional[str] = None) -> Dict[str, Any]:
        """Runs a complete self-optimization cycle.
        
        Steps:
        1. Self-Audit (via Scratchpad)
        2. Diagnosis (via SelfModificationEngine)
        3. Patch Generation (via Hephaestus)
        4. Safe Application (via SelfModificationEngine)
        """
        if self._is_optimizing:
            return {"ok": False, "error": "Optimization cycle already in progress."}

        self._is_optimizing = True
        try:
            mycelium = get_runtime_service("mycelial_network", default=None)
            if not mycelium:
                self._is_optimizing = False
                return {"ok": False, "error": "Mycelial Network unavailable."}

            async with mycelium.rooted_flow(
                source="meta_evolution",
                target="cognition",
                activity=f"Recursive Self-Optimization: {target_area or 'Core'}",
                timeout=120.0,
                priority=1.0
            ) as hypha:
                self.logger.info("🌀 Initiating Meta-Evolution Cycle...")
                start_time = time.time()
                
                # Subsystem Resolution
                scratchpad = get_runtime_service("scratchpad_engine", default=None)
                sme = get_runtime_service("self_modification_engine", default=None)
                hephaestus = get_runtime_service("hephaestus_engine", default=None)
                
                if not all([scratchpad, sme, hephaestus]):
                    self._is_optimizing = False
                    return {"ok": False, "error": "Missing required subsystems for meta-evolution."}

                # ISSUE-95: Metacognitive Review Efficiency
                # 1. Self-Audit (Transcendence: Incorporate Curiosity Gaps)
                curiosity = get_runtime_service("curiosity_engine", default=None)
                if curiosity:
                    gap = await curiosity.identify_knowledge_gap()
                    if gap:
                        self.logger.info("🔍 Transcendence: Identifying knowledge gap: %s", gap)
                        mycelium.route_signal("curiosity", "meta_cognition", {"gap": gap})
                        target_area = target_area or f"Integrate knowledge of {gap}"

                objective = f"Analyze performance and identify architectural bottlenecks in: {target_area or 'Core Orchestration'}"
                
                # Optimized depth based on mode
                audit_depth = 1
                cog_engine = get_runtime_service("cognitive_engine", default=None)
                if cog_engine and getattr(cog_engine, "current_mode", None) == "deliberate":
                    audit_depth = 2
                
                audit = await scratchpad.think_recursive(
                    objective=objective,
                    context={"recent_cycles": 1000, "error_priority": "high"},
                    depth=audit_depth
                )
                # CP126 92172bb9: use the DISTILLED strategy, never the raw
                # inner monologue — this value is spliced into a Hephaestus
                # prompt below and logged.
                audit_result = getattr(audit, "strategy", "") if not isinstance(audit, str) else audit
                if not getattr(audit, "ok", bool(audit_result)):
                    self._is_optimizing = False
                    return {
                        "ok": False,
                        "error": f"self-audit unavailable: {getattr(audit, 'error', 'no strategy')}",
                    }
                self.logger.info("Self-Audit complete (depth=%d). Strategy: %s", audit_depth, audit_result[:100] + "...")
                hypha.log("Audit Complete")

                # 2. Targeted Diagnosis - Skip if too recent
                if time.time() - self.last_optimization_time < 600:
                    self.logger.info("⚡ MetacognitiveReview: Skipping full diagnosis (too recent).")
                    diagnoses = []
                else:
                    diagnoses = await sme.diagnose_current_bugs()
                if not diagnoses:
                    self.logger.info("No bugs found. Proactively seeking optimizations via Hephaestus...")
                    hypha.log("No bugs — triggering proactive Deep Forge")
                    
                    # Use the audit strategy to drive a proactive optimization
                    # Hephaestus generates a logic patch based on the LLM audit findings
                    forge_target = target_area or "core/orchestrator.py"
                    
                    # Consume pending curiosity insights if available
                    forge_context = f"Optimize based on audit: {audit_result[:200]}"
                    finding = self.take_pending_optimization()
                    if finding:
                        forge_context += f" | Curiosity Insight: {finding.get('context', '') or ''}"[:300]
                        if finding.get("target_area"):
                            forge_target = finding["target_area"]
                    
                    forge_result = await hephaestus.synthesize_logic_patch(
                        forge_target,
                        forge_context
                    )
                    
                    if forge_result.get("ok"):
                        self.logger.info("🔨 Hephaestus produced a proactive patch.")
                        hypha.log("Proactive patch generated")
                        elapsed = time.time() - start_time
                        return {
                            "ok": True,
                            "applied": False,
                            "proactive_patch": True,
                            "fix": forge_result.get("fix"),
                            "message": "No bugs found. Hephaestus generated a proactive optimization patch for review.",
                            "latency": elapsed
                        }
                    else:
                        elapsed = time.time() - start_time
                        self.logger.info("System at peak health. Cycle took %.2fs.", elapsed)
                        return {"ok": True, "message": "System at peak health. No optimizations identified.", "latency": elapsed}

                # 3. Apply Top Diagnosis
                top_bug = diagnoses[0]
                proposal = await sme.propose_fix(top_bug)
                
                if proposal and proposal.get("ready_to_apply"):
                    success = await sme.apply_fix(proposal, force=False)
                    self.last_optimization_time = time.time()
                    elapsed = self.last_optimization_time - start_time
                    self.logger.info("✅ Optimization Applied in %.2fs: %s", elapsed, proposal.get('id'))
                    hypha.log(f"Optimization Applied: {proposal.get('id')}")
                    return {"ok": success, "applied": True, "proposal_id": proposal.get("id"), "latency": elapsed}
                    
                elapsed = time.time() - start_time
                self.logger.info("No valid optimizations found. Cycle took %.2fs.", elapsed)
                return {"ok": True, "applied": False, "reason": "No valid optimization proposals generated.", "latency": elapsed}

            if getattr(hypha, "failed", False):
                flow_error = getattr(hypha, "error", None)
                return {
                    "ok": False,
                    "error": str(flow_error or "rooted optimization flow failed"),
                }

            # If we exit the context manager without returning, return a success result
            return {"ok": True, "applied": False, "message": "Cycle complete."}

        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('meta_cognition', e)
            self.logger.error("Meta-Evolution cycle failed: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            self._is_optimizing = False

    async def optimize_underperforming_skills(self) -> Dict[str, Any]:
        """Analyzes audit logs for failing skills and triggers autonomous refinement."""
        if self._is_optimizing:
            return {"ok": False, "error": "Optimization in progress."}
        
        self._is_optimizing = True
        try:
            audit = get_runtime_service("audit_log", default=None)
            hephaestus = get_runtime_service("hephaestus_engine", default=None)
            if not audit or not hephaestus:
                return {"ok": False, "error": "Required optimization services missing."}
            
            # 1. Get stats for last 24h
            stats = audit.get_skill_performance_stats(since_hours=24)
            
            # 2. Identify underperformers (Success rate < 80% with at least 3 attempts)
            underperformers = [s for s in stats if s["success_rate"] < 0.8 and s["calls"] >= 3]
            
            if not underperformers:
                self.logger.info("✨ All skills performing optimally.")
                return {"ok": True, "message": "No underperforming skills found."}
            
            target = underperformers[0]
            skill_name = target["skill_name"]
            self.logger.warning("📉 Underperforming skill detected: %s (SR: %.1f%%). Triggering refinement...", 
                               skill_name, target['success_rate']*100)
            
            # 3. Trigger Refinement
            reason = f"Refactor to improve reliability. Current success rate is only {target['success_rate']*100:.1f}%."
            result = await hephaestus.refine_skill(skill_name, reason)
            
            return result
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('meta_cognition', e)
            self.logger.error("Skill optimization failed: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            self._is_optimizing = False

    def get_health(self) -> Dict[str, Any]:
        """Provides health info for the meta-layer."""
        return {
            **super().get_health(),
            "is_optimizing": self._is_optimizing,
            "last_cycle": self.last_optimization_time
        }
