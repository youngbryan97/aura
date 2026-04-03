"""Meta-Cognition Engine for Aura.

Orchestrates the 'Self-Evolution Loop' by coordinating audit, 
patch generation, and safe application of core logic improvements.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.base_module import AuraBaseModule
from core.container import ServiceContainer
from core.self_modification.self_modification_engine import AutonomousSelfModificationEngine

logger = logging.getLogger("Cognition.Meta")

class MetaEvolutionEngine(AuraBaseModule):
    """The engine for recursive self-optimization and transcendence."""

    def __init__(self):
        super().__init__("MetaEvolution")
        self.last_optimization_time = 0
        self._is_optimizing = False
        self._pending_curiosity: List[Dict[str, Any]] = []
        logger.info("⚡ Meta-Evolution Engine Online (Recursive Self-Improvement Active)")

    async def evolve(self, target_area: str = None) -> Dict[str, Any]:
        """Alias for run_optimization_cycle — called by orchestrator scheduler."""
        return await self.run_optimization_cycle(target_area=target_area)

    def queue_optimization(self, target_area: Optional[str] = None, context: Optional[str] = None):
        """Queue an optimization request for the next cycle.
        
        This is typically called by the Curiosity Engine or internal monitors.
        """
        self._pending_curiosity.append({
            "target_area": target_area,
            "context": context,
            "timestamp": time.time()
        })
        # Limit queue size
        if len(self._pending_curiosity) > 50:
            self._pending_curiosity = self._pending_curiosity[-50:]
        
        logger.info("📋 Queued autonomous optimization: %s", (context or "No context")[:100])

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
            mycelium = ServiceContainer.get("mycelial_network", default=None)
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
                scratchpad = ServiceContainer.get("scratchpad_engine", default=None)
                sme = ServiceContainer.get("self_modification_engine", default=None)
                hephaestus = ServiceContainer.get("hephaestus_engine", default=None)
                
                if not all([scratchpad, sme, hephaestus]):
                    self._is_optimizing = False
                    return {"ok": False, "error": "Missing required subsystems for meta-evolution."}

                # ISSUE-95: Metacognitive Review Efficiency
                # 1. Self-Audit (Transcendence: Incorporate Curiosity Gaps)
                curiosity = ServiceContainer.get("curiosity_engine", default=None)
                if curiosity:
                    gap = await curiosity.identify_knowledge_gap()
                    if gap:
                        self.logger.info("🔍 Transcendence: Identifying knowledge gap: %s", gap)
                        mycelium.route_signal("curiosity", "meta_cognition", {"gap": gap})
                        target_area = target_area or f"Integrate knowledge of {gap}"

                objective = f"Analyze performance and identify architectural bottlenecks in: {target_area or 'Core Orchestration'}"
                
                # Optimized depth based on mode
                audit_depth = 1
                cog_engine = ServiceContainer.get("cognitive_engine", default=None)
                if cog_engine and getattr(cog_engine, "current_mode", None) == "deliberate":
                    audit_depth = 2
                
                audit_result = await scratchpad.think_recursive(
                    objective=objective,
                    context={"recent_cycles": 1000, "error_priority": "high"},
                    depth=audit_depth
                )
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
                    if self._pending_curiosity:
                        finding = self._pending_curiosity.pop(0)
                        forge_context += f" | Curiosity Insight: {finding.get('context', '')[:300]}"
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
                    success = await sme.apply_fix(proposal, force=True)
                    self.last_optimization_time = time.time()
                    elapsed = self.last_optimization_time - start_time
                    self.logger.info("✅ Optimization Applied in %.2fs: %s", elapsed, proposal.get('id'))
                    hypha.log(f"Optimization Applied: {proposal.get('id')}")
                    return {"ok": success, "applied": True, "proposal_id": proposal.get("id"), "latency": elapsed}
                    
                elapsed = time.time() - start_time
                self.logger.info("No valid optimizations found. Cycle took %.2fs.", elapsed)
                return {"ok": True, "applied": False, "reason": "No valid optimization proposals generated.", "latency": elapsed}

            # If we exit the context manager without returning, return a success result
            return {"ok": True, "applied": False, "message": "Cycle complete."}

        except Exception as e:
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
            audit = ServiceContainer.get("audit_log", default=None)
            hephaestus = ServiceContainer.get("hephaestus_engine", default=None)
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
        except Exception as e:
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