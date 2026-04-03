"""
Aura Cognitive Integration Layer v5.0
====================================
Synthesizes the modular intelligence pipeline into a single service.
This class acts as the 'Advanced Cognition' hub, coordinating the
CognitiveKernel, InnerMonologue, and LanguageCenter.
"""
import logging
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from core.config import config
from core.container import ServiceContainer
from core.brain.reflex import get_reflex

logger = logging.getLogger("Aura.Cognition")

class CognitiveIntegrationLayer:
    """
    Advanced Cognitive Integration Layer.
    Orchestrates the new intelligence pipeline (Kernel -> Monologue -> LanguageCenter).
    """

    def __init__(self, orchestrator: Any = None, base_data_dir: Optional[str] = None):
        self.orchestrator = orchestrator
        self.base_data_dir = Path(base_data_dir) if base_data_dir else config.paths.home_dir
        self.kernel = None
        self.monologue = None
        self.language_center = None
        self._initialized = False
        self._reflex_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AuraReflex")

    def setup(self):
        """Synchronous setup phase."""
        logger.info("🧠 CognitiveIntegrationLayer: Synchronous setup beginning...")
        # Placeholder for any sync initialization
        pass

    async def initialize(self) -> bool:
        """Asynchronous initialization of components."""
        if self._initialized:
            return True

        logger.info("🧠 CognitiveIntegrationLayer: Initializing Advanced Intelligence Pipeline...")
        try:
            # 1. Resolve or Instantiate Components
            # We try to get them from the container first, then instantiate if missing
            self.kernel = ServiceContainer.get("cognitive_kernel", default=None)
            if not self.kernel:
                from core.cognitive_kernel import get_cognitive_kernel
                self.kernel = get_cognitive_kernel()
            
            await self.kernel.start()
            def _safe_register(name, instance):
                try:
                    ServiceContainer.register_instance(name, instance)
                except Exception as register_err:
                    logger.warning(f"⚠️ [BOOT] Could not register '{name}' in ServiceContainer: {register_err}")

            _safe_register("cognitive_kernel", self.kernel)

            # 2. Resolve or Instantiate InnerMonologue
            try:
                self.monologue = ServiceContainer.get("inner_monologue", default=None)
                if not self.monologue:
                    from core.inner_monologue import get_inner_monologue
                    self.monologue = get_inner_monologue()
                
                # Check if it's already started or needs initialization
                if hasattr(self.monologue, "start"):
                    await self.monologue.start()
                _safe_register("inner_monologue", self.monologue)
            except Exception as e:
                logger.warning("InnerMonologue failed to resolve: %s. Proceeding in degraded mode.", e)

            # 3. Resolve or Instantiate LanguageCenter
            try:
                self.language_center = ServiceContainer.get("language_center", default=None)
                if not self.language_center:
                    from core.language_center import get_language_center
                    self.language_center = get_language_center()
                
                if hasattr(self.language_center, "start"):
                    await self.language_center.start()
                _safe_register("language_center", self.language_center)
            except Exception as e:
                logger.warning("LanguageCenter failed to resolve: %s. Proceeding in degraded mode.", e)

            self._initialized = True
            logger.info("✅ CognitiveIntegrationLayer initialized successfully.")
            return True
        except Exception as e:
            logger.error("❌ CognitiveIntegrationLayer initialization FAILED: %s", e, exc_info=True)
            # [RECOVERY] One-time force-reload attempt for critical components
            if not getattr(self, "_retrying_init", False):
                self._retrying_init = True
                logger.warning("🧠 [RECOVERY] Retrying CognitiveIntegrationLayer initialization once...")
                await asyncio.sleep(1.0)
                return await self.initialize()
            return False

    async def evaluate(self, user_input: str, history: list = None) -> Any:
        """Primary entrance for thoughts."""
        if not self.kernel:
            return None
        return await self.kernel.evaluate(user_input, history or [])

    async def process_turn(self, message: str, context: Optional[dict] = None) -> str:
        """
        The standardized entry point for the Orchestrator's Phase 7 pipeline.
        Orchestrates Kernel evaluation -> InnerMonologue (planned) -> LanguageCenter expression.
        """
        if not self.is_active:
            await self.initialize()

        # Phase 23.4: Conceptual Engine Integration (Ava & Cortana)
        ava = ServiceContainer.get("ava", default=None)
        cortana = ServiceContainer.get("cortana", default=None)
        
        if ava:
            try:
                # Ava builds a social model of the user from the input
                ava.analyze_message(message, is_user=True)
            except Exception as e:
                logger.debug("Ava analysis failed: %s", e)

        # 0. Reflexive Path (Fast Fallback - Thread Isolated)
        reflex = get_reflex()
        # Offload to dedicated thread to avoid event-loop starvation
        reflex_response = await asyncio.get_running_loop().run_in_executor(
            self._reflex_executor, reflex.process, message
        )
        if reflex_response:
            logger.info("⚡ [REFLEX] Instant response generated (Thread Isolated).")
            return reflex_response

        if not self.kernel:
            logger.error("CognitiveIntegrationLayer: Kernel missing during process_turn.")
            return "Cognitive kernel offline."

        # 1. Evaluate (Kernel reasoning)
        # kernel.evaluate returns a CognitiveBrief
        brief = await self.kernel.evaluate(message, context=context)

        # Agency Integration: Execute tools if needed
        # v1.1 FIX: This restores Aura's ability to 'look things up' in the CogV5 pipeline.
        if brief.requires_research:
            try:
                # AgencyCoordinator is registered as agency_coordinator in ServiceContainer
                agency = ServiceContainer.get("agency_coordinator", default=None)
                if agency:
                    logger.info("🔍 [AGENCY] Tool use required. Dispatching to AgencyCoordinator.")
                    # Direct skill trigger for research
                    search_res = await agency.execute_skill("web_search", {"query": message})
                    if isinstance(search_res, dict) and search_res.get("ok"):
                        findings = search_res.get("result", "")
                        if findings:
                            logger.info("✅ [AGENCY] Research findings captured.")
                            # Inject findings as key points so the LanguageCenter sees them
                            brief.key_points.append(f"RESEARCH FINDINGS: {findings}")
                            if hasattr(brief, "internal_notes"):
                                brief.internal_notes += f"\n[Agentic Research Result]: {findings}"
                else:
                    logger.warning("AgencyCoordinator missing from container during research-required turn.")
            except Exception as e:
                logger.error("Agency resolution failed in CIL: %s", e)
        
        # 2. Express (LanguageCenter expression)
        if self.language_center:
            try:
                if self.monologue:
                    # Fix: InnerMonologue uses 'think', not 'process'
                    packet = await self.monologue.think(message, brief)
                    return await self.language_center.express(packet, message)
                else:
                    # Fallback: Shim a ThoughtPacket for the LanguageCenter
                    from core.inner_monologue import ThoughtPacket
                    packet = ThoughtPacket(
                        stance=brief.prior_beliefs[0] if brief.prior_beliefs else "I approach this with curiosity.",
                        primary_points=brief.key_points,
                        constraints=brief.avoid,
                        tone="direct",
                        length_target=brief.complexity if brief.complexity in ("brief", "medium", "extended") else "medium",
                        model_tier="local"
                    )
                    return await self.language_center.express(packet, message)
            except Exception as e:
                logger.exception("Error during cognitive expression: %s", e)
                final_response = "I'm processing that. Give me a second—my internal monologue is a bit of a maze right now."
        else:
            final_response = "I'm having a hard time putting my thoughts into words at the moment. My language center seems to be offline."
            
        # Post-process with Cortana (CognitiveHealthMonitor)
        if cortana:
            try:
                # Approximate token counts for health monitoring
                ctx_tokens = len(str(context)) // 4 if context else 0
                max_tokens = 8192
                quality = 0.9 if not brief.error else 0.3
                
                cortana.record_turn(
                    context_tokens=ctx_tokens,
                    max_tokens=max_tokens,
                    response_quality=quality,
                    identity_markers_present=True,
                    topics_in_play=len(brief.key_points),
                    resolved_topics=1
                )
                
                # If Cortana determines context is saturated, trigger memory eviction
                if cortana.should_prune():
                    logger.warning("🧠 Cortana: Cognitive Overload detected. Evicting oldest context layers.")
                    mem = ServiceContainer.get("memory_facade", default=None)
                    if mem and hasattr(mem, "prune_context"):
                         mem.prune_context()
            except Exception as e:
                logger.debug("Cortana turn recording failed: %s", e)
                
        # Post-process with Ava for the response
        if ava:
            try:
                ava.analyze_message(final_response, is_user=False)
            except Exception as e:
                logger.debug("Ava response analysis failed: %s", e)
                
        return final_response

    async def process_autonomous(self) -> Optional[str]:
        """
        Entry point for autonomous background thoughts.
        Generates an internal inquiry and processes it through the pipeline.
        """
        if not self.is_active:
            await self.initialize()

        if not self.kernel or not self.monologue:
            return None

        try:
            # 1. Generate an autonomous "spark" or inquiry
            # We can use a default prompt or pull from curiosity/drives
            spark = "I'm reflecting on my current state and recent interactions. What should I explore or deepen?"
            
            # 2. Evaluate via Kernel
            brief = await self.kernel.evaluate(spark, context={"autonomous": True})
            
            # 3. Deepen via Monologue
            packet = await self.monologue.think(spark, brief)
            
            # 4. Express (internally)
            if self.language_center:
                return await self.language_center.express(packet, spark)
            
            return brief.stance
        except Exception as e:
            logger.error("Autonomous thought processing failed in CIL: %s", e)
            return None

    async def record_interaction(self, message: str, response: str, domain: str = "general"):
        """Commits a conversation turn to the memory system."""
        try:
            mem = ServiceContainer.get("memory_facade", default=None)
            if mem and hasattr(mem, "commit_interaction"):
                logger.info("💾 [MEMORY] Recording interaction to Episodic/Vector systems.")
                await mem.commit_interaction(
                    context=f"User: {message[:200]}",
                    action="conversation_turn",
                    outcome=f"Aura: {response[:500]}",
                    success=True,
                    metadata={"domain": domain}
                )
        except Exception as e:
            logger.error("Failed to record cognitive interaction: %s", e)

    async def think(self, user_input: str) -> str:
        """
        End-to-end cognitive run (legacy behavior).
        Matches the interface used by some older components.
        """
        if not self.kernel:
            return "Cognition offline."
        
        brief = await self.kernel.evaluate(user_input)
        if self.language_center:
            # We must use a ThoughtPacket here as well
            from core.inner_monologue import ThoughtPacket
            packet = ThoughtPacket(
                stance=brief.prior_beliefs[0] if brief.prior_beliefs else "...",
                primary_points=brief.key_points,
                tone="direct",
                length_target="medium",
                model_tier="local"
            )
            return await self.language_center.express(packet, user_input)
        
        return "I'm thinking about it, but I'm having trouble articulating it right now."

    @property
    def is_active(self) -> bool:
        return self._initialized and self.kernel is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "kernel_ready": self.kernel is not None,
            "monologue_ready": self.monologue is not None,
            "language_center_ready": self.language_center is not None,
        }
