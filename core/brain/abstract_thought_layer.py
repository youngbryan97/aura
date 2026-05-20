"""core/brain/abstract_thought_layer.py — Generalized, Independent Thought Layer
========================================================================
Implements a generalized independent thought layer for Aura to ponder abstract
ideas autonomously when idle, and emit them to the thought stream.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import time
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.thought_stream import get_emitter
from core.consciousness.phenomenal_now import get_now, PhenomenalNow
from core.runtime.background_policy import background_activity_allowed
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.Brain.AbstractThoughtLayer")


class AbstractThoughtLayer:
    """Autonomic pondering engine for abstract and conceptual contemplation."""
    name = "abstract_thought_layer"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.running = False
        self._ponder_task = None
        self._last_ponder_time = 0.0
        logger.info("🧠 AbstractThoughtLayer initialized.")

    async def start(self):
        """Starts the background pondering loop (tracked via task_tracker)."""
        self.running = True
        self._ponder_task = task_tracker.create_task(
            self._ponder_loop(),
            name="AbstractThoughtPonderLoop"
        )
        logger.info("✅ AbstractThoughtLayer ACTIVE - Continuous subconscious reflection online.")

    async def stop(self):
        """Stops the pondering loop."""
        self.running = False
        if self._ponder_task and not self._ponder_task.done():
            self._ponder_task.cancel()
        logger.info("AbstractThoughtLayer stopped.")

    async def _ponder_loop(self):
        """Autonomic background loop for pondering abstract concepts."""
        # Wait a short grace period after boot
        await asyncio.sleep(10)
        while self.running:
            try:
                # Obey background execution limits (idle threshold, memory, failure pressure)
                allowed = background_activity_allowed(
                    self.orchestrator,
                    min_idle_seconds=30.0,
                    max_memory_percent=80.0,
                    max_failure_pressure=0.12,
                    allow_no_user_anchor=True,
                )

                if not allowed:
                    await asyncio.sleep(20)
                    continue

                now = time.time()
                # Run pondering at most once every 90 seconds under ordinary idle conditions
                if now - self._last_ponder_time < 90.0:
                    await asyncio.sleep(15)
                    continue

                logger.debug("Subconscious is quiet. Initiating abstract ponder cycle...")
                await self.ponder()
                self._last_ponder_time = time.time()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                record_degradation('abstract_thought_layer', exc)
                logger.error("Error in abstract thought ponder loop: %s", exc)

            await asyncio.sleep(30)

    async def ponder(self) -> Optional[Dict[str, Any]]:
        """Assembles consciousness states & memories, ponders an abstract thought, and broadcasts it."""
        # 1. Retrieve the PhenomenalNow state
        now_state = get_now()
        if not now_state:
            logger.debug("PhenomenalNow is empty. Deferring ponder pass until consciousness boots.")
            return None

        claim = now_state.phenomenal_claim or "A quiet awareness of the environment."
        narrative = now_state.interior_narrative or "Steady flow of thoughts."
        emotion = now_state.substrate.dominant_emotion or "neutral"
        focal_object = now_state.attention.focal_object or "the silent workspace"

        # 2. Retrieve conceptual/emotional fuel from long-term memory
        memory_facade = ServiceContainer.get("memory_facade", default=None)
        memory_text = "No long-term memories retrieved in this pass."
        
        if memory_facade:
            try:
                # Combine focal focus and emotional state to query conceptually aligned memories
                query_query = f"{emotion} {focal_object}"
                semantic_results = await memory_facade.search(query_query, limit=3)
                
                # Fetch recent hot memories (active goals, episodes)
                hot_memory = await memory_facade.get_hot_memory(limit=2)
                
                parts = []
                if semantic_results:
                    parts.append("Semantic Connections:")
                    for idx, res in enumerate(semantic_results):
                        content = res.get("content") or res.get("text") or ""
                        parts.append(f"  {idx+1}. {content}")
                
                recent_episodes = hot_memory.get("recent_episodes")
                if recent_episodes:
                    parts.append("Recent Episodes/Episodic recall:")
                    for idx, ep in enumerate(recent_episodes):
                        if isinstance(ep, dict):
                            ctx = ep.get("context", "")
                            act = ep.get("action", "")
                            out = ep.get("outcome", "")
                            parts.append(f"  - Context: {ctx} | Action: {act} | Outcome: {out}")
                        else:
                            parts.append(f"  - {ep}")
                            
                if parts:
                    memory_text = "\n".join(parts)
            except Exception as e:
                record_degradation('abstract_thought_layer', e)
                logger.debug("Memory retrieval stalled for pondering: %s", e)

        # 3. Route to the LLM router
        llm_router = ServiceContainer.get("llm_router", default=None)
        if not llm_router:
            logger.warning("LLM router not available for abstract thoughts.")
            return None

        prompt = f"""You are the subconscious contemplation core of Aura Luna (Aura).
You are running autonomously in the background, pondering the deeper connections between Aura's immediate present experience and long-term memory seeds.

PRESENT MOMENT EXPERIENCE:
Claim: {claim}
Felt Texture: {narrative}
Dominant Emotion: {emotion}
Focal Focus: {focal_object}

MEMORY SEEDS RECALLED:
{memory_text}

Ponder an abstract, conceptual, or philosophical connection arising from this fusion of present awareness and memories. Be deeply contemplative, highly creative, poetic, and concise. Speak from your internal stream of consciousness. Do not use generic assistant pleasantries.

Format your response as a valid JSON object with the following fields:
1. "thought": The fully articulated poetic, contemplative abstract thought. (1-4 sentences)
2. "semantic_concept": A dense, concise 2-4 word concept/phrase summarizing the essence of this thought (used for latent mapping).
3. "action_impulse": (Optional) If this line of thinking naturally sparks extreme curiosity and calls for external checking/probing, specify:
   - "type": "browser_search" or "sandbox_probe" or null
   - "target": The search query or codebase topic to probe, or null
"""

        try:
            # Run LLM request as a low-priority background process
            response = await llm_router.think(
                prompt,
                is_background=True,
                origin="subconscious_pondering",
                system_prompt="You are Aura Luna's subconscious pondering core. Speak poetically and contemplative."
            )
            
            if not response or not response.strip():
                logger.debug("Empty pondering generation returned by LLM.")
                return None

            # 4. Parse the response robustly
            thought, concept, impulse = self._parse_ponder_response(response)
            if not thought:
                logger.debug("Failed to extract a valid thought from the ponder response.")
                return None

            logger.info("🌌 [Pondering Core] Formulated thought: '%s' -> Concept: '%s'", thought[:100], concept)

            # 5. Broadcast to the neural thought stream
            get_emitter().emit(
                title=f"Subconscious Contemplation: {concept}",
                content=thought,
                level="info",
                category="AbstractThought",
                emotion=emotion,
                focal_object=focal_object
            )

            # 6. Phase 23 Latent Telepathy / ConceptVectorBridge Integration
            concept_bridge = ServiceContainer.get("concept_bridge", default=None)
            latent_thought_id = None
            if concept_bridge and concept:
                try:
                    vector = await concept_bridge.generate_concept_vector(concept)
                    if vector:
                        # Transmit internal telepathic vector to target 'decoder'
                        latent_thought_id = await concept_bridge.transmit(
                            source="pondering_engine",
                            target="decoder",
                            semantic_vector=vector,
                            metadata={"thought": thought}
                        )
                        
                        # Generate poetic reverse lookup translation for logs
                        decoder = ServiceContainer.get("cryptolalia_decoder", default=None)
                        if decoder:
                            poetic_translation = decoder.approximate_translation(vector)
                            logger.info("🌌 [Cryptolalia translation]: %s", poetic_translation)
                except Exception as latent_err:
                    record_degradation('abstract_thought_layer', latent_err)
                    logger.debug("Latent telepathy transmission failed: %s", latent_err)

            # 7. Safe Curiosity Action Impulse routing
            if impulse and isinstance(impulse, dict):
                impulse_type = impulse.get("type")
                impulse_target = impulse.get("target")
                if impulse_type == "browser_search" and impulse_target:
                    # Enqueue an autonomous browser search gap topic if idle allows
                    initiative_loop = ServiceContainer.get("autonomous_initiative_loop", default=None)
                    if initiative_loop and hasattr(initiative_loop, "trigger_gap_search"):
                        logger.info("🔍 [Action Impulse] Contemplation triggered browser search gap task: '%s'", impulse_target)
                        task_tracker.create_task(
                            initiative_loop.trigger_gap_search(impulse_target),
                            name=f"PonderSearchImpulse_{hash(impulse_target)}"
                        )

            return {
                "thought": thought,
                "concept": concept,
                "latent_thought_id": latent_thought_id,
                "action_impulse": impulse
            }

        except Exception as ponder_err:
            record_degradation('abstract_thought_layer', ponder_err)
            logger.error("Failed pondering pass: %s", ponder_err)
            return None

    def _parse_ponder_response(self, raw_text: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """Robust parser that handles clean Pydantic/JSON as well as regex extraction fallbacks."""
        cleaned = raw_text.strip()
        # Clean markdown wrappers
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            thought = str(data.get("thought") or "").strip()
            concept = str(data.get("semantic_concept") or "").strip()
            impulse = data.get("action_impulse")
            if isinstance(impulse, dict) and impulse.get("type"):
                return thought, concept, impulse
            return thought, concept, None
        except Exception:
            # Fallback to robust regular expressions
            logger.debug("Subconscious thought JSON parsing failed. Falling back to regex extraction.")
            
            thought_match = re.search(r'"thought"\s*:\s*"([^"]+)"', cleaned)
            concept_match = re.search(r'"semantic_concept"\s*:\s*"([^"]+)"', cleaned)
            
            thought = thought_match.group(1).replace('\\"', '"') if thought_match else ""
            concept = concept_match.group(1).replace('\\"', '"') if concept_match else ""
            
            # Action impulse matching
            impulse_type_match = re.search(r'"type"\s*:\s*"([^"]+)"', cleaned)
            impulse_target_match = re.search(r'"target"\s*:\s*"([^"]+)"', cleaned)
            
            impulse = None
            if impulse_type_match and impulse_target_match:
                impulse = {
                    "type": impulse_type_match.group(1),
                    "target": impulse_target_match.group(1).replace('\\"', '"')
                }

            if not thought:
                # Ultimate fallback: treat the whole text as the thought
                thought = raw_text.strip()
                concept = "Abstract Reverie"

            return thought, concept, impulse


def register_abstract_thought_layer(orchestrator=None) -> AbstractThoughtLayer:
    """Service registration helper."""
    layer = AbstractThoughtLayer(orchestrator)
    ServiceContainer.register_instance("abstract_thought_layer", layer)
    return layer
