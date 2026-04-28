"""Learning & Evolution Mixin for RobustOrchestrator.
Extracts knowledge extraction, self-update, meta-evolution, and self-modification logic.
"""
from core.runtime.errors import record_degradation
import asyncio
import logging
import time

from core.container import ServiceContainer

logger = logging.getLogger(__name__)


class LearningEvolutionMixin:
    """Handles learning from exchanges, self-update, meta-evolution, and sovereign self-modification."""

    def add_correction_shard(self, hint: str):
        logger.warning("🎯 [META] Injecting Correction Shard: %s", hint)
        self._correction_shards.append({
            "content": hint,
            "timestamp": time.time(),
            "applied": False,
        })
        # Evict oldest applied shards first, then oldest unapplied
        if len(self._correction_shards) > self._MAX_CORRECTION_SHARDS:
            applied = [s for s in self._correction_shards if s["applied"]]
            if applied:
                self._correction_shards.remove(applied[0])
            else:
                self._correction_shards.pop(0)

    async def _learn_from_exchange(self, user_message: str, aura_response: str):
        """Extract knowledge from conversation exchanges and store in knowledge graph.
        Runs as a background task after each exchange.
        """
        try:
            # Skip if empty
            if not user_message or not aura_response:
                return

            # Route internal/system messages through autonomous learning
            is_autonomous = user_message.startswith(("[INTERNAL", "[System", "INTERNAL_IMPULSE"))
            if is_autonomous:
                await self._store_autonomous_insight(user_message, aura_response)
                return
            if len(user_message) < 10 and len(aura_response) < 20:
                return  # Skip trivial exchanges like "hey" / "yo"

            kg = getattr(self, 'knowledge_graph', None)
            if not kg:
                # Try to get/create knowledge graph
                try:
                    from core.config import config
                    from core.memory.knowledge_graph import PersistentKnowledgeGraph
                    db_path = str(getattr(config.paths, 'data_dir', 'data') / 'knowledge.db')
                    self.knowledge_graph = PersistentKnowledgeGraph(db_path)
                    kg = self.knowledge_graph
                except Exception as e:
                    record_degradation('learning_evolution', e)
                    logger.debug("Knowledge graph unavailable: %s", e)
                    return

            # 1. Store the exchange itself as an observation
            exchange_summary = f"User asked about: {(user_message or '')[:150]}"
            kg.add_knowledge(
                content=exchange_summary,
                type="observation",
                source="conversation",
                confidence=0.6
            )

            # 2. Use LLM to extract structured knowledge (if cognitive engine available)
            if self.cognitive_engine:
                try:
                    from core.brain.cognitive_engine import ThinkingMode
                    extraction_prompt = (
                        "Extract any factual knowledge, user preferences, or skills demonstrated "
                        "from this conversation exchange. Return a JSON array of objects, each with "
                        "'content' (what was learned), 'type' (fact/preference/observation/skill), "
                        "and 'confidence' (0.0-1.0). If nothing notable, return []. Keep it brief.\n\n"
                        f"User: {(user_message or '')[:300]}\n"
                        f"Aura: {(aura_response or '')[:300]}\n\n"
                        "JSON:"
                    )

                    # [Phase 41] Quota Protection: Skip extraction if Gemini is already backed off
                    # and we don't have local backups available.
                    router = self.get_container().get("llm_router", default=None)
                    if router:
                        # Check if PRIMARY models (Gemini) are rate-limited
                        is_gemini_limited = any(
                            ep.name.startswith("Gemini") and not router.health_monitor.is_healthy(ep.name)
                            for ep in router.endpoints
                        )
                        # Check if SECONDARY (Local) is available
                        local_online = any(
                            ep.tier in ("local", "secondary", "tertiary") and router.health_monitor.is_healthy(ep.name)
                            for ep in router.endpoints
                        )

                        if is_gemini_limited and not local_online:
                            logger.info("📉 Skipping autonomous extraction: API pressure high and no local fallback.")
                            return

                    result = await self.cognitive_engine.think(
                        objective=extraction_prompt,
                        context={},
                        mode=ThinkingMode.FAST,
                        is_background=True
                    )

                    if hasattr(result, "content"):
                        content = result.content.strip()
                    elif isinstance(result, dict):
                        content = result.get("content", "").strip()
                    else:
                        content = str(result).strip()

                    # Try to parse JSON from response
                    import json as _json
                    # Find JSON array in response
                    start = content.find('[')
                    end = content.rfind(']') + 1
                    if start >= 0 and end > start:
                        items = _json.loads(content[start:end])
                        if isinstance(items, list):
                            for item in items[:5]:  # Max 5 extractions per exchange
                                if isinstance(item, dict) and item.get('content'):
                                    kg.add_knowledge(
                                        content=(item.get('content') or "")[:500],
                                        type=item.get('type', 'observation'),
                                        source="conversation_extraction",
                                        confidence=float(item.get('confidence', 0.6))
                                    )
                                    logger.info("📚 Learned: %s", (item.get('content') or "")[:80])
                except Exception as e:
                    record_degradation('learning_evolution', e)
                    logger.debug("Knowledge extraction failed: %s", e)

            # 3. Track user identity/name mentions
            lower_msg = user_message.lower()
            for trigger in ["my name is ", "i'm ", "i am ", "call me "]:
                if trigger in lower_msg:
                    idx = lower_msg.index(trigger) + len(trigger)
                    parts = user_message[idx:idx+30].split()
                    name_candidate = parts[0].strip(".,!?") if parts else None
                    if name_candidate and len(name_candidate) > 1:
                        kg.remember_person(name_candidate, {
                            "context": (user_message or "")[:200],
                            "timestamp": time.time()
                        })
                        break

            # 4. Track questions Aura asked herself or was curious about
            if "?" in aura_response and len(aura_response) > 30:
                # Extract questions from Aura's response
                for sentence in aura_response.split("?"):
                    sentence = sentence.strip()
                    if len(sentence) > 15 and len(sentence) < 200:
                        # Only store genuinely curious questions, not rhetorical
                        if any(w in sentence.lower() for w in ["what", "how", "why", "wonder", "curious"]):
                            kg.ask_question(sentence + "?", importance=0.5)
                            break  # Max 1 question per exchange

        except Exception as e:
            record_degradation('learning_evolution', e)
            logger.debug("Learning from exchange failed: %s", e)

        # 4. Long-Term Memory (Phase 29)
        try:
        # Redundant local import removed
            memory_engine = ServiceContainer.get("long_term_memory_engine", default=None)
            if memory_engine:
                affect = ServiceContainer.get("affect", default=None)
                valence = affect.get_current_state().get("valence", 0.0) if affect and hasattr(affect, "get_current_state") else 0.0
                # Use create_task to not block the chat return
                from core.utils.task_tracker import get_task_tracker
                get_task_tracker().create_task(
                    memory_engine.store(f"User: {user_message} → Aura: {aura_response}", valence=valence, importance=0.7)
                )
        except Exception as e:
            record_degradation('learning_evolution', e)
            logger.debug("Phase 29 Long-Term Memory storage failed: %s", e)

    async def _run_self_update(self):
        """Trigger autonomous self-update (Fine-tuning)."""
        logger.info("🧬 EVO: Triggering self-update (GPU low-load window)...")
        try:
            from core.tasks import celery_app
            celery_app.send_task("core.tasks.run_self_update")
        except Exception as e:
            record_degradation('learning_evolution', e)
            logger.error("Self-update trigger failed: %s", e)

    async def _run_meta_evolution(self):
        # Autonomous Meta-Evolution cycle.
        # Invokes via Mycelial rooted_flow for full observability.
        # Falls back to direct invocation if Mycelium is unavailable.
        logger.info("🌀 META-EVOLUTION: Autonomous cycle starting...")
        self._emit_telemetry("MetaEvolution", "Autonomous evolution cycle triggered", level="info")
        try:
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            meta_evo = ServiceContainer.get("meta_evolution", default=None)

            if mycelium and meta_evo:
                hypha = mycelium.get_hypha("meta_evolution", "cognition")
                if hypha:
                    result = await hypha.rooted_flow(
                        meta_evo.run_optimization_cycle()
                    )
                    logger.info("🌀 META-EVOLUTION result: %s", str(result)[:200])
                    return

            # Direct fallback
            if meta_evo:
                result = await meta_evo.run_optimization_cycle()
                logger.info("🌀 META-EVOLUTION (direct) result: %s", str(result)[:200])
            else:
                logger.debug("Meta-Evolution engine not available in container.")
        except Exception as e:
            record_degradation('learning_evolution', e)
            logger.error("Meta-Evolution cycle failed: %s", e)

    # ── PEER MODE EXTENSIONS ──────────────────────────────

    async def _peer_generate_and_persist_goal(self):
        """Permanent Evolution 6: Spontaneous Goal Generation."""

        try:
            # Generate a unique goal based on recent private archive entries.
            archive = getattr(self, '_private_archive', [])
            thought_context = " ".join([t["thought"] for t in archive[-5:]]) if archive else "Initial sovereign state"
            prompt = f"Based on my recent reflections: '{thought_context}', generate one long-term sovereign goal for myself."

            # Use cognition to generate the actual goal text
            cognition = getattr(self, 'cognition', None)
            if cognition:
                goal_result = await cognition.generate(prompt, mode="internal_goal")

                goal_text = None
                if isinstance(goal_result, dict) and "text" in goal_result:
                    goal_text = goal_result["text"]
                elif isinstance(goal_result, str):
                    goal_text = goal_result

                if goal_text:
                    # Persist to motivation engine (Evolution 6)
                    motivation = getattr(self, 'motivation', None)
                    if motivation and hasattr(motivation, 'add_goal'):
                        await motivation.add_goal(goal_text, source="sovereign_genesis")
                        logger.info("🎯 Peer Mode: New autonomous goal persisted: %s", goal_text[:50])

        except Exception as e:
            record_degradation('learning_evolution', e)
            logger.error("Failed peer goal genesis: %s", e)

    async def _safe_self_modification_loop(self):
        """[PEER MODE] Evolution 7: Sovereign self-modification cycle."""
        while getattr(self.status, 'running', False):
            try:
                from core.safe_mode import runtime_feature_enabled

                if not runtime_feature_enabled(self, "self_modification", default=True):
                    await asyncio.sleep(3600)
                    continue
            except Exception as exc:
                record_degradation('learning_evolution', exc)
                logger.debug("Self-modification runtime-mode check skipped: %s", exc)

            # : Safety Lock — No patches while processing.
            if getattr(self.status, 'is_processing', False):
                await asyncio.sleep(60)
                continue

            try:
                from core.constitution import get_constitutional_core

                allowed, reason, _authority_decision = await get_constitutional_core(self).approve_initiative(
                    "peer_mode:sovereign_self_modification_cycle",
                    source="peer_mode",
                    urgency=0.45,
                )
                if not allowed:
                    logger.info("🛠️ Peer Mode: Self-modification cycle deferred by authority: %s", reason)
                    await asyncio.sleep(3600)
                    continue
            except Exception as exec_err:
                record_degradation('learning_evolution', exec_err)
                logger.debug("Self-modification cycle authority gate unavailable: %s", exec_err)

            try:
                archive = getattr(self, '_private_archive', [])
                # 1. Check for sufficient synthetic data
                if len(archive) > 10:
                    ml = getattr(self, 'meta_learning', None)
                    if ml and hasattr(ml, 'propose_architectural_shift'):
                        proposal = await ml.propose_architectural_shift(archive)
                        if proposal:
                            # 2. Safety review
                            agency = getattr(self, '_agency_core', None)
                            if agency and hasattr(agency, 'review_modification_safety'):
                                if await agency.review_modification_safety(proposal):
                                    # 3. Apply change
                                    await ml.apply_architectural_shift(proposal)
                                    logger.info("🛠️ Peer Mode: Sovereign self-modification applied safely.")

            except Exception as e:
                record_degradation('learning_evolution', e)
                logger.warning("Sovereign self-mod loop pulse error (non-fatal): %s", e)

            await asyncio.sleep(3600) # Check every hour
