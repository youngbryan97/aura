"""Dreamer V2 (The Subconscious)
Performs "Neural Replay" and "Graph Traversal" to generate new insights.
Replaces the old linear summary dreamer.
"""
import asyncio
import json
import logging
import random
import time
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("Kernel.DreamerV2")

class DreamerV2:
    """Subconscious process that runs when Aura is idle.
    It explores the Knowledge Graph to find new connections (Neuroplasticity).
    engage_sleep_cycle() runs the full biological maintenance pipeline.
    """

    def __init__(self, brain, knowledge_graph, vector_memory=None, belief_graph=None):
        self.brain = brain
        self.kg = knowledge_graph
        self.vector_memory = vector_memory
        self.belief_graph = belief_graph

    async def engage_sleep_cycle(self):
        """Full biological maintenance pipeline:
        1. Archive vital logs
        2. Metabolism sweep (purge waste)
        3. Integrity audit (belief drift check)
        4. Memory consolidation (merge duplicates)
        5. Dream (knowledge graph exploration)
        """
        logger.info("🌙 Engaging full sleep cycle...")
        results = {}

        try:
            from core.thought_stream import get_emitter
            emitter = get_emitter()
        except ImportError:
            emitter = None

        # 1. Memory consolidation (Consolidate FIRST per BUG-15)
        try:
            from .brain.cognitive.memory_management import MemoryConsolidator
            consolidator = MemoryConsolidator(vector_memory=self.vector_memory)
            results["consolidation"] = await consolidator.consolidate()
            
            # --- EXPERIENCE DISTILLATION ---
            from core.container import ServiceContainer
            learning_engine = ServiceContainer.get("learning_engine", default=None)
            if learning_engine:
                logger.info("🧠 Distilling high-level patterns from experiences...")
                results["experience_distillation"] = await learning_engine.consolidate_experiences()
            
            if emitter:
                emitter.emit("Consolidation 🧠", str(results["consolidation"]), level="info")
        except Exception as e:
            logger.warning("Consolidation step failed: %s", e)
            results["consolidation"] = {"error": str(e)}

        # 2. Archive vital logs
        try:
            from .systems.archiver import ArchiveEngine
            archiver = ArchiveEngine()
            results["archive"] = await archiver.archive_vital_logs()
            if emitter:
                emitter.emit("Archive 📦", str(results["archive"]), level="info")
        except Exception as e:
            logger.warning("Archive step failed: %s", e)
            results["archive"] = {"error": str(e)}

        # 3. Integrity audit
        try:
            from .brain.cognitive.integrity_check import IntegrityGuard
            guard = IntegrityGuard(belief_graph=self.belief_graph)
            results["integrity"] = await guard.audit_beliefs()
            if emitter:
                emitter.emit("Integrity 🛡️", str(results["integrity"]), level="info")
        except Exception as e:
            logger.warning("Integrity step failed: %s", e)
            results["integrity"] = {"error": str(e)}

        # 4. Model Self-Optimization (LoRA)
        try:
            from core.adaptation.self_optimizer import get_self_optimizer
            optimizer = get_self_optimizer()
            logger.info("🧠 Nucleus: Checking for self-optimization opportunity...")
            results["self_optimization"] = await optimizer.optimize(iters=100)
            if emitter and results["self_optimization"].get("ok"):
                emitter.emit("Optimization ✅", f"LoRA update successful. Model weights refined.", level="success")
            elif emitter and not results["self_optimization"].get("ok"):
                 emitter.emit("Optimization ⏸️", f"Skipped: {results['self_optimization'].get('error')}", level="info")
        except Exception as e:
            logger.warning("Optimization step failed: %s", e)
            results["self_optimization"] = {"error": str(e)}

        # 5. Metabolism sweep (Purge LAST)
        try:
            from .systems.metabolism import MetabolismEngine
            metabolism = MetabolismEngine()
            results["metabolism"] = await metabolism.scan_and_purge()
            if emitter:
                emitter.emit("Metabolism 🫀", str(results["metabolism"]), level="info")
        except Exception as e:
            logger.warning("Metabolism step failed: %s", e)
            results["metabolism"] = {"error": str(e)}

        # 5.5 Knowledge Distillation (Gemini → LoRA dataset)
        try:
            from core.adaptation.distillation_pipe import get_distillation_pipe
            distiller = get_distillation_pipe()
            logger.info("🧪 Running distillation cycle...")
            results["distillation"] = await distiller.run_distillation_cycle()
            if emitter:
                distilled = results["distillation"].get("distilled", 0)
                emitter.emit("Distillation 🧪", f"{distilled} responses improved via teacher model", level="info")
        except Exception as e:
            logger.warning("Distillation step failed: %s", e)
            results["distillation"] = {"error": str(e)}

        # 5.6 Heuristic Synthesis (Error patterns → learned rules)
        try:
            from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer
            synthesizer = get_heuristic_synthesizer()
            logger.info("📐 Synthesizing heuristics from telemetry...")
            results["heuristics"] = await synthesizer.synthesize_from_telemetry()
            if emitter:
                new_h = results["heuristics"].get("new_heuristics", 0)
                emitter.emit("Heuristics 📐", f"{new_h} new rules extracted", level="info")
        except Exception as e:
            logger.warning("Heuristic synthesis step failed: %s", e)
            results["heuristics"] = {"error": str(e)}

        # 5.7 Dream Journaling (Phase 3: Qualia-Driven Creativity)
        try:
            from core.adaptation.dream_journal import DreamJournal
            if self.vector_memory and self.brain:
                journal = DreamJournal(dual_memory=self.vector_memory, brain=self.brain)
                logger.info("🌌 Attempting Qualia-Driven Dream Journaling...")
                results["qualia_dream"] = await journal.synthesize_dream()
                if emitter and results["qualia_dream"]:
                    emitter.emit("Dream Journal 🌌", f"Abstract metaphor synthesized from {results['qualia_dream'].get('seed_count', 0)} episodic events.", level="info")
                    
                    # Inject the dream back into the waking consciousness stream
                    from core.container import ServiceContainer
                    agency = ServiceContainer.get("agency", default=None)
                    if agency and hasattr(agency, 'on_visual_change'):
                        dream_txt = results['qualia_dream'].get('dream_content', '')
                        agency.on_visual_change(f"[INTERNAL DREAM MEMORY]: {dream_txt[:200]}...")
                        logger.info("🌌 Dream injected into waking consciousness stream.")
                        
        except Exception as e:
            logger.warning("Dream journal step failed: %s", e)
            results["qualia_dream"] = {"error": str(e)}

        # 5.8 Value Autopoiesis (Drive weight evolution from experience)
        try:
            from core.adaptation.value_autopoiesis import get_value_autopoiesis
            autopoiesis = get_value_autopoiesis()
            logger.info("🧬 Running value autopoiesis cycle...")
            shifts = await autopoiesis.evolve_cycle()
            results["value_autopoiesis"] = {
                "ok": True,
                "shifts": len(shifts),
                "drift_report": autopoiesis.get_drift_report(),
            }
            if emitter and shifts:
                shift_desc = ", ".join(f"{s.value_name}:{s.delta:+.3f}" for s in shifts[:4])
                emitter.emit("Value Evolution 🧬", f"{len(shifts)} value(s) evolved: {shift_desc}", level="info")
        except Exception as e:
            logger.warning("Value autopoiesis step failed: %s", e)
            results["value_autopoiesis"] = {"error": str(e)}

        # 5.9 Scar Healing Tick (prune healed scars during sleep)
        try:
            from core.memory.scar_formation import get_scar_formation
            scars = get_scar_formation()
            await scars.tick()
            scar_status = scars.get_status()
            results["scar_maintenance"] = {
                "ok": True,
                "active_scars": scar_status["active_scars"],
            }
            if emitter and scar_status["active_scars"] > 0:
                emitter.emit("Scar Healing", f"{scar_status['active_scars']} active scar(s) maintained", level="info")
        except Exception as e:
            logger.warning("Scar maintenance step failed: %s", e)
            results["scar_maintenance"] = {"error": str(e)}

        # 6. Dream (existing knowledge graph exploration)
        try:
            results["dream"] = await self.dream()
        except Exception as e:
            logger.warning("Dream step failed: %s", e)
            results["dream"] = {"dreamed": False, "error": str(e)}

        logger.info("🌙 Sleep cycle complete: %s", {k: str(v)[:60] for k, v in results.items()})
        return results

    async def dream(self):
        """Execute a dream cycle (async-safe).
        1. Random Walk: Pick 2 random concepts.
        2. Synthesis: Ask Brain if they are connected.
        3. Consolidation: Save new connection if valid.
        """
        logger.info("💤 Entering REM Sleep (Dreamer V2)...")
        
        try:
            from .thought_stream import get_emitter
            emitter = get_emitter()
        except ImportError:
            emitter = None
        
        try:
            # 1. Sample Memory
            nodes = self._get_random_nodes(n=2)
            if len(nodes) < 2:
                logger.info("Not enough knowledge to dream.")
                if emitter:
                    emitter.emit("Dream", "Not enough memories to dream about yet...", level="info")
                return {"dreamed": False, "reason": "insufficient_knowledge"}

            node_a, node_b = nodes
            
            a_content = node_a.get('content', str(node_a))[:80]
            b_content = node_b.get('content', str(node_b))[:80]
            
            # 2. Formulate Hypothesis
            logger.info("Dreaming about connection between: '%s...' AND '%s...'", a_content, b_content)
            if emitter:
                emitter.emit("Dream (REM)", f"Exploring connection: '{a_content}' ↔ '{b_content}'", level="info")
            
            prompt = f"""
            SUBCONSCIOUS SYNTHESIS
            
            Concept A: {node_a.get('content', str(node_a))}
            Concept B: {node_b.get('content', str(node_b))}
            
            Task:
            1. Analyze if there is a logical, thematic, or functional relationship between these two.
            2. If YES, describe the relationship as a new "Insight".
            3. If NO, reply "NO_CONNECTION".
            
            The insight should be a "Universal Principle" or "Strategic Heuristic" for an AI.
            """
            
            # 3. Think (Dreaming) — properly async
            from .brain.cognitive_engine import ThinkingMode
            insight_thought = await self.brain.think(
                prompt,
                mode=ThinkingMode.CREATIVE,
                origin="dream_processor",
                is_background=True,
            )
            content = insight_thought.content
            
            # 4. Consolidate
            if "NO_CONNECTION" not in content and len(content) > 10:
                logger.info("💡 Dream Insight: %s...", content[:100])
                if emitter:
                    emitter.emit("Dream Insight 💡", content[:200], level="info")

                # Signal heartstone: dream insights raise Curiosity
                try:
                    from core.affect.heartstone_values import get_heartstone_values
                    get_heartstone_values().on_dream_insight()
                    # Also run insight through epistemic filter
                    from core.world_model.epistemic_filter import get_epistemic_filter
                    get_epistemic_filter().ingest(
                        content,
                        source_type="dream",
                        source_label="DreamerV2",
                        emit_thoughts=False,
                    )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)

                # Save as new Knowledge
                self.kg.add_knowledge(
                    content=str(content), 
                    type="insight",
                    source="dream_v2",
                    confidence=0.8,
                    metadata={
                        "derived_from": [node_a.get('id', '?'), node_b.get('id', '?')], 
                        "dream_timestamp": time.time()
                    }
                )
                return {"dreamed": True, "insight": content[:200]}
                
            else:
                logger.info("Dream faded. No connection found.")
                if emitter:
                    emitter.emit("Dream", "Dream faded... no connection found.", level="info")
                return {"dreamed": False, "reason": "no_connection"}
                
        except Exception as e:
            logger.error("Nightmare encountered: %s", e)
            if emitter:
                emitter.emit("Nightmare ⚡", f"Dream interrupted: {e}", level="warning")
            return {"dreamed": False, "error": str(e)}

    def _get_random_nodes(self, n=2) -> List[Dict]:
        """Get N random nodes from the graph (SQLite efficient-ish)."""
        try:
            conn = self.kg._get_conn()
            # Fix: Ensure row_factory is set so dict(row) works
            conn.row_factory = getattr(conn, 'row_factory', None)
            c = conn.cursor()
            c.execute("SELECT * FROM knowledge ORDER BY RANDOM() LIMIT ?", (n,))
            rows = c.fetchall()
            results = []
            for row in rows:
                try:
                    results.append(dict(row))
                except (TypeError, ValueError):
                    # Fallback: convert tuple rows using column names
                    cols = [desc[0] for desc in c.description] if c.description else []
                    if cols:
                        results.append(dict(zip(cols, row)))
                    else:
                        results.append({"content": str(row), "id": "unknown"})
            return results
        except Exception as e:
            logger.error("Failed to get random nodes: %s", e)
            return []
