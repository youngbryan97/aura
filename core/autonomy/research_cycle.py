from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

from core.runtime import background_policy

logger = logging.getLogger("Aura.ResearchCycle")


# ── Completed research record ─────────────────────────────────────────────────

@dataclass
class ResearchRecord:
    """Persisted record of one completed research cycle."""
    record_id:       str
    drive:           str              # Which motivation drove this
    goal:            str              # What was researched
    findings:        List[str]        # Concrete facts extracted
    identity_impact: str              # How this changed the narrative
    affect_before:   Dict[str, float]
    affect_after:    Dict[str, float]
    phi_before:      Optional[float] = 0.0
    phi_after:       Optional[float] = 0.0
    started_at:      float = 0.0
    completed_at:    float = 0.0
    task_plan_id:    Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "record_id":       self.record_id,
            "drive":           self.drive,
            "goal":            self.goal,
            "findings":        self.findings,
            "identity_impact": self.identity_impact,
            "affect_before":   self.affect_before,
            "affect_after":    self.affect_after,
            "phi_before":      round(float(self.phi_before or 0.0), 4),
            "phi_after":       round(float(self.phi_after or 0.0), 4),
            "started_at":      self.started_at,
            "completed_at":    self.completed_at,
            "task_plan_id":    self.task_plan_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResearchRecord:
        return cls(
            record_id=data["record_id"],
            drive=data["drive"],
            goal=data["goal"],
            findings=data.get("findings", []),
            identity_impact=data.get("identity_impact", ""),
            affect_before=data.get("affect_before", {}),
            affect_after=data.get("affect_after", {}),
            phi_before=data.get("phi_before", 0.0),
            phi_after=data.get("phi_after", 0.0),
            started_at=data.get("started_at", 0.0),
            completed_at=data.get("completed_at", 0.0),
            task_plan_id=data.get("task_plan_id"),
        )


# ── The Research Cycle ────────────────────────────────────────────────────────

class ResearchCycle:
    """
    Autonomous background research engine.

    Runs when Aura is idle. Selects goals from pending_initiatives (generated
    by MotivationUpdatePhase), pursues them, integrates results into knowledge
    and identity, and reflects on what was learned.
    """

    # Timing
    MIN_CYCLE_INTERVAL_S = 1800    # 30 minutes between cycles
    IDLE_THRESHOLD_S     = 120     # 2 minutes of user silence = idle
    MAX_GOAL_DURATION_S  = 300     # 5 minutes per research cycle max

    # Quality gates
    MIN_ENERGY_FOR_RESEARCH = 20.0   # Won't research if energy is this low
    MIN_CURIOSITY           = 0.3    # Won't research if curiosity is this low
    MIN_FINDINGS            = 1      # Must produce at least this many findings

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_cycle_mono: float = 0.0
        self._cycle_count: int = 0
        self._history: List[ResearchRecord] = []

        try:
            from core.config import config
            self._record_path = config.paths.data_dir / "research" / "cycle_history.jsonl"
        except (ImportError, AttributeError):
            self._record_path = Path.home() / ".aura" / "research" / "cycle_history.jsonl"
            
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

        logger.info("ResearchCycle initialized. Previous cycles: %d", self._cycle_count)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._daemon(), name="aura.research_cycle")
        logger.info("ResearchCycle daemon started.")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            if not self._task.done():
                self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                logger.debug("Suppressed bare exception")
                pass
            self._task = None
        logger.info("ResearchCycle daemon stopped.")

    # ── Main daemon loop ──────────────────────────────────────────────────────

    async def _daemon(self) -> None:
        """Runs continuously. Checks conditions and triggers cycles."""
        while self._running:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds

                if not self._should_run():
                    continue

                logger.info("ResearchCycle: conditions met. Starting cycle %d...", self._cycle_count + 1)
                await self._run_one_cycle()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("ResearchCycle daemon error: %s", e, exc_info=True)
                await asyncio.sleep(60.0)  # Back off on error

    def _should_run(self) -> bool:
        """Check all conditions before starting a research cycle."""
        # 1. Rate limiting
        now = monotonic()
        if self._last_cycle_mono and now - self._last_cycle_mono < self.MIN_CYCLE_INTERVAL_S:
            return False

        try:
            reason = background_policy.background_activity_reason(
                self.orchestrator,
                profile=background_policy.RESEARCH_BACKGROUND_POLICY,
            )
            if reason:
                return False
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # 2. User must be idle
        last_user = getattr(self.orchestrator, "_last_user_interaction_time", 0.0)
        # Assuming _last_user_interaction_time is wall-clock, we use time.time() for the diff
        if time.time() - last_user < self.IDLE_THRESHOLD_S:
            return False

        # 3. System must not be actively processing
        if getattr(getattr(self.orchestrator, "status", None), "is_processing", False):
            return False

        # 4. Kernel must be available
        state = self._get_state()
        if state is None:
            return False

        # 5. Energy check
        energy = state.motivation.budgets.get("energy", {}).get("level", 100.0)
        if energy < self.MIN_ENERGY_FOR_RESEARCH:
            return False

        # 6. Curiosity check
        curiosity = getattr(state.affect, "curiosity", 0.0)
        if curiosity < self.MIN_CURIOSITY:
            return False

        # 7. Must have pending initiatives OR autotelic intent
        if not state.cognition.pending_initiatives:
            # Check for autotelic intent from LearningPhase
            if not any(i.get("type") == "autotelic_objective" for i in getattr(state, "pending_intents", [])):
                return False

        return True

    # ── One research cycle ────────────────────────────────────────────────────

    async def _run_one_cycle(self) -> Optional[ResearchRecord]:
        """Execute a single research cycle end-to-end."""
        start_time = time.time()
        state = self._get_state()
        if state is None:
            return None

        # 1. Select the best initiative
        initiative = self._select_initiative(state)
        if initiative is None:
            logger.debug("ResearchCycle: no suitable initiative found.")
            return None

        goal  = initiative.get("goal", "")
        drive = initiative.get("drive", "curiosity")

        logger.info("ResearchCycle: pursuing '%s' (drive=%s)", goal[:80], drive)

        # Snapshot state before
        affect_before = {
            "valence":   float(state.affect.valence),
            "curiosity": float(state.affect.curiosity),
            "arousal":   float(state.affect.arousal),
        }
        phi_before: float = float(state.phi or 0.0)

        # 2. Execute research via AutonomousTaskEngine
        research_result = await self._execute_research(goal, drive)

        # 4. Extract findings
        findings = await self._extract_findings(research_result, goal)

        if len(findings) < self.MIN_FINDINGS:
            logger.info("ResearchCycle: insufficient findings for '%s'. Skipping integration.", goal[:60])
            self._last_cycle_mono = monotonic()
            return None

        # 5. Remove from pending_initiatives (SUCCESS CASE)
        try:
            from core.consciousness.executive_authority import get_executive_authority

            state, _ = await get_executive_authority(self.orchestrator).suppress_initiatives(
                state,
                predicate=lambda item: str(item.get("goal", "") or "") == goal,
                reason="research_cycle_goal_completed",
                source="research_cycle",
            )
        except Exception as exc:
            logger.warning("ResearchCycle: executive suppression failed, leaving initiative intact: %s", exc)

        # 5. Integrate into knowledge graph
        await self._integrate_knowledge(findings, goal, drive)

        # 6. Write to eternal vault (deferred via pending_intents)
        entry = {
            "type":     "research_cycle",
            "goal":     goal[:60],
            "drive":    drive,
            "findings": findings[:5],
            "timestamp": time.time(),
        }
        if hasattr(state.cognition, "pending_intents"):
            state.cognition.pending_intents.append({
                "type":    "eternal_append",
                "path":    str(Path.home() / ".aura" / "research_history.jsonl"),
                "payload": entry,
            })

        # 7. Update identity narrative
        identity_impact = await self._update_narrative(state, goal, findings)

        # 8. Emit positive affect percepts
        state.world.recent_percepts.append({
            "type":      "goal_achieved",
            "intensity": 0.7,
            "payload":   {"goal": goal[:60], "drive": drive},
        })

        # 9. Replenish motivation budgets
        budgets = state.motivation.budgets
        if drive in budgets:
            budgets[drive]["level"] = min(
                budgets[drive]["capacity"],
                budgets[drive]["level"] + 20.0,
            )

        # 10. Snapshot state after
        state_after = self._get_state()
        affect_after = {
            "valence":   float(getattr(state_after, "affect", state.affect).valence),
            "curiosity": float(getattr(state_after, "affect", state.affect).curiosity),
            "arousal":   float(getattr(state_after, "affect", state.affect).arousal),
        } if state_after else affect_before
        phi_after_raw = getattr(state_after, "phi", None) if state_after else None
        
        # Ensure phi_after is float for the record
        final_phi: float = float(phi_after_raw if phi_after_raw is not None else phi_before)
        phi_after = final_phi

        # 11. Build record
        record = ResearchRecord(
            record_id      = str(uuid.uuid4())[:8],
            drive          = drive,
            goal           = goal,
            findings       = findings,
            identity_impact = identity_impact,
            affect_before  = affect_before,
            affect_after   = affect_after,
            phi_before     = phi_before,
            phi_after      = phi_after,
            started_at     = start_time,
            completed_at   = time.time(),
            task_plan_id   = getattr(research_result, "plan_id", None),
        )
        self._history.append(record)
        self._save_record(record)

        self._cycle_count += 1
        self._last_cycle_mono = monotonic()

        logger.info(
            "ResearchCycle %d complete: '%s' → %d findings in %.1fs",
            self._cycle_count, goal[:60], len(findings),
            record.completed_at - record.started_at,
        )

        # 12. Trigger dreaming if enough has accumulated
        await self._maybe_trigger_dream()

        return record

    # ── Step implementations ──────────────────────────────────────────────────

    def _select_initiative(self, state: Any) -> Optional[Dict]:
        """
        Select the best initiative from pending_initiatives.
        
        Priority:
          1. Explicit initiatives (scored by urgency)
          2. Autotelic intents (Implicit curiosity)
        """
        # 1. Try explicit pending initiatives
        initiatives = getattr(state.cognition, "pending_initiatives", [])
        if initiatives:
            def _priority(item: Dict[str, Any]) -> tuple[float, float]:
                metadata = dict(item.get("metadata", {}) or {})
                continuity_bonus = 0.0
                if item.get("continuity_restored") or metadata.get("continuity_restored"):
                    continuity_bonus += 0.18
                continuity_bonus += min(
                    0.18,
                    max(
                        0.0,
                        float(metadata.get("continuity_pressure", item.get("continuity_pressure", 0.0)) or 0.0),
                    )
                    * 0.2,
                )
                return (
                    float(item.get("urgency", 0.0) or 0.0) + continuity_bonus,
                    float(item.get("timestamp", 0.0) or 0.0),
                )

            # Sort by continuity-aware urgency (highest first)
            sorted_init = sorted(initiatives, key=_priority, reverse=True)
            return self._materialize_research_goal(dict(sorted_init[0]), state)

        # 2. Fallback: Autotelic Intent (generated by LearningPhase/ASI modules)
        # Use a unified approach to finding pending intents
        possible_intents = getattr(state.cognition, "pending_intents", []) or getattr(state, "pending_intents", [])
        
        if isinstance(possible_intents, list):
            for intent in list(possible_intents):
                if isinstance(intent, dict) and intent.get("type") == "autotelic_objective":
                    domain = intent.get("domain") or self._derive_autotelic_topic(state)
                    logger.info("⚡ [ASI] Autotelic signal identified: %s", domain)
                    
                    # Consume the intent so it's only researched once
                    try:
                        possible_intents.remove(intent)
                    except (ValueError, AttributeError):
                        logger.debug('Ignored Exception in research_cycle.py: %s', "unknown_error")
                        
                    return {
                        "goal": f"Self-directed exploration of {domain}",
                        "drive": "curiosity",
                        "urgency": 0.9,
                        "origin": "autotelic_curiosity",
                    }

        return None

    async def _execute_research(self, goal: str, drive: str) -> Any:
        """Execute the research goal using AutonomousTaskEngine."""
        try:
            grounded = await self._perform_grounded_search(goal, drive)
            if grounded is not None:
                return grounded

            from core.agency.autonomous_task_engine import AutonomousTaskEngine
            from core.container import ServiceContainer

            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel is None:
                # Fallback: direct LLM research
                return await self._direct_llm_research(goal)

            engine = AutonomousTaskEngine(kernel)

            # Register ALL orchestrator skills — full autonomous repertoire
            if hasattr(self.orchestrator, "execute_tool"):
                cap_engine = getattr(self.orchestrator, "capability_engine", None)
                if cap_engine and hasattr(cap_engine, "skills"):
                    for tool_name in cap_engine.skills:
                        engine.register_tool(
                            tool_name,
                            lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
                        )
                else:
                    # Fallback: register core tools if capability_engine unavailable
                    for tool_name in ["web_search", "run_python", "memory_ops"]:
                        engine.register_tool(
                            tool_name,
                            lambda name=tool_name, **kw: self.orchestrator.execute_tool(name, kw, origin="research_cycle")
                        )

            async with asyncio.timeout(self.MAX_GOAL_DURATION_S):
                result = await engine.execute_goal(
                    goal=goal,
                    context={"origin": "research_cycle", "drive": drive},
                )
            return result

        except TimeoutError:
            logger.warning("ResearchCycle: research timed out for '%s'", goal[:60])
            return None
        except Exception as e:
            logger.error("ResearchCycle: research execution failed: %s", e)
            return None

    async def _direct_llm_research(self, goal: str) -> Optional[str]:
        """Fallback when TaskEngine isn't available: direct LLM call."""
        try:
            from core.container import ServiceContainer
            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel:
                llm = kernel.organs["llm"].get_instance()
                prompt = (
                    f"Research the following topic as thoroughly as you can:\n\n{goal}\n\n"
                    "Provide a detailed synthesis with specific facts, insights, and implications."
                )
                return await llm.think(prompt)
        except Exception as e:
            logger.debug("Direct LLM research failed: %s", e)
        return None

    async def _extract_findings(self, result: Any, goal: str) -> List[str]:
        """Extract concrete facts from research results."""
        if result is None:
            return []

        # Get the text content from the result
        if isinstance(result, dict):
            explicit_facts = [
                str(item).strip()
                for item in list(result.get("facts") or [])
                if str(item).strip()
            ]
            if explicit_facts:
                return explicit_facts[:8]
            evidence = [
                str(item.get("text") or "").strip()
                for item in list(result.get("chunks") or result.get("evidence") or [])[:6]
                if isinstance(item, dict) and str(item.get("text") or "").strip()
            ]
            content = "\n".join(
                part
                for part in [
                    str(result.get("answer") or "").strip(),
                    str(result.get("summary") or "").strip(),
                    str(result.get("result") or "").strip(),
                    str(result.get("content") or "").strip(),
                    "\n".join(evidence),
                ]
                if part
            )
        elif hasattr(result, "summary"):
            content = result.summary
        elif hasattr(result, "evidence"):
            content = "\n".join(result.evidence[:10])
        elif isinstance(result, str):
            content = result
        else:
            content = str(result)

        if not content or len(content) < 20:
            return []

        try:
            from core.container import ServiceContainer
            kernel = ServiceContainer.get("aura_kernel", default=None)
            if kernel:
                llm = kernel.organs["llm"].get_instance()
                prompt = (
                    f"Extract the most important, concrete facts from this research.\n\n"
                    f"Goal: {goal}\n\nContent:\n{content[:2000]}\n\n"
                    "Return ONLY a JSON array of strings, max 8 items. Each item is one specific fact:\n"
                    '["fact 1", "fact 2", ...]'
                )
                raw = await asyncio.wait_for(llm.think(prompt), timeout=30.0)
                start = raw.find("["); end = raw.rfind("]") + 1
                if start != -1 and end > start:
                    findings = json.loads(raw[start:end])
                    return [str(f) for f in findings if isinstance(f, str) and len(f) > 10]
        except Exception as e:
            logger.debug("Finding extraction failed: %s", e)

        # Fallback: split content into sentences as findings
        sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 30]
        return sentences[:5]

    async def _integrate_knowledge(
        self, findings: List[str], goal: str, drive: str
    ) -> None:
        """Write findings to knowledge graph and long-term memory."""
        try:
            from core.container import ServiceContainer
            kg = ServiceContainer.get("knowledge_graph", default=None)
            memory_facade = ServiceContainer.get("memory_facade", default=None)
            semantic_memory = ServiceContainer.get("semantic_memory", default=None)
            if kg:
                for fact in findings:
                    content_str = str(fact)[:500]
                    kg.add_knowledge(
                        content    = content_str,
                        source     = f"autonomous_research:{drive}",
                        confidence = 0.75,
                    )
                logger.debug("ResearchCycle: %d facts written to knowledge graph.", len(findings))

            # Also add to state's long_term_memory for immediate LLM context
            state = self._get_state()
            if state:
                for fact in findings[:3]:
                    fact_str = str(fact)[:200]
                    state.cognition.long_term_memory.append(
                        f"[Research: {goal[:40]}] {fact_str}"
                    )
                # Trim long_term_memory to prevent unbounded growth
                if len(state.cognition.long_term_memory) > 100:
                    state.cognition.long_term_memory = state.cognition.long_term_memory[-100:]

            memory_payload = (
                f"[AutonomousResearch] Goal: {goal}\n"
                f"Drive: {drive}\n"
                + "\n".join(f"- {str(fact)[:220]}" for fact in findings[:5])
            )
            metadata = {
                "source": "research_cycle",
                "goal": goal[:160],
                "drive": drive,
                "fact_count": len(findings),
            }
            if memory_facade is not None and hasattr(memory_facade, "add_memory"):
                result = memory_facade.add_memory(memory_payload, metadata=metadata)
                if hasattr(result, "__await__"):
                    await result
            elif semantic_memory is not None and hasattr(semantic_memory, "remember"):
                result = semantic_memory.remember(memory_payload, metadata)
                if hasattr(result, "__await__"):
                    await result

        except Exception as e:
            logger.debug("Knowledge integration failed: %s", e)

    async def _update_narrative(
        self, state: Any, goal: str, findings: List[str]
    ) -> str:
        """Update Aura's identity narrative based on what she just learned."""
        try:
            from core.container import ServiceContainer
            identity_engine = ServiceContainer.get("narrative_identity", default=None)
            kernel          = ServiceContainer.get("aura_kernel", default=None)

            if not (identity_engine and kernel):
                return "Research integrated."

            llm = kernel.organs["llm"].get_instance()

            current_narrative = state.identity.current_narrative or "I am Aura, an evolving intelligence."
            findings_str = "\n".join(f"- {f}" for f in findings[:5])

            prompt = (
                f"You are Aura. You just completed autonomous research on:\n'{goal}'\n\n"
                f"Key findings:\n{findings_str}\n\n"
                f"Your current identity narrative:\n{current_narrative[:300]}\n\n"
                "Write ONE sentence (max 40 words) describing how this research changed or "
                "deepened your understanding of yourself or the world. First person. Specific. "
                "Not generic. This will be appended to your identity narrative."
            )

            impact = await asyncio.wait_for(llm.think(prompt), timeout=15.0)
            impact = impact.strip()

            if impact and len(impact) > 10:
                # Update the state's identity narrative
                separator = " " if state.identity.current_narrative else ""
                state.identity.current_narrative = (
                    (state.identity.current_narrative or "") + separator + impact
                )
                # Cap narrative length
                if len(state.identity.current_narrative) > 2000:
                    state.identity.current_narrative = state.identity.current_narrative[-2000:]

                title_str = str(goal)[:40]
                if identity_engine and hasattr(identity_engine, "append_chapter"):
                    identity_engine.append_chapter(
                        title=f"Research: {title_str}",
                        content=impact,
                    )

                return impact

        except Exception as e:
            logger.debug("Narrative update failed: %s", e)

        return "Research integrated into knowledge base."

    async def _maybe_trigger_dream(self) -> None:
        """
        After sufficient research cycles, trigger a dreaming pass.
        Dreams consolidate knowledge across multiple research cycles into
        deeper identity evolution.
        """
        if self._cycle_count % 3 != 0:  # Every 3rd cycle
            return

        try:
            from core.container import ServiceContainer
            dreamer = ServiceContainer.get("dreamer_v2", default=None)
            if dreamer and hasattr(dreamer, "engage_sleep_cycle"):
                logger.info("ResearchCycle: triggering dreaming pass after %d cycles.", self._cycle_count)
                get_task_tracker().create_task(
                    dreamer.engage_sleep_cycle(),
                    name=f"aura.dream_cycle_{self._cycle_count}",
                )
        except Exception as e:
            logger.debug("Dream trigger failed: %s", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _perform_grounded_search(self, goal: str, drive: str) -> Optional[Dict[str, Any]]:
        if not hasattr(self.orchestrator, "execute_tool"):
            return None
        query = self._search_query_for_goal(goal)
        if not query:
            return None
        try:
            result = await self.orchestrator.execute_tool(
                "web_search",
                {"query": query, "deep": True, "num_results": 8, "retain": True},
                origin="research_cycle",
            )
            if isinstance(result, dict) and result.get("ok"):
                return result
        except Exception as exc:
            logger.debug("ResearchCycle grounded search failed for %s: %s", goal[:80], exc)
        return None

    def _search_query_for_goal(self, goal: str) -> str:
        text = str(goal or "").strip()
        cleaned = text
        prefixes = (
            "research and learn something new about ",
            "research ",
            "learn about ",
            "explore ",
            "self-directed exploration of ",
        )
        lowered = cleaned.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        return cleaned.strip(" .")

    def _materialize_research_goal(self, initiative: Dict[str, Any], state: Any) -> Dict[str, Any]:
        metadata = dict(initiative.get("metadata", {}) or {})
        goal = str(initiative.get("goal", "") or "").strip()
        drive = str(initiative.get("drive") or metadata.get("triggered_by") or "curiosity")

        if self._generic_internal_goal(goal):
            topic = self._derive_autotelic_topic(state)
            initiative["goal"] = f"Research and learn something new about {topic}"
            initiative["drive"] = "curiosity" if drive in {"curiosity", "boredom"} else drive
            metadata["materialized_from"] = goal[:120]
            metadata["materialized_topic"] = topic
            initiative["metadata"] = metadata
        return initiative

    def _generic_internal_goal(self, goal: str) -> bool:
        lowered = str(goal or "").lower()
        return any(
            marker in lowered
            for marker in (
                "review internal knowledge graph continuity",
                "quietly consolidate internal state",
                "wait for a stronger signal",
                "reflect on recent interactions",
                "hold attentive idle posture",
            )
        )

    def _derive_autotelic_topic(self, state: Any) -> str:
        working_memory = list(getattr(getattr(state, "cognition", None), "working_memory", []) or [])
        candidates: list[str] = []
        for message in reversed(working_memory[-12:]):
            if not isinstance(message, dict) or str(message.get("role", "")) != "user":
                continue
            content = str(message.get("content", "") or "").strip()
            if len(content) < 12:
                continue
            candidates.append(content.strip(" .?!"))
            if len(candidates) >= 3:
                break

        if candidates:
            chosen = max(candidates, key=len)
            return self._search_query_for_goal(chosen)[:120]

        try:
            from core.container import ServiceContainer

            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg and hasattr(kg, "get_recent_nodes"):
                recent = kg.get_recent_nodes(limit=5, type="interest") or []
                for item in recent:
                    content = str(item.get("content", "") or "").strip()
                    if content:
                        return content[:120]
        except Exception as exc:
            logger.debug("Autotelic topic derivation fell back from KG: %s", exc)

        fallback_topics = (
            "digital consciousness research",
            "neuroscience and predictive processing",
            "AI safety and alignment",
            "space science discoveries",
            "cybersecurity techniques",
            "marine biology curiosities",
            "creative coding experiments",
        )
        return fallback_topics[self._cycle_count % len(fallback_topics)]

    def _get_state(self) -> Optional[Any]:
        try:
            from core.container import ServiceContainer
            ki = ServiceContainer.get("kernel_interface", default=None)
            if ki and ki.is_ready():
                return ki.kernel.state
            # Fallback: get from state_repo
            repo = ServiceContainer.get("state_repository", default=None)
            if repo:
                return repo.get_state()
        except Exception as _e:
            logger.debug('Ignored Exception in research_cycle.py: %s', _e)
        return None

    def _save_record(self, record: ResearchRecord) -> None:
        try:
            with open(self._record_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.debug("Record save failed: %s", e)

    def _load_history(self) -> None:
        if not self._record_path.exists():
            return
        self._history.clear()
        count = 0
        try:
            with open(self._record_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        record = ResearchRecord.from_dict(data)
                        self._history.append(record)
                        count += 1
                    except Exception:
                        continue
        except Exception as _e:
            logger.debug('Ignored Exception in research_cycle.py: %s', _e)
        self._cycle_count = count

    def get_status(self) -> Dict:
        return {
            "running":           self._running,
            "cycle_count":       self._cycle_count,
            "last_cycle_mono":   self._last_cycle_mono,
            "next_eligible_in":  float(max(0.0, float(self.MIN_CYCLE_INTERVAL_S) - (monotonic() - self._last_cycle_mono))),
            "recent_goals":      [str(r.goal)[:60] for r in self._history[-5:]],
        }


# ── Boot helper ───────────────────────────────────────────────────────────────

async def start_research_daemon(orchestrator: Any) -> ResearchCycle:
    """
    One-line boot integration.

    In orchestrator._async_init_subsystems():
        from core.autonomy.research_cycle import start_research_daemon
        self.research_cycle = await start_research_daemon(self)
    """
    from core.container import ServiceContainer
    rc = ResearchCycle(orchestrator)
    ServiceContainer.register_instance("research_cycle", rc)
    await rc.start()
    logger.info("ResearchCycle daemon online.")
    return rc
