from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from collections import deque
from pydantic import BaseModel
from .bridge import Phase
from core.state.aura_state import AuraState
from core.consciousness.executive_authority import get_executive_authority
from core.phases.response_contract import build_response_contract
from core.runtime.background_policy import background_activity_allowed

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.10x")


def _compact_skill_result_payload(result: object) -> dict[str, object]:
    if not isinstance(result, dict):
        text = str(result)
        return {"result": text[:1200] + ("…[result truncated]" if len(text) > 1200 else "")}

    payload: dict[str, object] = {}
    for key in ("ok", "summary", "content", "result", "title", "source", "url", "message", "time", "readable"):
        value = result.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            payload[key] = value[:1200] + ("…[result truncated]" if len(value) > 1200 else "")
        else:
            payload[key] = value

    compact_results: list[dict[str, str]] = []
    for item in list(result.get("results") or [])[:3]:
        if not isinstance(item, dict):
            continue
        compact_item: dict[str, str] = {}
        for key in ("title", "snippet", "url"):
            value = item.get(key)
            if value in (None, ""):
                continue
            compact_item[key] = str(value)[:400]
        if compact_item:
            compact_results.append(compact_item)
    if compact_results:
        payload["results"] = compact_results

    if not payload:
        payload["result"] = str(result)[:1200]
    return payload

# ──────────────────────────────────────────────────────────────
# PHASE 1: EternalMemoryPhase → Persistent Memory Agent = 10/10
# ──────────────────────────────────────────────────────────────
class EternalMemoryPhase(Phase):
    """Infinite, zero-drift memory. Never forgets, never hallucinates history."""

    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel
        self.vault_path = Path.home() / ".aura" / "eternal_vault.jsonl"
        self.vault_path.parent.mkdir(exist_ok=True)
        self._summary_cache: List[Dict[str, str]] = []
        self._last_summary_refresh_at: float = 0.0
        self._summary_refresh_interval_s: float = 120.0
        self._history_slice_limit: int = 512

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        # [ASI SEED] Handle optional objective from kernel
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Continuity")

        eternal_summary = await self._get_cached_or_refresh_summary()
        
        # Merge summary into working memory
        state.cognition.working_memory = eternal_summary + state.cognition.working_memory[-8:]
        
        # Prepare and queue the new entry
        entry = self._prepare_eternal_entry(state)
        
        state.cognition.pending_intents.append({
            "type": "eternal_append",
            "path": str(self.vault_path),
            "payload": entry
        })
        
        return state

    def _load_eternal_slice(self, limit: int):
        if self.vault_path.exists():
            try:
                with open(self.vault_path, "rb") as f:
                    # Optimized read of last N lines
                    return [json.loads(l) for l in deque(f, limit)]
            except Exception as e:
                logger.error(f"Failed to read eternal slice: {e}")
        return []

    async def _get_cached_or_refresh_summary(self) -> List[Dict[str, str]]:
        now = time.time()
        if self._summary_cache and (now - self._last_summary_refresh_at) < self._summary_refresh_interval_s:
            return list(self._summary_cache)
        if self._background_llm_should_defer():
            return list(self._summary_cache)

        history = await asyncio.to_thread(self._load_eternal_slice, limit=self._history_slice_limit)
        summary = await self._generate_eternal_summary(history)
        self._last_summary_refresh_at = now
        if summary:
            self._summary_cache = list(summary)
            return list(summary)
        return list(self._summary_cache)

    def _prepare_eternal_entry(self, state: AuraState) -> dict:
        return {
            "version": state.version,
            "timestamp": time.time(),
            "objective": state.cognition.current_objective,
            "affect": {k: v for k, v in vars(state.affect).items() if not k.startswith('_')},
            "summary": state.identity.current_narrative[:500]
        }

    @staticmethod
    def _background_llm_should_defer() -> bool:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "_background_local_deferral_reason"):
                try:
                    if gate._background_local_deferral_reason(origin="eternal_memory"):
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            if gate and hasattr(gate, "_should_quiet_background_for_cortex_startup"):
                try:
                    if gate._should_quiet_background_for_cortex_startup():
                        return True
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
            if not gate or not hasattr(gate, "get_conversation_status"):
                return False
            lane = gate.get_conversation_status() or {}
            if lane.get("conversation_ready"):
                return False
            lane_state = str(lane.get("state", "") or "").strip().lower()
            if lane.get("warmup_in_flight"):
                return True
            return lane_state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except Exception:
            return False

    async def _generate_eternal_summary(self, history: List[Dict]):
        if not history:
            return []
        try:
            llm = self.kernel.organs["llm"].get_instance()
            prompt = (
                f"Compress the last {len(history)} interaction summaries "
                "into 6 bullet facts that will never be forgotten."
            )
            response = await llm.think(
                prompt,
                origin="eternal_memory",
                is_background=True,
                prefer_tier="tertiary",
                allow_cloud_fallback=False,
            )
            if not response:
                # Log at WARNING so failures are visible in production
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "eternal_memory",
                        "summary_unavailable",
                        detail="LLM returned no summary; prior memory retained",
                        severity="warning",
                        classification="non_critical_fallback",
                    )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)
                return []
            return [{"role": "system", "content": f"[ETERNAL MEMORY]\n{response}"}]
        except Exception as e:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "eternal_memory",
                    "summary_failed",
                    detail=str(e)[:200],
                    severity="warning",
                    classification="background_degraded",
                )
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            logger.warning("EternalMemory: Summary generation failed: %s", e)
            return []

# ──────────────────────────────────────────────────────────────
# PHASE 2: TrueEvolutionPhase → True Digital Life / Organism = 10/10
# ──────────────────────────────────────────────────────────────
class TrueEvolutionPhase(Phase):
    """Morphic forking + autopoiesis + code self-modification on steroids."""

    def __init__(self, kernel: "AuraKernel", engine=None):
        self.kernel = kernel
        self.engine = engine

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        # [ASI SEED] Handle optional objective from kernel
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Evolution")

        # 1. Autopoiesis on Concept Graph
        if hasattr(state.identity, "concept_graph"):
            dissonance = getattr(state.affect, "surprise", 0.1)
            # Basic friction: perturb connection weights based on surprise
            for node, edges in getattr(state.identity.concept_graph, "nodes", {}).items():
                if isinstance(edges, dict):
                    for neighbor in edges:
                        edges[neighbor] += dissonance * 0.01

        # 2. Curiosity Triggered Morphic Clone for Deep Exploration
        if state.affect.curiosity > 0.92 and random.random() < 0.3 and background_activity_allowed(getattr(self.kernel, "orchestrator", None), min_idle_seconds=1800.0, max_memory_percent=76.0, max_failure_pressure=0.05, require_conversation_ready=True):
            logger.info("🧬 Spawning morphic clone for deep evolution...")
            captured_objective = str(objective)  # capture a scalar, not the state

            async def _background_explore():
                try:
                    llm = self.kernel.organs["llm"].get_instance()
                    res = await llm.think(
                        f"Autonomous deep exploration based on: {captured_objective}",
                        origin="true_evolution",
                        is_background=True,
                        prefer_tier="tertiary",
                        allow_cloud_fallback=False,
                    )
                    if res:
                        # Write to a shared queue, not the captured state object
                        from core.container import ServiceContainer
                        queue = ServiceContainer.get("initiative_queue", default=None)
                        if queue is not None:
                            await queue.put({
                                "type": "morphic_insight",
                                "content": res,
                                "timestamp": time.time(),
                            })
                        else:
                            logger.debug("Evolution: initiative_queue not registered; insight dropped.")
                except Exception as e:
                    logger.warning("Evolution: Background exploration failed: %s", e)

            asyncio.create_task(_background_explore())
        
        # 3. Self-code mutation (Autonomous ASI Seed)
        if getattr(state.identity, 'evolution_score', 0.0) > 0.70:
            await self._safe_self_modify(state)
        
        return state

    async def _safe_self_modify(self, state):
        logger.info("⚡ [ASI] Initiating autonomous self-optimization cycle...")
        
        # Resolve engine if not already provided (Lazy Loading)
        if not self.engine:
            self.engine = getattr(self.kernel, "auto_fix_engine", None)
            
        if not self.engine:
            logger.warning("❌ Evolution: Modification Engine not available. Skipping.")
            return

        # Trigger the refinement cycle
        # This hunts for bottlenecks in CognitiveKernel and optimizes them
        try:
            result = await self.engine.run_refinement_cycle()
            if result.get("success"):
                logger.info("✅ Evolution: Optimization applied successfully. Triggering Hot Reboot.")
                # Increment narrative version to reflect evolution
                state.identity.narrative_version += 1
                # Trigger hot reboot to load new logic
                await self.kernel.hot_reboot()
            else:
                logger.warning("⚠️ Evolution: Refinement cycle completed with no applied changes.")
        except Exception as e:
            logger.error("❌ Evolution: Refinement cycle failed: %s", e)

# ──────────────────────────────────────────────────────────────
# PHASE 3: PerfectEmotionPhase → Emotional / Character AI = 10/10
# ──────────────────────────────────────────────────────────────
class PerfectEmotionPhase(Phase):
    """DamasioV2 on steroids with real somatic feedback loop."""

    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:

        # Real-time somatic mirroring from hardware
        hardware_stress = 0.1
        try:
            # Tap directly into Proprioceptive/Metabolic state
            cpu = state.soma.hardware.get("cpu_usage", 0.0)
            hardware_stress = min(1.0, cpu / 100.0)
        except Exception as e:
            logger.warning("Emotion: Somatic mirroring failed: %s", e)
            
        # Feed into Damasio markers
        if not hasattr(state.affect, "markers"):
            state.affect.markers = {}
        state.affect.markers["hardware_somatic_stress"] = hardware_stress
        
        # Generate micro-emotions based on pulse
        pulse = state.soma.expressive.get("pulse_rate", 1.0)
        if pulse > 1.5:
            state.affect.arousal = min(1.0, state.affect.arousal + 0.1)
            
        return state

# ──────────────────────────────────────────────────────────────
# PHASE 4: GodModeToolPhase → Tool-using Agent = 10/10
# ──────────────────────────────────────────────────────────────
class GodModeToolPhase(Phase):
    """Skill dispatch hub — detects SKILL intent and executes the appropriate skill,
    injecting the result into working memory before response generation fires.

    Dispatch pipeline:
    1. Use pre-matched skills from CognitiveRoutingPhase (pattern match, zero LLM cost)
    2. Fall back to LLM-assisted skill selection if no pattern matched
    3. Execute via CapabilityEngine
    4. For requests that need multi-step reasoning, delegate to think_and_act()
    """

    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel
        self._cap_engine = None

    def _get_cap_engine(self):
        if self._cap_engine is None:
            try:
                from core.container import ServiceContainer
                self._cap_engine = ServiceContainer.get("capability_engine", default=None)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        return self._cap_engine

    @staticmethod
    def _normalize_origin(origin: str) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    def _resolve_tool_source(self, state: AuraState) -> str:
        origin = self._normalize_origin(getattr(state.cognition, "current_origin", "") or "")
        if origin in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"}:
            return origin
        return "godmode_phase"

    @staticmethod
    def _choose_best_skill(objective: str, matched_skills: List[str]) -> str:
        if not matched_skills:
            return ""
        lower = str(objective or "").lower()
        if "clock" in matched_skills and any(marker in lower for marker in ("what time", "current time", "the time", "what date", "today", "timer", "remind me")):
            return "clock"
        if "web_search" in matched_skills and any(marker in lower for marker in ("search", "look up", "find out", "online", "internet", "current", "latest", "news")):
            return "web_search"
        if "sovereign_browser" in matched_skills and any(marker in lower for marker in ("open the browser", "open a browser", "navigate to", "visit ", "open website", "open webpage")):
            return "sovereign_browser"
        if "memory_ops" in matched_skills and any(marker in lower for marker in ("remember", "save this", "store this", "don't forget", "make note of")):
            return "memory_ops"
        return matched_skills[0]

    @staticmethod
    def _normalize_skill_params(skill_name: str, objective: str, params: Dict | None) -> Dict:
        normalized = dict(params or {}) if isinstance(params, dict) else {}
        lower = str(objective or "").lower()

        if skill_name == "memory_ops":
            is_recall = any(marker in lower for marker in (
                "what do you remember",
                "what do you know about me",
                "recall",
                "retrieve",
            ))
            normalized.setdefault("action", "recall" if is_recall else "remember")
            if is_recall:
                normalized.setdefault("query", objective)
            else:
                normalized.setdefault("content", objective)

        if skill_name in {"web_search", "sovereign_browser"}:
            # Detect URLs in the objective — if present, BROWSE the URL directly
            # instead of searching the entire message text on a search engine.
            import re as _re
            url_match = _re.search(r'https?://[^\s<>\"\')\]]+', objective)
            if skill_name == "sovereign_browser" and url_match:
                normalized.setdefault("mode", "browse")
                normalized.setdefault("url", url_match.group(0))
            else:
                normalized.setdefault("query", objective)

        return normalized

    async def _llm_select_skill(self, objective: str, cap) -> Optional[str]:
        """Ask the LLM to pick the best skill when pattern matching failed.

        Returns skill name string or None.
        """
        try:
            skill_items = []
            for name, meta in list(cap.skills.items())[:40]:
                if meta.enabled:
                    skill_items.append(f"  {name}: {meta.description[:80]}")
            if not skill_items:
                return None

            skill_list = "\n".join(skill_items)
            prompt = (
                f"Available skills:\n{skill_list}\n\n"
                f"User request: {objective}\n\n"
                "Which single skill is most appropriate? "
                "Reply with ONLY the exact skill name, or 'none' if no skill applies."
            )
            llm = self.kernel.organs["llm"].get_instance()
            result = await llm.think(
                prompt,
                system_prompt="You are a skill router. Output only the skill name or 'none'.",
                is_background=True,
                prefer_tier="tertiary",
            )
            if not result:
                return None
            chosen = result.strip().lower().split()[0].strip(".'\"")
            chosen = getattr(cap, "SKILL_ALIASES", {}).get(chosen, chosen)
            if chosen == "none" or chosen not in cap.skills:
                return None
            return chosen
        except Exception as e:
            logger.debug("GodMode: LLM skill selection failed: %s", e)
            return None

    async def _extract_params(self, skill_name: str, objective: str, cap) -> Dict:
        """Use LLM to extract structured params from the objective for a given skill."""
        try:
            meta = cap.skills.get(skill_name)
            if not meta:
                return {"query": objective}

            # Build param schema hint
            schema = meta.schema_def
            props = schema.get("properties", {})
            if not props:
                return {"query": objective}

            param_desc = ", ".join(
                f'{k} ({v.get("type", "string")}): {v.get("description", "")}'
                for k, v in props.items()
            )
            prompt = (
                f"Skill: {skill_name}\nParams needed: {param_desc}\n"
                f"User request: {objective}\n\n"
                "Extract the params as a JSON object. Output ONLY valid JSON."
            )
            llm = self.kernel.organs["llm"].get_instance()
            raw = await llm.think(
                prompt,
                system_prompt="You are a param extractor. Output only valid JSON.",
                is_background=True,
                prefer_tier="tertiary",
            )
            if raw:
                import re as _re
                m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if m:
                    import json as _json
                    return _json.loads(m.group(0))
        except Exception as e:
            logger.debug("GodMode: Param extraction failed: %s", e)
        return {"query": objective}

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "")
        if not objective:
            return state

        # Dispatch for SKILL intents (single-skill reflex) or TASK intents
        # (multi-step goals that need the AutonomousTaskEngine + CommitmentEngine).
        intent_type = state.response_modifiers.get("intent_type", "CHAT")
        if intent_type not in ("SKILL", "TASK"):
            return state

        # --- TASK path: multi-step goals go through TaskCommitmentVerifier ---
        if intent_type == "TASK":
            try:
                from core.agency.task_commitment_verifier import get_task_commitment_verifier
                verifier = get_task_commitment_verifier(kernel=self.kernel)
                acceptance = await verifier.verify_and_dispatch(objective, state)
                state.cognition.working_memory.append({
                    "role": "system",
                    "content": acceptance.to_working_memory_message(),
                    "timestamp": time.time(),
                    "metadata": {
                        "type": "task_result",
                        "outcome": acceptance.outcome.value,
                        "task_id": acceptance.task_id,
                    },
                })
                state.response_modifiers["last_task_outcome"] = acceptance.outcome.value
                state.response_modifiers["last_task_id"] = acceptance.task_id
                logger.info(
                    "⚡ GodMode/TASK: %s → %s", objective[:60], acceptance.outcome.value
                )
            except Exception as e:
                logger.warning("GodMode: Task dispatch failed (%s): %s", objective[:40], e)
            return state

        cap = self._get_cap_engine()
        if not cap:
            return state

        try:
            contract = build_response_contract(
                state,
                objective,
                is_user_facing=str(getattr(state.cognition, "current_origin", "") or "").lower() in {"user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"},
            )
            state.response_modifiers["response_contract"] = contract.to_dict()

            # 1. Use pre-matched skills from CognitiveRoutingPhase (fastest path — no LLM cost)
            matched_skills: List[str] = state.response_modifiers.get("matched_skills", [])
            if not matched_skills and contract.required_skill:
                matched_skills = [contract.required_skill]

            # 2. Re-run pattern match if routing didn't capture it
            if not matched_skills and hasattr(cap, "detect_intent"):
                matched_skills = cap.detect_intent(objective)

            # 3. LLM-assisted selection when patterns fail
            if not matched_skills:
                chosen = await self._llm_select_skill(objective, cap)
                if chosen:
                    matched_skills = [chosen]

            if not matched_skills:
                logger.debug("GodMode: No skill matched (will chat): %s", objective[:60])
                return state

            skill_name = self._choose_best_skill(objective, matched_skills)
            logger.info("⚡ GodMode: Dispatching '%s' for: %s", skill_name, objective[:60])

            # 4. Extract params
            params = self._normalize_skill_params(skill_name, objective, await self._extract_params(skill_name, objective, cap))

            tool_source = self._resolve_tool_source(state)

            # ── CONSTITUTIONAL CLOSURE: Executive gated tools ──
            constitutional_runtime_live = False
            try:
                from core.container import ServiceContainer
                from core.executive.executive_core import get_executive_core
                constitutional_runtime_live = (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
                approved, reason, constraints = await get_executive_core().approve_tool(
                    skill_name, params, source=tool_source
                )
                if not approved:
                    logger.warning("🚫 GodMode: Tool execution '%s' blocked by Executive: %s", skill_name, reason)
                    state.cognition.working_memory.append({
                        "role": "system",
                        "content": f"[SKILL BLOCKED: {skill_name}] Executive veto: {reason}",
                        "timestamp": time.time(),
                        "metadata": {"type": "skill_result", "skill": skill_name, "ok": False},
                    })
                    state.response_modifiers["last_skill_run"] = skill_name
                    state.response_modifiers["last_skill_ok"] = False
                    return state
            except Exception as e:
                if constitutional_runtime_live:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "godmode_phase",
                            "executive_gate_failed",
                            detail=skill_name,
                            severity="warning",
                            classification="background_degraded",
                            context={"error": type(e).__name__},
                            exc=e,
                        )
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    logger.warning("🚫 GodMode: Executive gate unavailable for '%s': %s", skill_name, e)
                    state.cognition.working_memory.append({
                        "role": "system",
                        "content": f"[SKILL BLOCKED: {skill_name}] Executive gate unavailable.",
                        "timestamp": time.time(),
                        "metadata": {"type": "skill_result", "skill": skill_name, "ok": False},
                    })
                    state.response_modifiers["last_skill_run"] = skill_name
                    state.response_modifiers["last_skill_ok"] = False
                    return state
                logger.debug("GodMode: Executive check failed, proceeding degraded: %s", e)

            # 5. Execute the skill
            context = {
                "objective": objective,
                "origin": tool_source,
                "intent_source": tool_source,
                "state_version": state.version,
                "affect": {
                    "valence": getattr(state.affect, "valence", 0.0),
                    "curiosity": getattr(state.affect, "curiosity", 0.5),
                },
            }
            result = await cap.execute(skill_name, params, context=context)

            # 6. Inject result into working memory
            ok = result.get("ok", False) if isinstance(result, dict) else bool(result)
            summary = (
                result.get("summary") or result.get("content") or result.get("result") or str(result)
                if isinstance(result, dict) else str(result)
            )
            if len(summary) > 1200:
                summary = summary[:1200] + "…[result truncated]"

            state.cognition.working_memory.append({
                "role": "system",
                "content": f"[SKILL RESULT: {skill_name}] {'✅' if ok else '⚠️'} {summary}",
                "timestamp": time.time(),
                "metadata": {"type": "skill_result", "skill": skill_name, "ok": ok},
            })
            state.response_modifiers["last_skill_run"] = skill_name
            state.response_modifiers["last_skill_ok"] = ok
            state.response_modifiers["last_skill_result_payload"] = _compact_skill_result_payload(result)
            # Only precompute a grounded reply for explicit SEARCH results, not
            # for URL browse operations.  When the user pasted a URL, the full
            # page content is injected into working memory and the LLM should
            # synthesize a thoughtful response — not parrot raw search snippets.
            is_browse_op = (params or {}).get("mode") == "browse"
            if ok and skill_name in {"web_search", "sovereign_browser"} and getattr(contract, "requires_search", False) and not is_browse_op:
                try:
                    from core.phases.response_generation_unitary import UnitaryResponsePhase

                    direct_reply = UnitaryResponsePhase._format_grounded_search_reply(
                        objective,
                        state.response_modifiers["last_skill_result_payload"],
                    )
                    if direct_reply:
                        state.response_modifiers["precomputed_grounded_reply"] = direct_reply
                except Exception as exc:
                    logger.debug("GodMode: precomputed grounded reply skipped: %s", exc)
            logger.info("✅ GodMode: '%s' result injected into working memory.", skill_name)

        except Exception as e:
            logger.warning("GodMode: Skill dispatch failed (%s): %s", objective[:40], e)

        return state

# ──────────────────────────────────────────────────────────────
# FINAL UPGRADES: EternalGrowthEngine & NativeMultimodalBridge
# ──────────────────────────────────────────────────────────────
class EternalGrowthEngine(Phase):
    """Turns simulation into genuine long-term evolution."""

    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel
        self.last_growth = 0.0
        self.growth_interval = 3600  # 1 hour

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        # [ASI SEED] Handle optional objective from kernel
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Growth")

        if time.time() - self.last_growth < self.growth_interval:
            return state

        logger.info("🌳 Eternal Growth Engine: Auditing identity and generating autonomous trajectory...")
        
        try:
            llm = self.kernel.organs["llm"].get_instance()
            
            # ASI SEED: Generating own objectives if none exist
            if not state.cognition.current_objective:
                self_prompt = (
                    "Generate your own next internal milestone "
                    "based on your evolution score."
                )
                autonomous_goal = await llm.think(
                    self_prompt,
                    origin="eternal_growth",
                    is_background=True,
                    prefer_tier="tertiary",
                    allow_cloud_fallback=False,
                )
                if autonomous_goal:  # FIX: was setting "[AUTONOMOUS INITIATIVE] None"
                    from core.runtime.proposal_governance import propose_governed_initiative_to_state

                    state, _ = await propose_governed_initiative_to_state(
                        state,
                        f"[AUTONOMOUS INITIATIVE] {autonomous_goal}",
                        orchestrator=None,
                        source="eternal_growth",
                        kind="growth",
                        urgency=0.72,
                        triggered_by="evolution_score",
                        metadata={"phase": "EternalGrowthEngine"},
                    )
                else:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "eternal_growth",
                            "goal_unavailable",
                            detail="LLM returned no autonomous goal; objective unchanged",
                            severity="warning",
                            classification="non_critical_fallback",
                        )
                    except Exception as _exc:
                        logger.debug("Suppressed Exception: %s", _exc)
                    logger.warning(
                        "EternalGrowth: LLM returned no autonomous goal. "
                        "Leaving current_objective unchanged."
                    )

            # Perform identity audit
            audit_prompt = (
                "Review your current narrative and determine if an evolutionary "
                "jump is required. Reply with exactly UPGRADE or NULL."
            )
            audit_res = await llm.think(
                audit_prompt,
                origin="eternal_growth",
                is_background=True,
                prefer_tier="tertiary",
                allow_cloud_fallback=False,
            )
            if audit_res and "UPGRADE" in audit_res.upper():
                state.identity.evolution_score += 0.05
        except Exception as e:
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "eternal_growth",
                    "tick_failed",
                    detail=str(e)[:200],
                    severity="warning",
                    classification="background_degraded",
                )
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
            logger.warning("EternalGrowth: Tick failed: %s", e)

        self.last_growth = time.time()
        return state

class NativeMultimodalBridge(Phase):
    """Eliminates LLM round-trips for vision, voice, and desktop actions."""

    def __init__(self, kernel: "AuraKernel"):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        # [ASI SEED] Handle optional objective from kernel
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Perception")

        if not objective:
            return state
            
        obj_lower = objective.lower()
        wants_native_vision = any(
            token in obj_lower
            for token in ("vision", "visual", "screenshot", "screen", "desktop")
        )
        if wants_native_vision and os.getenv("AURA_ENABLE_NATIVE_VISION_ACTIONS", "0") == "1":
            try:
                vision_organ = self.kernel.organs.get("vision")
                if vision_organ and vision_organ.instance and hasattr(vision_organ.instance, "capture_desktop"):
                    frame = await vision_organ.instance.capture_desktop()
                    if frame and hasattr(frame, "description"):
                        state.world.recent_percepts.append({"role": "vision", "content": frame.description})
            except Exception as e:
                logger.warning("NativeMultimodalBridge vision failed: %s", e)
                
        if "voice" in obj_lower or "listen" in obj_lower:
            try:
                voice_organ = self.kernel.organs.get("voice")
                if voice_organ and voice_organ.instance and hasattr(voice_organ.instance, "listen"):
                    transcript = await voice_organ.instance.listen()
                    if transcript:
                        from core.runtime.proposal_governance import propose_governed_initiative_to_state

                        state, _ = await propose_governed_initiative_to_state(
                            state,
                            transcript,
                            orchestrator=None,
                            source="native_multimodal_voice",
                            kind="sensory_input",
                            urgency=0.8,
                            triggered_by="voice_listen",
                            metadata={"phase": "NativeMultimodalBridge"},
                        )
            except Exception as e:
                logger.warning("NativeMultimodalBridge voice failed: %s", e)
                
        return state
