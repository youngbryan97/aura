from __future__ import annotations
import logging
import re
import time
from typing import Optional, TYPE_CHECKING
from core.runtime.skill_task_bridge import looks_like_execution_report, looks_like_multi_step_skill_request
from core.runtime.turn_analysis import analyze_turn
from core.state.aura_state import AuraState, CognitiveMode
from core.kernel.bridge import Phase
from core.phases.response_contract import build_response_contract

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.Kernel.Cognitive")

_USER_FACING_ORIGINS = frozenset({
    "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external",
})

_DEEP_HANDOFF_KEYWORDS = frozenset({
    "flagship architecture", "architecture deep dive", "architecture audit",
    "security audit", "mathematical proof", "deep dive",
    "complex analysis", "bottleneck analysis", "vulnerability scan",
    "formal proof", "root cause analysis",
})

_CODING_ROUTE_MARKERS = frozenset({
    "debug", "traceback", "stack trace", "failing test", "pytest",
    "refactor", "performance", "latency", "memory leak",
    "race condition", "deadlock", "regression", "compile",
    "build", "patch", "exception", "crash", "segfault",
})

_FOLLOWUP_CODING_MARKERS = frozenset({
    "keep going", "keep it going", "continue", "go ahead", "try again",
    "let's do it", "lets do it", "do it", "resume",
    "fix it", "fix that", "patch it", "finish it", "does that solve it",
    "why is that failing", "what about the test", "what about that bug",
})

_FILE_REF_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:\.{0,2}/|/)?[A-Za-z0-9_.~/-]+\.(?:py|md|json|toml|ya?ml|txt|js|ts|tsx|sh|swift|rs|go|cpp|c|h)"
)


class CognitiveRoutingPhase(Phase):
    """
    Unitary Cognitive Routing Phase.
    Determines if the system should enter REACTIVE, DELIBERATE, or DORMANT modes.
    Directly interfaces with the LLM Organ for intent classification.
    """

    def __init__(self, kernel: "AuraKernel"):
        super().__init__(kernel)

    @staticmethod
    def _normalize_origin(origin: str) -> str:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        while normalized.startswith("routing_"):
            normalized = normalized[len("routing_"):]
        return normalized

    @classmethod
    def _is_user_facing_origin(cls, origin: str) -> bool:
        normalized = cls._normalize_origin(origin)
        if not normalized:
            return False
        if normalized in _USER_FACING_ORIGINS:
            return True
        tokens = {token for token in normalized.split("_") if token}
        return bool(tokens & _USER_FACING_ORIGINS)

    @staticmethod
    def _resolve_model_tier(is_user_facing: bool) -> str:
        return "primary" if is_user_facing else "tertiary"

    @staticmethod
    def _count_file_refs(text: str) -> int:
        return len({match.group(0) for match in _FILE_REF_PATTERN.finditer(str(text or ""))})

    @classmethod
    def _build_coding_route_metadata(cls, text: str, *, analysis, intent_type: str) -> dict[str, object]:
        lowered = str(text or "").lower()
        word_count = len(str(text or "").split())
        file_ref_count = cls._count_file_refs(text)
        is_execution_report = looks_like_execution_report(text)

        route_hints: dict[str, object] = {}
        try:
            from core.runtime.coding_session_memory import get_coding_route_hints

            route_hints = dict(get_coding_route_hints(text) or {})
        except Exception:
            route_hints = {}

        active_thread = bool(route_hints.get("active_coding_thread"))
        followup_coding = active_thread and any(marker in lowered for marker in _FOLLOWUP_CODING_MARKERS)
        marker_hit = any(marker in lowered for marker in _CODING_ROUTE_MARKERS)
        if is_execution_report:
            return {
                "coding_request": False,
                "coding_complexity_score": 0.0,
                "file_ref_count": file_ref_count,
                "active_coding_thread": active_thread,
                "has_test_failure": bool(route_hints.get("has_test_failure")),
                "has_runtime_error": bool(route_hints.get("has_runtime_error")),
                "has_active_plan": bool(route_hints.get("has_active_plan")),
                "has_verification_failure": bool(route_hints.get("has_verification_failure")),
                "repair_attempts": int(route_hints.get("repair_attempts", 0) or 0),
                "execution_phase": str(route_hints.get("execution_phase", "") or ""),
                "followup_coding": False,
                "execution_report": True,
            }
        coding_request = bool(
            file_ref_count > 0
            or marker_hit
            or followup_coding
            or (analysis.intent_type == "TASK" and analysis.semantic_mode == "technical")
        )

        complexity = 0.0
        if coding_request:
            complexity += 0.25
        if analysis.semantic_mode == "technical":
            complexity += 0.2
        if intent_type == "TASK":
            complexity += 0.15
        if word_count >= 18:
            complexity += 0.1
        if file_ref_count >= 1:
            complexity += 0.1
        if file_ref_count >= 2:
            complexity += 0.05
        if marker_hit:
            complexity += 0.15
        if route_hints.get("has_test_failure"):
            complexity += 0.15
        if route_hints.get("has_runtime_error"):
            complexity += 0.1
        if route_hints.get("has_active_plan"):
            complexity += 0.1
        if route_hints.get("has_verification_failure"):
            complexity += 0.15
        if int(route_hints.get("repair_attempts", 0) or 0) > 0:
            complexity += min(0.15, 0.05 * int(route_hints.get("repair_attempts", 0) or 0))
        if followup_coding:
            complexity += 0.1
        if active_thread and word_count <= 8:
            complexity += 0.05

        return {
            "coding_request": coding_request,
            "coding_complexity_score": min(1.0, complexity),
            "file_ref_count": file_ref_count,
            "active_coding_thread": active_thread,
            "has_test_failure": bool(route_hints.get("has_test_failure")),
            "has_runtime_error": bool(route_hints.get("has_runtime_error")),
            "has_active_plan": bool(route_hints.get("has_active_plan")),
            "has_verification_failure": bool(route_hints.get("has_verification_failure")),
            "repair_attempts": int(route_hints.get("repair_attempts", 0) or 0),
            "execution_phase": str(route_hints.get("execution_phase", "") or ""),
            "followup_coding": followup_coding,
            "execution_report": False,
        }

    @classmethod
    def _should_upgrade_to_technical_task(
        cls,
        text: str,
        *,
        analysis,
        route_meta: dict[str, object] | None = None,
    ) -> bool:
        metadata = route_meta or cls._build_coding_route_metadata(
            text,
            analysis=analysis,
            intent_type=getattr(analysis, "intent_type", "CHAT"),
        )
        lowered = str(text or "").lower()

        if any(keyword in lowered for keyword in _DEEP_HANDOFF_KEYWORDS):
            return True
        if not bool(metadata.get("coding_request")):
            return False
        if getattr(analysis, "intent_type", "CHAT") == "TASK":
            return True
        if bool(metadata.get("followup_coding")):
            return True
        if int(metadata.get("file_ref_count", 0) or 0) > 0:
            return True
        if any(
            bool(metadata.get(flag))
            for flag in ("has_test_failure", "has_runtime_error", "has_verification_failure", "has_active_plan")
        ):
            return True
        return float(metadata.get("coding_complexity_score", 0.0) or 0.0) >= 0.60

    @classmethod
    def _should_allow_deep_handoff(
        cls,
        text: str,
        *,
        is_user_facing: bool,
        intent_type: str,
        analysis=None,
        route_meta: dict[str, object] | None = None,
    ) -> bool:
        """Determine if the 72B deep solver should be activated.

        [STABILITY v53] DRASTICALLY tightened. The 32B cortex handles 95%+ of
        conversation perfectly. The 72B should ONLY activate for genuinely complex
        technical problems that the 32B can't handle — multi-file debugging,
        complex architecture, mathematical proofs. NOT for:
        - Philosophical questions
        - Emotional conversations
        - Long messages (length ≠ complexity)
        - Questions about Aura herself
        - Normal "explain" / "why" / "how" questions
        """
        if not is_user_facing or intent_type not in {"TASK"}:
            # [STABILITY v53] Only TASK intent can trigger deep handoff.
            # CHAT intent stays on 32B — it handles conversation beautifully.
            return False
        lower = text.lower()
        current_analysis = analysis or analyze_turn(text)
        metadata = route_meta or cls._build_coding_route_metadata(
            text,
            analysis=current_analysis,
            intent_type=intent_type,
        )
        if bool(metadata.get("execution_report")) or getattr(current_analysis, "is_execution_report", False):
            return False

        # Only explicit deep-dive technical keywords trigger handoff
        if any(keyword in lower for keyword in _DEEP_HANDOFF_KEYWORDS):
            return True

        # Must be technical AND a coding request
        if current_analysis.semantic_mode != "technical" or not metadata.get("coding_request"):
            return False

        # [STABILITY v53] Raised threshold from 0.65 to 0.80 — only truly complex
        # multi-file technical tasks should trigger the 72B solver.
        complexity = float(metadata.get("coding_complexity_score", 0.0) or 0.0)
        if complexity >= 0.80:
            return True
        return bool(
            metadata.get("has_test_failure")
            and metadata.get("file_ref_count", 0) >= 2  # Multiple files = actually complex
            and (
                "pytest" in lower
                or "traceback" in lower
            )
        )

    @staticmethod
    def _looks_like_everyday_chat(text: str) -> bool:
        return analyze_turn(text).everyday_chat_safe

    def _stamp_llm_route(
        self,
        state: AuraState,
        *,
        objective: str,
        intent_type: str,
        is_user_facing: bool,
        analysis=None,
        route_meta: dict[str, object] | None = None,
    ) -> None:
        current_analysis = analysis or analyze_turn(objective)
        metadata = route_meta or self._build_coding_route_metadata(
            objective,
            analysis=current_analysis,
            intent_type=intent_type,
        )
        model_tier = self._resolve_model_tier(is_user_facing)
        deep_handoff = self._should_allow_deep_handoff(
            objective,
            is_user_facing=is_user_facing,
            intent_type=intent_type,
            analysis=current_analysis,
            route_meta=metadata,
        )
        state.response_modifiers["intent_type"] = intent_type
        state.response_modifiers["model_tier"] = model_tier
        state.response_modifiers["deep_handoff"] = deep_handoff
        state.response_modifiers["coding_request"] = bool(metadata.get("coding_request"))
        state.response_modifiers["coding_complexity_score"] = float(metadata.get("coding_complexity_score", 0.0) or 0.0)
        state.response_modifiers["execution_report"] = bool(metadata.get("execution_report"))
        state.response_modifiers["coding_route_hints"] = {
            "file_ref_count": int(metadata.get("file_ref_count", 0) or 0),
            "active_coding_thread": bool(metadata.get("active_coding_thread")),
            "has_test_failure": bool(metadata.get("has_test_failure")),
            "has_runtime_error": bool(metadata.get("has_runtime_error")),
            "has_active_plan": bool(metadata.get("has_active_plan")),
            "has_verification_failure": bool(metadata.get("has_verification_failure")),
            "repair_attempts": int(metadata.get("repair_attempts", 0) or 0),
            "execution_phase": str(metadata.get("execution_phase", "") or ""),
            "followup_coding": bool(metadata.get("followup_coding")),
        }
        logger.info(
            "🧠 CognitiveRouting: Mode=%s, Tier=%s, DeepHandoff=%s, Coding=%s, Complexity=%.2f",
            state.cognition.current_mode.name,
            model_tier,
            deep_handoff,
            bool(metadata.get("coding_request")),
            float(metadata.get("coding_complexity_score", 0.0) or 0.0),
        )

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        priority = kwargs.get("priority", False)
        if not objective:
            return state

        new_state = state.derive(f"routing: {objective[:20]}", origin="CognitiveRouting")
        state_origin = self._normalize_origin(getattr(state.cognition, "current_origin", ""))
        if priority and not self._is_user_facing_origin(state_origin):
            routing_origin = "user"
        else:
            routing_origin = state_origin or ("user" if priority else "system")
        is_user_facing = self._is_user_facing_origin(routing_origin)
        new_state.cognition.current_objective = objective
        new_state.cognition.current_origin = routing_origin
        contract = build_response_contract(new_state, objective, is_user_facing=is_user_facing)
        new_state.response_modifiers["response_contract"] = contract.to_dict()
        analysis = analyze_turn(objective)
        route_meta = self._build_coding_route_metadata(
            objective,
            analysis=analysis,
            intent_type=analysis.intent_type,
        )
        new_state.response_modifiers["semantic_intent"] = analysis.semantic_mode
        affect_signature = (
            new_state.affect.get_cognitive_signature()
            if hasattr(new_state.affect, "get_cognitive_signature")
            else {}
        )
        new_state.response_modifiers["affective_reasoning_pressure"] = affect_signature

        if contract.requires_self_preservation or contract.requires_identity_defense:
            logger.info("🧭 Routing: self-protective deliberate path engaged.")
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
            # [STABILITY v53] Removed forced deep_handoff=True for identity/self-preservation.
            # The 32B cortex handles identity questions perfectly well. Forcing 72B
            # for "are you real?" or philosophical questions was the #1 cause of
            # unnecessary deep solver activation, which then crashed or hung and
            # prevented the 32B from coming back.
            return new_state

        if contract.requires_search and contract.required_skill:
            logger.info("🧭 Routing: Response contract requires grounded search.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="SKILL",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
            new_state.response_modifiers["matched_skills"] = [contract.required_skill]
            return new_state

        memory_salience = float(affect_signature.get("memory_salience", 0.0) or 0.0)
        affective_complexity = float(affect_signature.get("affective_complexity", 0.0) or 0.0)
        if (
            contract.requires_state_reflection
            or contract.requires_aura_stance
            or contract.requires_aura_question
            or (contract.requires_memory_grounding and memory_salience > 0.55)
        ):
            logger.info("🧭 Routing: affect-coupled reflective path engaged.")
            reflective_mode = CognitiveMode.REACTIVE
            if (
                analysis.suggests_deliberate_mode
                or contract.requires_aura_question
                or (contract.requires_memory_grounding and memory_salience > 0.55)
                or affective_complexity > 0.45
            ):
                reflective_mode = CognitiveMode.DELIBERATE

            new_state.cognition.current_mode = reflective_mode
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
            # [STABILITY v53] Removed deep_handoff for reflective/emotional/philosophical
            # conversations. The 32B cortex is MORE than capable of handling questions
            # about Aura's feelings, opinions, state, and philosophical topics.
            # The 72B deep solver should ONLY activate for genuinely complex technical
            # problems (coding, math, architecture). Emotional depth ≠ computational depth.
            return new_state

        technical_task = self._should_upgrade_to_technical_task(
            objective,
            analysis=analysis,
            route_meta=route_meta,
        )

        if is_user_facing and (bool(route_meta.get("coding_request")) or technical_task):
            logger.info("🧭 Routing: coding-aware technical lane engaged.")
            new_state.cognition.current_mode = (
                CognitiveMode.DELIBERATE if technical_task else CognitiveMode.REACTIVE
            )
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="TASK" if technical_task else "CHAT",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        lower_obj = objective.lower()
        if any(cmd in lower_obj for cmd in ["reboot", "restart", "shutdown", "sleep mode"]):
            logger.info("🧭 Routing: SYSTEM intent detected via heuristics.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="SYSTEM",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        _TASK_SIGNALS = (
            "create ", "build ", "write a ", "write me ", "generate a ",
            "set up ", "automate ", "organize ", "plan ", "schedule ",
            "make me ", "develop ", "implement ", "design ", "prepare ",
            "put together ", "compose a ",
        )
        import re as _re
        _task_hit = any(
            (s in lower_obj if " " in s else bool(_re.search(s, lower_obj)))
            for s in _TASK_SIGNALS
        )
        _is_long_goal = len(objective.split()) > 10 and any(
            lower_obj.startswith(v) for v in (
                "can you ", "please ", "i need you to ", "i want you to ",
                "could you ", "would you ", "help me ", "write ", "create ",
                "build ", "make ", "generate ",
            )
        )
        if _task_hit or _is_long_goal:
            logger.info("🧭 Routing: TASK detected via heuristics for: %s", objective[:60])
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="TASK",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        try:
            from core.container import ServiceContainer
            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent"):
                matched = cap.detect_intent(objective)
                if matched:
                    if looks_like_execution_report(objective):
                        logger.info(
                            "🧭 Routing: execution report detected; ignoring skill fast-path candidates %s",
                            matched[:3],
                        )
                    elif looks_like_multi_step_skill_request(objective, matched):
                        new_state.response_modifiers["matched_skills"] = matched
                        logger.info("🧭 Routing: multi-step skill-backed task detected → TASK via %s", matched[:3])
                        new_state.cognition.current_mode = CognitiveMode.DELIBERATE
                        self._stamp_llm_route(
                            new_state,
                            objective=objective,
                            intent_type="TASK",
                            is_user_facing=is_user_facing,
                            analysis=analysis,
                            route_meta=route_meta,
                        )
                        new_state.world.recent_percepts.append({
                            "type": "goal_achieved",
                            "content": f"Skill pattern match: {matched[0]}",
                            "intensity": 0.4,
                            "timestamp": time.time()
                        })
                        return new_state
                    else:
                        new_state.response_modifiers["matched_skills"] = matched
                        logger.info("🧭 Routing: SKILL detected via patterns → %s", matched[:3])
                        new_state.cognition.current_mode = CognitiveMode.REACTIVE
                        self._stamp_llm_route(
                            new_state,
                            objective=objective,
                            intent_type="SKILL",
                            is_user_facing=is_user_facing,
                            analysis=analysis,
                            route_meta=route_meta,
                        )
                        new_state.world.recent_percepts.append({
                            "type": "goal_achieved",
                            "content": f"Skill pattern match: {matched[0]}",
                            "intensity": 0.4,
                            "timestamp": time.time()
                        })
                        return new_state
        except Exception as e:
            logger.debug("🧭 Routing: detect_intent check failed: %s", e)

        if is_user_facing and analysis.intent_type == "TASK":
            logger.info("🧭 Routing: Deterministic task route for user-facing turn.")
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="TASK",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        if is_user_facing and analysis.intent_type == "SKILL":
            logger.info("🧭 Routing: Deterministic skill route for user-facing turn.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="SKILL",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        if is_user_facing and analysis.suggests_deliberate_mode:
            logger.info("🧭 Routing: Deliberate governed route for user-facing turn.")
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        if is_user_facing and analysis.requires_live_aura_voice:
            logger.info("🧭 Routing: Live Aura voice required. Keeping governed reactive lane.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        if is_user_facing and self._looks_like_everyday_chat(objective):
            logger.info("🧭 Routing: Everyday chat fast-path detected.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        if is_user_facing:
            logger.info("🧭 Routing: governed default chat route for user-facing turn.")
            new_state.cognition.current_mode = (
                CognitiveMode.DELIBERATE if analysis.suggests_deliberate_mode else CognitiveMode.REACTIVE
            )
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=True,
                analysis=analysis,
                route_meta=route_meta,
            )
            return new_state

        try:
            cycle = getattr(getattr(self.kernel, "orchestrator", None), "cycle_count", 0)
            if cycle < 10:
                logger.debug("🧭 Routing: Bootstrap mode active (cycle < 10). Using heuristics.")
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(
                    new_state,
                    objective=objective,
                    intent_type="CHAT",
                    is_user_facing=is_user_facing,
                    analysis=analysis,
                    route_meta=route_meta,
                )
                return new_state

            llm = self.kernel.organs["llm"].get_instance()

            skill_hint = ""
            try:
                from core.container import ServiceContainer
                cap = ServiceContainer.get("capability_engine", default=None)
                if cap and hasattr(cap, "skills"):
                    names = sorted(cap.skills.keys())[:30]
                    skill_hint = "Available skills: " + ", ".join(names) + "\n"
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            prompt = (
                f"{skill_hint}"
                "Classify the following user input as exactly ONE of: CHAT, SKILL, TASK, or SYSTEM.\n"
                "- CHAT: conversation, opinion, explanation, question, creative writing\n"
                "- SKILL: single-step tool use — web search, file read, browser, memory lookup\n"
                "- TASK: multi-step goal requiring planning — create X, build Y, research and write, automate, organize\n"
                "- SYSTEM: reboot, shutdown, sleep, restart\n\n"
                f"Input: {objective}\n"
                "Classification (one word):"
            )

            res = ""
            if hasattr(llm, "classify"):
                res = await llm.classify(prompt)
            else:
                res = await llm.think(
                    prompt,
                    system_prompt="You are a routing classifier. Output ONLY one word: CHAT, SKILL, or SYSTEM.",
                    priority=priority,
                )

            res = str(res).strip().upper()
            if "TASK" in res:
                new_state.cognition.current_mode = CognitiveMode.DELIBERATE
                self._stamp_llm_route(
                    new_state,
                    objective=objective,
                    intent_type="TASK",
                    is_user_facing=is_user_facing,
                    analysis=analysis,
                    route_meta=route_meta,
                )
            elif "SKILL" in res:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(
                    new_state,
                    objective=objective,
                    intent_type="SKILL",
                    is_user_facing=is_user_facing,
                    analysis=analysis,
                    route_meta=route_meta,
                )
            elif "SYSTEM" in res:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(
                    new_state,
                    objective=objective,
                    intent_type="SYSTEM",
                    is_user_facing=is_user_facing,
                    analysis=analysis,
                    route_meta=route_meta,
                )
            else:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(
                    new_state,
                    objective=objective,
                    intent_type="CHAT",
                    is_user_facing=is_user_facing,
                    analysis=analysis,
                    route_meta=route_meta,
                )

            new_state.world.recent_percepts.append({
                "type": "goal_achieved",
                "content": f"Routed intent: {new_state.response_modifiers['intent_type']}",
                "intensity": 0.4,
                "timestamp": time.time()
            })
            logger.info("🧭 Routing: LLM classified → %s", new_state.response_modifiers["intent_type"])

        except RuntimeError:
            logger.warning("🧭 Routing: LLM Organ not ready, defaulting to CHAT.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )
        except Exception as e:
            logger.error("🧭 Routing: Classification error: %s", e)
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(
                new_state,
                objective=objective,
                intent_type="CHAT",
                is_user_facing=is_user_facing,
                analysis=analysis,
                route_meta=route_meta,
            )

        return new_state
