from __future__ import annotations
import logging
import time
from typing import Optional, TYPE_CHECKING
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
    "architect", "security audit", "mathematical proof", "deep dive",
    "complex analysis", "bottleneck analysis", "vulnerability scan",
    "formal proof", "root cause analysis", "flagship architecture",
})


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
    def _should_allow_deep_handoff(text: str, *, is_user_facing: bool, intent_type: str) -> bool:
        if not is_user_facing or intent_type not in {"CHAT", "TASK"}:
            return False
        lower = text.lower()
        word_count = len(text.split())
        if word_count >= 120 or len(text) >= 900:
            return True
        return any(keyword in lower for keyword in _DEEP_HANDOFF_KEYWORDS)

    @staticmethod
    def _looks_like_everyday_chat(text: str) -> bool:
        stripped = str(text or "").strip()
        if not stripped:
            return False
        lower = stripped.lower()
        if len(stripped.split()) > 32 or len(stripped) > 220:
            return False
        if any(keyword in lower for keyword in _DEEP_HANDOFF_KEYWORDS):
            return False
        return True

    def _stamp_llm_route(self, state: AuraState, *, objective: str, intent_type: str, is_user_facing: bool) -> None:
        model_tier = self._resolve_model_tier(is_user_facing)
        deep_handoff = self._should_allow_deep_handoff(
            objective,
            is_user_facing=is_user_facing,
            intent_type=intent_type,
        )
        state.response_modifiers["intent_type"] = intent_type
        state.response_modifiers["model_tier"] = model_tier
        state.response_modifiers["deep_handoff"] = deep_handoff
        logger.info(
            "🧠 CognitiveRouting: Mode=%s, Tier=%s, DeepHandoff=%s",
            state.cognition.current_mode.name,
            model_tier,
            deep_handoff,
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
        affect_signature = (
            new_state.affect.get_cognitive_signature()
            if hasattr(new_state.affect, "get_cognitive_signature")
            else {}
        )
        new_state.response_modifiers["affective_reasoning_pressure"] = affect_signature

        if contract.requires_self_preservation or contract.requires_identity_defense:
            logger.info("🧭 Routing: self-protective deliberate path engaged.")
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)
            new_state.response_modifiers["deep_handoff"] = True
            return new_state

        if contract.requires_search and contract.required_skill:
            logger.info("🧭 Routing: Response contract requires grounded search.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(new_state, objective=objective, intent_type="SKILL", is_user_facing=is_user_facing)
            new_state.response_modifiers["matched_skills"] = [contract.required_skill]
            return new_state

        memory_salience = float(affect_signature.get("memory_salience", 0.0) or 0.0)
        affective_complexity = float(affect_signature.get("affective_complexity", 0.0) or 0.0)
        if contract.requires_state_reflection or (contract.requires_memory_grounding and memory_salience > 0.55):
            logger.info("🧭 Routing: affect-coupled reflective path engaged.")
            new_state.cognition.current_mode = CognitiveMode.DELIBERATE
            self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)
            if contract.requires_state_reflection and affective_complexity > 0.45:
                new_state.response_modifiers["deep_handoff"] = True
            return new_state

        lower_obj = objective.lower()
        if any(cmd in lower_obj for cmd in ["reboot", "restart", "shutdown", "sleep mode"]):
            logger.info("🧭 Routing: SYSTEM intent detected via heuristics.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(new_state, objective=objective, intent_type="SYSTEM", is_user_facing=is_user_facing)
            return new_state

        _TASK_SIGNALS = (
            "create ", "build ", "write a ", "write me ", "generate a ",
            "set up ", "automate ", "organize ", "plan ", "schedule ",
            "make me ", "develop ", "implement ", "design ", "prepare ",
            "put together ", "compose a ", " and then ", "first.*then",
            "step by step", "multiple ", "a series of",
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
            self._stamp_llm_route(new_state, objective=objective, intent_type="TASK", is_user_facing=is_user_facing)
            return new_state

        try:
            from core.container import ServiceContainer
            cap = ServiceContainer.get("capability_engine", default=None)
            if cap and hasattr(cap, "detect_intent"):
                matched = cap.detect_intent(objective)
                if matched:
                    logger.info("🧭 Routing: SKILL detected via patterns → %s", matched[:3])
                    new_state.cognition.current_mode = CognitiveMode.REACTIVE
                    self._stamp_llm_route(new_state, objective=objective, intent_type="SKILL", is_user_facing=is_user_facing)
                    new_state.response_modifiers["matched_skills"] = matched
                    new_state.world.recent_percepts.append({
                        "type": "goal_achieved",
                        "content": f"Skill pattern match: {matched[0]}",
                        "intensity": 0.4,
                        "timestamp": time.time()
                    })
                    return new_state
        except Exception as e:
            logger.debug("🧭 Routing: detect_intent check failed: %s", e)

        if is_user_facing and self._looks_like_everyday_chat(objective):
            logger.info("🧭 Routing: Everyday chat fast-path detected.")
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=True)
            return new_state

        try:
            cycle = getattr(getattr(self.kernel, "orchestrator", None), "cycle_count", 0)
            if cycle < 10:
                logger.debug("🧭 Routing: Bootstrap mode active (cycle < 10). Using heuristics.")
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)
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
                self._stamp_llm_route(new_state, objective=objective, intent_type="TASK", is_user_facing=is_user_facing)
            elif "SKILL" in res:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(new_state, objective=objective, intent_type="SKILL", is_user_facing=is_user_facing)
            elif "SYSTEM" in res:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(new_state, objective=objective, intent_type="SYSTEM", is_user_facing=is_user_facing)
            else:
                new_state.cognition.current_mode = CognitiveMode.REACTIVE
                self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)

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
            self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)
        except Exception as e:
            logger.error("🧭 Routing: Classification error: %s", e)
            new_state.cognition.current_mode = CognitiveMode.REACTIVE
            self._stamp_llm_route(new_state, objective=objective, intent_type="CHAT", is_user_facing=is_user_facing)

        return new_state
