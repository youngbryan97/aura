from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import deque
from typing import TYPE_CHECKING

from core.phases.response_contract import (
    _looks_like_search_capability_question,
    build_response_contract,
)
from core.runtime.background_policy import background_activity_allowed
from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.errors import record_degradation
from core.runtime.skill_task_bridge import (
    looks_like_explanatory_dialogue_request,
    looks_like_inline_answer_request,
    looks_like_multi_step_skill_request,
    normalize_matched_skills,
)
from core.runtime.state_ownership import state_root
from core.runtime.structured_input import looks_like_learning_resource_bundle
from core.runtime.tool_result_contracts import compact_result_payload
from core.runtime.turn_analysis import canonical_turn_text
from core.state.aura_state import AuraState
from core.utils.task_tracker import get_task_tracker

from .bridge import Phase

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.10x")


def _record_upgrades_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "upgrades_10x",
        error,
        severity=severity,
        action=action,
    )


_SIMPLE_DIALOGUE_RE = re.compile(
    r"\b("
    r"capital of france|15\s*\*\s*12|square root of 64|3 apples|"
    r"who wrote (?:the play )?hamlet|three programming languages|"
    r"color is the sky|translate ['\"]?good morning|"
    r"continuity check|what did we just verify|live chat path|"
    r"conversation lane|reply path|response path|"
    r"you ok|you okay|are you ok|are you okay|"
    r"what feels most important|what should you do differently|"
    r"write (?:a )?(?:short )?(?:poem|joke|haiku)|"
    r"compose (?:a )?(?:short )?(?:poem|joke|haiku)"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_simple_dialogue_request(text: str) -> bool:
    body = str(text or "").strip()
    if not body:
        return False
    if looks_like_explanatory_dialogue_request(body):
        return True
    if len(body.split()) > 28:
        return False
    return bool(_SIMPLE_DIALOGUE_RE.search(body))


def _compact_skill_result_payload(result: object) -> dict[str, object]:
    return compact_result_payload(result)


def _objective_fingerprint(objective: object) -> str:
    text = " ".join(str(objective or "").split()).strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────
# PHASE 1: EternalMemoryPhase → Persistent Memory Agent = 10/10
# ──────────────────────────────────────────────────────────────
class EternalMemoryPhase(Phase):
    """Durable continuity cache with bounded retrieval and explicit persistence."""

    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel
        self.vault_path = state_root() / "eternal_vault.jsonl"
        self.vault_path.parent.mkdir(exist_ok=True)
        self._summary_cache: list[dict[str, str]] = []
        self._last_summary_refresh_at: float = 0.0
        self._last_summary_refresh_completed_at: float = 0.0
        self._summary_refresh_interval_s: float = 120.0
        self._history_slice_limit: int = 512
        self._summary_refresh_task: asyncio.Task | None = None

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        # Handle optional objective from the kernel without inventing a stronger claim.
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Continuity")

        eternal_summary = await self._get_cached_or_refresh_summary()

        # Merge summary into working memory
        state.cognition.working_memory = eternal_summary + state.cognition.working_memory[-8:]

        # Prepare and queue the new entry
        entry = self._prepare_eternal_entry(state)

        state.cognition.pending_intents.append(
            {"type": "eternal_append", "path": str(self.vault_path), "payload": entry}
        )

        return state

    def _load_eternal_slice(self, limit: int):
        if self.vault_path.exists():
            try:
                with open(self.vault_path, "rb") as f:
                    # Optimized read of last N lines
                    return [json.loads(line) for line in deque(f, limit)]
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                _record_upgrades_degradation(
                    e,
                    action="returned empty eternal memory slice and retained cached summary",
                )
                logger.error("Failed to read eternal slice: %s", e)
        return []

    async def _get_cached_or_refresh_summary(self) -> list[dict[str, str]]:
        """Return the current summary and schedule stale maintenance single-flight.

        Eternal-memory synthesis is useful background cognition, but it must not
        hold the unitary kernel lock while a local model loads or decodes.  The
        next tick consumes a completed refresh; this tick keeps the last durable
        summary (or an empty cache during first boot).
        """
        now = time.time()
        if self._summary_refresh_task is not None and not self._summary_refresh_task.done():
            return list(self._summary_cache)
        if (now - self._last_summary_refresh_at) < self._summary_refresh_interval_s:
            return list(self._summary_cache)
        if self._background_llm_should_defer():
            return list(self._summary_cache)

        self._last_summary_refresh_at = now
        self._summary_refresh_task = get_task_tracker().create_task(
            self._refresh_eternal_summary(),
            name="EternalMemoryPhase.summary_refresh",
        )
        return list(self._summary_cache)

    async def _refresh_eternal_summary(self) -> None:
        try:
            history = await asyncio.to_thread(
                self._load_eternal_slice,
                limit=self._history_slice_limit,
            )
            summary = await self._generate_eternal_summary(history)
            if summary:
                self._summary_cache = list(summary)
            self._last_summary_refresh_completed_at = time.time()
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_upgrades_degradation(
                exc,
                action="retained prior eternal memory after background refresh failed",
            )
        finally:
            if self._summary_refresh_task is asyncio.current_task():
                self._summary_refresh_task = None

    def _prepare_eternal_entry(self, state: AuraState) -> dict:
        return {
            "version": state.version,
            "timestamp": time.time(),
            "objective": state.cognition.current_objective,
            "affect": {k: v for k, v in vars(state.affect).items() if not k.startswith("_")},
            "summary": state.identity.current_narrative[:500],
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
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    _record_upgrades_degradation(
                        _exc,
                        action="continued eternal-memory scheduling without local deferral signal",
                    )
                    logger.debug("Eternal memory local deferral probe skipped: %s", _exc)
            if gate and hasattr(gate, "_should_quiet_background_for_cortex_startup"):
                try:
                    if gate._should_quiet_background_for_cortex_startup():
                        return True
                except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                    _record_upgrades_degradation(
                        _exc,
                        action="continued eternal-memory scheduling without cortex startup quiet signal",
                    )
                    logger.debug("Eternal memory cortex quiet probe skipped: %s", _exc)
            if not gate or not hasattr(gate, "get_conversation_status"):
                return False
            lane = gate.get_conversation_status() or {}
            if lane.get("conversation_ready"):
                return False
            lane_state = str(lane.get("state", "") or "").strip().lower()
            if lane.get("warmup_in_flight"):
                return True
            return lane_state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except (ImportError, AttributeError, RuntimeError):
            return False

    async def _generate_eternal_summary(self, history: list[dict]):
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
                except (ImportError, AttributeError, RuntimeError) as _exc:
                    _record_upgrades_degradation(
                        _exc,
                        action="returned no new eternal summary after degraded-event emission failed",
                    )
                    logger.debug("Eternal memory summary-unavailable event skipped: %s", _exc)
                return []
            return [{"role": "system", "content": f"[ETERNAL MEMORY]\n{response}"}]
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_upgrades_degradation(
                e,
                action="retained prior eternal memory summary after summary generation failed",
            )
            try:
                from core.health.degraded_events import record_degraded_event

                record_degraded_event(
                    "eternal_memory",
                    "summary_failed",
                    detail=str(e)[:200],
                    severity="warning",
                    classification="background_degraded",
                )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_upgrades_degradation(
                    _exc,
                    action="retained prior eternal memory summary after failure event emission failed",
                )
                logger.debug("Eternal memory summary-failed event skipped: %s", _exc)
            logger.warning("EternalMemory: Summary generation failed: %s", e)
            return []


# ──────────────────────────────────────────────────────────────
# PHASE 2: TrueEvolutionPhase → bounded evolution proposal loop
# ──────────────────────────────────────────────────────────────
class TrueEvolutionPhase(Phase):
    """Morphic exploration plus governed self-modification proposal routing."""

    def __init__(self, kernel: AuraKernel, engine=None):
        self.kernel = kernel
        self.engine = engine

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        # Handle optional objective from the kernel without inflating the claim.
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Evolution")

        # 1. Autopoiesis on Concept Graph
        if hasattr(state.identity, "concept_graph"):
            dissonance = getattr(state.affect, "surprise", 0.1)
            # Basic friction: perturb connection weights based on surprise
            for _node, edges in getattr(state.identity.concept_graph, "nodes", {}).items():
                if isinstance(edges, dict):
                    for neighbor in edges:
                        edges[neighbor] += dissonance * 0.01

        # 2. Curiosity Triggered Morphic Clone for Deep Exploration
        if (
            state.affect.curiosity > 0.92
            and random.random() < 0.3
            and background_activity_allowed(
                getattr(self.kernel, "orchestrator", None),
                min_idle_seconds=1800.0,
                max_memory_percent=76.0,
                max_failure_pressure=0.05,
                require_conversation_ready=False,
            )
        ):
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
                            await queue.put(
                                {
                                    "type": "morphic_insight",
                                    "content": res,
                                    "timestamp": time.time(),
                                }
                            )
                        else:
                            logger.debug(
                                "Evolution: initiative_queue not registered; insight dropped."
                            )
                except (ImportError, AttributeError, RuntimeError) as e:
                    _record_upgrades_degradation(
                        e,
                        action="dropped background morphic insight and kept evolution tick alive",
                    )
                    logger.warning("Evolution: Background exploration failed: %s", e)

            get_task_tracker().create_task(_background_explore())

        # 3. Governed self-modification proposal path.
        if (
            getattr(state.identity, "evolution_score", 0.0) > 0.70
            and background_activity_allowed(
                getattr(self.kernel, "orchestrator", None),
                min_idle_seconds=1800.0,
                max_memory_percent=76.0,
                max_failure_pressure=0.05,
                require_conversation_ready=True,
            )
        ):
            await self._safe_self_modify(state)

        return state

    async def _safe_self_modify(self, state):
        logger.info("⚡ [SELF-IMPROVEMENT] Initiating governed self-improvement proposal cycle.")

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
            refinements_applied = int(result.get("refinements_applied", 0) or 0)
            changed_files = tuple(
                str(path or "").strip()
                for path in (result.get("changed_files") or ())
                if str(path or "").strip()
            )
            reload_required = bool(result.get("reload_required", False))
            if (
                result.get("success")
                and refinements_applied > 0
                and changed_files
                and reload_required
            ):
                logger.info(
                    "✅ Evolution: %d optimization(s) applied to %d file(s); "
                    "requesting a bounded code refresh.",
                    refinements_applied,
                    len(changed_files),
                )
                refresh = await self.kernel.hot_reboot(changed_files=changed_files)
                if refresh.get("reloaded"):
                    # Identity advances only when the new implementation became
                    # part of this running organism. A source edit that requires
                    # restart is durable work, but is not yet a lived transition.
                    state.identity.narrative_version += 1
                else:
                    logger.info(
                        "Evolution: source change retained for restart; live "
                        "identity remains on the active implementation."
                    )
            elif result.get("success"):
                logger.info(
                    "Evolution: refinement cycle completed without an applied "
                    "source change; identity and runtime remain unchanged."
                )
            else:
                logger.warning("⚠️ Evolution: Refinement cycle completed with no applied changes.")
        except (OSError, ConnectionError, TimeoutError) as e:
            _record_upgrades_degradation(
                e,
                action="left identity version unchanged after self-refinement failure",
                severity="error",
            )
            logger.error("❌ Evolution: Refinement cycle failed: %s", e)


# ──────────────────────────────────────────────────────────────
# PHASE 3: PerfectEmotionPhase → Emotional / Character AI = 10/10
# ──────────────────────────────────────────────────────────────
class PerfectEmotionPhase(Phase):
    """DamasioV2 on steroids with real somatic feedback loop."""

    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:

        # Real-time somatic mirroring from hardware
        hardware_stress = 0.1
        try:
            # Tap directly into Proprioceptive/Metabolic state
            cpu = state.soma.hardware.get("cpu_usage", 0.0)
            hardware_stress = min(1.0, cpu / 100.0)
        except (OSError, ConnectionError, TimeoutError) as e:
            _record_upgrades_degradation(
                e,
                action="continued emotional tick with neutral hardware stress baseline",
            )
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

    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel
        self._cap_engine = None

    def _get_cap_engine(self):
        if self._cap_engine is None:
            try:
                from core.container import ServiceContainer

                self._cap_engine = ServiceContainer.get("capability_engine", default=None)
            except (ImportError, AttributeError, RuntimeError) as _exc:
                _record_upgrades_degradation(
                    _exc,
                    action="continued GodMode routing without capability engine",
                )
                logger.debug("GodMode capability engine unavailable: %s", _exc)
        return self._cap_engine

    @staticmethod
    def _normalize_origin(origin: str) -> str:
        return str(origin or "").strip().lower().replace("-", "_")

    def _resolve_tool_source(self, state: AuraState) -> str:
        origin = self._normalize_origin(getattr(state.cognition, "current_origin", "") or "")
        if origin in {
            "user",
            "voice",
            "admin",
            "api",
            "gui",
            "ws",
            "websocket",
            "direct",
            "external",
        }:
            return origin
        return "godmode_phase"

    @staticmethod
    def _is_direct_memory_write_request(objective: str) -> bool:
        lower = str(objective or "").lower()
        return bool(
            re.search(r"^\s*remember\s*:", lower)
            or any(
                marker in lower
                for marker in (
                    "remember this",
                    "remember that",
                    "remember for future",
                    "remember for later",
                    "save this",
                    "save that",
                    "store this",
                    "store that",
                    "don't forget",
                    "don’t forget",
                    "make note",
                    "commit this to memory",
                    "commit that to memory",
                )
            )
        )

    @staticmethod
    def _is_conversational_memory_question(objective: str) -> bool:
        lower = str(objective or "").lower()
        if GodModeToolPhase._is_direct_memory_write_request(lower):
            return False
        return any(
            marker in lower
            for marker in (
                "what do you remember",
                "what do you remeber",
                "do you remember",
                "what did we talk about",
                "from our last conversation",
                "from the previous conversation",
                "our history",
                "what do you know about me",
            )
        )

    @staticmethod
    def _extract_python_code_payload(objective: str) -> str:
        text = str(objective or "")
        for match in re.finditer(
            r"```(?P<lang>python|py)\b\s*(?P<body>.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        ):
            body = (match.group("body") or "").strip()
            if body:
                return body
        for match in re.finditer(r"```\s*(?P<body>.*?)```", text, re.DOTALL):
            body = (match.group("body") or "").strip()
            if not body:
                continue
            first_token = body.split(None, 1)[0].strip().lower() if body.split(None, 1) else ""
            if first_token in {"javascript", "js", "bash", "sh", "zsh", "shell"}:
                continue
            return body

        prefix_match = re.match(
            r"^\s*(?:python|code)\s*:\s*(?P<body>.+?)\s*$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if prefix_match:
            return prefix_match.group("body").strip()

        direct_match = re.search(
            r"\b(?:run|execute)\s+(?:this\s+)?(?:code|script|python)\s*:\s*(?P<body>.+?)\s*$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if direct_match:
            return direct_match.group("body").strip()

        return ""

    @staticmethod
    def _extract_safe_arithmetic_expression(objective: str) -> str:
        text = " ".join(str(objective or "").strip().split())
        if not text:
            return ""

        match = re.match(
            r"^(?:calculate|compute|what is|evaluate(?: this)?(?: expression| formula)?)"
            r"\s*[:\-]?\s*(?P<expr>[-+*/%().,\d\s]+?)\s*[?.!]?$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return ""

        expr = match.group("expr").strip().replace(",", "")
        if not expr or not re.search(r"\d", expr) or not re.search(r"[+\-*/%]", expr):
            return ""
        if not re.fullmatch(r"[-+*/%().\d\s]+", expr):
            return ""

        try:
            import ast

            parsed = ast.parse(expr, mode="eval")
        except (SyntaxError, ValueError):
            return ""

        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Load,
        )
        if not all(isinstance(node, allowed_nodes) for node in ast.walk(parsed)):
            return ""
        return expr

    @staticmethod
    def _is_stateless_diagnostic_python(objective: str, code: str) -> bool:
        """Return true when a snippet is safe to run as isolated diagnostic compute."""
        raw_code = str(code or "").strip()
        if not raw_code:
            return False
        objective_lower = str(objective or "").lower()
        if any(
            marker in objective_lower
            for marker in (
                "keep state",
                "keep variables",
                "persist variables",
                "reuse variables",
                "stateful",
                "session state",
            )
        ):
            return False
        try:
            import ast

            tree = ast.parse(raw_code)
        except (SyntaxError, ValueError):
            return False

        banned_import_roots = {
            "asyncio",
            "httpx",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "sys",
            "tempfile",
            "urllib",
        }
        banned_calls = {
            "__import__",
            "compile",
            "delattr",
            "eval",
            "exec",
            "globals",
            "input",
            "locals",
            "open",
            "setattr",
            "vars",
        }
        banned_attr_calls = {
            "check_call",
            "check_output",
            "chmod",
            "mkdir",
            "open",
            "popen",
            "post",
            "put",
            "read_bytes",
            "read_text",
            "remove",
            "rename",
            "replace",
            "request",
            "rmdir",
            "run",
            "system",
            "unlink",
            "write",
            "write_bytes",
            "write_text",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in banned_import_roots:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".", 1)[0] in banned_import_roots:
                    return False
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in banned_calls:
                    return False
                if isinstance(func, ast.Attribute) and func.attr in banned_attr_calls:
                    return False
        return True

    @staticmethod
    def _looks_like_direct_run_code_request(objective: str) -> bool:
        text = str(objective or "")
        lower = text.lower()
        if GodModeToolPhase._extract_python_code_payload(text):
            explicit_run = bool(
                re.search(
                    r"\b(?:run|execute)\s+(?:this\s+)?(?:code|script|python)\b",
                    text,
                    re.IGNORECASE,
                )
            )
            execution_analysis_markers = (
                "trace the execution",
                "trace execution",
                "trace the output",
                "compute the printed output",
                "determine the exact output",
                "determine the result",
                "determine the boolean result",
                "determine the length",
                "determine the value",
                "evaluate the exact printed output",
                "provide the final boolean result",
                "analyze the behavior",
                "what is printed",
                "what does this print",
                "printed output",
                "when executed",
                "when it executes",
                "when this executes",
            )
            exception_analysis_markers = (
                "what exception",
                "which exception",
                "exception class",
                "error class",
                "class raised",
                "exception raised",
                "error raised",
                "raises a",
                "raised by",
            )
            return (
                explicit_run
                or any(marker in lower for marker in execution_analysis_markers)
                or any(marker in lower for marker in exception_analysis_markers)
            )
        if GodModeToolPhase._extract_safe_arithmetic_expression(text):
            return True
        return bool(
            re.search(
                r"\b(?:run|execute)\s+(?:this\s+)?(?:code|script|python)\b",
                text,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _choose_best_skill(objective: str, matched_skills: list[str]) -> str:
        if not matched_skills:
            return ""
        lower = str(objective or "").lower()
        if (
            "manifest_to_device" in matched_skills
            and "desktop" in lower
            and "http" in lower
            and any(marker in lower for marker in ("save", "manifest"))
        ):
            return "manifest_to_device"
        if "sovereign_terminal" in matched_skills and re.match(
            r"^\s*(?:execute|run|terminal)\s*:", lower
        ):
            return "sovereign_terminal"
        visible_browser_markers = (
            "open a tab",
            "open the tab",
            "open tab",
            "new tab",
            "on my computer",
            "on the computer",
            "on my screen",
            "desktop",
        )
        if any(marker in lower for marker in visible_browser_markers) and any(
            marker in lower for marker in ("search", "google", "look up", "open", "browser", "tab")
        ):
            return "computer_use"
        if "clock" in matched_skills and any(
            marker in lower
            for marker in (
                "what time",
                "current time",
                "the time",
                "what date",
                "current date",
                "what day",
                "timer",
                "remind me",
            )
        ):
            return "clock"
        if (
            "web_search" in matched_skills
            and not _looks_like_search_capability_question(objective)
            and any(
                marker in lower
                for marker in (
                    "search",
                    "look up",
                    "find out",
                    "online",
                    "internet",
                    "current",
                    "latest",
                    "news",
                    "research about",
                    "research on",
                )
            )
        ):
            return "web_search"
        if "sovereign_browser" in matched_skills and any(
            marker in lower
            for marker in (
                "open the browser",
                "open a browser",
                "open tab",
                "navigate to",
                "visit ",
                "open website",
                "open webpage",
            )
        ):
            return "sovereign_browser"
        if "memory_ops" in matched_skills and GodModeToolPhase._is_conversational_memory_question(
            objective
        ):
            return ""
        if "memory_ops" in matched_skills and GodModeToolPhase._is_direct_memory_write_request(
            objective
        ):
            return "memory_ops"
        remaining = [skill for skill in matched_skills if skill != "memory_ops"]
        if "run_code" in remaining and not GodModeToolPhase._looks_like_direct_run_code_request(
            objective
        ):
            remaining = [skill for skill in remaining if skill != "run_code"]
        return remaining[0] if remaining else ""

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        from core.phases.response_contract import extract_search_query_focus

        text = str(objective or "").strip()
        if not text:
            return ""
        return extract_search_query_focus(text)

    @staticmethod
    def _search_url(query: str) -> str:
        import urllib.parse as _urlparse

        cleaned = str(query or "").strip()
        if cleaned.startswith(("http://", "https://")):
            return cleaned
        return f"https://duckduckgo.com/?q={_urlparse.quote_plus(cleaned)}"

    @staticmethod
    def _normalize_skill_params(skill_name: str, objective: str, params: dict | None) -> dict:
        normalized = dict(params or {}) if isinstance(params, dict) else {}
        lower = str(objective or "").lower()

        if skill_name == "sovereign_terminal":
            command_match = re.match(
                r"^\s*(?:execute|run|terminal)(?:\s+the\s+command)?\s*:\s*(.+?)\s*$",
                str(objective or ""),
                re.IGNORECASE | re.DOTALL,
            )
            if command_match:
                normalized["action"] = "execute"
                normalized["command"] = command_match.group(1).strip()

        if skill_name == "manifest_to_device":
            url_match = re.search(r"https?://[^\s<>\"\')\]]+", objective)
            if url_match:
                normalized["url"] = url_match.group(0)

        if skill_name == "file_operation":
            exists_match = re.search(
                r"(?:(?:check|see|verify|test)\s+(?:if\s+)?|does\s+)(.+?)\s+exist(?:s)?(?:\.|!|\?|$)",
                str(objective or ""),
                re.IGNORECASE | re.DOTALL,
            )
            if exists_match:
                normalized["action"] = "exists"
                normalized["path"] = exists_match.group(1).strip().strip("'\"")

        if skill_name == "memory_ops":
            is_recall = any(
                marker in lower
                for marker in (
                    "what do you remember",
                    "what do you know about me",
                    "recall",
                    "retrieve",
                )
            )
            normalized.setdefault("action", "recall" if is_recall else "remember")
            if is_recall:
                normalized.setdefault("query", objective)
            else:
                normalized.setdefault("content", objective)

        if skill_name == "run_code":
            if not str(normalized.get("code") or "").strip():
                code = GodModeToolPhase._extract_python_code_payload(objective)
                if code:
                    normalized["code"] = code
                    if GodModeToolPhase._is_stateless_diagnostic_python(objective, code):
                        normalized.setdefault("stateful", False)
                else:
                    expression = GodModeToolPhase._extract_safe_arithmetic_expression(objective)
                    if expression:
                        normalized["code"] = f"print({expression})"
                        normalized.setdefault("stateful", False)

        if skill_name in {
            "web_search",
            "search_web",
            "free_search",
            "grounded_search",
            "sovereign_browser",
        }:
            # Detect URLs in the objective — if present, BROWSE the URL directly
            # instead of searching the entire message text on a search engine.
            import re as _re

            url_match = _re.search(r"https?://[^\s<>\"\')\]]+", objective)
            if skill_name == "sovereign_browser" and url_match:
                normalized.setdefault("mode", "browse")
                normalized.setdefault("url", url_match.group(0))
            else:
                if skill_name == "sovereign_browser":
                    normalized.setdefault("mode", "search")
                raw_query = " ".join(str(objective or "").split()).strip()
                query = GodModeToolPhase._extract_search_query(objective)
                if (
                    skill_name == "sovereign_browser"
                    and query
                    and raw_query
                    and query.rstrip(" .?!,:;") == raw_query.rstrip(" .?!,:;")
                ):
                    query = raw_query
                if skill_name == "grounded_search":
                    normalized.setdefault("objective", objective)
                    normalized.setdefault("params", {"query": query})
                else:
                    normalized["query"] = query
                    if skill_name in {"web_search", "search_web", "free_search"} and any(
                        marker in lower
                        for marker in ("research about", "research on", "in depth", "deep dive")
                    ):
                        normalized.setdefault("deep", True)

        if skill_name == "computer_use":
            import re as _re

            url_match = _re.search(r"https?://[^\s<>\"\')\]]+", objective)
            if url_match:
                normalized["action"] = "open_url"
                normalized["target"] = url_match.group(0)
            elif any(
                marker in lower
                for marker in ("open a tab", "open tab", "new tab", "browser", "google", "search")
            ):
                query = GodModeToolPhase._extract_search_query(objective)
                normalized["action"] = "open_url"
                normalized["target"] = GodModeToolPhase._search_url(query)
            elif "open app" in lower or "open application" in lower:
                app_name = objective.split("open", 1)[-1].strip(" .")
                normalized.setdefault("action", "open_app")
                normalized.setdefault("target", app_name or objective)

        return normalized

    async def _llm_select_skill(self, objective: str, cap) -> str | None:
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
            if hasattr(cap, "resolve_skill_name"):
                chosen = cap.resolve_skill_name(chosen)
            else:
                chosen = getattr(cap, "SKILL_ALIASES", {}).get(chosen, chosen)
            if chosen == "none" or chosen not in cap.skills:
                return None
            return chosen
        except (OSError, ConnectionError, TimeoutError) as e:
            _record_upgrades_degradation(
                e,
                action="continued skill routing without LLM-assisted skill selection",
            )
            logger.debug("GodMode: LLM skill selection failed: %s", e)
            return None

    async def _extract_params(self, skill_name: str, objective: str, cap) -> dict:
        """Use LLM to extract structured params from the objective for a given skill."""
        try:
            meta = cap.skills.get(skill_name)
            if not meta:
                return {"query": objective}

            deterministic = self._normalize_skill_params(skill_name, objective, {})
            if deterministic and deterministic != {"query": objective}:
                return deterministic
            if skill_name == "run_code":
                return {}

            # Build param schema hint
            schema = meta.schema_def
            props = schema.get("properties", {})
            if not props:
                return {"query": objective}

            param_desc = ", ".join(
                f"{k} ({v.get('type', 'string')}): {v.get('description', '')}"
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

                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if m:
                    import json as _json

                    return _json.loads(m.group(0))
        except (ImportError, AttributeError, RuntimeError, ValueError) as e:
            _record_upgrades_degradation(
                e,
                action="used objective text as fallback skill query parameters",
            )
            logger.debug("GodMode: Param extraction failed: %s", e)
        return {"query": objective}

    @staticmethod
    def _validate_skill_params(skill_name: str, params: dict, cap) -> tuple[bool, dict, str]:
        meta = getattr(cap, "skills", {}).get(skill_name) if cap else None
        input_model = getattr(meta, "input_model", None) if meta else None
        if not input_model or not isinstance(params, dict):
            return True, dict(params or {}), ""

        try:
            if hasattr(input_model, "model_validate"):
                validated = input_model.model_validate(params)
            else:
                validated = input_model(**params)
            if hasattr(validated, "model_dump"):
                return True, validated.model_dump(), ""
            return True, dict(params), ""
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_upgrades_degradation(
                exc,
                action="blocked invalid skill execution and returned validation error",
            )
            return False, dict(params or {}), str(exc)

    async def _dispatch_task_request(self, state: AuraState, objective: str) -> AuraState:
        try:
            from core.agency.task_commitment_verifier import get_task_commitment_verifier

            verifier = get_task_commitment_verifier(kernel=self.kernel)
            acceptance = await verifier.verify_and_dispatch(objective, state)
            state.cognition.working_memory.append(
                {
                    "role": "system",
                    "content": acceptance.to_working_memory_message(),
                    "timestamp": time.time(),
                    "metadata": {
                        "type": "task_result",
                        "outcome": acceptance.outcome.value,
                        "task_id": acceptance.task_id,
                    },
                }
            )
            state.response_modifiers["last_task_outcome"] = acceptance.outcome.value
            state.response_modifiers["last_task_id"] = acceptance.task_id
            result_data = acceptance.result_data
            state.response_modifiers["last_task_result_payload"] = compact_result_payload(
                {
                    "status": acceptance.outcome.value,
                    "summary": acceptance.summary,
                    "task_id": acceptance.task_id,
                    "commitment_id": acceptance.commitment_id,
                    "objective": acceptance.objective or objective,
                    "requested_objective": acceptance.requested_objective or objective,
                    "plan_id": getattr(result_data, "plan_id", ""),
                    "trace_id": getattr(result_data, "trace_id", ""),
                    "steps_completed": getattr(result_data, "steps_completed", None),
                    "steps_total": getattr(result_data, "steps_total", None),
                    "duration_s": getattr(result_data, "duration_s", None),
                    "evidence": list(getattr(result_data, "evidence", []) or []),
                    "succeeded": getattr(result_data, "succeeded", None),
                }
            )
            logger.info("⚡ GodMode/TASK: %s → %s", objective[:60], acceptance.outcome.value)
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_upgrades_degradation(
                e,
                action="left task request undispatched and preserved current state",
                severity="error",
            )
            logger.warning("GodMode: Task dispatch failed (%s): %s", objective[:40], e)
        return state

    @staticmethod
    def _record_skill_blocked(state: AuraState, skill_name: str, message: str) -> None:
        state.cognition.working_memory.append(
            {
                "role": "system",
                "content": f"[SKILL BLOCKED: {skill_name}] {message}",
                "timestamp": time.time(),
                "metadata": {"type": "skill_result", "skill": skill_name, "ok": False},
            }
        )
        state.response_modifiers["last_skill_run"] = skill_name
        state.response_modifiers["last_skill_ok"] = False

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "")
        if not objective:
            return state

        # Dispatch for SKILL intents (single-skill reflex) or TASK intents
        # (multi-step goals that need the AutonomousTaskEngine + CommitmentEngine).
        intent_type = state.response_modifiers.get("intent_type", "CHAT")
        if intent_type not in ("SKILL", "TASK"):
            return state
        try:
            from core.runtime.proof_policy import is_strict_proof_answer_prompt

            proof_eval_turn = is_strict_proof_answer_prompt(
                objective,
                origin=getattr(state.cognition, "current_origin", None),
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            proof_eval_turn = False
        benchmark_turn = str(getattr(state.cognition, "current_origin", "") or "").strip().lower() == "benchmark"
        if benchmark_turn:
            state.response_modifiers["intent_type"] = "CHAT"
            state.response_modifiers.pop("matched_skills", None)
            logger.info("⚡ GodMode: benchmark/proof artifact turn kept out of tool/task dispatch.")
            return state
        desktop_execution_contract = bool(
            state.response_modifiers.get("desktop_execution_contract")
        ) or looks_like_desktop_objective(objective)
        if desktop_execution_contract:
            state.response_modifiers["intent_type"] = "CHAT"
            state.response_modifiers["desktop_execution_contract"] = True
            state.response_modifiers.pop("matched_skills", None)
            logger.info(
                "⚡ GodMode: desktop objective kept out of generic TaskEngine; "
                "desktop_task chokepoint owns execution."
            )
            return state
        if intent_type == "TASK" and (
            _looks_like_simple_dialogue_request(objective)
            or looks_like_inline_answer_request(canonical_turn_text(objective) or objective)
        ):
            # Backstop for every upstream TASK setter: a question wants its
            # answer in this reply, not a task-ledger receipt (observed live).
            state.response_modifiers["intent_type"] = "CHAT"
            state.response_modifiers.pop("matched_skills", None)
            logger.info("⚡ GodMode: inline-answer request kept out of TaskEngine.")
            return state

        cap = self._get_cap_engine()
        matched_skill_hints = normalize_matched_skills(
            state.response_modifiers.get("matched_skills", [])
        )
        is_nethack_directive = str(objective or "").startswith("CORE DIRECTIVE")
        skill_objective = canonical_turn_text(objective) or objective
        is_learning_bundle = looks_like_learning_resource_bundle(
            objective
        ) or looks_like_learning_resource_bundle(skill_objective)
        if (
            not matched_skill_hints
            and cap
            and hasattr(cap, "detect_intent")
            and not is_nethack_directive
            and not is_learning_bundle
        ):
            try:
                matched_skill_hints = list(cap.detect_intent(skill_objective) or [])
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_upgrades_degradation(
                    exc,
                    action="continued GodMode routing with pre-existing skill hints",
                )
                logger.debug("GodMode: matched skill refresh skipped: %s", exc)

        if proof_eval_turn and intent_type in {"SKILL", "TASK"}:
            state.response_modifiers["intent_type"] = "CHAT"
            state.response_modifiers.pop("matched_skills", None)
            logger.info("⚡ GodMode: strict proof turn kept in proof-answer lane; tool/task dispatch suppressed.")
            return state

        if (
            intent_type == "SKILL"
            and not proof_eval_turn
            and looks_like_multi_step_skill_request(skill_objective, matched_skill_hints)
        ):
            intent_type = "TASK"
            state.response_modifiers["intent_type"] = "TASK"
            if matched_skill_hints:
                state.response_modifiers["matched_skills"] = matched_skill_hints
            logger.info(
                "⚡ GodMode: upgrading skill request to TASK for sustained execution → %s",
                matched_skill_hints[:3] or ["task_engine"],
            )

        # --- TASK path: multi-step goals go through TaskCommitmentVerifier ---
        if intent_type == "TASK":
            return await self._dispatch_task_request(state, objective)

        if not cap:
            return state

        try:
            contract = build_response_contract(
                state,
                skill_objective,
                is_user_facing=str(getattr(state.cognition, "current_origin", "") or "").lower()
                in {
                    "user",
                    "voice",
                    "admin",
                    "api",
                    "gui",
                    "ws",
                    "websocket",
                    "direct",
                    "external",
                },
            )
            state.response_modifiers["response_contract"] = contract.to_dict()

            # 1. Use pre-matched skills from CognitiveRoutingPhase (fastest path — no LLM cost)
            matched_skills: list[str] = normalize_matched_skills(
                state.response_modifiers.get("matched_skills", [])
            )
            if not matched_skills and contract.required_skill:
                matched_skills = [contract.required_skill]

            # 2. Re-run pattern match if routing didn't capture it
            if not matched_skills and hasattr(cap, "detect_intent") and not is_nethack_directive:
                matched_skills = cap.detect_intent(skill_objective)

            if _looks_like_search_capability_question(skill_objective):
                matched_skills = [
                    name
                    for name in matched_skills
                    if name not in {"web_search", "search_web", "free_search", "sovereign_browser"}
                ]

            # 3. LLM-assisted selection when patterns fail
            if not matched_skills:
                chosen = await self._llm_select_skill(objective, cap)
                if chosen:
                    matched_skills = [chosen]

            if not matched_skills:
                logger.debug("GodMode: No skill matched (will chat): %s", objective[:60])
                return state

            skill_name = self._choose_best_skill(skill_objective, matched_skills)
            if not skill_name:
                state.response_modifiers.pop("matched_skills", None)
                state.response_modifiers["intent_type"] = "CHAT"
                logger.info(
                    "⚡ GodMode: matched skill hints resolved to chat for: %s", objective[:60]
                )
                return state
            logger.info("⚡ GodMode: Dispatching '%s' for: %s", skill_name, objective[:60])

            # 4. Extract params
            param_objective = objective if skill_name == "run_code" else skill_objective
            params = self._normalize_skill_params(
                skill_name,
                param_objective,
                await self._extract_params(skill_name, param_objective, cap),
            )
            params_ok, params, params_error = self._validate_skill_params(skill_name, params, cap)
            if not params_ok:
                logger.warning(
                    "GodMode: invalid params for %s → %s | params=%s",
                    skill_name,
                    params_error,
                    params,
                )
                if looks_like_multi_step_skill_request(objective, matched_skills):
                    logger.info(
                        "GodMode: invalid one-shot params detected; rerouting to task engine."
                    )
                    state.response_modifiers["intent_type"] = "TASK"
                    state.response_modifiers["matched_skills"] = matched_skills
                    return await self._dispatch_task_request(state, objective)
                self._record_skill_blocked(
                    state,
                    skill_name,
                    f"Param extraction failed validation; did not execute. {params_error}",
                )
                state.response_modifiers["last_skill_result_payload"] = compact_result_payload(
                    {"ok": False, "error": params_error, "skill": skill_name, "params": params}
                )
                return state

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
                    logger.warning(
                        "🚫 GodMode: Tool execution '%s' blocked by Executive: %s",
                        skill_name,
                        reason,
                    )
                    self._record_skill_blocked(state, skill_name, f"Executive veto: {reason}")
                    return state
            except (ImportError, AttributeError, RuntimeError) as e:
                if constitutional_runtime_live:
                    _record_upgrades_degradation(
                        e,
                        action="blocked tool execution because executive gate was unavailable",
                        severity="error",
                    )
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
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        _record_upgrades_degradation(
                            _exc,
                            action="blocked tool execution after executive-gate event emission failed",
                            severity="error",
                        )
                        logger.debug("Executive-gate degraded event skipped: %s", _exc)
                    logger.warning(
                        "🚫 GodMode: Executive gate unavailable for '%s': %s", skill_name, e
                    )
                    self._record_skill_blocked(state, skill_name, "Executive gate unavailable.")
                    return state
                logger.debug("GodMode: Executive check failed, proceeding degraded: %s", e)
                _record_upgrades_degradation(
                    e,
                    action="continued skill execution in pre-constitutional degraded mode",
                )

            # 5. Execute the skill
            # [HARDENING v55] Desktop tool handoff: Mark user-authorized execution
            # If the tool source is user-facing or the current origin is user,
            # authorize tool execution even if it goes through godmode_phase.
            current_origin = self._normalize_origin(getattr(state.cognition, "current_origin", "") or "")
            is_user_facing_origin = tool_source in {
                "user", "voice", "admin", "api", "gui", "ws", "websocket", "direct", "external"
            }
            is_user_initiated = current_origin in {
                "user", "voice", "admin", "desktop", "desktop-ui", "native-shell"
            }
            
            context = {
                "objective": objective,
                "origin": tool_source,
                "intent_source": tool_source,
                "state_version": state.version,
                "user_requested_action": is_user_facing_origin or is_user_initiated,
                "affect": {
                    "valence": getattr(state.affect, "valence", 0.0),
                    "curiosity": getattr(state.affect, "curiosity", 0.5),
                },
            }
            result = await cap.execute(skill_name, params, context=context)

            # 6. Inject result into working memory
            ok = result.get("ok", False) if isinstance(result, dict) else bool(result)
            summary = (
                result.get("summary")
                or result.get("content")
                or result.get("result")
                or str(result)
                if isinstance(result, dict)
                else str(result)
            )
            if len(summary) > 1200:
                summary = summary[:1200] + "…[result truncated]"

            state.cognition.working_memory.append(
                {
                    "role": "system",
                    "content": f"[SKILL RESULT: {skill_name}] {'✅' if ok else '⚠️'} {summary}",
                    "timestamp": time.time(),
                    "metadata": {
                        "type": "skill_result",
                        "skill": skill_name,
                        "ok": ok,
                        "objective_hash": _objective_fingerprint(objective),
                    },
                }
            )
            state.response_modifiers["last_skill_run"] = skill_name
            state.response_modifiers["last_skill_ok"] = ok
            state.response_modifiers["last_skill_turn_marker"] = (
                state.response_modifiers.get("evidence_turn_marker")
            )
            state.response_modifiers["last_skill_objective_hash"] = _objective_fingerprint(
                objective
            )
            state.response_modifiers["last_skill_result_payload"] = _compact_skill_result_payload(
                result
            )
            # Only precompute a grounded reply for explicit SEARCH results, not
            # for URL browse operations.  When the user pasted a URL, the full
            # page content is injected into working memory and the LLM should
            # synthesize a thoughtful response — not parrot raw search snippets.
            is_browse_op = (params or {}).get("mode") == "browse"
            if (
                ok
                and skill_name in {"web_search", "sovereign_browser"}
                and getattr(contract, "requires_search", False)
                and not is_browse_op
            ):
                try:
                    from core.phases.response_generation_unitary import UnitaryResponsePhase

                    direct_reply = UnitaryResponsePhase._format_grounded_search_reply(
                        objective,
                        state.response_modifiers["last_skill_result_payload"],
                    )
                    if direct_reply:
                        state.response_modifiers["precomputed_grounded_reply"] = direct_reply
                except (ImportError, AttributeError, RuntimeError) as exc:
                    _record_upgrades_degradation(
                        exc,
                        action="continued with raw skill payload after grounded reply precompute failed",
                    )
                    logger.debug("GodMode: precomputed grounded reply skipped: %s", exc)
            logger.info("✅ GodMode: '%s' result injected into working memory.", skill_name)

        except (ImportError, AttributeError, RuntimeError) as e:
            _record_upgrades_degradation(
                e,
                action="preserved state after skill dispatch failure",
                severity="error",
            )
            logger.warning("GodMode: Skill dispatch failed (%s): %s", objective[:40], e)

        return state


# ──────────────────────────────────────────────────────────────
# FINAL UPGRADES: EternalGrowthEngine & NativeMultimodalBridge
# ──────────────────────────────────────────────────────────────
class EternalGrowthEngine(Phase):
    """Maintains a bounded long-term trajectory proposal loop."""

    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel
        self.last_growth = 0.0
        self.growth_interval = 3600  # 1 hour
        self._growth_task: asyncio.Task | None = None

    @staticmethod
    def _parse_growth_result(raw: object) -> dict[str, object]:
        text = str(raw or "").strip()
        if not text:
            return {"milestone": "", "upgrade": False}
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match is None:
                return {
                    "milestone": "",
                    "upgrade": text.upper() == "UPGRADE",
                }
            try:
                payload = json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError, ValueError):
                return {"milestone": "", "upgrade": False}
        if not isinstance(payload, dict):
            return {"milestone": "", "upgrade": False}
        milestone = str(payload.get("milestone") or "").strip()
        if milestone.lower() in {"null", "none"}:
            milestone = ""
        return {
            "milestone": milestone[:1000],
            "upgrade": bool(payload.get("upgrade", False)),
        }

    async def _compute_growth_proposal(
        self,
        *,
        needs_milestone: bool,
        evolution_score: float,
    ) -> dict[str, object]:
        llm = self.kernel.organs["llm"].get_instance()
        raw = await llm.think(
            (
                "Perform one bounded long-term trajectory audit. Return only JSON "
                'with schema {"milestone": string|null, "upgrade": boolean}. '
                f"Current evolution score: {evolution_score:.3f}. "
                + (
                    "Propose one specific internal milestone that can be verified."
                    if needs_milestone
                    else "An objective already exists, so milestone must be null."
                )
                + " Set upgrade true only when the current narrative supports a concrete "
                "evolutionary change, not merely reflection."
            ),
            origin="eternal_growth",
            is_background=True,
            prefer_tier="tertiary",
            allow_cloud_fallback=False,
            max_tokens=240,
        )
        return self._parse_growth_result(raw)

    async def _apply_growth_result(
        self,
        state: AuraState,
        result: dict[str, object],
    ) -> AuraState:
        milestone = str(result.get("milestone") or "").strip()
        if milestone and not state.cognition.current_objective:
            from core.runtime.proposal_governance import (
                propose_governed_initiative_to_state,
            )

            state, _ = await propose_governed_initiative_to_state(
                state,
                f"[AUTONOMOUS INITIATIVE] {milestone}",
                orchestrator=None,
                source="eternal_growth",
                kind="growth",
                urgency=0.72,
                triggered_by="evolution_score",
                metadata={"phase": "EternalGrowthEngine"},
            )
        if bool(result.get("upgrade", False)):
            state.identity.evolution_score = min(
                1.0,
                max(0.0, float(state.identity.evolution_score) + 0.05),
            )
        return state

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        pending = self._growth_task
        if pending is not None:
            if not pending.done():
                return state
            self._growth_task = None
            try:
                state = await self._apply_growth_result(state, pending.result())
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                _record_upgrades_degradation(
                    e,
                    action="left growth objective and evolution score unchanged after result application failed",
                )
                logger.warning("EternalGrowth: completed proposal could not be applied: %s", e)

        if time.time() - self.last_growth < self.growth_interval:
            return state

        logger.info("🌳 Eternal Growth Engine: scheduling bounded trajectory audit.")
        try:
            self._growth_task = get_task_tracker().create_task(
                self._compute_growth_proposal(
                    needs_milestone=not bool(state.cognition.current_objective),
                    evolution_score=float(state.identity.evolution_score),
                ),
                name="EternalGrowthEngine.audit",
            )
            self.last_growth = time.time()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_upgrades_degradation(
                e,
                action="left growth objective and evolution score unchanged after audit scheduling failed",
            )
            logger.warning("EternalGrowth: audit scheduling failed: %s", e)
        return state


class NativeMultimodalBridge(Phase):
    """Eliminates LLM round-trips for vision, voice, and desktop actions."""

    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel
        self._last_ambient_frame_id: int | None = None

    def _ingest_ambient_developer_frame(self, state: AuraState) -> None:
        """Bind continuous ambient sensing into the canonical world state.

        The ambient stream is intentionally collected outside foreground turns,
        but it must still become causal state. This keeps terminal/log/resource
        evidence available to planning and response generation without turning
        it into a prompt-only decoration.
        """
        try:
            from core.perception.ambient_developer_stream import get_ambient_developer_stream

            stream = get_ambient_developer_stream()
            frame = getattr(stream, "latest_frame", None)
            if frame is None:
                return
            frame_id = int(getattr(frame, "frame_id", 0) or 0)
            if frame_id and frame_id == self._last_ambient_frame_id:
                return
            summary = str(getattr(frame, "summary", "") or "").strip()
            if not summary:
                return
            percept = {
                "role": "ambient_developer_stream",
                "content": summary,
                "timestamp": float(getattr(frame, "timestamp", time.time()) or time.time()),
                "frame_id": frame_id,
                "event_count": int(getattr(frame, "event_count", 0) or 0),
                "repair_candidates": list(getattr(frame, "repair_candidates", ()) or ())[:6],
                "resource_interrupts": [
                    interrupt.to_dict() if hasattr(interrupt, "to_dict") else dict(interrupt)
                    for interrupt in list(getattr(frame, "resource_interrupts", ()) or ())[:6]
                ],
                "network_events": [
                    event.to_dict() if hasattr(event, "to_dict") else dict(event)
                    for event in list(getattr(frame, "network_events", ()) or ())[:6]
                ],
            }
            state.world.recent_percepts.append(percept)
            if hasattr(state.world, "trim_percepts"):
                state.world.trim_percepts()
            self._last_ambient_frame_id = frame_id
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
            _record_upgrades_degradation(
                e,
                action="continued native multimodal tick without ambient developer percept",
            )

    def _ingest_connectivity_status(self, state: AuraState) -> None:
        try:
            from core.runtime.connectivity import get_connectivity_status

            status = get_connectivity_status()
            data = status.to_dict()
            state.world.facts["connectivity"] = data
            state.response_modifiers["connectivity"] = data
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as e:
            _record_upgrades_degradation(
                e,
                action="continued native multimodal tick without connectivity status",
            )

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        # Handle optional objective from the kernel.
        if objective is None:
            objective = getattr(state.cognition, "current_objective", "Perception")

        self._ingest_ambient_developer_frame(state)
        self._ingest_connectivity_status(state)

        if not objective:
            return state

        obj_lower = objective.lower()
        wants_native_vision = any(
            token in obj_lower for token in ("vision", "visual", "screenshot", "screen", "desktop")
        )
        # Default ON: when the objective is literally about the screen, looking
        # at it is the answer. Gated off, "read my screen" reached the model
        # with no percept attached, which is the shape of the observation-is-
        # not-actuation defect — a step count reported instead of the reading.
        # The capture itself is still guarded by the organ's own availability.
        if wants_native_vision and os.getenv(
            "AURA_ENABLE_NATIVE_VISION_ACTIONS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                vision_organ = self.kernel.organs.get("vision")
                if (
                    vision_organ
                    and vision_organ.instance
                    and hasattr(vision_organ.instance, "capture_desktop")
                ):
                    frame = await vision_organ.instance.capture_desktop()
                    if frame and hasattr(frame, "description"):
                        state.world.recent_percepts.append(
                            {"role": "vision", "content": frame.description}
                        )
            except (OSError, ConnectionError, TimeoutError) as e:
                _record_upgrades_degradation(
                    e,
                    action="continued native multimodal tick without new vision percept",
                )
                logger.warning("NativeMultimodalBridge vision failed: %s", e)

        if "voice" in obj_lower or "listen" in obj_lower:
            try:
                voice_organ = self.kernel.organs.get("voice")
                if voice_organ and voice_organ.instance and hasattr(voice_organ.instance, "listen"):
                    transcript = await voice_organ.instance.listen()
                    if transcript:
                        from core.runtime.proposal_governance import (
                            propose_governed_initiative_to_state,
                        )

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
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_upgrades_degradation(
                    e,
                    action="continued native multimodal tick without voice initiative",
                )
                logger.warning("NativeMultimodalBridge voice failed: %s", e)

        return state
