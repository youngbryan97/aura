"""core/planning/task_decomposer.py — Natural Language → TaskGraph DAG
======================================================================
Converts vague natural-language objectives into structured TaskGraph DAGs
with dependency ordering, verification predicates, rollback actions,
and fallback alternatives.

Uses the LLM router to decompose, then validates against the AppRegistry
to ensure all required apps/capabilities exist. Falls back to pattern-
matching heuristics if LLM is unavailable.

The decomposer never hardcodes specific demo flows. It produces general
task graphs that the MissionState executor runs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.conversation.word_markers import names_any
from core.planning.task_graph import TaskGraph, TaskNode
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.TaskDecomposer")

# ---------------------------------------------------------------------------
# Task decomposition prompt template
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """You are a task planner for an AI desktop assistant on macOS.

Given a natural-language objective, decompose it into an ordered list of concrete steps.
Each step is a single atomic action using one of these primitives:

AVAILABLE PRIMITIVES:
- launch_app: Open an application. Params: {name: str}
- focus_app: Bring app to front. Params: {name: str}
- close_app: Quit an app. Params: {name: str}
- create_folder: Create a folder. Params: {path: str}
- create_text_file: Write text to file. Params: {path: str, content: str}
- create_pdf: Render text to PDF. Params: {path: str, title: str, body: str}
- move_file: Move a file. Params: {source: str, destination: str}
- type_text: Type text into active app. Params: {text: str}
- hotkey: Press keyboard shortcut. Params: {keys: [str]}
- menu_select: Click a menu item. Params: {app: str, path: [str]}
- click_at: Click screen coordinates. Params: {x: int, y: int}
- open_url: Open URL in browser. Params: {url: str}
- search_web: Search and open results. Params: {query: str, count: int}
- search_images: Search for images. Params: {query: str}
- download_image: Download an image. Params: {url: str, save_dir: str}
- set_wallpaper: Set desktop wallpaper. Params: {image_path: str}
- get_wallpaper: Get current wallpaper path. Params: {}
- take_screenshot: Capture screen. Params: {save_path: str}
- get_screen_text: OCR the screen. Params: {}
- run_command: Run a shell command. Params: {command: str}
- extract_article: Extract article text from URL. Params: {url: str}
- summarize_sources: Summarize multiple sources. Params: {sources: [str]}
- set_clipboard: Set clipboard content. Params: {text: str}
- paste: Paste clipboard. Params: {}
- wait: Wait seconds. Params: {seconds: float}
- notify_user: Show notification. Params: {message: str}

AVAILABLE APPS ON THIS MACHINE:
{available_apps}

CURRENT STATE:
{current_state}

COGNITIVE SITUATION:
{cognitive_situation}

OBJECTIVE: {objective}

Respond with a JSON array of steps:
```json
[
  {{
    "id": "t1",
    "action": "launch_app",
    "params": {{"name": "Notes"}},
    "depends_on": [],
    "verify": "app_is_frontmost",
    "verify_args": {{"name": "Notes"}},
    "rollback": "close_app",
    "rollback_params": {{"name": "Notes"}},
    "fallback": "",
    "fallback_params": {{}},
    "risk": "low",
    "description": "Open Notes app",
    "critical": true
  }}
]
```

Rules:
1. Each step must use exactly ONE primitive from the list above.
2. Steps must have explicit dependencies (depends_on) for ordering.
3. Each step must have a verification predicate.
4. Non-critical steps (like opening a tab for visual reference) can have critical: false.
5. Include rollback actions where possible.
6. Include fallback alternatives (e.g., TextEdit if Notes fails).
7. Use the most reliable method first (direct API > AppleScript > UI clicking).
8. For long text, use create_text_file or clipboard+paste, NOT keystroke typing.
9. Do NOT hardcode specific content — use placeholders like {{generated_content}}.
10. Keep the plan to 20 steps or fewer.
"""


class TaskDecomposer:
    """Converts natural-language objectives into TaskGraph DAGs.

    Usage:
        decomposer = get_task_decomposer()
        graph = await decomposer.decompose("Find an image of a mountain and set it as wallpaper")
    """

    def __init__(self) -> None:
        self._decomposition_count = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("task_decomposer", self, required=False)
        self._started = True
        logger.info("TaskDecomposer ONLINE")

    async def decompose(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
    ) -> TaskGraph:
        """Decompose a natural-language objective into a TaskGraph.

        1. Gather available apps and current state
        2. Ask LLM to decompose into steps
        3. Parse and validate the response
        4. Build TaskGraph with dependencies
        5. Fall back to heuristic decomposition if LLM fails
        """
        context = context or {}
        self._decomposition_count += 1
        mission_id = f"m_{int(time.time())}_{hashlib.sha256(objective.encode()).hexdigest()[:8]}"

        # Gather environment info
        available_apps = await self._get_available_apps()
        current_state = self._get_current_state()

        context = dict(context)
        cognitive_situation = self._cognitive_situation_for_objective(objective, context)
        if cognitive_situation and "cognitive_situation_frame" not in context:
            context["cognitive_situation_frame"] = cognitive_situation

        # Lessons from prior episodes: strategies that have failed for this CLASS of goal
        # are surfaced as structured guidance so the decomposition can steer away from them.
        # A learned statistic (see PlanFailureMemory), not a heuristic — best-effort.
        try:
            from core.planning.plan_failure_memory import get_plan_failure_memory

            _guidance = get_plan_failure_memory().guidance(objective)
            if _guidance.has_lessons:
                context["plan_failure_guidance"] = _guidance.to_dict()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass

        # Try LLM decomposition first
        steps = await self._llm_decompose(
            objective,
            available_apps,
            current_state,
            context,
        )

        # Fall back to heuristic if LLM failed
        if not steps:
            logger.info("LLM decomposition failed, using heuristic fallback")
            steps = self._heuristic_decompose(objective, context)

        # Build the TaskGraph
        graph = TaskGraph(
            mission_id=mission_id,
            objective=objective,
            metadata={
                "planning_context": {
                    "cognitive_situation_frame": cognitive_situation,
                    "current_state": current_state,
                    "available_apps_digest": hashlib.sha256(
                        available_apps.encode("utf-8", errors="ignore")
                    ).hexdigest()[:16],
                }
            },
        )
        for step in steps:
            node = TaskNode(
                task_id=step.get("id", f"t{len(graph.nodes) + 1}"),
                action=step.get("action", ""),
                params=step.get("params", {}),
                preconditions=step.get("depends_on", []),
                verification=step.get("verify", "true"),
                verification_args=step.get("verify_args", {}),
                rollback_action=step.get("rollback", ""),
                rollback_params=step.get("rollback_params", {}),
                fallback_action=step.get("fallback", ""),
                fallback_params=step.get("fallback_params", {}),
                risk_level=step.get("risk", "low"),
                description=step.get("description", step.get("action", "")),
                critical=step.get("critical", True),
                timeout_s=float(step.get("timeout", 30.0)),
            )
            graph.add_node(node)

        # Validate
        warnings = graph.validate()
        for w in warnings:
            logger.warning("TaskGraph validation: %s", w)

        logger.info(
            "Decomposed '%s' into %d steps (mission=%s)",
            objective[:60], graph.total_steps, mission_id,
        )
        return graph

    def _cognitive_situation_for_objective(
        self,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the same semantic/body grounding frame used by live cognition."""

        supplied = context.get("cognitive_situation_frame")
        if isinstance(supplied, dict):
            return supplied
        try:
            from core.brain.cognitive_situation import get_cognitive_situation_engine

            frame = get_cognitive_situation_engine().frame(
                objective,
                context=context,
                origin=str(context.get("origin") or "task_decomposer"),
            )
            return frame.to_dict()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "task_decomposer.cognitive_situation",
                exc,
                action="planned without cognitive situation frame after frame construction failed",
            )
            return {}

    async def _get_available_apps(self) -> str:
        """Get available apps summary from AppRegistry."""
        try:
            registry = ServiceContainer.get("app_registry", default=None)
            if registry:
                report = registry.get_capability_report()
                categories = report.get("categories", {})
                lines = []
                for cat, apps in categories.items():
                    lines.append(f"  {cat}: {', '.join(apps)}")
                return "\n".join(lines) if lines else "Unknown apps"
            return "App registry not available"
        except (ImportError, AttributeError, RuntimeError):
            return "App registry not available"

    def _get_current_state(self) -> str:
        """Get current environment state summary."""
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws:
                parts = []
                if hasattr(ws, "active_foreground_app") and ws.active_foreground_app:
                    parts.append(f"Active app: {ws.active_foreground_app}")
                if hasattr(ws, "active_window_title") and ws.active_window_title:
                    parts.append(f"Window: {ws.active_window_title}")
                parts.append(f"Time: {ws.time_of_day}")
                return "; ".join(parts) if parts else "No state available"
            return "WorldState not available"
        except (ImportError, AttributeError, RuntimeError):
            return "WorldState not available"

    async def _llm_decompose(
        self,
        objective: str,
        available_apps: str,
        current_state: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Use the LLM router to decompose the objective."""
        try:
            # Get LLM router
            router = ServiceContainer.get("llm_router", default=None)
            if router is None:
                return []

            prompt = self._render_decompose_prompt(
                objective=objective,
                available_apps=available_apps,
                current_state=current_state,
                cognitive_situation=self._render_cognitive_situation_for_planning(context),
            )

            # Route to the fastest available model
            response = await router.route(
                prompt=prompt,
                system="You are a precise task planner. Respond ONLY with valid JSON.",
                temperature=0.3,
                max_tokens=2000,
                route_hint="planning",
            )

            if not response or not hasattr(response, "text"):
                return []

            text = response.text if hasattr(response, "text") else str(response)
            return self._parse_llm_response(text)

        except (ImportError, AttributeError, RuntimeError, TypeError) as e:
            record_degradation("task_decomposer.llm", e)
            logger.debug("LLM decomposition failed: %s", e)
            return []

    @staticmethod
    def _render_decompose_prompt(
        *,
        objective: str,
        available_apps: str,
        current_state: str,
        cognitive_situation: str,
    ) -> str:
        """Render the planning prompt without interpreting JSON examples as format fields."""

        return (
            DECOMPOSE_PROMPT.replace("{available_apps}", available_apps)
            .replace("{current_state}", current_state)
            .replace("{cognitive_situation}", cognitive_situation)
            .replace("{objective}", objective)
        )

    @staticmethod
    def _render_cognitive_situation_for_planning(context: dict[str, Any]) -> str:
        frame = context.get("cognitive_situation_frame")
        if not isinstance(frame, dict) or not frame:
            return "No cognitive situation frame available; use direct objective and current state."

        def _num(name: str) -> float:
            try:
                return float(frame.get(name, 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                return 0.0

        affordances = frame.get("embodied_affordances")
        interpretations = frame.get("semantic_interpretations")
        bridges = frame.get("analogy_bridges")
        causal_effects = frame.get("causal_effects")
        lines = [
            (
                f"semantic_flexibility={_num('semantic_flexibility'):.2f}; "
                f"analogical_leap_pressure={_num('analogical_leap_pressure'):.2f}; "
                f"sensorimotor_grounding={_num('sensorimotor_grounding'):.2f}; "
                f"verification_pressure={_num('verification_pressure'):.2f}; "
                f"social_uncertainty={_num('social_uncertainty'):.2f}; "
                f"social_repair_pressure={_num('social_repair_pressure'):.2f}"
            ),
            "Planning effects: preserve multiple valid interpretations, use analogous known workflows when useful, and bind every screen/tool step to observable verification.",
        ]
        if isinstance(interpretations, list) and interpretations:
            labels = [
                str(item.get("label") or item.get("focus") or "")[:80]
                for item in interpretations[:3]
                if isinstance(item, dict)
            ]
            labels = [item for item in labels if item]
            if labels:
                lines.append("Candidate interpretations: " + "; ".join(labels))
        if isinstance(bridges, list) and bridges:
            labels = [
                str(item.get("bridge") or item.get("source") or "")[:80]
                for item in bridges[:3]
                if isinstance(item, dict)
            ]
            labels = [item for item in labels if item]
            if labels:
                lines.append("Analogical bridges: " + "; ".join(labels))
        if isinstance(affordances, list) and affordances:
            lines.append("Embodied affordances: " + ", ".join(map(str, affordances[:6])))
        if isinstance(causal_effects, dict):
            constraints = causal_effects.get("perception_planning_constraints")
            if isinstance(constraints, list) and constraints:
                lines.append(
                    "Perception constraints: "
                    + ", ".join(str(item)[:120] for item in constraints[:8])
                )
            repairs = causal_effects.get("perception_repair_requirements")
            if isinstance(repairs, list) and repairs:
                lines.append(
                    "Evidence repair before irreversible action: "
                    + ", ".join(str(item)[:120] for item in repairs[:8])
                )
            social_constraints = causal_effects.get("social_planning_constraints")
            if isinstance(social_constraints, list) and social_constraints:
                lines.append(
                    "Social/consent constraints: "
                    + ", ".join(str(item)[:120] for item in social_constraints[:8])
                )
        return "\n".join(lines)

    def _parse_llm_response(self, text: str) -> list[dict[str, Any]]:
        """Parse JSON steps from LLM response text."""
        # Try to extract JSON from various formats
        candidates = []

        # Try fenced code blocks first
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
            candidates.append(match.group(1).strip())

        # Try bare JSON
        for bracket_open, bracket_close in [("[", "]"), ("{", "}")]:
            start = text.find(bracket_open)
            end = text.rfind(bracket_close)
            if start >= 0 and end > start:
                candidates.append(text[start:end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [s for s in parsed if isinstance(s, dict) and "action" in s][:20]
                if isinstance(parsed, dict) and "steps" in parsed:
                    steps = parsed["steps"]
                    if isinstance(steps, list):
                        return [s for s in steps if isinstance(s, dict) and "action" in s][:20]
            except (json.JSONDecodeError, TypeError):
                continue

        return []

    # ------------------------------------------------------------------
    # Heuristic decomposition (fallback when LLM unavailable)
    # ------------------------------------------------------------------

    def _heuristic_decompose(
        self,
        objective: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Pattern-matching decomposition for common task patterns."""
        text = objective.lower()
        steps: list[dict[str, Any]] = []
        step_id = 0

        def _next_id() -> str:
            nonlocal step_id
            step_id += 1
            return f"t{step_id}"

        # Pattern: contains "wallpaper"
        if "wallpaper" in text:
            query = self._extract_subject(text, "wallpaper")
            save_dir = str(Path.home() / "Documents" / "Aura" / "images")
            img_id = _next_id()
            steps.extend([
                {
                    "id": _next_id(),
                    "action": "create_folder",
                    "params": {"path": save_dir},
                    "depends_on": [],
                    "verify": "folder_exists",
                    "verify_args": {"path": save_dir},
                    "description": "Create images folder",
                    "risk": "low",
                    "critical": False,
                },
                {
                    "id": _next_id(),
                    "action": "get_wallpaper",
                    "params": {},
                    "depends_on": [],
                    "verify": "true",
                    "description": "Save current wallpaper for rollback",
                    "risk": "low",
                    "critical": False,
                },
                {
                    "id": (img_id := _next_id()),
                    "action": "search_images",
                    "params": {"query": query or "nature landscape"},
                    "depends_on": [],
                    "verify": "true",
                    "description": f"Search for images of {query or 'nature'}",
                    "risk": "medium",
                    "critical": True,
                },
                {
                    "id": _next_id(),
                    "action": "download_image",
                    "params": {"save_dir": save_dir},
                    "depends_on": [img_id],
                    "verify": "file_is_image",
                    "verify_args": {},
                    "description": "Download selected image",
                    "risk": "medium",
                    "critical": True,
                },
                {
                    "id": _next_id(),
                    "action": "set_wallpaper",
                    "params": {},
                    "depends_on": [f"t{step_id}"],
                    "verify": "wallpaper_changed",
                    "description": "Set as desktop wallpaper",
                    "risk": "medium",
                    "critical": True,
                },
            ])

        # Pattern: contains "note" or "document" or "write"
        if names_any(text, ("note", "document", "write", "journal", "pdf")):
            folder_path = str(Path.home() / "Documents" / "Aura")
            folder_id = _next_id()
            file_id = _next_id()
            steps.extend([
                {
                    "id": folder_id,
                    "action": "create_folder",
                    "params": {"path": folder_path},
                    "depends_on": [],
                    "verify": "folder_exists",
                    "verify_args": {"path": folder_path},
                    "description": "Create documents folder",
                    "risk": "low",
                    "critical": False,
                },
                {
                    "id": file_id,
                    "action": "create_text_file",
                    "params": {"path": f"{folder_path}/note.txt", "content": "{{generated_content}}"},
                    "depends_on": [folder_id],
                    "verify": "file_exists",
                    "verify_args": {"path": f"{folder_path}/note.txt"},
                    "description": "Create text file",
                    "risk": "low",
                    "critical": True,
                },
            ])
            if "pdf" in text:
                steps.append({
                    "id": _next_id(),
                    "action": "create_pdf",
                    "params": {"path": f"{folder_path}/document.pdf", "title": "Document", "body": "{{generated_content}}"},
                    "depends_on": [file_id],
                    "verify": "file_is_pdf",
                    "verify_args": {"path": f"{folder_path}/document.pdf"},
                    "description": "Render PDF",
                    "risk": "low",
                    "critical": True,
                })

        # Pattern: contains "search" or "research" or "article"
        if names_any(text, ("search", "research", "article", "browse", "find information")):
            query = self._extract_subject(text, "search|research|find|browse")
            steps.append({
                "id": _next_id(),
                "action": "search_web",
                "params": {"query": query or objective[:100], "count": 3},
                "depends_on": [],
                "verify": "browser_has_tabs",
                "verify_args": {"min_count": 1},
                "description": f"Search for: {query or objective[:50]}",
                "risk": "low",
                "critical": True,
            })

        # Pattern: contains "open" + app name
        app_match = re.search(r"open\s+(\w[\w\s]*?)(?:\s+and|\s*$|,)", text)
        if app_match:
            app_name = app_match.group(1).strip().title()
            steps.append({
                "id": _next_id(),
                "action": "launch_app",
                "params": {"name": app_name},
                "depends_on": [],
                "verify": "app_is_frontmost",
                "verify_args": {"name": app_name},
                "rollback": "close_app",
                "rollback_params": {"name": app_name},
                "description": f"Open {app_name}",
                "risk": "low",
                "critical": True,
            })

        # If nothing matched, create a generic observation step
        if not steps:
            steps.append({
                "id": _next_id(),
                "action": "get_screen_text",
                "params": {},
                "depends_on": [],
                "verify": "true",
                "description": "Observe current desktop state",
                "risk": "low",
                "critical": False,
            })
            steps.append({
                "id": _next_id(),
                "action": "notify_user",
                "params": {"message": f"I need more detail to plan: {objective[:100]}"},
                "depends_on": [f"t{step_id}"],
                "verify": "true",
                "description": "Ask for clarification",
                "risk": "low",
                "critical": False,
            })

        return steps

    @staticmethod
    def _extract_subject(text: str, marker: str) -> str:
        """Extract the subject/topic from text near a marker word."""
        # Try "X of Y" pattern
        match = re.search(
            rf"(?:{marker})\s+(?:an?\s+)?(?:image|picture|photo)?\s*(?:of\s+)?(.+?)(?:\s+and|\s*$|,|\.)",
            text, re.IGNORECASE,
        )
        if match:
            subj = match.group(1).strip()
            # Remove common suffixes
            subj = re.sub(r"\s*(?:for me|please|now|quickly)\s*$", "", subj, flags=re.IGNORECASE)
            if subj and len(subj) > 2:
                return subj[:100]

        # Try extracting after common verbs
        match = re.search(r"(?:find|get|search|set)\s+(?:me\s+)?(?:an?\s+)?(.+?)(?:\s+and|\s*$|,)", text)
        if match:
            return match.group(1).strip()[:100]

        return ""

    def get_status(self) -> dict[str, Any]:
        return {
            "decompositions": self._decomposition_count,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: TaskDecomposer | None = None


def get_task_decomposer() -> TaskDecomposer:
    global _instance
    if _instance is None:
        _instance = TaskDecomposer()
    return _instance


__all__ = ["TaskDecomposer", "get_task_decomposer"]
