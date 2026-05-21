"""core/agency/skill_library.py — Persistent Macro and Skill Storage
===================================================================
The Skill Library allows Aura to save successful sequences of actions as
parameterized 'Skills' (macros). This means she doesn't have to reason from
first principles for every repetitive task.

Skills can be:
1. A sequence of standard tool calls.
2. An orchestration of other skills (recursive composition).

This fulfills Phase 22.9 for persistent procedure learning.
"""

from __future__ import annotations

import ast
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import config
from core.container import ServiceContainer
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.SkillLibrary")

_SKILL_LIBRARY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
)


def _record_skill_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    classification: FallbackClassification = FallbackClassification.SAFE_FALLBACK,
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=classification,
        receipt_required=True,
        extra=extra,
    )


@dataclass
class SkillStep:
    """A single step in a skill macro."""

    tool_name: str
    arguments: dict[str, Any]  # Can contain template variables like '{{target_dir}}'


@dataclass
class LearnedSkill:
    """A parameterized macro of tool calls."""

    name: str
    description: str
    parameters: list[str]  # Expected kwargs when executing
    steps: list[SkillStep]
    successes: int = 0
    failures: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def reliability(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5


class SkillLibrary:
    """Persistent storage and execution router for learned skills."""

    name = "skill_library"

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.data_path = config.paths.data_dir / "skills.json"

        self.skills: dict[str, LearnedSkill] = {}
        self._load()

    def learn_skill(
        self, name: str, description: str, parameters: list[str], steps: list[dict[str, Any]]
    ):
        """
        Save a new skill.
        `steps` is a list of dicts: [{'tool_name': '...', 'arguments': {...}}]
        """
        name = name.lower().replace(" ", "_")
        if not name:
            raise ValueError("Skill name cannot be empty.")

        # AST Validation for dynamic code blocks
        for s in steps:
            if not isinstance(s, dict):
                raise ValueError("Skill step must be a dict.")
            tool_name = str(s.get("tool_name") or "").strip()
            if not tool_name:
                raise ValueError("Skill step is missing tool_name.")
            arguments = s.get("arguments", {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError(f"Skill step '{tool_name}' arguments must be a dict.")
            if tool_name in ["run_python_script", "execute_code"]:
                code = s.get("arguments", {}).get("code", "")
                if code:
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        logger.error(
                            "Skill '%s' rejected: Syntax error in step %s", name, tool_name
                        )
                        raise ValueError(f"Invalid Python syntax in skill: {e}") from e

        skill_steps = []
        for s in steps:
            skill_steps.append(
                SkillStep(tool_name=str(s["tool_name"]).strip(), arguments=s.get("arguments") or {})
            )

        skill = LearnedSkill(
            name=name, description=description, parameters=parameters, steps=skill_steps
        )

        self.skills[name] = skill
        self._save()
        self._update_system_health()
        logger.info("🧠 Learned new skill: %s (Reliability: pending)", name)

    async def execute_skill(self, name: str, kwargs: dict[str, Any]) -> list[Any]:
        """
        Execute a macro by resolving templates and running tool calls sequentially.
        """
        name = name.lower()
        if name not in self.skills:
            raise ValueError(f"Skill '{name}' not found in library.")

        skill = self.skills[name]

        # Verify parameters
        missing = [p for p in skill.parameters if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters for skill '{name}': {missing}")

        results = []
        tool_orchestrator = ServiceContainer.get("tool_orchestrator", default=None)

        if not tool_orchestrator:
            error = RuntimeError("tool_orchestrator not found")
            _record_skill_degradation(
                "skill_library_execution",
                error,
                action="failed macro skill execution because tool orchestrator was unavailable",
                extra={"skill": name, "step": 0},
            )
            skill.failures += 1
            self._save()
            self._update_system_health()
            raise RuntimeError("Cannot execute skill: tool_orchestrator not found.")

        step_index = 0
        step_name = "preflight"
        try:
            for i, step in enumerate(skill.steps):
                step_index = i + 1
                step_name = step.tool_name
                # Resolve template arguments
                resolved_args = self._resolve_arguments(step.arguments, kwargs)

                logger.info(
                    "Executing skill '%s' step %d/%d: %s",
                    name,
                    i + 1,
                    len(skill.steps),
                    step.tool_name,
                )

                # We need to dispatch this through the tool orchestrator
                if hasattr(tool_orchestrator, "execute_tool"):
                    result = await tool_orchestrator.execute_tool(step.tool_name, resolved_args)
                    results.append(result)
                else:
                    raise RuntimeError("tool_orchestrator lacks execute_tool method")
                if isinstance(result, dict) and (
                    result.get("error")
                    or result.get("success") is False
                    or result.get("ok") is False
                ):
                    raise RuntimeError(
                        f"tool '{step.tool_name}' returned failure: {result.get('error') or result}"
                    )

            # Record success
            skill.successes += 1
            self._save()
            self._update_system_health()
            return results

        except (RuntimeError, AttributeError, TypeError, ValueError, OSError, TimeoutError) as e:
            _record_skill_degradation(
                "skill_library_execution",
                e,
                action="failed macro skill execution and persisted failure count",
                extra={"skill": name, "step": step_index},
            )
            skill.failures += 1
            self._save()
            self._update_system_health()
            raise RuntimeError(
                f"Skill '{name}' failed at step {step_index} ({step_name}): {e}"
            ) from e

    def _update_system_health(self):
        """Wire aggregated metrics into AuraState.health (Digital Metabolism)."""
        orchestrator = ServiceContainer.get("orchestrator", default=None)
        if orchestrator and hasattr(orchestrator, "state"):
            state = orchestrator.state

            # Aggregate stats
            total_skills = len(self.skills)
            avg_reliability = (
                sum(s.reliability for s in self.skills.values()) / total_skills
                if total_skills > 0
                else 1.0
            )

            # Update health capability section
            if "capabilities" not in state.health:
                state.health["capabilities"] = {}

            state.health["capabilities"]["skill_library"] = {
                "reliability": round(avg_reliability, 2),
                "count": total_skills,
                "status": "nominal" if avg_reliability > 0.7 else "degraded",
            }

    def _resolve_arguments(
        self, raw_args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Simple templating: replaces '{{param_name}}' with context['param_name']."""
        resolved = {}
        for k, v in raw_args.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                param_key = v[2:-2].strip()
                resolved[k] = context.get(param_key, v)
            elif isinstance(v, dict):
                resolved[k] = self._resolve_arguments(v, context)
            else:
                resolved[k] = v
        return resolved

    def get_available_skills_prompt(self) -> str:
        """Returns a formatted string of available skills for the LLM to use."""
        if not self.skills:
            return ""

        lines = []
        # Only show reliable skills
        reliable_skills = [
            s for s in self.skills.values() if s.reliability > 0.4 or (s.successes + s.failures) < 3
        ]

        for s in reliable_skills:
            params = ", ".join(s.parameters)
            lines.append(
                f"- **{s.name}**({params}): {s.description} (Reliability: {s.reliability:.2f})"
            )

        return "\n### AVAILABLE MACRO SKILLS\n" + "\n".join(lines) + "\n"

    def _save(self):
        """Persist the library to disk atomically."""
        try:
            from core.utils.file_utils import atomic_write_json

            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"skills": {k: asdict(v) for k, v in self.skills.items()}}
            atomic_write_json(self.data_path, data)
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as e:
            _record_skill_degradation(
                "skill_library_persistence",
                e,
                action="kept in-memory skill library but failed to persist it",
                severity="warning",
                classification=FallbackClassification.AUDIT_GAP,
            )
            logger.error("Failed to save Skill Library: %s", e)

    def _load(self):
        if not self.data_path.exists():
            return
        try:
            # SL-001: Force utf-8 encoding
            with open(self.data_path, encoding="utf-8") as f:
                data = json.load(f)

            for k, dict_v in data.get("skills", {}).items():
                steps = [SkillStep(**s) for s in dict_v.pop("steps", [])]
                self.skills[k] = LearnedSkill(steps=steps, **dict_v)

        except _SKILL_LIBRARY_RECOVERABLE_ERRORS as e:
            _record_skill_degradation(
                "skill_library_load",
                e,
                action="started with empty in-memory skill library after load failed",
                severity="warning",
            )
            logger.error("Failed to load Skill Library: %s", e)


def register_skill_library(orchestrator=None):
    lib = SkillLibrary(orchestrator)
    ServiceContainer.register_instance("skill_library", lib)
    return lib
