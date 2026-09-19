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
from core.runtime.service_access import resolve_orchestrator

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

#: Publication additionally meets the catalog's own refusals — an unrecognised
#: effect scope, a name clash, a malformed declaration — all of which
#: ``register_skill`` raises as ValueError or TypeError. Named separately so the
#: publication path cannot quietly widen into swallowing something else.
_SKILL_LIBRARY_RECOVABLE_OR_REGISTRATION_ERRORS = _SKILL_LIBRARY_RECOVERABLE_ERRORS


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
    created_at: float = field(default_factory=lambda: time.time())

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
        self.publish_as_tool(skill)
        logger.info("🧠 Learned new skill: %s (Reliability: pending)", name)

    def publish_as_tool(self, skill: LearnedSkill) -> bool:
        """Register a macro into the live catalog so the model can call it.

        Until this existed a learned macro could be described to a turn and not
        invoked — it reached the model as prose and not as a tool, so the only
        way to use one was to re-derive its steps, which is what learning it was
        meant to avoid.

        Failure is never fatal. A macro that cannot be published is still a
        macro: it stays in the library, still executes through
        :meth:`execute_skill`, and the next reload can try again.
        """
        try:
            from core.agency.macro_skill import MacroSkill, derive_effect_scope
            from core.container import ServiceContainer

            engine = ServiceContainer.get("capability_engine", default=None)
            if engine is None or not hasattr(engine, "register_skill"):
                return False

            def _scope_for(tool: str) -> str:
                meta = getattr(engine, "skills", {}).get(tool)
                return str(getattr(meta, "effect_scope", "") or "")

            scope = derive_effect_scope([s.tool_name for s in skill.steps], _scope_for)
            engine.register_skill(
                MacroSkill(
                    macro_name=skill.name,
                    description=skill.description,
                    parameters=list(skill.parameters),
                    effect_scope=scope,
                ),
                replace=True,
            )
            logger.info("🔧 Macro '%s' published as a tool (scope: %s)", skill.name, scope)
            return True
        except _SKILL_LIBRARY_RECOVABLE_OR_REGISTRATION_ERRORS as e:
            _record_skill_degradation(
                "skill_library_publication",
                e,
                action="kept the macro in the library but did not publish it as a tool",
                severity="warning",
            )
            return False

    def publish_all(self) -> int:
        """Publish every reliable macro. Called after the catalog reloads.

        A catalog reload rebuilds ``skills`` from source discovery, which drops
        every runtime registration — so without re-publishing here, macros
        would silently stop being callable the first time anything reloaded.
        """
        return sum(1 for skill in self._reliable_skills() if self.publish_as_tool(skill))

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
        orchestrator = resolve_orchestrator()
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

    def _reliable_skills(self) -> list[LearnedSkill]:
        """Skills worth offering: proven, or too new to have been disproven."""
        return [
            s
            for s in self.skills.values()
            if s.reliability > 0.4 or (s.successes + s.failures) < 3
        ]

    def documents(self) -> list[Any]:
        """This library's skills, as retrieval documents.

        Registered with the process-wide retriever so a learned macro can be
        found by what it does. Until this existed the library was write-only:
        ``learn_skill`` persisted macros and the only way to read them back
        listed every one of them, which is why nothing called it.
        """
        from core.skills.skill_retrieval import SkillDocument

        return [
            SkillDocument(
                name=s.name,
                description=f"{s.description} parameters: {', '.join(s.parameters)}",
                source="macro",
            )
            for s in self._reliable_skills()
        ]

    def retrieve(self, objective: str, *, k: int = 3) -> list[LearnedSkill]:
        """The macros most relevant to ``objective``, best first.

        Voyager retrieves the top few skills for the task at hand rather than
        showing the agent its whole library, and the reason is a budget one: a
        library that grows is a prompt that grows, until the skills crowd out
        the problem. Retrieval is what makes learning more skills cheap.
        """
        if not self.skills:
            return []
        try:
            from core.skills.skill_retrieval import get_skill_retriever

            retriever = get_skill_retriever()
            retriever.register_provider("learned_macros", self.documents)
            hits = retriever.retrieve(objective, k=k)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
            _record_skill_degradation(
                "skill_library_retrieval",
                e,
                action="offered no macro skills because retrieval failed",
                severity="warning",
            )
            return []
        return [
            self.skills[hit.name]
            for hit in hits
            if hit.source == "macro" and hit.name in self.skills
        ]

    def get_available_skills_prompt(self, objective: str = "", *, k: int = 3) -> str:
        """The macros worth showing for this objective, as prompt text.

        With no objective this still lists everything, because a caller that
        cannot say what it is doing has no basis for a subset — but the ranked
        path is the one the live context uses.
        """
        if not self.skills:
            return ""

        selected = self.retrieve(objective, k=k) if objective.strip() else self._reliable_skills()
        if not selected:
            return ""

        lines = []
        for s in selected:
            params = ", ".join(s.parameters)
            lines.append(
                f"- **{s.name}**({params}): {s.description} (Reliability: {s.reliability:.2f})"
            )

        return "\n### AVAILABLE MACRO SKILLS\n" + "\n".join(lines) + "\n"

    def _save(self):
        """Persist the library to disk atomically."""
        try:
            from core.utils.file_utils import atomic_write_json

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
    # Macros loaded from disk are as callable as ones learned this session.
    # Without this, a restart turned every previously learned macro back into
    # prose the model could read and not invoke.
    lib.publish_all()
    return lib
