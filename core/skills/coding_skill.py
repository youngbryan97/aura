"""
Dedicated code generation with existing reasoning and sandbox execution checks.
"""

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict

from infrastructure import BaseSkill

logger = logging.getLogger("Skills.Coding")

class CodingSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "coding_skill"
    description = "Dedicated skill for writing, refactoring, and debugging complex code using step-by-step reasoning."

    def __init__(self):
        self.brain = None

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        params = goal.get("params", {})
        task = params.get("task", goal.get("objective", ""))
        language = params.get("language", "auto")

        if not task:
            return {"ok": False, "error": "No coding task provided"}

        logger.info("Executing coding task in %s", language)

        system_prompt = (
            "You are an expert software engineer. "
            f"Write clean, efficient, and well-documented {language} code. "
            "Think through the architecture, edge cases, and design patterns before implementing."
        )

        try:
            if self.brain is None:
                from core.container import ServiceContainer

                self.brain = ServiceContainer.get("cognitive_engine", default=None)
            if self.brain is None:
                return {"ok": False, "error": "Cognitive engine unavailable for coding_skill."}

            from core.brain.reasoning_strategies import StrategyType

            checks = params.get("checks")
            verification_args = {}
            if checks:
                verification_args = {"force_strategy": StrategyType.CONSISTENCY,
                                     "verification_context": {"function_examples": checks}}
            raw_result = await self.brain.generate(
                prompt=f"Task: {task}",
                system_prompt=system_prompt,
                origin=str(context.get("origin") or "api"),
                purpose="coding",
                prefer_tier="primary",
                deep_handoff=bool(context.get("deep_handoff", False)),
                max_tokens=int(context.get("max_tokens", 4096) or 4096),
                temperature=float(context.get("temperature", 0.2) or 0.2),
                use_strategies=True,
                **verification_args,
            )
            if isinstance(raw_result, dict):
                code = str(raw_result.get("response") or raw_result.get("text") or "")
                thought = str(raw_result.get("thought") or raw_result.get("thought_process") or "")
            else:
                code = str(raw_result or "")
                thought = ""

            # Don't claim success on an empty draft. The cognitive engine can
            # return "" under load (empty_cognitive_engine_reply is a real live
            # event); reporting ok=True with no code is the exact "technically
            # true but useless" failure a user hits as a blank answer.
            if not code.strip():
                return {
                    "ok": False,
                    "error": "The coding lane returned no code for that task (model produced an empty draft).",
                    "thought_process": thought,
                }

            verification = {"verified": False, "unmeasured": "execution checks are available for Python"}
            if str(language).lower() in ("python", "py"):
                from core.self_modification.checking_python import FunctionExample, check_python
                from core.utils.python_source_extraction import extract_python_code

                code = extract_python_code(code) if "```" in code else code.strip()
                if not code.strip():
                    return {"ok": False, "error": "the response did not contain Python code"}
                examples = params.get("checks")
                verification = await check_python(code, None if examples is None else [FunctionExample.model_validate(e) for e in examples])
                verification["oracle"] = "supplied independent examples" if examples is not None else "draft self-examples"
                verification["independent"] = examples is not None
                if not verification.get("syntax") or (verification.get("cases") and not verification.get("verified")):
                    return {"ok": False, "code": code, "verification": verification,
                            "error": verification.get("error") or "the draft did not pass its executable examples"}

            return {
                "ok": True,
                "code": code,
                "thought_process": thought,
                "note": "Generated through foreground coding reasoning",
                "verification": verification,
            }
        except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as e:
            record_degradation('coding_skill', e)
            logger.error("Coding skill failed: %s", e)
            return {"ok": False, "error": str(e)}
