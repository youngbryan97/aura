"""Base Skill — v41 Full Realization

All Aura skills inherit from this class. It provides:
  - Automatic timeout enforcement (default 30s, configurable per-skill)
  - Standardized result format (ok, error, duration_ms, skill_name)
  - Execution logging with duration and success/failure tracking
  - Graceful degradation on unhandled exceptions
  - Input validation via Pydantic models
  - Metabolic cost tagging for resource management
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel


logger = logging.getLogger("Skills")


class SkillResult(BaseModel):
    """Standardized result from any skill execution."""
    ok: bool
    skill: str = ""
    summary: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values for cleanliness."""
        d = self.model_dump(exclude_none=True)
        return d


class BaseSkill(ABC):
    """Base class for all Aura skills.

    Provides automatic timeout enforcement, error handling, input validation,
    and standardized result formatting. Subclasses implement `run()` with
    their specific logic; the `safe_execute()` wrapper handles the rest.

    Class Attributes:
        name: Unique identifier for this skill.
        description: Human-readable description for LLM tool routing.
        input_model: Optional Pydantic model for input validation.
        timeout_seconds: Maximum execution time (default 30s).
        metabolic_cost: 0=Core, 1=Light, 2=Medium, 3=Heavy.
        is_core_personality: True if this skill defines personality traits.
        requires_approval: True if destructive actions need user confirmation.
    """

    name: str = "base_skill"
    description: str = "Base skill description"

    # Input validation
    input_model: Optional[Type[BaseModel]] = None

    # Execution limits
    timeout_seconds: float = 30.0

    # Metabolic Tagging (0=Core, 1=Light, 2=Medium, 3=Heavy)
    metabolic_cost: int = 1
    is_core_personality: bool = False
    requires_approval: bool = False

    # Execution stats (instance-level)
    _total_executions: int = 0
    _total_failures: int = 0

    @abstractmethod
    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the skill's core logic.

        Subclasses MUST implement this method. It should focus purely on
        the skill's business logic — error handling, timeouts, and logging
        are handled by `safe_execute()`.

        Args:
            params: Validated input (Pydantic model or dict).
            context: Execution context (agent state, user info, etc.).

        Returns:
            Result dict. Must include 'ok' (bool). May include 'summary',
            'error', 'content', or any skill-specific keys.
        """
        pass

    async def safe_execute(self, params: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute with timeout, error handling, and standardized results.

        This is the PUBLIC entry point that the skill router should call.
        It wraps `execute()` with:
          - Input validation (if input_model is defined)
          - Timeout enforcement
          - Exception catching with graceful error messages
          - Duration tracking
          - Execution stats

        Args:
            params: Raw input (dict or Pydantic model).
            context: Optional execution context.

        Returns:
            Standardized result dict with 'ok', 'skill', 'summary',
            'duration_ms', and optionally 'error'.
        """
        context = context or {}
        start = time.monotonic()

        # Input validation
        if self.input_model and isinstance(params, dict):
            try:
                params = self.input_model(**params)
            except Exception as e:
                return self._error_result(
                    f"Invalid input: {e}",
                    time.monotonic() - start
                )

        # Execute with timeout
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await self.execute(params, context)

        except (asyncio.TimeoutError, TimeoutError):
            self._total_failures += 1
            duration = (time.monotonic() - start) * 1000
            logger.error(
                "⏱️ Skill '%s' timed out after %.0fms (limit: %.0fs)",
                self.name, duration, self.timeout_seconds
            )
            return self._error_result(
                f"Skill timed out after {self.timeout_seconds}s",
                time.monotonic() - start
            )

        except PermissionError as e:
            self._total_failures += 1
            logger.warning("🔒 Skill '%s' permission denied: %s", self.name, e)
            return self._error_result(
                f"Permission denied: {e}",
                time.monotonic() - start
            )

        except Exception as e:
            self._total_failures += 1
            duration = (time.monotonic() - start) * 1000
            logger.error(
                "💥 Skill '%s' crashed after %.0fms: %s",
                self.name, duration, e, exc_info=True
            )
            return self._error_result(
                f"Skill error: {type(e).__name__}: {e}",
                time.monotonic() - start
            )

        # Standardize the result
        duration_ms = (time.monotonic() - start) * 1000
        self._total_executions += 1

        if not isinstance(result, dict):
            result = {"ok": True, "content": str(result)}

        # Inject standard fields
        result.setdefault("ok", True)
        result["skill"] = self.name
        result["duration_ms"] = round(duration_ms, 1)

        if result.get("ok"):
            logger.info(
                "✅ Skill '%s' completed in %.0fms",
                self.name, duration_ms
            )
        else:
            self._total_failures += 1
            logger.warning(
                "⚠️ Skill '%s' returned error in %.0fms: %s",
                self.name, duration_ms, result.get("error", "unknown")
            )

        return result

    def _error_result(self, error: str, elapsed: float) -> Dict[str, Any]:
        """Build a standardized error result."""
        return {
            "ok": False,
            "skill": self.name,
            "error": error,
            "duration_ms": round(elapsed * 1000, 1)
        }

    def get_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for the skill's input parameters."""
        if self.input_model:
            return self.input_model.model_json_schema()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics for this skill."""
        return {
            "name": self.name,
            "executions": self._total_executions,
            "failures": self._total_failures,
            "success_rate": (
                round(1 - self._total_failures / max(1, self._total_executions), 3)
            ),
            "metabolic_cost": self.metabolic_cost
        }

    def match(self, goal: Dict[str, Any]) -> bool:
        """Check if this skill can handle the given goal.

        Default implementation returns False. Skills that want to be
        auto-matched by the skill router should override this.
        """
        return False

    def __repr__(self) -> str:
        return f"<Skill:{self.name} cost={self.metabolic_cost}>"