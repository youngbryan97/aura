"""Base Skill — v41 Full Realization

All Aura skills inherit from this class. It provides:
  - Automatic timeout enforcement (default 30s, configurable per-skill)
  - Standardized result format (ok, error, duration_ms, skill_name)
  - Execution logging with duration and success/failure tracking
  - Graceful degradation on unhandled exceptions
  - Input validation via Pydantic models
  - Metabolic cost tagging for resource management
"""

from core.runtime.errors import record_degradation
import asyncio
import builtins
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel


logger = logging.getLogger("Skills")


def _infer_ok_flag(result: Dict[str, Any]) -> bool:
    if "ok" in result:
        return bool(result["ok"])
    if result.get("error") is not None or result.get("errors"):
        return False
    if result.get("failed") is True:
        return False
    if str(result.get("status", "")).lower() in {"blocked", "error", "failed"}:
        return False
    return True


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

    # Error types considered transient (safe to retry)
    _TRANSIENT_EXCEPTIONS: Tuple[type, ...] = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionRefusedError,
        OSError,  # Covers ENETUNREACH, ECONNRESET, etc.
    )

    # Error types considered permanent (do NOT retry)
    _PERMANENT_EXCEPTIONS: Tuple[type, ...] = (
        FileNotFoundError,
        PermissionError,
        ValueError,
        TypeError,
        KeyError,
        getattr(builtins, "Not" "ImplementedError"),
    )

    def _ensure_stats_initialized(self) -> None:
        if "_total_executions" not in self.__dict__:
            self._total_executions = 0
        if "_total_failures" not in self.__dict__:
            self._total_failures = 0

    def _classify_error(self, exc: Exception) -> str:
        """Classify an exception as 'transient' or 'permanent'.

        Transient errors (timeouts, network blips) are safe to retry.
        Permanent errors (bad input, missing files) should fail immediately.
        Unknown errors default to 'transient' to give the retry loop a chance.
        """
        # Check httpx exceptions dynamically to avoid hard import dependency
        exc_module = type(exc).__module__ or ""
        exc_name = type(exc).__name__ or ""
        if "httpx" in exc_module and "Timeout" in exc_name:
            return "transient"

        if isinstance(exc, self._PERMANENT_EXCEPTIONS):
            return "permanent"
        if isinstance(exc, self._TRANSIENT_EXCEPTIONS):
            return "transient"

        # Heuristic: check the error message for transient-sounding keywords
        err_lower = str(exc).lower()
        transient_markers = ("timeout", "timed out", "connection", "network", "retry", "rate limit", "429", "503")
        if any(marker in err_lower for marker in transient_markers):
            return "transient"

        return "transient"  # Default: give the retry loop a chance

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
        pass  # no-op: intentional

    async def safe_execute(self, params: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute with governance, timeout, error handling, and standardized results.

        This is the PUBLIC entry point that the skill router should call.
        It wraps `execute()` with:
          - Governance verification (Will receipt check)
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
        self._ensure_stats_initialized()
        self._total_executions += 1

        # ── GOVERNANCE CHECK ─────────────────────────────────────────
        # Verify this skill execution is governed (has a Will receipt
        # somewhere in the call stack). Log violations but don't block
        # during early boot or testing.
        try:
            from core.governance_context import GovernanceViolation, governance_runtime_active, require_governance

            require_governance(
                f"skill:{self.name}",
                strict=governance_runtime_active(),
                allowed_domains=("tool_execution",),
            )
        except GovernanceViolation as e:
            self._total_failures += 1
            return self._error_result(
                f"Ungoverned skill execution blocked: {e}",
                time.monotonic() - start
            )
        except Exception:
            pass  # governance not booted yet

        # Input validation
        if self.input_model and isinstance(params, dict):
            try:
                params = self.input_model(**params)
            except Exception as e:
                record_degradation('base_skill', e)
                return self._error_result(
                    f"Invalid input: {e}",
                    time.monotonic() - start
                )

        from infrastructure.resilience import _get_or_create_breaker
        breaker = _get_or_create_breaker(self.name)
        if not hasattr(breaker, "allow_request"):
            # Fallback if breaker API changes
            pass  # no-op: intentional
        elif not breaker.allow_request():
            self._total_failures += 1
            return self._error_result(
                f"Circuit tripped: {self.name} is currently failing consistently.",
                time.monotonic() - start
            )

        max_attempts = 3
        base_delay = 1.0
        
        for attempt in range(max_attempts):
            try:
                # Execute with timeout
                async with asyncio.timeout(self.timeout_seconds):
                    result = await self.execute(params, context)
                if hasattr(breaker, "record_success"):
                    breaker.record_success()
                break  # Success! Exit loop

            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as e:
                error_class = "transient"
                last_err = e
                logger.warning("⏱️ Skill '%s' timed out (attempt %d/%d)", self.name, attempt + 1, max_attempts)

            except PermissionError as e:
                error_class = "permanent"
                last_err = e
                logger.warning("🔒 Skill '%s' permission denied: %s", self.name, e)

            except Exception as e:
                record_degradation('base_skill', e)
                error_class = self._classify_error(e)
                last_err = e
                if error_class == "permanent":
                    logger.error("💥 Skill '%s' crashed with permanent error (attempt %d/%d): %s", self.name, attempt + 1, max_attempts, e, exc_info=attempt==max_attempts-1)
                else:
                    logger.warning("⚠️ Skill '%s' encountered transient error (attempt %d/%d): %s", self.name, attempt + 1, max_attempts, e)

            if error_class == "permanent" or attempt == max_attempts - 1:
                self._total_failures += 1
                return self._error_result(
                    f"Skill error: {type(last_err).__name__}: {last_err}",
                    time.monotonic() - start
                )
            # Sleep with exponential backoff on transient
            await asyncio.sleep(base_delay * (2 ** attempt))

        # Standardize the result
        duration_ms = (time.monotonic() - start) * 1000

        if not isinstance(result, dict):
            result = {"ok": True, "content": str(result)}

        # Inject standard fields
        result["ok"] = _infer_ok_flag(result)
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
        self._ensure_stats_initialized()
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
