import asyncio
import contextlib
import inspect
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.utils.paths import aura_data_dir
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.DynamicRouter")

MAX_HISTORY_MODELS = 64
MAX_HISTORY_PER_MODEL = 100
MAX_MODEL_NAME_CHARS = 120
MAX_TASK_TYPE_CHARS = 64
MAX_PROMPT_FINGERPRINT_CHARS = 12000
BACKGROUND_SAVE_INTERVAL_S = 300.0
STOP_TIMEOUT_S = 2.0


@dataclass
class RouteDecision:
    model: str
    reason: str
    confidence: float
    expected_tokens: int
    first_person_thought: str


def _emit_router_fault(
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    stage: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    if stage:
        metadata["stage"] = stage
    try:
        record_degradation(
            "dynamic_router",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata or None,
        )
    except TypeError:
        record_degradation("dynamic_router", error)


def _safe_text(value: Any, default: str = "", *, max_chars: int = 1000) -> str:
    if value is None:
        return default
    try:
        text = str(value)
    except (RuntimeError, TypeError, ValueError):
        return default
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "success", "ok"}
    return bool(value)


def _safe_int(value: Any, default: int = 0, *, minimum: int = 0, maximum: int = 1_000_000) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(number, minimum), maximum)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_model_name(value: Any) -> str:
    name = _safe_text(value, default="", max_chars=MAX_MODEL_NAME_CHARS).strip()
    return name or "Cortex"


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class DynamicRouter:
    name = "dynamic_router"

    def __init__(self):
        self.llm_router = None
        self.performance_history: dict[str, list[dict[str, Any]]] = {}
        self.db_path = aura_data_dir() / "routing_history.json"
        self.running = False
        self._learning_task: asyncio.Task | None = None
        self._dirty_outcomes = 0

        # Task fingerprints (tiers + smart weights)
        self.task_types = {
            "fast_fact": 0.2,
            "deep_reasoning": 0.9,
            "creative": 0.7,
            "tool_heavy": 0.85,
            "self_reflection": 0.95,
            "autonomous_goal": 1.0,
        }

    async def start(self):
        if self.running:
            return
        try:
            self.llm_router = ServiceContainer.get("intelligent_llm_router", default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self.llm_router = None
            _emit_router_fault(
                exc,
                action="started with deterministic Cortex fallback after LLM router lookup failed",
                severity="warning",
                stage="start.llm_router",
            )

        self._load_history()
        self.running = True
        learner = self._background_learner()
        try:
            self._learning_task = task_tracker.create_task(learner, name="DynamicRouter")
        except (RuntimeError, TypeError, ValueError) as exc:
            self.running = False
            with contextlib.suppress(RuntimeError):
                learner.close()
            _emit_router_fault(
                exc,
                action="failed closed because routing learner could not be supervised",
                severity="critical",
                stage="start.task_tracker",
            )
            raise
        self._learning_task.add_done_callback(self._observe_learning_task)

        logger.info("Dynamic Router online: model selection and outcome learning active.")

        try:
            await get_event_bus().publish(
                "mycelium.register",
                {
                    "component": "dynamic_router",
                    "hooks_into": [
                        "cognitive_engine",
                        "planner",
                        "critic_engine",
                        "belief_revision",
                    ],
                },
            )
        except (ImportError, AttributeError, RuntimeError) as e:
            _emit_router_fault(
                e,
                action="continued routing while mycelium registration is deferred",
                severity="warning",
                stage="start.event_bus",
            )
            logger.debug("Event bus publish missed for Mycelium hook: %s", e)

    async def stop(self):
        self.running = False
        if self._learning_task and not self._learning_task.done():
            self._learning_task.cancel()
            try:
                await asyncio.wait_for(self._learning_task, timeout=STOP_TIMEOUT_S)
            except asyncio.CancelledError:
                pass
            except TimeoutError as exc:
                _emit_router_fault(
                    exc,
                    action="continued shutdown after learner cancellation timeout",
                    severity="warning",
                    stage="stop.learning_task",
                )
        self._learning_task = None
        self._save_history()

    def _load_history(self):
        if not self.db_path.exists():
            return
        try:
            raw = json.loads(self.db_path.read_text(encoding="utf-8"))
            self.performance_history = self._sanitize_history(raw)
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.performance_history = {}
            self._quarantine_history()
            _emit_router_fault(
                exc,
                action="quarantined invalid routing history and reset learning state",
                severity="degraded",
                stage="load_history",
            )
            logger.debug("Could not load routing history: %s", exc)

    def _save_history(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._sanitize_history(self.performance_history)
            atomic_write_text(self.db_path, json.dumps(payload, indent=2, allow_nan=False))
            self._dirty_outcomes = 0
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            _emit_router_fault(
                e,
                action="continued with in-memory routing history after persistence failure",
                severity="degraded",
                stage="save_history",
            )
            logger.error("Router history save failed: %s", e)

    def _quarantine_history(self) -> None:
        if not self.db_path.exists():
            return
        quarantine = self.db_path.with_name(
            f"{self.db_path.stem}.corrupt.{int(time.time())}{self.db_path.suffix}"
        )
        try:
            self.db_path.replace(quarantine)
        except OSError as exc:
            _emit_router_fault(
                exc,
                action="continued with reset routing history after quarantine rename failed",
                severity="warning",
                stage="quarantine_history",
            )

    def _sanitize_history(self, raw: Any) -> dict[str, list[dict[str, Any]]]:
        if isinstance(raw, dict) and isinstance(raw.get("performance_history"), dict):
            raw = raw["performance_history"]
        if not isinstance(raw, dict):
            raise TypeError("routing history root must be an object")

        cleaned: dict[str, list[dict[str, Any]]] = {}
        for raw_model, raw_entries in list(raw.items())[:MAX_HISTORY_MODELS]:
            model = _safe_model_name(raw_model)
            if not isinstance(raw_entries, list):
                continue
            entries: list[dict[str, Any]] = []
            for raw_entry in raw_entries[-MAX_HISTORY_PER_MODEL:]:
                entry = self._coerce_history_entry(raw_entry)
                if entry is not None:
                    entries.append(entry)
            cleaned[model] = entries[-MAX_HISTORY_PER_MODEL:]
        return cleaned

    def _coerce_history_entry(self, raw_entry: Any) -> dict[str, Any] | None:
        if isinstance(raw_entry, dict):
            success = _safe_bool(raw_entry.get("success", False))
            tokens = _safe_int(raw_entry.get("tokens", 0))
            task_type = _safe_text(
                raw_entry.get("task_type", "autonomous_goal"),
                default="autonomous_goal",
                max_chars=MAX_TASK_TYPE_CHARS,
            )
            recorded_at = _safe_float(
                raw_entry.get("recorded_at", time.time()), default=time.time()
            )
        elif isinstance(raw_entry, (list, tuple)) and len(raw_entry) >= 3:
            success = _safe_bool(raw_entry[0])
            tokens = _safe_int(raw_entry[1])
            task_type = _safe_text(
                raw_entry[2],
                default="autonomous_goal",
                max_chars=MAX_TASK_TYPE_CHARS,
            )
            recorded_at = time.time()
        else:
            return None
        if task_type not in self.task_types:
            task_type = "autonomous_goal"
        return {
            "success": success,
            "tokens": tokens,
            "task_type": task_type,
            "recorded_at": recorded_at,
        }

    def _observe_learning_task(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except (RuntimeError, asyncio.CancelledError):
            return
        if exc is not None:
            self.running = False
            _emit_router_fault(
                exc,
                action="marked routing learner offline after background failure",
                severity="degraded",
                stage="background_learner",
            )

    async def route(self, prompt: str, context: dict[str, Any] | None = None) -> RouteDecision:
        """Main public API — called before every LLM generation."""
        if not isinstance(context, dict):
            context = {}

        task_type = self._fingerprint_task(prompt, context)
        available_models = self._get_available_models()

        # Score each model
        scores = {}
        for model in available_models:
            score = self._score_model(model, task_type, context)
            scores[model] = score

        # Pick winner
        if scores:
            best_model = max(scores.items(), key=lambda x: x[1])[0]
        else:
            best_model = "Cortex"

        confidence = _safe_float(scores.get(best_model, 0.5), default=0.5)

        reason = f"Chose {best_model} for {task_type} task (score: {confidence:.2f})"

        # First-person thought for CEL
        thought = (
            f"I'm routing this to {best_model} because it needs {task_type.replace('_', ' ')}."
        )

        decision = RouteDecision(
            model=best_model,
            reason=reason,
            confidence=confidence,
            expected_tokens=300 if "self_reflection" in task_type else 800,
            first_person_thought=thought,
        )

        # Emit to CEL so she feels the decision
        cel = None
        try:
            cel = ServiceContainer.get("constitutive_expression_layer", default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _emit_router_fault(
                exc,
                action="continued route without constitutive expression emission",
                severity="warning",
                stage="route.cel_lookup",
            )
        if cel:
            try:
                await _maybe_await(
                    cel.emit(
                        {
                            "first_person": thought,
                            "phi": confidence,
                            "origin": "dynamic_router",
                            "task_type": task_type,
                            "selected_model": best_model,
                        }
                    )
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as _e:
                _emit_router_fault(
                    _e,
                    action="continued route after constitutive expression emission failed",
                    severity="warning",
                    stage="route.cel_emit",
                    extra={"model": best_model, "task_type": task_type},
                )
                logger.debug("Dynamic router CEL emission failed: %s", _e)

        logger.debug("DynamicRouter -> %s | confidence %s", best_model, f"{confidence:.2f}")
        return decision

    def _fingerprint_task(self, prompt: str, context: dict[str, Any]) -> str:
        """Lightning-fast task fingerprinting."""
        explicit_type = _safe_text(
            context.get("task_type"),
            default="",
            max_chars=MAX_TASK_TYPE_CHARS,
        )
        if explicit_type in self.task_types:
            return explicit_type

        lower = _safe_text(prompt, max_chars=MAX_PROMPT_FINGERPRINT_CHARS).lower()
        if any(k in lower for k in ["reflect", "think about myself", "who am i", "my feelings"]):
            return "self_reflection"
        if (
            context.get("requires_tools")
            or context.get("tool_call")
            or context.get("external_io")
            or any(
                k in lower
                for k in ["tool", "execute", "search", "terminal", "browser", "shell", "click"]
            )
        ):
            return "tool_heavy"
        if "goal" in lower or "plan" in lower or "research" in lower:
            return "deep_reasoning"
        if any(k in lower for k in ["debug", "implement", "refactor", "patch", "test", "code"]):
            return "deep_reasoning"
        if len(lower) < 80 and "?" in lower:
            return "fast_fact"
        return (
            "creative"
            if any(k in lower for k in ["create", "write", "imagine"])
            else "autonomous_goal"
        )

    def _get_available_models(self) -> list:
        """Respects your existing tier system + health."""
        if not self.llm_router:
            return ["Cortex"]

        try:
            tiers = (
                self.llm_router.get_tier_layout()
                if hasattr(self.llm_router, "get_tier_layout")
                else {}
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _emit_router_fault(
                exc,
                action="fell back to adapter inventory after tier layout failed",
                severity="warning",
                stage="available_models.tiers",
            )
            tiers = {}
        if not tiers:
            # Fallback if method doesn't exist
            try:
                adapters = getattr(self.llm_router, "adapters", {})
                if isinstance(adapters, dict):
                    return self._dedupe_models(adapters.keys()) or ["Cortex"]
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _emit_router_fault(
                    exc,
                    action="fell back to Cortex after adapter inventory failed",
                    severity="warning",
                    stage="available_models.adapters",
                )
            return ["Cortex"]

        healthy = []
        for tier in ["PRIMARY", "SECONDARY", "TERTIARY"]:
            tier_models = tiers.get(tier, []) if isinstance(tiers, dict) else []
            if isinstance(tier_models, str):
                tier_models = [tier_models]
            for raw_model in tier_models:
                model = _safe_model_name(raw_model)
                if self._is_model_healthy(model):
                    healthy.append(model)
        return self._dedupe_models(healthy) or ["Cortex"]  # safe fallback

    def _is_model_healthy(self, model: str) -> bool:
        if not self.llm_router or not hasattr(self.llm_router, "is_unhealthy"):
            return True
        try:
            return not bool(self.llm_router.is_unhealthy(model))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _emit_router_fault(
                exc,
                action="excluded model after health check failed",
                severity="warning",
                stage="available_models.health",
                extra={"model": model},
            )
            return False

    @staticmethod
    def _dedupe_models(models: Any) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for raw_model in models:
            model = _safe_model_name(raw_model)
            if model not in seen:
                seen.add(model)
                deduped.append(model)
        return deduped

    def _score_model(self, model: str, task_type: str, context: dict[str, Any]) -> float:
        base_score = self.task_types.get(task_type, 0.5)

        # History bonus/penalty
        history = self.performance_history.get(model, [])
        if history:
            recent = [entry for entry in history[-10:] if isinstance(entry, dict)]
            if recent:
                recent_success = sum(1 for entry in recent if entry.get("success")) / len(recent)
                base_score += (recent_success - 0.5) * 0.3

        # Autonomous override — if she's in deep self-mode, prefer strongest model
        if context.get("origin") == "autonomous_volition":
            base_score += (
                0.4
                if "Pro" in model
                or model in {"Cortex", "Solver", "Brainstem", "Reflex"}
                or "MLX" in model
                else 0.0
            )

        model_lower = model.lower()
        if task_type == "tool_heavy" and any(
            k in model_lower for k in ("tool", "agent", "action", "planner")
        ):
            base_score += 0.2

        # Give local MLX a minor boost for normal tasks to save tokens
        if (
            model in {"Cortex", "Solver", "Brainstem", "Reflex"} or "MLX" in model
        ) and task_type in ("fast_fact", "creative"):
            base_score += 0.15

        return min(1.0, max(0.0, _safe_float(base_score, default=0.5)))

    async def record_outcome(self, model: str, success: bool, tokens: int, task_type: str):
        """Called after every LLM call."""
        model = _safe_model_name(model)
        entry = self._coerce_history_entry(
            {
                "success": success,
                "tokens": tokens,
                "task_type": task_type,
                "recorded_at": time.time(),
            }
        )
        if entry is None:
            return
        if model not in self.performance_history:
            self.performance_history[model] = []
        self.performance_history[model].append(entry)
        # Keep last 100 per model
        if len(self.performance_history[model]) > 100:
            self.performance_history[model] = self.performance_history[model][-100:]
        self._dirty_outcomes += 1
        if self._dirty_outcomes >= 10:
            self._save_history()

    async def _background_learner(self):
        while self.running:
            await asyncio.sleep(BACKGROUND_SAVE_INTERVAL_S)
            self._save_history()


# Singleton
_router_instance = None


def get_dynamic_router():
    global _router_instance
    if _router_instance is None:
        _router_instance = DynamicRouter()
    return _router_instance
