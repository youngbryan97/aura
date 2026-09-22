"""Cognitive Background Mixin for RobustOrchestrator.
Handles background reflection, learning, RL training, and memory hygiene.
"""

import asyncio
import inspect
import logging
import time

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
from core.utils.exceptions import capture_and_log

logger = logging.getLogger(__name__)
_COGNITIVE_BACKGROUND_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    Exception,
)


def _record_background_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("cognitive_background", error, severity=severity, action=action)


def _dispose_awaitable(result):
    if inspect.iscoroutine(result):
        result.close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()


def _task_scheduled(result):
    return asyncio.isfuture(result) or isinstance(result, asyncio.Task)


def _bg_task_exception_handler(task: asyncio.Task):
    if not task.cancelled() and task.exception():
        e = task.exception()
        logging.getLogger("Aura.BgTasks").debug(f"Task exception handler itself failed: {e}")


class CognitiveBackgroundMixin:
    """Handles background learning, reflection, and memory management loops."""

    async def _run_rl_training(self):
        """Legacy shim for RL training. (Deprecated / Moved to Learning Engine)"""
        logger.debug(
            "Shim: _run_rl_training is deprecated. Redirecting to live_learner if available."
        )
        if getattr(self, "live_learner", None) and hasattr(self.live_learner, "train"):
            await self.live_learner.train()

    def _manage_memory_hygiene(self):
        # 1. Hard Limit — rolling window (HARDENING: bounded memory)
        if hasattr(self, "conversation_history") and isinstance(self.conversation_history, list):
            if len(self.conversation_history) > 150:
                # v Zenith: Robust slicing
                self.conversation_history = self.conversation_history[-150:]

            # 1b. Time-based eviction — drop entries older than 2 hours
            try:
                cutoff = time.time() - 7200
                self.conversation_history = [
                    msg
                    for msg in self.conversation_history
                    if isinstance(msg, dict) and msg.get("timestamp", time.time()) > cutoff
                ] or self.conversation_history[-20:]  # Keep at least 20
            except _COGNITIVE_BACKGROUND_ERRORS as e:
                _record_background_degradation(
                    e,
                    action="continued memory hygiene with existing conversation history after timestamp pruning failed",
                )
                capture_and_log(e, {"module": __name__})

        # 2. Deduplication
        if hasattr(self, "conversation_history") and len(self.conversation_history) > 2:
            self._deduplicate_history()

        status = getattr(self, "status", None)
        if (
            status
            and hasattr(self, "conversation_history")
            and len(self.conversation_history) > 100
        ):
            if getattr(status, "cycle_count", 0) % 100 == 0 and not getattr(
                status, "is_processing", False
            ):
                from core.utils.task_tracker import get_task_tracker

                get_task_tracker().bounded_track(self._prune_history_async, name="prune_history")

    def _trigger_background_reflection(self, response: str):
        reflect_coro = None
        try:
            from core.conversation_reflection import get_reflector
            from core.utils.task_tracker import get_task_tracker

            reflect_coro = get_reflector().maybe_reflect(
                self.conversation_history,
                self.cognitive_engine,
                mood=self._get_current_mood(),
                time_str=self._get_current_time_str(),
            )
            try:
                reflect_task = get_task_tracker().create_task(
                    reflect_coro,
                    name="cognitive_background.reflection",
                )
            # not a failure: no loop to create the task on; the coroutine is disposed rather
            # than left un-awaited.
            except RuntimeError:
                _dispose_awaitable(reflect_coro)
                return
            if not _task_scheduled(reflect_task):
                _dispose_awaitable(reflect_coro)
                return
            reflect_task.add_done_callback(_bg_task_exception_handler)
        except _COGNITIVE_BACKGROUND_ERRORS as e:
            _record_background_degradation(
                e,
                action="skipped background reflection scheduling after reflector setup failed",
            )
            if reflect_coro is not None:
                _dispose_awaitable(reflect_coro)
            logger.debug("Background reflection setup failed: %s", e)

    def _trigger_background_learning(self, message: str, response: str):
        learn_coro = None
        try:
            original_msg = message.replace("Impulse: ", "").replace("Thought: ", "")
            from core.utils.task_tracker import get_task_tracker

            tracker = get_task_tracker()
            learn_coro = self._learn_from_exchange(original_msg, response)
            try:
                learn_task = tracker.create_task(
                    learn_coro,
                    name="cognitive_background.learn_from_exchange",
                )
            # not a failure: the same, for the learning coroutine.
            except RuntimeError:
                _dispose_awaitable(learn_coro)
                return
            if not _task_scheduled(learn_task):
                _dispose_awaitable(learn_coro)
                return
            learn_task.add_done_callback(_bg_task_exception_handler)

            # Feed curiosity engine from conversation
            if (
                hasattr(self, "curiosity")
                and self.curiosity
                and hasattr(self.curiosity, "extract_curiosity_from_conversation")
            ):
                curiosity_result = self.curiosity.extract_curiosity_from_conversation(original_msg)
                if inspect.isawaitable(curiosity_result):
                    try:
                        curiosity_task = tracker.create_task(
                            curiosity_result,
                            name="cognitive_background.curiosity_extract",
                        )
                    except RuntimeError:
                        _dispose_awaitable(curiosity_result)
                    else:
                        if _task_scheduled(curiosity_task):
                            curiosity_task.add_done_callback(_bg_task_exception_handler)
                        else:
                            _dispose_awaitable(curiosity_result)

            try:
                belief_engine = get_runtime_service("belief_revision_engine", default=None)
                update_belief = getattr(belief_engine, "update_belief_from_conversation", None)
                if callable(update_belief):
                    world_context = (
                        self._get_world_context() if hasattr(self, "_get_world_context") else {}
                    )
                    belief_result = update_belief(
                        message=original_msg,
                        response=response,
                        world_context=world_context,
                    )
                    if inspect.isawaitable(belief_result):
                        try:
                            belief_task = tracker.create_task(
                                belief_result,
                                name="cognitive_background.belief_revision",
                            )
                        except RuntimeError:
                            _dispose_awaitable(belief_result)
                        else:
                            if _task_scheduled(belief_task):
                                belief_task.add_done_callback(_bg_task_exception_handler)
                            else:
                                _dispose_awaitable(belief_result)
            except _COGNITIVE_BACKGROUND_ERRORS as e:
                _record_background_degradation(
                    e,
                    action="continued background learning after belief revision scheduling failed",
                )
                logger.debug("Background belief revision setup failed: %s", e)

        except _COGNITIVE_BACKGROUND_ERRORS as e:
            _record_background_degradation(
                e,
                action="skipped background learning scheduling after setup failed",
            )
            if learn_coro is not None:
                _dispose_awaitable(learn_coro)
            logger.debug("Background learning setup failed: %s", e)
