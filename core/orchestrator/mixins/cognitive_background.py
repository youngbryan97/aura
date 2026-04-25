"""Cognitive Background Mixin for RobustOrchestrator.
Handles background reflection, learning, RL training, and memory hygiene.
"""
import inspect
import logging
import time
import asyncio
from core.utils.exceptions import capture_and_log

logger = logging.getLogger(__name__)


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
        logger.debug("Shim: _run_rl_training is deprecated. Redirecting to live_learner if available.")
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
                    msg for msg in self.conversation_history
                    if isinstance(msg, dict) and msg.get("timestamp", time.time()) > cutoff
                ] or self.conversation_history[-20:]  # Keep at least 20
            except Exception as e:
                capture_and_log(e, {'module': __name__})
            
        # 2. Deduplication
        if hasattr(self, "conversation_history") and len(self.conversation_history) > 2:
            self._deduplicate_history()
            
        status = getattr(self, "status", None)
        if status and hasattr(self, "conversation_history") and len(self.conversation_history) > 100:
            if getattr(status, "cycle_count", 0) % 100 == 0 and not getattr(status, "is_processing", False):
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
            except RuntimeError:
                _dispose_awaitable(reflect_coro)
                return
            reflect_task.add_done_callback(_bg_task_exception_handler)
        except Exception as e:
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
            except RuntimeError:
                _dispose_awaitable(learn_coro)
                return
            learn_task.add_done_callback(_bg_task_exception_handler)

            # Feed curiosity engine from conversation
            if hasattr(self, 'curiosity') and self.curiosity and hasattr(self.curiosity, 'extract_curiosity_from_conversation'):
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
                        curiosity_task.add_done_callback(_bg_task_exception_handler)

            # Phase 26: Belief Revision Engine
            from core.container import ServiceContainer
            belief_engine = ServiceContainer.get("belief_revision_engine", default=None)
            if belief_engine:
                belief_coro = belief_engine.update_belief_from_conversation(
                    user_input=original_msg,
                    aura_response=response,
                    context={"world_state": self._get_world_context()}
                )
                try:
                    belief_task = tracker.create_task(
                        belief_coro,
                        name="cognitive_background.belief_revision",
                    )
                except RuntimeError:
                    _dispose_awaitable(belief_coro)
                else:
                    belief_task.add_done_callback(_bg_task_exception_handler)

        except Exception as e:
            if learn_coro is not None:
                _dispose_awaitable(learn_coro)
            logger.debug("Background learning setup failed: %s", e)
