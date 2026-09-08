"""Compatibility bridge for Aura's internal MLX inference lane.

Older callers still import ``LocalBrain`` for summarization, actuator synthesis,
or agent loops.  This module deliberately contains no HTTP model-server path:
all generation delegates to :class:`core.brain.unified_inference.UnifiedInferenceEngine`.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from core.config import config
from core.conversation.word_markers import names_any
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.LocalBrain")

_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)

_CODING_KEYWORDS = {
    "api",
    "bug",
    "build",
    "class",
    "code",
    "compile",
    "database",
    "debug",
    "deploy",
    "docker",
    "error",
    "exception",
    "fix",
    "function",
    "git",
    "implement",
    "javascript",
    "python",
    "query",
    "refactor",
    "script",
    "sql",
    "test",
    "traceback",
    "typescript",
}


def _record_llm_degradation(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    severity: str = "degraded",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        subsystem,
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def detect_task_tier(prompt: str, system_prompt: str = "") -> str:
    """Detect a lightweight generation tier for callers that still ask."""
    combined = f"{prompt} {system_prompt or ''}".lower()
    if names_any(combined, _CODING_KEYWORDS):
        return "coding"
    if names_any(combined, ("summarize", "compress", "distill")):
        return "summary"
    return "chat"


class LocalBrain:
    """MLX-only compatibility facade.

    This class intentionally refuses to fall through to raw external endpoints.
    If unified inference fails, the caller receives an empty structured failure
    and a degradation receipt instead of a generic assistant reply.
    """

    THOUGHT_STREAM_PREFIX = "__THOUGHT__:"
    THOUGHT_STREAM_SUFFIX = ":__ENDTHOUGHT__"

    def __init__(self, model_name: str | None = None):
        self.model = model_name or str(getattr(config.llm, "model", None) or "Aura-MLX")
        self.timeout = getattr(config.llm, "timeout", config.llm_request_timeout_s)
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    async def close(self) -> None:
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    #: Lane states in which a visible turn can actually be served. Everything
    #: else — cold, spawning, handshaking, warming, recovering, fenced,
    #: closed, failed, retired — either has no model resident or is mid
    #: transition.
    _SERVING_LANE_STATES = frozenset({"ready"})

    def check_health(self) -> bool:
        """True only when the lane can serve a turn RIGHT NOW.

        CP126 5a18fe70: this excluded exactly two states, ``failed`` and
        ``retired``, so a cold lane with no model resident, a lane still
        spawning its worker, one mid-handshake, one warming, one recovering
        from a kill and one already closed all reported healthy. Callers use
        this to decide whether to route a turn here; each of those states
        answers "yes, send it" and then cannot produce a token.

        A lane that is merely LOADABLE is reported by ``can_become_ready``.
        The two questions have different answers and had one method.
        """
        return self._lane_state() in self._SERVING_LANE_STATES

    def can_become_ready(self) -> bool:
        """True when the lane is not terminally out of service.

        Cold, warming and recovering lanes belong here: a caller willing to
        wait through a load should still try. A failed, retired or closed
        lane will not become ready without operator action, and an unknown
        state ("") has not been shown to be recoverable — it has only
        declined to answer, which is not the same thing.
        """
        state = self._lane_state()
        return bool(state) and state not in {"failed", "retired", "closed"}

    def _lane_state(self) -> str:
        """The lane's state, or "" when it cannot be established.

        An unreachable client is not a healthy one: "" satisfies neither
        predicate above, so an unknown lane fails closed in both directions.
        """
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            if not client:
                return ""
            status = client.get_lane_status() if hasattr(client, "get_lane_status") else {}
            return str((status or {}).get("state") or "").lower()
        except _RECOVERABLE_ERRORS:
            return ""

    async def check_health_async(self) -> bool:
        return await asyncio.to_thread(self.check_health)

    async def warmup(self) -> bool:
        try:
            from core.brain.llm.mlx_client import get_mlx_client

            client = get_mlx_client()
            if hasattr(client, "warmup"):
                result = await client.warmup()
                return bool(result)
            return bool(client)
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_brain_mlx_warmup",
                exc,
                action="left internal MLX lane cold after compatibility warmup failed",
                severity="warning",
                extra={"model": self.model},
            )
            return False

    def _check_circuit(self) -> bool:
        if self._circuit_open and time.time() < self._circuit_open_until:
            logger.warning("Circuit breaker OPEN; skipping compatibility generation")
            return False
        if self._circuit_open:
            self._circuit_open = False
        return True

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._circuit_open = False
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self._circuit_open = True
            self._circuit_open_until = time.time() + 30.0

    @staticmethod
    def _extract_think_segments(text: str) -> tuple[str, str]:
        thoughts: list[str] = []
        for match in re.finditer(r"<think>(.*?)</think>", text or "", flags=re.DOTALL):
            thought_text = match.group(1).strip()
            if thought_text:
                thoughts.append(thought_text)
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
        cleaned = cleaned.replace("</think>", "").replace("<think>", "")
        return cleaned.strip(), "\n\n".join(thoughts)

    def _strip_think_tags(self, text: str) -> str:
        return self._extract_think_segments(text)[0]

    @staticmethod
    def _is_usable_result(result: Any) -> bool:
        """Did inference actually produce something a turn can use?

        CP126 c428ec85: both paths called ``_record_success()`` the moment
        ``generate_unified`` RETURNED, without looking at what it returned.
        The unified engine reports its own failures in-band — it returns
        ``{"response": "", "error": "..."}`` rather than raising — so an
        explicitly failed generation reset the failure streak and healed the
        circuit breaker. The breaker therefore could not trip on the failure
        mode it exists for: inference that keeps answering, emptily.
        """
        if not isinstance(result, dict):
            return False
        if result.get("error"):
            return False
        return bool(str(result.get("response") or "").strip())

    async def _run_inference(self, **call: Any) -> dict[str, str]:
        """One inference call: deadline enforced, outcome graded, circuit fed.

        CP126 c5e2f2f9: ``self.timeout`` was read from config in __init__ and
        then never used — no wait_for, no deadline, no wrapper. The advertised
        timeout was a field on an object. A compatibility call could block
        for as long as the resident model took, which under load is
        unbounded, while the caller believed it had a bound.
        """
        from core.brain.unified_inference import UnifiedInferenceEngine

        deadline = self._deadline_seconds()
        try:
            if deadline > 0:
                result = await asyncio.wait_for(
                    UnifiedInferenceEngine().generate_unified(**call), timeout=deadline
                )
            else:
                result = await UnifiedInferenceEngine().generate_unified(**call)
        except TimeoutError as exc:
            _record_llm_degradation(
                "local_llm_unified_timeout",
                exc,
                action=(
                    "abandoned a compatibility generation that exceeded the "
                    f"configured {deadline:.0f}s timeout"
                ),
                severity="error",
                extra={"model": self.model, "timeout_s": deadline},
            )
            self._record_failure()
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_timeout",
            }

        if self._is_usable_result(result):
            self._record_success()
        else:
            # Not an exception, and not a success either. Feeding the breaker
            # here is the whole point: a lane returning empties forever is
            # exactly what it should open on.
            self._record_failure()
        return result

    def _deadline_seconds(self) -> float:
        """The configured timeout, or 0 when it is unusable.

        A non-numeric, non-finite or non-positive timeout is not a licence to
        run unbounded, but neither is it something to invent a number for: it
        is reported, and the call runs under whatever bound the engine itself
        applies.
        """
        try:
            deadline = float(self.timeout)
        except (TypeError, ValueError):
            deadline = 0.0
        if deadline != deadline or deadline in (float("inf"), float("-inf")):
            deadline = 0.0
        if deadline <= 0:
            _record_llm_degradation(
                "local_llm_timeout_unusable",
                ValueError(f"unusable llm timeout: {self.timeout!r}"),
                action=(
                    "ran a compatibility generation under the engine's own "
                    "bound because the configured timeout was not a usable "
                    "positive number"
                ),
                severity="warning",
                extra={"model": self.model, "configured": repr(self.timeout)},
            )
            return 0.0
        return deadline

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        if not self._check_circuit():
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_circuit_open",
            }

        try:
            return await self._run_inference(
                prompt=prompt,
                system_prompt=system_prompt,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="refused raw external generation fallback after unified MLX inference failed",
                severity="error",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            self._record_failure()
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_unified_inference_failed",
            }

    async def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        if not self._check_circuit():
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_circuit_open",
            }

        try:
            return await self._run_inference(
                prompt="",
                messages=messages,
                options=options,
                endpoint_name=kwargs.get("endpoint_name"),
                **kwargs,
            )
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_unified_fallback",
                exc,
                action="refused raw external chat fallback after unified MLX inference failed",
                severity="error",
                extra={"model": self.model, "endpoint": kwargs.get("endpoint_name")},
            )
            self._record_failure()
            return {
                "response": "",
                "thought": "",
                "error": "internal_mlx_unified_inference_failed",
            }

    @staticmethod
    def _streaming_lane() -> Any:
        """The resident nucleus, or None when there is no streaming lane.

        Resolved from the service container, never constructed: a second
        NucleusManager is a second owner of the same model weights, and on a
        64GB host a duplicated 32B lane is how the runtime dies. The
        container's instance is the one that already holds them.
        """
        try:
            from core.container import container

            nucleus = container.get("nucleus", None)
        except _RECOVERABLE_ERRORS:
            return None
        if nucleus is None or not hasattr(nucleus, "generate_stream_async"):
            return None
        return nucleus

    async def _stream_or_buffer(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        messages: list[dict[str, str]] | None,
        cancel_event,
        options: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ):
        """Incremental text, stopped at the token cancellation arrives.

        CP126 63396370: both stream methods checked ``cancel_event`` once,
        BEFORE calling the non-streaming path, then yielded the finished
        answer as a single chunk. So there was no stream — the caller waited
        the full generation and received it whole — and cancellation raised
        during generation did nothing at all, because nothing looked at the
        event again and nothing told the resident model to stop. A user who
        pressed stop kept paying for every remaining token.

        The nucleus stream is a real token stream and stops its worker when
        the generator is closed, so cancellation here reaches the model
        rather than merely discarding its output. When no streaming lane is
        available the buffered path is still used — one chunk is a poor
        stream but a correct answer — and the cancel event is honoured
        between the call and the yield.
        """
        cancelled = (
            cancel_event.is_set
            if cancel_event is not None and hasattr(cancel_event, "is_set")
            else (lambda: False)
        )
        if cancelled():
            return

        stream = None
        try:
            nucleus = self._streaming_lane()
            if messages is None and nucleus is not None:
                stream = nucleus.generate_stream_async(
                    prompt, system_prompt=system_prompt, **kwargs
                )
        except _RECOVERABLE_ERRORS as exc:
            _record_llm_degradation(
                "local_llm_stream_unavailable",
                exc,
                action=(
                    "fell back to a buffered single-chunk stream after the "
                    "incremental lane was unavailable"
                ),
                severity="warning",
                extra={"model": self.model},
            )
            stream = None

        if stream is not None:
            produced = False
            try:
                async for chunk in stream:
                    if cancelled():
                        break
                    if chunk:
                        produced = True
                        yield chunk
            finally:
                # Closing the generator sets the worker's stop event, which
                # is what actually stops resident inference. Without this the
                # model keeps generating into a queue nobody reads.
                await stream.aclose()
            if produced:
                # A stream that delivered tokens is a working lane; the
                # breaker must see it, or a healthy streaming path would
                # never clear a streak left by the buffered one.
                self._record_success()
                return
            if cancelled():
                # Stopped on request. Neither a success nor a lane fault.
                return
            # The streaming lane yielded nothing at all. Fall through to the
            # buffered path rather than returning an empty stream, which the
            # caller cannot distinguish from a refusal.

        if messages is None:
            result = await self.generate(
                prompt, system_prompt=system_prompt, options=options, **kwargs
            )
        else:
            result = await self.chat(messages, options=options, **kwargs)

        if cancelled():
            return
        response = result.get("response", "")
        thought = result.get("thought", "")
        if response:
            yield response
        if thought:
            yield f"{self.THOUGHT_STREAM_PREFIX}{thought}{self.THOUGHT_STREAM_SUFFIX}"
        if result.get("error") and not response:
            yield f" [Error: Sovereign stream interrupted: {result['error']}]"

    async def generate_text_stream_async(
        self,
        prompt: str,
        system_prompt: str | None = None,
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        async for chunk in self._stream_or_buffer(
            prompt=prompt,
            system_prompt=system_prompt,
            messages=None,
            cancel_event=cancel_event,
            options=options,
            kwargs=kwargs,
        ):
            yield chunk

    async def chat_stream_async(
        self,
        messages: list[dict[str, str]],
        cancel_event=None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        async for chunk in self._stream_or_buffer(
            prompt="",
            system_prompt=None,
            messages=messages,
            cancel_event=cancel_event,
            options=options,
            kwargs=kwargs,
        ):
            yield chunk
