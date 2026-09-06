"""Production code generation adapter for clean-room reconstruction.

This module is the LLM-layer implementation of the ``CodeGenerator``
protocol used by ``core.self_improvement``.  The reconstruction lab remains
deterministic at the authority boundary: the model only proposes code, while
syntax checks, guardrails, sandbox tests, and the promotion gate decide whether
anything is acceptable.
"""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import inspect
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.brain.llm.deferral_record import explain_empty_generation
from core.runtime.errors import FallbackClassification, record_degradation

logger = logging.getLogger("Aura.LLMCodeGenerator")

_CODE_GENERATOR_RECOVERABLE_ERRORS = (
    AttributeError,
    RuntimeError,
    SyntaxError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_code_generator_degradation(
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


_CODE_GEN_SYSTEM_PROMPT = (
    "You are Aura's production Python code generator for clean-room module "
    "reconstruction.\n\n"
    "Return only complete, valid Python source code. Do not include markdown, "
    "analysis, prose, shell commands, or test output. Preserve the public "
    "interface described in the specification. Implement real behavior from "
    "the contract; do not hardcode expected test values. Include required "
    "imports and keep side effects minimal. Never use eval(), exec(), compile(), "
    "__import__(), network access, subprocesses, or filesystem reads of the "
    "original implementation."
)


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.IGNORECASE | re.DOTALL)
#: Just the opening marker, for output that never got to close it.
_OPENING_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n?", re.IGNORECASE)


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized request sent to Aura's LLM runtime."""

    prompt: str
    system_prompt: str
    prefer_tier: str
    max_tokens: int
    temperature: float
    origin: str = "reimplementation_lab"
    is_background: bool = True


def _first_pythonish_line(text: str) -> int:
    starters = (
        "from ",
        "import ",
        "class ",
        "def ",
        "async def ",
        "@",
        '"""',
        "'''",
        "#",
        "__all__",
    )
    for idx, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith(starters):
            return idx
    return 0


def extract_python_code(text: str) -> str:
    """Extract Python source from a model response without trusting wrappers."""

    raw = str(text or "").strip()
    if not raw:
        return ""

    fenced = _FENCE_RE.findall(raw)
    if fenced:
        candidates = [candidate.strip() for candidate in fenced if candidate.strip()]
        if candidates:
            return max(candidates, key=len).strip()

    # An opening fence with no closing one. The pattern requires both, so a
    # generation that ran out of tokens mid-block matched nothing and the raw
    # text — fence marker and all — went to the parser, which reported
    # "invalid syntax" on line 1 and lost an otherwise usable implementation.
    # Truncation is ordinary; throwing the whole answer away for it is not.
    opening = _OPENING_FENCE_RE.search(raw)
    if opening:
        tail = raw[opening.end():]
        closing = tail.find("```")
        body = (tail[:closing] if closing >= 0 else tail).strip()
        if body:
            return body

    lines = raw.splitlines()
    start = _first_pythonish_line(raw)
    if start:
        raw = "\n".join(lines[start:]).strip()

    # Some models append a short explanatory tail after otherwise valid code.
    # Prefer the full response if it parses; otherwise progressively trim the
    # tail until the candidate is syntactically valid.
    try:
        ast.parse(raw)
        return raw
    except SyntaxError as _exc:
        logger.debug("Suppressed %s in core.brain.llm.code_generator: %s", type(_exc).__name__, _exc)

    trimmed = raw.splitlines()
    for end in range(len(trimmed) - 1, 0, -1):
        candidate = "\n".join(trimmed[:end]).rstrip()
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return raw


def _coerce_response_text(response: Any) -> str:
    """Normalize the different LLM client return shapes used in Aura."""

    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, tuple):
        if len(response) >= 2 and isinstance(response[0], bool):
            return str(response[1] or "") if response[0] else ""
        for item in response:
            text = _coerce_response_text(item)
            if text:
                return text
        return ""
    if isinstance(response, dict):
        if response.get("ok") is False:
            return ""
        for key in ("text", "content", "response", "output"):
            if response.get(key):
                return str(response[key])
        message = response.get("message")
        if isinstance(message, dict) and message.get("content"):
            return str(message["content"])
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            return _coerce_response_text(choices[0])
        return ""
    for attr in ("content", "text", "response"):
        value = getattr(response, attr, None)
        if value:
            return str(value)
    return str(response)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass only supported kwargs unless the callable accepts arbitrary kwargs."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)

    filtered: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters:
            filtered[key] = value

    if "system" in parameters and "system_prompt" in kwargs and "system" not in filtered:
        filtered["system"] = kwargs["system_prompt"]

    if "context" in parameters and "context" not in filtered:
        filtered["context"] = dict(kwargs)

    return filtered


class LLMCodeGenerator:
    """CodeGenerator implementation backed by Aura's LLM runtime.

    ``generate_async`` is the preferred production path.  ``generate`` remains
    for the existing synchronous protocol and for tests that use a simple
    pluggable generator.
    """

    def __init__(
        self,
        *,
        router: Any | None = None,
        service_names: Iterable[str] = ("inference_gate", "llm_router", "cognitive_engine"),
        prefer_tier: str = "primary",
        prefer_endpoint: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.25,
        timeout_s: float = 180.0,
        fallback_to_stub: bool = False,
    ) -> None:
        self._router = router
        self.service_names = tuple(service_names)
        self.prefer_tier = prefer_tier
        self.prefer_endpoint = prefer_endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.fallback_to_stub = fallback_to_stub
        self.is_background = True

    def generate(self, prompt: str, context: dict[str, Any]) -> str:
        """Synchronous protocol adapter."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return asyncio.run(
                    asyncio.wait_for(
                        self.generate_async(prompt, context), timeout=self.timeout_s + 5.0
                    )
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"LLM generation exceeded timeout of {self.timeout_s + 5.0}s"
                ) from exc

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(lambda: asyncio.run(self.generate_async(prompt, context)))
        try:
            return future.result(timeout=self.timeout_s + 5.0)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"LLM generation exceeded timeout of {self.timeout_s + 5.0}s") from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    async def generate_async(self, prompt: str, context: dict[str, Any]) -> str:
        try:
            max_tokens = int(context.get("max_tokens", self.max_tokens))
        except (TypeError, ValueError):
            max_tokens = self.max_tokens
        try:
            temperature = float(context.get("temperature", self.temperature))
        except (TypeError, ValueError):
            temperature = self.temperature
        prefer_tier = str(context.get("prefer_tier", self.prefer_tier) or self.prefer_tier)
        # A caller that took the trouble to write a system prompt meant it.
        # The clean-room reconstruction lane sends "you are NOT given, and must
        # NOT assume, the original source" — the single instruction that makes
        # the output clean-room — and it was being discarded here in favour of
        # the generic one, silently, on every call.
        system_prompt = str(context.get("system_prompt") or "").strip() or _CODE_GEN_SYSTEM_PROMPT
        # A caller who says someone is waiting is telling the truth about
        # priority, and is the only party that knows it.
        is_background = bool(context.get("is_background", self.is_background))
        request = GenerationRequest(
            prompt=self._augment_prompt(prompt, context),
            system_prompt=system_prompt,
            prefer_tier=prefer_tier,
            max_tokens=max(64, min(max_tokens, self.max_tokens)),
            temperature=max(0.0, min(temperature, 2.0)),
            is_background=is_background,
        )

        try:
            router = self._resolve_router()
            if router is None:
                raise RuntimeError("no LLM router, inference gate, or cognitive engine is registered")

            response = await self._call_router(router, request)
            raw = _coerce_response_text(response)
            code = extract_python_code(raw)
            if not code:
                # The extractor only returns nothing when the response was
                # blank, and a blank response means one of two very different
                # things: a model that produced nothing, or a request that was
                # never run at all. The reconstruction lane reported "0/14
                # held-out positions reproduced" off the back of the second,
                # blaming verification for a generation that never happened.
                raise RuntimeError(
                    "LLM returned no Python source; "
                    + (explain_empty_generation() or "the model returned nothing at all")
                )

            try:
                ast.parse(code)
            except SyntaxError as exc:
                # A bare "invalid syntax" is the other undiagnosable ending: an
                # apology or a prose answer reaches here intact, and what the
                # model actually said is the entire diagnosis.
                # Keep the line structure. Collapsing it to one line was how
                # this failure first read as "`pythonimport randomdef move(...)"
                # — unreadable, and indistinguishable from a model that had
                # genuinely emitted no newlines. The shape IS the diagnosis
                # when the complaint is a syntax error.
                preview = "\n".join(str(code).splitlines()[:8])[:400]
                raise RuntimeError(
                    f"LLM returned no valid Python source "
                    f"({exc.msg} at line {exc.lineno}); model said:\n{preview}"
                ) from exc
            # Parsing says it is Python. It does not say the Python does what
            # it reads as doing. LIVE: a run delivered in 335 seconds whose
            # code never awaited its queued coroutines, could deadlock waiting
            # on queue data while holding the producer's lock, and mutated a
            # queue's internal deque instead of calling put(). Delivery
            # succeeded and semantic correctness was zero.
            self.last_async_findings = ()
            try:
                from core.verify.is_this_async_code_correct import what_is_wrong_with

                self.last_async_findings = what_is_wrong_with(code)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                _record_code_generator_degradation(
                    "llm_code_generator",
                    exc,
                    action="served generated code without the async correctness check",
                )
            if self.last_async_findings:
                _record_code_generator_degradation(
                    "llm_code_generator",
                    RuntimeError(
                        "; ".join(
                            one.what_happens for one in self.last_async_findings[:3]
                        )
                    ),
                    action=(
                        f"served {len(self.last_async_findings)} async mistake(s) in "
                        "generated code; the findings are on last_async_findings"
                    ),
                    extra={"module_path": str(context.get("module_path", ""))},
                )
                logger.warning(
                    "Generated code has %d async mistake(s): %s",
                    len(self.last_async_findings),
                    "; ".join(str(one) for one in self.last_async_findings[:3]),
                )
            logger.info(
                "Generated reconstruction candidate for %s (%d chars)",
                context.get("module_path", "<unknown>"),
                len(code),
            )
            return code
        except _CODE_GENERATOR_RECOVERABLE_ERRORS as exc:
            _record_code_generator_degradation(
                "llm_code_generator",
                exc,
                action="failed code generation and refused to synthesize placeholder code",
                extra={"module_path": str(context.get("module_path", ""))},
            )
            logger.warning("LLM code generation failed: %s", exc)
            if self.fallback_to_stub:
                fallback = self._validated_explicit_fallback(context)
                if fallback:
                    return fallback
            raise

    def _resolve_router(self) -> Any:
        if self._router is not None:
            return self._router
        try:
            from core.container import ServiceContainer

            for service_name in self.service_names:
                service = ServiceContainer.get(service_name, default=None)
                if service is not None:
                    self._router = service
                    return service
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_code_generator_degradation(
                "llm_code_generator_router",
                exc,
                action="treated LLM service lookup failure as no registered generator",
                severity="warning",
            )
            logger.debug("Could not resolve LLM service for code generation: %s", exc)
        return None

    def _validated_explicit_fallback(self, context: dict[str, Any]) -> str:
        fallback = str(context.get("stub_code") or "").strip()
        if not fallback:
            raise RuntimeError("fallback_to_stub requested but context.stub_code is empty")
        ast.parse(fallback)
        return fallback

    def _augment_prompt(self, prompt: str, context: dict[str, Any]) -> str:
        module_path = context.get("module_path", "<unknown>")
        attempt = context.get("attempt", 1)
        return (
            f"{prompt}\n\n"
            "## Production Constraints\n"
            f"- Target module: {module_path}\n"
            f"- Reconstruction attempt: {attempt}\n"
            "- Generate a complete module, not a patch.\n"
            "- Use only the specification above and the public interface stub.\n"
            "- If behavior is underspecified, choose the safest deterministic implementation.\n"
            "- CRITICAL: If feedback from a previous attempt is provided, analyze the discrepancies and ensure the new implementation addresses the root causes.\n"
        )

    async def _call_router(self, router: Any, request: GenerationRequest) -> Any:
        kwargs: dict[str, Any] = {
            "system_prompt": request.system_prompt,
            "prefer_tier": request.prefer_tier,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "origin": request.origin,
            "is_background": request.is_background,
            # This is source, not speech. Without it the reply pipeline shapes
            # the output as conversation and the indentation does not survive.
            "code_generation_contract": True,
        }
        if self.prefer_endpoint:
            kwargs["prefer_endpoint"] = self.prefer_endpoint
        if request.prefer_tier.lower() in {"secondary", "deep", "api_deep", "local_deep"}:
            kwargs["deep_handoff"] = True
            kwargs["allow_deep_handoff"] = True

        for method_name in ("think", "generate", "call", "generate_text_async", "generate_text"):
            method = getattr(router, method_name, None)
            if not callable(method):
                continue
            call_kwargs = _filter_kwargs(method, kwargs)
            try:
                return await _maybe_await(method(request.prompt, **call_kwargs))
            except TypeError as exc:
                logger.debug("LLM method %s rejected kwargs: %s", method_name, exc)
                continue

        raise RuntimeError(f"{type(router).__name__} exposes no supported generation method")


__all__ = ["LLMCodeGenerator", "GenerationRequest", "extract_python_code"]
