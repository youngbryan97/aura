"""GenAI spans in the OpenTelemetry semantic convention.

Aura has had `opentelemetry` installed and a capable home-grown tracer for a
long time, and not one `gen_ai.*` attribute anywhere. The consequence is
narrow but real: every model call, tool execution and agent invocation was
traced under names only this codebase understands, so none of it could be read
by a standard backend, compared against another system, or joined with anything
else's traces. Instrumentation nobody else can parse is instrumentation with an
audience of one.

This layer sits on top of `core.observability.tracing` rather than beside it —
same tracer, same head-based sampling, same OTLP export. What it adds is the
vocabulary: `chat {model}`, `execute_tool {tool}`, `invoke_agent {agent}`, with
the attribute names the spec fixes, and the two standard metrics
(`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`).

**Attribute names are a contract.** Same rule as the telemetry dictionary: they
are read by things outside this repo, so they are never renamed or repurposed.
That is why they are constants here rather than string literals at call sites.

**Content capture is off, and redacted when on.** The spec makes recording
prompts and completions opt-in. Aura goes further: even when enabled, message
content passes through the log sink's redactor first, because a span is an
*egress path* — it leaves for a collector — and the last time an egress
boundary here was checked it scrubbed values but not dict keys. Default-off is
not enough on its own; a flag someone flips in an incident is exactly when raw
conversation would leak.
"""
from __future__ import annotations

import os
import time
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from enum import StrEnum
from typing import Any

from core.observability import histograms
from core.observability.logging_config import redact_text
from core.observability.tracing import Span, SpanStatus, get_tracer

__all__ = [
    "GenAIOperation",
    "GenAIAttr",
    "content_capture_enabled",
    "chat_span",
    "embeddings_span",
    "tool_span",
    "agent_span",
    "workflow_span",
    "memory_span",
    "annotate_response",
    "record_token_usage",
    "TOKEN_USAGE_INPUT_METRIC",
    "TOKEN_USAGE_OUTPUT_METRIC",
    "OPERATION_DURATION_METRIC",
]


class GenAIOperation(StrEnum):
    """``gen_ai.operation.name`` values. Spec-fixed; do not invent members."""

    CHAT = "chat"
    GENERATE_CONTENT = "generate_content"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    RETRIEVAL = "retrieval"
    EXECUTE_TOOL = "execute_tool"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    PLAN = "plan"
    INVOKE_WORKFLOW = "invoke_workflow"
    CREATE_MEMORY = "create_memory"
    UPDATE_MEMORY = "update_memory"
    DELETE_MEMORY = "delete_memory"
    SEARCH_MEMORY = "search_memory"


class GenAIAttr(StrEnum):
    """Attribute keys. A contract with every consumer outside this repo."""

    OPERATION_NAME = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"
    CONVERSATION_ID = "gen_ai.conversation.id"

    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"

    RESPONSE_MODEL = "gen_ai.response.model"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
    USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"

    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_TYPE = "gen_ai.tool.type"

    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    AGENT_VERSION = "gen_ai.agent.version"

    WORKFLOW_NAME = "gen_ai.workflow.name"

    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"

    #: Not a gen_ai.* key — the general OTel convention for a failed span.
    ERROR_TYPE = "error.type"


#: Standard metric names. Histograms, per spec.
#:
#: The spec carries direction as a ``gen_ai.token.type`` dimension; the local
#: histogram registry is unlabelled, so the split is in the name instead. That
#: is the honest trade: input and output tokens price differently and a summed
#: total cannot be decomposed afterwards, whereas two histograms can always be
#: added. Model breakdown stays on the spans, where cardinality is affordable.
TOKEN_USAGE_INPUT_METRIC = "gen_ai.client.token.usage.input"
TOKEN_USAGE_OUTPUT_METRIC = "gen_ai.client.token.usage.output"
OPERATION_DURATION_METRIC = "gen_ai.client.operation.duration"

#: Who to ask about these numbers. A repo-relative FILE, because that is
#: what an owner is for and what the ratchet checks: a path that does not
#: exist answers nobody. "observability/genai" was neither a file nor a
#: directory, and the three histograms below failed the owner check on
#: every run that imported this module beside it.
_OWNER = "core/observability/genai_semconv.py"

histograms.declare_histogram(
    TOKEN_USAGE_INPUT_METRIC,
    description="Input tokens per GenAI operation.",
    owner=_OWNER,
    unit="{token}",
    minimum=1.0,
    maximum=2_000_000.0,
)
histograms.declare_histogram(
    TOKEN_USAGE_OUTPUT_METRIC,
    description="Output tokens per GenAI operation.",
    owner=_OWNER,
    unit="{token}",
    minimum=1.0,
    maximum=2_000_000.0,
)
histograms.declare_histogram(
    OPERATION_DURATION_METRIC,
    description="Wall-clock duration of a GenAI operation.",
    owner=_OWNER,
    unit="ms",
    minimum=1.0,
    maximum=600_000.0,
)

_CONTENT_CAPTURE_ENV = "AURA_GENAI_CAPTURE_CONTENT"


def content_capture_enabled() -> bool:
    """Whether prompt/completion content may be attached to spans.

    Read per call rather than cached at import: this is the switch that decides
    whether conversation text leaves the process, and a cached value would make
    turning it back *off* take a restart.
    """
    return os.getenv(_CONTENT_CAPTURE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _redacted_messages(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Role + redacted text. Never the raw mapping.

    Passing the original through would carry whatever else the caller happened
    to put in it — ids, tool payloads, system internals — to a collector that
    asked for none of it.
    """
    out: list[dict[str, str]] = []
    for message in messages:
        out.append({
            "role": redact_text(str(message.get("role", "")))[:64],
            "content": redact_text(str(message.get("content", ""))),
        })
    return out


def _set(span: Span, key: GenAIAttr | str, value: Any) -> None:
    """Set an attribute, skipping None so absent stays absent.

    An attribute present with a null value reads as "measured, and it was
    nothing", which is a different claim from "not measured".
    """
    if value is None:
        return
    span.set_attribute(str(key), value)


def _record_duration(seconds: float) -> None:
    histograms.record_duration(OPERATION_DURATION_METRIC, seconds)


def record_token_usage(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> None:
    """Emit the standard token-usage histograms, split by direction."""
    if input_tokens is not None:
        histograms.record(TOKEN_USAGE_INPUT_METRIC, float(input_tokens))
    if output_tokens is not None:
        histograms.record(TOKEN_USAGE_OUTPUT_METRIC, float(output_tokens))


@contextmanager
def _genai_span(
    span_name: str,
    operation: GenAIOperation,
    *,
    model: str | None = None,
    provider: str | None = None,
    conversation_id: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Shared span body: naming, required attributes, error typing, duration."""
    started = time.perf_counter()
    with get_tracer().span(span_name) as span:
        _set(span, GenAIAttr.OPERATION_NAME, operation.value)
        _set(span, GenAIAttr.PROVIDER_NAME, provider)
        _set(span, GenAIAttr.REQUEST_MODEL, model)
        _set(span, GenAIAttr.CONVERSATION_ID, conversation_id)
        for key, value in (attributes or {}).items():
            _set(span, key, value)
        try:
            yield span
        except BaseException as exc:
            # error.type is the low-cardinality class name, per spec — the
            # message is high-cardinality and belongs in the status, not a
            # metric dimension.
            _set(span, GenAIAttr.ERROR_TYPE, type(exc).__name__)
            span.set_status(SpanStatus.ERROR.value, str(exc)[:200])
            raise
        finally:
            _record_duration(time.perf_counter() - started)


@contextmanager
def chat_span(
    model: str,
    *,
    provider: str | None = None,
    conversation_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    input_messages: Sequence[Mapping[str, Any]] | None = None,
) -> Generator[Span, None, None]:
    """A model inference call. Span name: ``chat {model}``."""
    with _genai_span(
        f"{GenAIOperation.CHAT.value} {model}",
        GenAIOperation.CHAT,
        model=model,
        provider=provider,
        conversation_id=conversation_id,
    ) as span:
        _set(span, GenAIAttr.REQUEST_TEMPERATURE, temperature)
        _set(span, GenAIAttr.REQUEST_TOP_P, top_p)
        _set(span, GenAIAttr.REQUEST_MAX_TOKENS, max_tokens)
        if input_messages and content_capture_enabled():
            _set(span, GenAIAttr.INPUT_MESSAGES, _redacted_messages(input_messages))
        yield span


@contextmanager
def embeddings_span(
    model: str, *, provider: str | None = None
) -> Generator[Span, None, None]:
    """Span name: ``embeddings {model}``."""
    with _genai_span(
        f"{GenAIOperation.EMBEDDINGS.value} {model}",
        GenAIOperation.EMBEDDINGS,
        model=model,
        provider=provider,
    ) as span:
        yield span


@contextmanager
def tool_span(
    tool_name: str,
    *,
    call_id: str | None = None,
    description: str | None = None,
    tool_type: str | None = None,
    conversation_id: str | None = None,
) -> Generator[Span, None, None]:
    """A tool execution. Span name: ``execute_tool {tool_name}``."""
    with _genai_span(
        f"{GenAIOperation.EXECUTE_TOOL.value} {tool_name}",
        GenAIOperation.EXECUTE_TOOL,
        conversation_id=conversation_id,
    ) as span:
        _set(span, GenAIAttr.TOOL_NAME, tool_name)
        _set(span, GenAIAttr.TOOL_CALL_ID, call_id)
        _set(span, GenAIAttr.TOOL_DESCRIPTION, description)
        _set(span, GenAIAttr.TOOL_TYPE, tool_type)
        yield span


@contextmanager
def agent_span(
    agent_name: str,
    *,
    operation: GenAIOperation = GenAIOperation.INVOKE_AGENT,
    agent_id: str | None = None,
    description: str | None = None,
    version: str | None = None,
    model: str | None = None,
    conversation_id: str | None = None,
) -> Generator[Span, None, None]:
    """An agent invocation. Span name: ``invoke_agent {agent_name}``."""
    if operation not in {
        GenAIOperation.INVOKE_AGENT,
        GenAIOperation.CREATE_AGENT,
        GenAIOperation.PLAN,
    }:
        raise ValueError(
            f"{operation} is not an agent operation; use the span helper that "
            "matches it so the span name and attributes stay consistent"
        )
    with _genai_span(
        f"{operation.value} {agent_name}",
        operation,
        model=model,
        conversation_id=conversation_id,
    ) as span:
        _set(span, GenAIAttr.AGENT_NAME, agent_name)
        _set(span, GenAIAttr.AGENT_ID, agent_id)
        _set(span, GenAIAttr.AGENT_DESCRIPTION, description)
        _set(span, GenAIAttr.AGENT_VERSION, version)
        yield span


@contextmanager
def workflow_span(
    workflow_name: str, *, conversation_id: str | None = None
) -> Generator[Span, None, None]:
    """Span name: ``invoke_workflow {workflow_name}``."""
    with _genai_span(
        f"{GenAIOperation.INVOKE_WORKFLOW.value} {workflow_name}",
        GenAIOperation.INVOKE_WORKFLOW,
        conversation_id=conversation_id,
    ) as span:
        _set(span, GenAIAttr.WORKFLOW_NAME, workflow_name)
        yield span


@contextmanager
def memory_span(
    operation: GenAIOperation, *, conversation_id: str | None = None
) -> Generator[Span, None, None]:
    """A memory-store operation. Span name is the bare operation, per spec.

    Memory spans carry no record id in the name deliberately: ids are
    high-cardinality and would shatter the span-name aggregation they exist to
    support.
    """
    if not operation.value.endswith("memory"):
        raise ValueError(f"{operation} is not a memory operation")
    with _genai_span(
        operation.value, operation, conversation_id=conversation_id
    ) as span:
        yield span


def annotate_response(
    span: Span,
    *,
    model: str | None = None,
    response_id: str | None = None,
    finish_reasons: Sequence[str] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    output_messages: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Attach response-side attributes once the call returns.

    Separate from the span opener because none of this is knowable when the
    span starts, and a span that has to be opened after the work is done cannot
    time the work.
    """
    _set(span, GenAIAttr.RESPONSE_MODEL, model)
    _set(span, GenAIAttr.RESPONSE_ID, response_id)
    if finish_reasons:
        _set(span, GenAIAttr.RESPONSE_FINISH_REASONS, list(finish_reasons))
    _set(span, GenAIAttr.USAGE_INPUT_TOKENS, input_tokens)
    _set(span, GenAIAttr.USAGE_OUTPUT_TOKENS, output_tokens)
    _set(span, GenAIAttr.USAGE_REASONING_OUTPUT_TOKENS, reasoning_tokens)
    _set(span, GenAIAttr.USAGE_CACHE_READ_INPUT_TOKENS, cache_read_tokens)
    _set(span, GenAIAttr.USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_tokens)
    if output_messages and content_capture_enabled():
        _set(span, GenAIAttr.OUTPUT_MESSAGES, _redacted_messages(output_messages))

    record_token_usage(input_tokens=input_tokens, output_tokens=output_tokens)
