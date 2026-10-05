"""Surface markers both halves of the worker read.

``mlx_worker`` imports the surface-quality helpers at module scope and
``mlx_worker_surface_quality`` imported one marker back the same way, so
the pair only resolved when the worker was imported first. On its own:

    ImportError: cannot import name '_BACKEND_SYMBOLIC_SURFACE_MARKERS'
    from partially initialized module

A test or a tool reaching for the surface-quality module alone hit that,
and the only way through was to import the worker first and say why — an
ordering that looks like style and is load-bearing.

Neither marker belongs to either module. They are compiled patterns with
no dependencies, read from both sides, so they live here and the edge
runs one way: both import this, this imports neither.

``_FUSION_MODEL_IDENTITY`` deliberately does NOT move here. It is mutable
module state on the worker — reassigned at runtime and monkeypatched by
``tests/test_fusion_gate_in_worker.py`` — which is why the two places that
read it import it *inside* the function that needs it, so each call sees
the current value. Hoisting those would bind the name once at import and
every later patch would stop being seen.

The role labels that end a decode live here too, with the two readers that
apply them only to the reply. LIVE, 5 October 2026: asked the two-trains
bird problem, the cortex reasoned four times in its private channel and each
pass stopped between 212 and 237 tokens with no text. Every other way a pass
ends leaves a line in the log, and none was there; the clip for a model
simulating the next chat turn ran over the whole buffer, reasoning included,
so a role label written while working the problem out ended the pass, and
the turn went to the smaller model. A label in her reasoning quotes a turn or
plans one. It is a new turn only in the reply.
"""

from __future__ import annotations

import re

__all__ = [
    "_BACKEND_SYMBOLIC_SURFACE_MARKERS",
    "_CORRUPT_LANGUAGE_MARKERS",
    "_first_stop_in_reply",
    "_strip_leading_chatml_prefix",
    "_truncate_reply_role_continuation",
    "_truncate_role_continuation",
]

#: Words the decoder emits when the language itself has come apart. Not a
#: vocabulary of bad answers — these are not words.
_CORRUPT_LANGUAGE_MARKERS = re.compile(
    r"\b(?:xublcate|ingediate|evocer)\b",
    re.IGNORECASE,
)

#: Internal symbols that belong to the machinery and never to a reply.
_BACKEND_SYMBOLIC_SURFACE_MARKERS = re.compile(
    r"\b(?:PROCEEDING|TOOL_ACTION|CONVERGE_UNION|CONFORMED_METHODS|"
    r"TACTICAL_ORGANIZE|UI_SHUTDOWN_OR_DURATIVE_TIMEOUT|"
    r"MySelfEpsilon|CanonicalStabilityAnchor|currentInferenceProblem|"
    r"fieldOfPlay|INTRUSTION_DETECTED|INTRUSION_DETECTED|"
    r"ExistenceHash)\b"
)


def _strip_leading_chatml_prefix(text: str) -> str:
    cleaned = str(text or "")
    prefixes = (
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<｜Assistant｜>",
        "Assistant:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip("\n")
                changed = True
    return cleaned


_ROLE_CONTINUATION_RE = re.compile(
    r"(?is)(?:<\|im_end\|>\s*)?<\|im_start\|>\s*"
    r"(?:user|human|system|assistant|aura)\b.*$"
    r"|(?:^|\n|(?<=[.!?]))\s*(?:User|Human|System|Assistant|Aura)\s*[:：].*$"
)
_LEADING_GENERATION_ROLE_RE = re.compile(
    r"^\s*(?:<\|im_start\|>\s*)?(?:User|Human|Assistant|Aura|System)\s*[:：]\s*",
    re.IGNORECASE,
)
_LEADING_ROLE_NO_SEPARATOR_RE = re.compile(
    r"^\s*(?:user|human|assistant|system)(?=(?:i['’]?m\b|i\b|you\b|"
    r"what\b|who\b|when\b|where\b|why\b|how\b|yes\b|no\b|the\b))",
    re.IGNORECASE,
)
_USER_CONTINUATION_NO_COLON_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:User|Human)\s+"
    r"(?=(?:what|who|when|where|why|how|can|could|would|if|i\b|you\b|"
    r"yes\b|no\b|tell\b|translate\b|name\b|write\b|hello\b|hi\b|[\"'0-9])).*$"
)
_ROLE_SUFFIX_RE = re.compile(r"(?is)_user\b.*$")


def _truncate_role_continuation(text: str, *, final: bool = False) -> tuple[str, bool]:
    """Clip generation when the model starts simulating another chat turn.

    ``final`` decides whether trailing whitespace may be trimmed, and it is the
    whole reason this parameter exists.

    This runs on the ENTIRE accumulated buffer after every token. The trailing
    ``.strip()`` therefore deleted the newline the model had just emitted,
    every time it emitted one, before the next token could arrive. Asked for
    three fruits one per line, the live runtime returned::

        AppleBananaOrange

    and asked to echo a Python block, ``import randomdef f(x): return x + 1``
    — which is why the 2048 reconstruction kept failing with "invalid syntax
    at line 1" on code the model had written correctly. In Python, whitespace
    IS the syntax; in prose it is every paragraph she has ever written.

    Mid-stream the buffer is not finished, so its trailing whitespace is not
    trailing — it is the next line beginning.
    """
    cleaned = _strip_leading_chatml_prefix(str(text or ""))
    for _ in range(2):
        stripped = _LEADING_GENERATION_ROLE_RE.sub("", cleaned).lstrip()
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = _LEADING_ROLE_NO_SEPARATOR_RE.sub("", cleaned).lstrip()
    original = cleaned
    cleaned = _ROLE_CONTINUATION_RE.sub("", cleaned)
    cleaned = _USER_CONTINUATION_NO_COLON_RE.sub("", cleaned)
    cleaned = _ROLE_SUFFIX_RE.sub("", cleaned)
    return (cleaned.strip() if final else cleaned), cleaned != original


#: Where the private channel closes in a native-thinking decode.
_CHANNEL_CLOSE = "</think>"


def _reply_starts_at(text: str, *, native_thinking: bool) -> int | None:
    """Where the reply begins in a decode buffer; None while the channel is open.

    Qwen's template opens the channel in the prompt, so with native thinking
    on nothing generated is the reply until ``</think>`` arrives.
    """
    if native_thinking is not True:
        return 0
    boundary = text.find(_CHANNEL_CLOSE)
    if boundary < 0:
        return None
    return boundary + len(_CHANNEL_CLOSE)


def _truncate_reply_role_continuation(
    text: str, *, native_thinking: bool
) -> tuple[str, bool]:
    """``_truncate_role_continuation`` over the reply alone, mid-stream."""
    start = _reply_starts_at(text, native_thinking=native_thinking)
    if start is None:
        return text, False
    if start == 0:
        return _truncate_role_continuation(text)
    reply = text[start:]
    body = reply.lstrip()
    clipped, hit = _truncate_role_continuation(body)
    return text[:start] + reply[: len(reply) - len(body)] + clipped, hit


def _first_stop_in_reply(
    text: str, stops: list[str] | tuple[str, ...], *, native_thinking: bool
) -> tuple[int, str]:
    """The first stop sequence found, as ``(index, stop)``, or ``(-1, "")``.

    A chat-control token ends the turn wherever it appears. A role label
    such as ``"\\nUser:"`` ends it only in the reply.
    """
    start = _reply_starts_at(text, native_thinking=native_thinking)
    for stop in stops:
        if stop.startswith("<|"):
            index = text.find(stop)
        elif start is None:
            continue
        else:
            index = text.find(stop, start)
        if index >= 0:
            return index, stop
    return -1, ""
