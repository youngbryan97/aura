"""When the question is about her own machinery, read her own machinery.

LIVE 2026-08-18: "you have a lock ordering system. what happens if two
subsystems take locks in opposite order, and how would you know before it
deadlocks?"

    You're hitting on a classic deadlock scenario... Statically, you could use
    a tool like `lock-order`... Go's model of communicating sequential
    processes, for example, lets you reason about concurrency without locks.

She has core/runtime/lockdep.py. It wraps every lock through checked_lock and
checked_async_lock, finds ABBA cycles without the deadlock having to happen,
and reports them in runtime_health_report()["integrity"]. That is the answer
to the question that was asked, and instead she described the general
literature and recommended a language she is not written in.

The question named HER. A question about her own mechanism is answerable from
her own source, which is on this disk and which the evidence provider already
searches well — so the answer stops depending on what a language model
remembers about deadlocks in general.

This is the same shape as every other observable here: a reading exists, the
turn that needs it never received it, and the model filled the space.
"""

from __future__ import annotations

import re

__all__ = [
    "SELF_SOURCE_HEADER",
    "asks_about_own_implementation",
    "self_source_block",
]

SELF_SOURCE_HEADER = "## YOUR OWN SOURCE, FOR THE THING BEING ASKED ABOUT"

#: Nouns that name a piece of machinery rather than a feeling or an opinion.
_MECHANISM_NOUN = (
    r"(?:system|subsystem|module|engine|gate|gateway|pipeline|lane|loop|"
    r"scheduler|verifier|validator|registry|contract|protocol|policy|"
    r"lock(?:ing|s|dep)?|deadlock|mutex|thread(?:ing)?|queue|cache|index|"
    r"memory\s+system|store|database|ledger|watchdog|monitor|sandbox|"
    r"router|dispatcher|planner|executor|parser|tokenizer|embedding|"
    r"telemetry|instrumentation|invariant|ratchet|baseline|checkpoint|"
    r"architecture|implementation|code|codebase|source)"
)

_ASKS_OWN_MECHANISM_RE = re.compile(
    # "your lock ordering system", "you have a lock ordering system"
    r"\b(?:your|you\s+have\s+(?:a|an|the))\b[^.?!]{0,60}?" + _MECHANISM_NOUN
    # "how do you detect deadlocks", "how do you handle X"
    + r"|\bhow\s+(?:do|does|did|would|can)\s+(?:you|aura)\b[^.?!]{0,40}?"
    r"\b(?:detect|handle|implement|enforce|verify|track|store|schedule|route|"
    r"dispatch|recover|guard|prevent|measure|instrument)\b"
    # "what happens in your X when", "where in your code"
    r"|\bwhere\s+in\s+your\s+(?:code|source|codebase)\b"
    r"|\bwhat\s+(?:module|file|component|class)\s+(?:of\s+yours\s+)?(?:does|handles|owns)\b",
    re.IGNORECASE,
)

#: Feelings, opinions and preferences are about her, and are not source code.
_NOT_A_MECHANISM_RE = re.compile(
    r"\bhow\s+(?:do|are)\s+you\s+(?:feel|feeling|doing|holding\s+up)\b"
    r"|\bwhat\s+do\s+you\s+(?:think|believe|prefer|like|want)\b"
    r"|\byour\s+(?:day|mood|feelings?|opinion|favou?rite)\b",
    re.IGNORECASE,
)


def asks_about_own_implementation(prompt: str) -> bool:
    """True when the turn asks how SHE does something, mechanically."""
    text = str(prompt or "")
    if not text.strip():
        return False
    if _NOT_A_MECHANISM_RE.search(text):
        return False
    return bool(_ASKS_OWN_MECHANISM_RE.search(text))


async def self_source_block(prompt: str) -> str:
    """Spans from her own source for whatever the question names."""
    if not asks_about_own_implementation(prompt):
        return ""
    try:
        from core.brain.evidence_provider import EvidenceProvider

        spans = await EvidenceProvider(memory_facade=None).gather(
            str(prompt or ""), task_type="repo_audit", limit=6
        )
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    repo = [span for span in (spans or []) if getattr(span, "source", "") == "repo"]
    if not repo:
        # A named absence beats an invented mechanism: "I could not find it"
        # is checkable, and "you could use a tool like lock-order" is not.
        return (
            "No file in your own source matched what was asked about. Say that "
            "rather than describing how such a thing is usually built."
        )
    lines = [
        "These are lines from your own source tree, read just now. The answer "
        "to a question about how YOU work is here, not in general practice:"
    ]
    for span in repo[:6]:
        ref = str(getattr(span, "ref", "") or "").strip()
        body = " ".join(str(getattr(span, "text", "") or "").split())[:240]
        if ref:
            lines.append(f"- {ref}: {body}")
    return "\n".join(lines)
