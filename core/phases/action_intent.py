"""User action-intent detection.

Closes half of the disconnect the user observed: Aura defers or talks
about actions even when the user has explicitly asked for them and
explicitly granted permission. This module detects those signals from
the user's own text and surfaces a ``user_granted_permission`` flag that
the Will, the skill router, and the inference gate can consume.

This is intentionally lenient. The goal is to bias toward action when
the user has clearly said "yes, do it". Safety gates upstream (output
guardrails, ontological boundary, skill-level permission checks) are
still in force.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from core.runtime.skill_task_bridge import strip_negated_action_spans


_ACTION_VERBS = (
    "open", "launch", "run", "start", "execute", "click", "type",
    "write", "search", "look up", "find", "close", "quit", "switch",
    "read", "check", "show", "display", "visit", "browse", "google",
    "navigate", "download", "save", "export", "copy", "paste", "move",
    "delete", "create", "make", "build", "send",
    # LIVE, 2026-08-10: "Put the text ORION-7 on my clipboard" was not an
    # action request here, while "copy that to my clipboard" was — so nothing
    # ran and she reported the clipboard set when it was empty.
    #
    # This is the third module with its own verb enumeration for the same
    # question, and the three disagreed. Until they are one list, the least
    # that has to hold is that an everyday word for doing a thing is in all of
    # them.
    "put", "place", "set", "add", "rename", "empty", "clear",
)

_PERMISSION_PHRASES = (
    r"\bdo it\b",
    r"\bgo ahead\b",
    r"\byou can\b",
    r"\byou have\b",
    r"\bi trust you\b",
    r"\bi('|\s+a)m giving you permission\b",
    r"\bi give you permission\b",
    r"\byou have permission\b",
    r"\bpermission granted\b",
    r"\byes,? do it\b",
    r"\bplease do\b",
    r"\bjust do it\b",
    r"\bactually do it\b",
    r"\btry it\b",
    r"\btry again\b",
    r"\bgo for it\b",
)


@dataclass(frozen=True)
class ActionIntent:
    has_action_request: bool
    has_permission_grant: bool
    verb: Optional[str]
    target: Optional[str]
    raw_excerpt: str

    @property
    def should_execute(self) -> bool:
        """True when we should skip deferral and try the real skill."""
        return self.has_action_request and self.has_permission_grant

    def as_dict(self) -> Dict[str, object]:
        return {
            "has_action_request": self.has_action_request,
            "has_permission_grant": self.has_permission_grant,
            "verb": self.verb,
            "target": self.target,
            "raw_excerpt": self.raw_excerpt,
            "should_execute": self.should_execute,
        }


_ACTION_VERB_RE = re.compile(
    r"\b(" + "|".join(_ACTION_VERBS) + r")\b[\s,:.;]+(?P<target>[A-Za-z0-9 \"'/\\._-]{2,80})",
    re.IGNORECASE,
)

_PERMISSION_RE = re.compile("|".join(_PERMISSION_PHRASES), re.IGNORECASE)

_MESSAGE_INITIAL_IMPERATIVE_RE = re.compile(
    r"^(please\s+)?(open|run|click|type|write|execute|launch|search|show|visit|"
    r"navigate|download|save|create|make|build|send|copy|paste|move|delete|"
    r"put|place|set|add|rename|empty|clear)\b",
    re.IGNORECASE,
)

# Clause boundaries a real message uses before getting to the ask: sentence
# enders, colons, semicolons, dashes, newlines, and "then"/"and then".
_CLAUSE_SPLIT_RE = re.compile(
    r"(?:[.!?;:\n]+|\s+—\s+|\s+-\s+|\s*\band then\b\s*|\s*\bthen\b\s*)",
    re.IGNORECASE,
)

# Narrower than the message-initial set on purpose: no "show"/"make", so a
# conversational "…, show me your reasoning" is not an execution request.
_CLAUSE_INITIAL_IMPERATIVE_RE = re.compile(
    r"^(please\s+)?(open|run|execute|launch|click|type|paste|write|save|export|"
    r"download|navigate|visit|search|send|build|install|compile)\b",
    re.IGNORECASE,
)


def detect_action_intent(text: str) -> ActionIntent:
    raw = str(text or "").strip()
    if not raw:
        return ActionIntent(False, False, None, None, "")

    lowered = strip_negated_action_spans(raw).lower()
    verb_match = _ACTION_VERB_RE.search(lowered)
    has_action = verb_match is not None
    verb = verb_match.group(1).lower() if verb_match else None
    target = (verb_match.group("target") if verb_match else "").strip(".,;:\"' ") or None

    has_permission = bool(_PERMISSION_RE.search(lowered))

    # A blunt imperative without a separate permission phrase also counts
    # as permission. "Open Notes and type X" is already the user asking
    # for the action; we should not require them to ALSO say "I trust you".
    if has_action and not has_permission:
        imperative = bool(_MESSAGE_INITIAL_IMPERATIVE_RE.match(lowered))
        # ...and an imperative after a preamble is still an imperative. The
        # message-initial anchor above meant ANY preamble defeated it, which is
        # exactly how people actually write. Measured live:
        #
        #   "Hey Aura, it's Bryan. Hold onto the codeword LANTERN for later.
        #    First real task: run a Python snippet ... and give me the two
        #    actual numbers it returned."
        #
        # `run_code`, `code_repl` and `internal_sandbox` were all READY, and
        # nothing dispatched: no permission was detected, so should_execute
        # stayed False and the turn never reached an executor. The codebase had
        # already learned this lesson once for the search-query extractor —
        # "every pattern below is anchored with .match(), so a preamble defeats
        # all of them".
        #
        # The clause-level verb set is deliberately narrower than the
        # message-initial one: it excludes conversational verbs like "show" and
        # "make" so "…, show me your reasoning" does not read as an execution
        # request just because it follows a comma.
        if not imperative:
            imperative = any(
                _CLAUSE_INITIAL_IMPERATIVE_RE.match(clause.strip())
                for clause in _CLAUSE_SPLIT_RE.split(lowered)
                if clause and clause.strip()
            )
        if imperative:
            has_permission = True

    excerpt = raw[:200]
    return ActionIntent(
        has_action_request=has_action,
        has_permission_grant=has_permission,
        verb=verb,
        target=target,
        raw_excerpt=excerpt,
    )


def apply_intent_to_context(text: str, context: Dict[str, object]) -> ActionIntent:
    """Stamp the detected intent onto a mutable context dict.

    Callers that want the Will / skill router / inference gate to
    consume the intent pass the same ``context`` object around.
    """
    intent = detect_action_intent(text)
    if intent.has_action_request:
        context["user_explicit_action_request"] = True
    if intent.has_permission_grant:
        context["user_granted_permission"] = True
    if intent.should_execute:
        context["user_requested_action"] = True
        context["action_intent"] = intent.as_dict()
    return intent
