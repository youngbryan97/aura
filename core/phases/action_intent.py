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


_ACTION_VERBS = (
    "open", "launch", "run", "start", "execute", "click", "type",
    "write", "search", "look up", "find", "close", "quit", "switch",
    "read", "check", "show", "display", "visit", "browse", "google",
    "navigate", "download", "save", "export", "copy", "paste", "move",
    "delete", "create", "make", "build", "send",
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


def detect_action_intent(text: str) -> ActionIntent:
    raw = str(text or "").strip()
    if not raw:
        return ActionIntent(False, False, None, None, "")

    lowered = raw.lower()
    verb_match = _ACTION_VERB_RE.search(lowered)
    has_action = verb_match is not None
    verb = verb_match.group(1).lower() if verb_match else None
    target = (verb_match.group("target") if verb_match else "").strip(".,;:\"' ") or None

    has_permission = bool(_PERMISSION_RE.search(lowered))

    # A blunt imperative without a separate permission phrase also counts
    # as permission. "Open Notes and type X" is already the user asking
    # for the action; we should not require them to ALSO say "I trust you".
    if has_action and not has_permission:
        imperative = bool(re.match(r"^(please\s+)?(open|run|click|type|write|execute|launch|search|show|visit|navigate|download|save|create|make|build|send|copy|paste|move|delete)\b", lowered))
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
