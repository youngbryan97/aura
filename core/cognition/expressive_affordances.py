"""Expressive affordances — the mind's menu of ways to act, not scripts.

Bryan's question: does Aura KNOW she can use her mind — that when an idea is
better shown than said, she can generate an image; that "I need a table"
means clarify-then-demonstrate on the real machine; that she can ask for a
photo, model out scenarios, or examine a file deeply — and does she CHOOSE
these from general cognition rather than hardcoded trigger paths?

The architecture answer is not a router of if/elif keyword matches. It is:

  1. a typed registry of AFFORDANCES (capabilities framed as things the mind
     can decide to do), each with a plain-language "when you'd reach for this"
     description and a governed realizer;
  2. a compact menu injected into the generation context so the model reasons
     WITH its own affordances present, the way a person knows their own hands
     are available;
  3. an intent grammar — the model emits ``⟦affordance:name arg=...⟧`` inline
     when its own judgment says an action would serve the moment;
  4. a parser + governed executor that realizes the chosen affordance and
     attaches the result to the reply.

This keeps the DECISION in general cognition (the model chooses, by context
and inference) while the MECHANISM stays a clean, extensible, governed layer.
Adding a new affordance is one registry entry; no new routing code.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.ExpressiveAffordances")

# The intent grammar the model emits when it decides to act. Deliberately
# distinctive so it never collides with ordinary prose or code fences.
_INTENT_RE = re.compile(
    r"⟦affordance:(?P<name>[a-z_]+)(?P<args>(?:\s+[a-z_]+=(?:\"[^\"]*\"|[^\s⟧]+))*)\s*⟧",
    re.IGNORECASE,
)
_ARG_RE = re.compile(r"([a-z_]+)=(\"[^\"]*\"|[^\s⟧]+)", re.IGNORECASE)
# Execution is strict; visibility is conservative. These patterns also catch
# unknown names, alternate brackets, and partial controls that cannot execute
# but still must never become user-facing prose.
_CONTROL_SPAN_RE = re.compile(
    r"(?:"
    r"⟦\s*affordance\s*:[^⟧]{0,4096}⟧"
    r"|\[\[\s*affordance\s*:[^\]]{0,4096}\]\]"
    r"|<<\s*affordance\s*:[^>]{0,4096}>>"
    r"|\[\s*affordance\s*:[^\]]{0,4096}\]"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_CONTROL_HEAD_RE = re.compile(
    r"(?:⟦|\[\[|<<|\[)\s*affordance\s*:\s*[a-z_][a-z0-9_-]*"
    r"(?:\s+[a-z_][a-z0-9_]*=(?:\"[^\"]*\"|'[^']*'|[^\s⟧\]>]+))*"
    r"\s*(?:⟧|\]\]|>>|\])?",
    re.IGNORECASE,
)
_CONTROL_START_RE = re.compile(
    r"(?:⟦|\[\[|<<|\[)\s*affordance\s*:",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AffordanceIntent:
    """A single action the mind chose to take, parsed from its own output."""

    name: str
    args: dict[str, str]
    raw: str


@dataclass(frozen=True)
class AffordanceSanitization:
    """Result of enforcing the private-control/user-prose boundary."""

    text: str
    changed: bool
    removed_controls: int
    malformed_controls: int


@dataclass
class Affordance:
    """A capability framed as a decision the mind can make.

    ``when`` is written in the second person and describes the JUDGMENT, not a
    keyword — it is the text the model reads to know this option exists. The
    ``realize`` coroutine performs the governed action and returns a result
    dict; it must be fail-open (never raise) and cheap to skip.
    """

    name: str
    when: str
    realize: Callable[[dict[str, str], dict[str, Any]], Awaitable[dict[str, Any]]]
    args_hint: str = ""
    metabolic_cost: int = 1


@dataclass
class AffordanceRegistry:
    """Aura's live menu of expressive actions."""

    _affordances: dict[str, Affordance] = field(default_factory=dict)

    def register(self, affordance: Affordance) -> None:
        self._affordances[affordance.name] = affordance

    def names(self) -> list[str]:
        return sorted(self._affordances)

    def get(self, name: str) -> Affordance | None:
        return self._affordances.get(name)

    def menu_text(self) -> str:
        """The block injected into generation context.

        Phrased as self-knowledge — 'these are things you can decide to do' —
        so the model reasons with its affordances present, the way a person
        knows their own capabilities are always on the table.
        """
        if not self._affordances:
            return ""
        lines = [
            "[Your expressive affordances — you decide, by context and your own "
            "judgment, when any of these would serve the moment better than words "
            "alone. To use one, emit its tag inline where it belongs; it will be "
            "realized on this machine and its result attached to your reply. Use "
            "them the way a person knows their own hands are available — never "
            "gratuitously, but never forgetting you have them.]",
        ]
        for name in self.names():
            aff = self._affordances[name]
            tag = f"⟦affordance:{name}{(' ' + aff.args_hint) if aff.args_hint else ''}⟧"
            lines.append(f"  • {tag} — {aff.when}")
        lines.append(
            "  Example — user: 'I can't name this piece of furniture, it's low "
            "with little drawers'  →  you: 'Sounds like a chest of drawers or a "
            "credenza. \u27e6affordance:show_sketch prompt=\"a low wooden credenza "
            "with small drawers\"\u27e7 Does that match?'  (emit the tag exactly, "
            "inline, and keep speaking naturally around it.)"
        )
        return "\n".join(lines) + "\n"

    def parse_intents(self, text: str) -> list[AffordanceIntent]:
        """Extract the affordance intents the model emitted in its output."""
        intents: list[AffordanceIntent] = []
        for match in _INTENT_RE.finditer(text or ""):
            name = match.group("name").lower()
            if name not in self._affordances:
                continue
            args = {
                k.lower(): v.strip('"')
                for k, v in _ARG_RE.findall(match.group("args") or "")
            }
            intents.append(AffordanceIntent(name=name, args=args, raw=match.group(0)))
        return intents

    @staticmethod
    def strip_intents(text: str) -> str:
        """Remove intent tags from user-visible prose (the result speaks for them)."""
        return sanitize_affordance_control_syntax(text).text

    async def realize(
        self, intent: AffordanceIntent, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Run one chosen affordance through its governed realizer (fail-open)."""
        aff = self._affordances.get(intent.name)
        if aff is None:
            return {"ok": False, "affordance": intent.name, "reason": "unknown_affordance"}
        try:
            result = await aff.realize(intent.args, context or {})
            if not isinstance(result, dict):
                result = {"ok": bool(result), "value": result}
            result.setdefault("ok", True)
            result["affordance"] = intent.name
            return result
        except (
            RuntimeError, AttributeError, TypeError, ValueError, KeyError,
            OSError, ImportError,
        ) as exc:  # fail-open by contract — never break the reply
            from core.runtime.errors import record_degradation

            record_degradation(
                "expressive_affordances",
                exc,
                severity="warning",
                action=f"skipped affordance '{intent.name}' after realizer error",
            )
            return {"ok": False, "affordance": intent.name, "reason": f"error:{type(exc).__name__}"}


def sanitize_affordance_control_syntax(text: str) -> AffordanceSanitization:
    """Strip complete and malformed affordance controls from visible prose.

    Complete spans are removed first. A malformed control then loses only its
    control head and parseable key/value arguments, preserving surrounding
    natural language whenever possible. The final start-marker pass guarantees
    that no executable-looking prefix survives.
    """

    original = str(text or "")
    complete = list(_CONTROL_SPAN_RE.finditer(original))
    cleaned = _CONTROL_SPAN_RE.sub("", original)
    partial = list(_CONTROL_HEAD_RE.finditer(cleaned))
    cleaned = _CONTROL_HEAD_RE.sub("", cleaned)
    residual = list(_CONTROL_START_RE.finditer(cleaned))
    cleaned = _CONTROL_START_RE.sub("", cleaned)
    if complete or partial or residual:
        from core.utils.text_after_removal import closed_up

        # The gap a removed control left is closed inside its line; indentation and fenced code stay as written.
        cleaned = closed_up(cleaned.replace("⟦", "").replace("⟧", ""))
    cleaned = cleaned.strip()
    removed = len(complete) + len(partial) + len(residual)
    return AffordanceSanitization(
        text=cleaned,
        changed=cleaned != original.strip(),
        removed_controls=removed,
        malformed_controls=len(partial) + len(residual),
    )


_REGISTRY: AffordanceRegistry | None = None


def get_affordance_registry() -> AffordanceRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AffordanceRegistry()
        _install_default_affordances(_REGISTRY)
    return _REGISTRY


def _install_default_affordances(registry: AffordanceRegistry) -> None:
    """Register the built-in affordances. Each realizer is fail-open and
    delegates to an existing governed subsystem — this layer only decides and
    wires; it never reimplements a capability."""
    from core.cognition import affordance_realizers

    registry.register(
        Affordance(
            name="show_sketch",
            when=(
                "an idea is clearer shown than described, or you're approximating "
                "something the user is reaching for — generate an image that says "
                "'does it look like this?'"
            ),
            args_hint='prompt="what to depict"',
            realize=affordance_realizers.realize_show_sketch,
            metabolic_cost=3,
        )
    )
    registry.register(
        Affordance(
            name="demonstrate_artifact",
            when=(
                "the user needs a concrete thing made (a table, a document, a "
                "small program) — build a real example on this machine and show "
                "it, saying 'something like this?'"
            ),
            args_hint='kind="table|doc|program" spec="what it should contain"',
            realize=affordance_realizers.realize_demonstrate_artifact,
            metabolic_cost=2,
        )
    )
    registry.register(
        Affordance(
            name="request_media",
            when=(
                "you'd understand far better if you could see or hear the thing — "
                "ask the user to share an image, video, file, or link, saying "
                "specifically what would help and why"
            ),
            args_hint='need="what to share"',
            realize=affordance_realizers.realize_request_media,
            metabolic_cost=1,
        )
    )
    registry.register(
        Affordance(
            name="model_scenarios",
            when=(
                "a choice or plan has real branches — model the options out, "
                "compare outcomes, and commit to the one you actually judge best "
                "with your reasoning"
            ),
            args_hint='options="A vs B vs ..."',
            realize=affordance_realizers.realize_model_scenarios,
            metabolic_cost=2,
        )
    )
    registry.register(
        Affordance(
            name="deep_examine",
            when=(
                "the user shared a file or image and deserves genuine considered "
                "feedback — examine it closely and react to what is actually there, "
                "not a skim or a summary"
            ),
            args_hint='target="path or reference"',
            realize=affordance_realizers.realize_deep_examine,
            metabolic_cost=2,
        )
    )
