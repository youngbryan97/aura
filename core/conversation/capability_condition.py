"""What she can do *right now*, in a form she can talk about.

Aura's capability engine already knows, per skill, whether it is available,
what state it is in, and why it is not — `_catalog_item_for_skill` computes
exactly that. None of it ever reached the part of her that speaks. So when a
skill was down she fell back to a string somebody wrote months earlier:

    I can't access external data right now, but based on what I know...

Three things are wrong with that. It is not her voice. It is fixed at a
moment that has nothing to do with this moment. And it collapses a
distinction that matters enormously to a person:

    "I can't search, there's no network"      — true now, false in a minute
    "I don't have a way to search at all"     — true until someone builds it

Bryan's framing, and it is the right one: awareness of a failed event is not
a catastrophe, it is communication. What she can do at one minute may not be
what she can do the next; she has to notice, say so in her own words, and
carry on.

This module is the evidence side of that. It reads the live catalog and
answers one question — for the capability this turn needs, what is true right
now — as a compact block the model can speak from. It deliberately produces
FACTS, not sentences: the moment this file starts writing prose, it becomes
the canned reply it exists to replace.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("Aura.CapabilityCondition")

__all__ = [
    "CapabilityStanding",
    "CapabilityCondition",
    "capability_condition_evidence",
    "condition_for",
    "needed_capabilities",
]


class CapabilityStanding(str, Enum):
    """The distinction a person actually cares about."""

    READY = "ready"
    #: Registered, but something about *now* is wrong. Recoverable.
    UNAVAILABLE_NOW = "unavailable_now"
    #: Nothing in the runtime provides this. Not a bad minute — a missing limb.
    ABSENT = "absent"
    #: She HAS the faculty and the world is not currently supplying what it
    #: needs. A person who knows how to search does not forget how when the
    #: wifi drops; they say "there's no internet". This is that state, and it
    #: is derived — capability AND preconditions — never asserted.
    BLOCKED_BY_PRECONDITION = "blocked_by_precondition"


@dataclass(frozen=True)
class CapabilityCondition:
    name: str
    standing: CapabilityStanding
    reason: str = ""
    detail: str = ""

    #: What the world is failing to supply, when that is the reason.
    missing_preconditions: tuple[str, ...] = ()

    @property
    def is_transient(self) -> bool:
        return self.standing in (
            CapabilityStanding.UNAVAILABLE_NOW,
            CapabilityStanding.BLOCKED_BY_PRECONDITION,
        )

    @property
    def faculty_intact(self) -> bool:
        """She has this capability, whatever the world is doing."""
        return self.standing is not CapabilityStanding.ABSENT


#: Which capability a turn is reaching for. Intentionally small: this is used
#: to decide whether to LOOK at the catalog, not to route anything, so a miss
#: costs nothing and a false positive only adds evidence nobody needed.
_CAPABILITY_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("web_search", (
        "search", "look up", "look it up", "google", "on the web", "online",
        "latest", "current price", "weather", "news", "who won", "how much is",
    )),
    ("file_operation", (
        "read the file", "open the file", "write a file", "save it to",
        "in my folder", "on disk", "the directory",
    )),
    ("code_execution", (
        "run this", "execute", "run the code", "calculate with python",
        "run it for real",
    )),
    ("computer_use", (
        "click", "open the app", "take a screenshot", "on my screen",
        "my desktop",
        # Asking what she can ACT on is a capability question with a factual
        # answer — core/perception/element_inventory.py reads the frontmost
        # window's controls. Without this cue the turn was answered from
        # general knowledge about screens instead of from her own screen.
        "what can you click", "what buttons", "what controls",
        "clickable", "interact with the screen",
    )),
    ("email_adapter", ("email", "inbox", "send a message to")),
    # Asking whether she has a mind of her own is a capability question, and
    # it has a factual answer sitting in the initiative record.
    ("initiative", (
        "initiate", "unprompted", "spontaneous", "spontaneously",
        "independent thought", "on your own", "by yourself", "first message",
        "message me first", "start a conversation", "reach out",
    )),
)


def needed_capabilities(user_message: Any) -> tuple[str, ...]:
    """Capabilities this turn plausibly reaches for, in cue order."""
    text = str(user_message or "").casefold()
    if not text:
        return ()
    found: list[str] = []

    # A cue is a WORD, not a substring. "Orca Research" contains "search",
    # so a folder name was pulling web_search evidence into a turn that never
    # mentioned the web. Measured 2026-07-28.
    for name, cues in _CAPABILITY_CUES:
        for cue in cues:
            if re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", text):
                if name not in found:
                    found.append(name)
                break

    # One definition of "this turn needs the desktop".
    #
    # The cue list missed "Open the Notes app and write a new note", so that
    # turn carried NO capability evidence at all — and she answered "I can't
    # actually open apps or write notes", denying a capability she has and
    # was about to use. The router that decides a turn is a desktop objective
    # already knows better; reusing it means these two can never disagree.
    try:
        from core.runtime.desktop_objective_intent import looks_like_desktop_objective

        if looks_like_desktop_objective(user_message):
            for name in ("desktop_task", "computer_use"):
                if name not in found:
                    found.append(name)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Desktop-objective cue unavailable: %s", exc)

    return tuple(found)


def _initiative_evidence() -> str:
    """What her own initiative has actually done, as a fact she can speak.

    She told Bryan "I respond to prompts, not initiate them unprompted... not
    independent thought generation", and produced a JSON block claiming
    {"initiative": false}. The log at that moment carried her own unprompted
    goal updates, and an hour earlier she had opened a conversation with "I
    noticed you went quiet. Everything alright?" — which he received.
    Believing a false thing about herself is the defect; the record is the
    correction.
    """
    try:
        from core.autonomy.proactive_presence import initiative_record

        record = initiative_record()
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Initiative record unavailable: %s", exc)
        return ""

    if record.get("has_spoken_unprompted"):
        count = int(record.get("count", 0))
        gap = record.get("seconds_since_last")
        when = (
            f" the most recent about {int(gap // 60)} minutes ago"
            if isinstance(gap, (int, float)) and gap >= 60
            else " the most recent just now"
            if isinstance(gap, (int, float))
            else ""
        )
        last = str(record.get("last_text") or "").strip()
        quoted = f' It began: "{last[:120]}"' if last else ""
        return (
            f"- initiative: YOU HAVE STARTED {count} CONVERSATION(S) YOURSELF "
            f"this session,{when}.{quoted} You have a proactive-presence "
            f"faculty that speaks from your own goals and state without being "
            f"asked. Do not claim you only respond to prompts — the record "
            f"says otherwise."
        )
    return (
        "- initiative: you have a proactive-presence faculty that can start a "
        "conversation from your own goals and state. It has not fired yet in "
        "this session, which is a fact about this session, not about whether "
        "you have it."
    )


def _catalog_rows(capability_engine: Any) -> Iterable[dict[str, Any]]:
    if capability_engine is None:
        return ()
    for attr in ("iter_tool_catalog", "get_tool_catalog"):
        reader = getattr(capability_engine, attr, None)
        if not callable(reader):
            continue
        try:
            rows = reader(include_inactive=True)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Capability catalog read failed via %s: %s", attr, exc)
            continue
        if rows is None:
            continue
        return rows
    return ()


def condition_for(name: str, *, capability_engine: Any = None) -> CapabilityCondition:
    """The live standing of one capability.

    ABSENT is reserved for "nothing here provides this". A registry that could
    not be read is NOT absence — reporting a missing limb because a lookup
    failed would be a confident lie about herself, so that case reports
    UNAVAILABLE_NOW with the read failure as its reason.
    """
    wanted = str(name or "").strip()
    if not wanted:
        return CapabilityCondition("", CapabilityStanding.ABSENT, "no_capability_named")

    if capability_engine is None:
        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, RuntimeError, AttributeError) as exc:
            logger.debug("Capability engine unavailable: %s", exc)
            capability_engine = None

    if capability_engine is None:
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            "capability_registry_unreadable",
        )

    # The world's side, computed first: a capability whose preconditions are
    # missing cannot work no matter how healthy the registry says it is.
    # Composed here rather than described in a prompt, so unplugging the
    # network changes the conclusion by itself.
    try:
        from core.conversation.capability_preconditions import failing_preconditions

        blocked_by = failing_preconditions(wanted)
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug("Precondition probe unavailable for %s: %s", wanted, exc)
        blocked_by = ()

    seen_any = False
    for row in _catalog_rows(capability_engine):
        if not isinstance(row, dict):
            continue
        seen_any = True
        row_name = str(row.get("name") or row.get("skill") or "").strip()
        if row_name.casefold() != wanted.casefold():
            continue
        if bool(row.get("available")):
            if blocked_by:
                return CapabilityCondition(
                    wanted,
                    CapabilityStanding.BLOCKED_BY_PRECONDITION,
                    "; ".join(state.fact for state in blocked_by),
                    missing_preconditions=tuple(state.name for state in blocked_by),
                )
            return CapabilityCondition(wanted, CapabilityStanding.READY)
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            str(row.get("availability_reason") or row.get("state") or "unavailable"),
            str(row.get("policy_state") or ""),
        )

    if not seen_any:
        return CapabilityCondition(
            wanted,
            CapabilityStanding.UNAVAILABLE_NOW,
            "capability_registry_empty",
        )
    return CapabilityCondition(wanted, CapabilityStanding.ABSENT, "not_registered")


#: Reason codes rendered as the plain fact underneath them. The model reads
#: these and says it however it wants; nothing here is a sentence she must use.
_REASON_FACTS: dict[str, str] = {
    "capability_registry_unreadable": "her own capability registry could not be read this turn",
    "capability_registry_empty": "no capabilities are loaded yet",
    "not_registered": "nothing in this runtime provides it",
    "disabled_by_policy": "it is switched off by policy",
    "inactive_by_policy": "it is switched off by policy",
    "dependency_not_ready": "something it depends on has not finished loading",
    "network_unavailable": "there is no network right now",
    "offline": "there is no network right now",
    "memory_pressure": "memory is too tight to load it right now",
    "quarantined": "it was quarantined after failing",
    "ERROR": "it errored the last time it ran",
}


def _fact_for(condition: CapabilityCondition) -> str:
    raw = str(condition.reason or "").strip()
    for key, fact in _REASON_FACTS.items():
        if key.casefold() in raw.casefold():
            return fact
    return raw.replace("_", " ") or "it is not available"


def capability_condition_evidence(
    user_message: Any,
    *,
    capability_engine: Any = None,
    already_used: Iterable[str] = (),
) -> str:
    """A block of live capability facts, or "" when the turn needs none.

    Facts only. The instruction to speak in her own words lives with the other
    surface contracts; if this function ever returns a ready-made apology, the
    canned reply has simply moved house.
    """
    wanted = needed_capabilities(user_message)
    if not wanted:
        return ""

    # A capability that already produced evidence THIS TURN is working, full
    # stop. The registry can say whatever it likes; the turn has a receipt.
    # Without this, a search that just succeeded could still be announced as
    # "not available this moment" — the same contradiction as reporting a
    # scan blocked while its results sit in the prompt.
    proven = {str(name or "").casefold() for name in already_used if str(name or "").strip()}

    lines: list[str] = []
    for name in wanted:
        if name.casefold() in proven:
            continue
        if name == "initiative":
            initiative_line = _initiative_evidence()
            if initiative_line:
                lines.append(initiative_line)
            continue
        condition = condition_for(name, capability_engine=capability_engine)
        if condition.standing is CapabilityStanding.READY:
            lines.append(f"- {name}: available right now")
        elif condition.standing is CapabilityStanding.BLOCKED_BY_PRECONDITION:
            lines.append(
                f"- {name}: YOU HAVE THIS, BUT IT CANNOT WORK RIGHT NOW — "
                f"{condition.reason}. The capability is intact; what it needs "
                f"is missing. Reason from that: it will work again when that "
                f"comes back."
            )
        elif condition.standing is CapabilityStanding.ABSENT:
            lines.append(
                f"- {name}: NOT SOMETHING YOU HAVE — {_fact_for(condition)}. "
                f"This is not a temporary outage."
            )
        else:
            lines.append(
                f"- {name}: NOT AVAILABLE THIS MOMENT — {_fact_for(condition)}. "
                f"You do have this capability; it may work again shortly."
            )
    if not lines:
        return ""

    return (
        "[LIVE CAPABILITY CONDITION]\n"
        + "\n".join(lines)
        + "\n"
        + "Say this in your own words, as part of your answer, the way you "
        "would mention any other fact about your situation. Do not apologise "
        "at length and do not treat it as a failure — it is information. Keep "
        "'not right now' and 'not something I can do' clearly different, and "
        "still answer whatever you can answer without it.\n"
        "[END LIVE CAPABILITY CONDITION]"
    )
