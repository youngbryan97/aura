"""Her operational self context, each part of it where the turn bears on it.

The block carries what she is, her capacities, the capabilities she has now, and her live internals (mood, runtime
state, memory in use). Until 11 November 2026 all of it was put in front of every message, with two sections of
"binding" rules on how she acts and how she speaks of herself; those were instructions in her turns, and Bryan said
the same day to take them out. Asked "Hey Aura, can you hear me okay?" in voice mode, she answered with her memory level and
thermal state; asked "What's two plus two?", she read the whole block before saying "Four." Bryan's word: only when it
bears. Each part goes in when the turn bears on it by meaning (core/cognition/evidence_relevance.py, the measure her
history reach uses), and the whole block goes in when the turn cannot be judged: no message, or no encoder to judge it
with. A turn that needed her capabilities is never left without them.

Lifted out of core/conversation/chat_preflight.py, which imports it back; what it uses from that module it imports at
call time.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def what_of_herself_bears(objective: str, parts: dict[str, str]) -> set[str] | None:
    """The names of the parts the turn bears on, by meaning; None when it cannot be judged."""
    request = " ".join(str(objective or "").split())
    if not request or not parts:
        return None
    from core.cognition.evidence_relevance import assess_evidence_alignments

    names = list(parts)
    verdicts = assess_evidence_alignments(request, [parts[name] for name in names])
    if not verdicts or not all(verdict.measured for verdict in verdicts):
        return None
    return {name for name, verdict in zip(names, verdicts, strict=True) if verdict.relevant}


def _joined(heading: str, items: Iterable[str]) -> str:
    return " ".join([heading, *items])


async def inject_operational_self_context(objective: str = "") -> str:
    """Inject the identity contract: live, truthful self context for the voice.

    This block is how the substrate and the voice stay one entity. It carries
    (1) substrate facts, (2) the verified capability inventory plus the live
    skill registry, (3) live internals — all evidence-bounded: runtime
    telemetry, not proof of private qualia, literal personhood, or proven
    consciousness. Each goes in where the turn bears on it (see the module's
    docstring). It says what is so; it does not tell her what to do.

    Returns formatted operational-self context block if available, empty string otherwise.
    """
    from core.conversation.chat_preflight import (
        _ARCHITECTURAL_CAPACITIES,
        _CHAT_PREFLIGHT_RECOVERABLE_ERRORS,
        _SUBSTRATE_FACTS,
        MAX_OPERATIONAL_SELF_CONTEXT_CHARS,
        _emit_chat_fault,
        _live_capability_affordance_block,
        _live_internals_summary,
        _reference_corpus_summary,
        _safe_truncated_text,
    )

    try:
        from core.consciousness.unified_self import get_unified_self

        unified_self = await get_unified_self()
        self_state = unified_self.get_state()
        internals = _live_internals_summary()
        state_lines = [
            f"Name: {self_state.name}",
            f"Runtime state: {self_state.current_state.value}",
            f"Mood: {self_state.current_mood}",
            f"Functional agency signal: {self_state.sense_of_agency:.0%}",
            f"Functional presence signal: {self_state.sense_of_presence:.0%}",
            f"Continuity: {self_state.continuity:.0%}",
        ]
        bears = what_of_herself_bears(objective, {
            "state": _joined("How I am right now, my live internals:", [*state_lines, *internals]),
            "what_i_am": _joined("What I am:", _SUBSTRATE_FACTS),
            "capacities": _joined("Architectural capacities:", _ARCHITECTURAL_CAPACITIES),
            "what_i_can_do_now": "What I can do right now: the capabilities and tools I have available this turn, "
                                 "and acting with them on what is asked.",
        })

        def wanted(part: str) -> bool:
            return bears is None or part in bears

        reference = _reference_corpus_summary(objective)
        if bears is not None and not bears and not reference:
            return ""

        lines = ["[Operational Self Context]"]
        if wanted("state"):
            lines += [*state_lines, "Evidence boundary: this is live runtime telemetry, "
                                    "not proof of private qualia, literal personhood, or proven consciousness."]
        if wanted("what_i_am"):
            lines += ["", "What I am:", *(f"  • {fact}" for fact in _SUBSTRATE_FACTS)]
        if wanted("capacities"):
            lines += ["", "Architectural capacities (current availability is listed separately):",
                      *(f"  • {capability}" for capability in _ARCHITECTURAL_CAPACITIES)]
        if wanted("what_i_can_do_now"):
            affordance_block = _live_capability_affordance_block(objective)
            lines += ["", "Current actionable affordances:"]
            if affordance_block:
                lines.extend(f"  {line}" for line in affordance_block.splitlines())
            else:
                lines.append("  The live capability catalog is unavailable for this turn.")

        if reference:
            lines.append("")
            lines.append(
                "From my offline reference corpus (these are the passages I actually "
                "hold; cite this corpus only for what appears here):"
            )
            lines.extend(f"  • {item}" for item in reference)

        if wanted("state") and internals:
            lines += ["", "Current internals (live):", *(f"  • {item}" for item in internals)]

        if wanted("what_i_am") and self_state.identity_commitments:
            lines += ["", "My commitments in this conversation:",
                      *(f"  • {commitment}" for commitment in self_state.identity_commitments[:3])]

        lines.append("[End operational self context]")

        return _safe_truncated_text(
            "\n".join(lines) + "\n\n",
            max_chars=MAX_OPERATIONAL_SELF_CONTEXT_CHARS,
        )

    except _CHAT_PREFLIGHT_RECOVERABLE_ERRORS as exc:
        _emit_chat_fault(
            exc,
            action="continued without operational self context",
            severity="degraded",
            stage="operational_self_context.inject",
        )
        return ""


__all__: list[Any] = ["inject_operational_self_context", "what_of_herself_bears"]
