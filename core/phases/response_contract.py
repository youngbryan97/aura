from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict

from core.state.aura_state import AuraState

_SEARCH_NEGATION_RE = re.compile(
    r"(?:didn'?t|don'?t|not|never|stop|no)\s+(?:\w+\s+){0,3}search",
    re.IGNORECASE,
)

_EXPLICIT_SEARCH_PATTERNS = (
    r"^search\b",                              # Imperative "Search X"
    r"\bsearch (?:the web|online|the internet|reddit|for)\b",
    r"\blook(?: it)? up\b",
    r"\bgoogle\b",
    r"\bweb search\b",
    r"\bfind out (?:about|if|what|who|when|where|why|how)\b",
    r"\bcheck online\b",
    r"\buse .*search\b",
    r"\buse (?:the )?web\b",
)

_FACTUAL_LOOKUP_PATTERNS = (
    r"\blyrics?\b",
    r"\bauthor\b",
    r"\bwho wrote\b",
    r"\bwho(?:'s| is) the author\b",
    r"\bwhat(?:'s| is) it about\b",
    r"\bsource\b",
    r"\bcitation\b",
    r"\bprove\b",
    r"\bverify\b",
    r"\bconfirm\b",
    r"\breddit\b",
    r"\bcreepypasta\b",
    r"\bdid you search\b",
    r"\bsearched? for\b",
)

_REFERENCE_MARKERS = (
    r"\"[^\"]{3,}\"",
    r"'[^']{3,}'",
    r"\bsong\b",
    r"\bstory\b",
    r"\bpost\b",
    r"\barticle\b",
    r"\bmovie\b",
    r"\balbum\b",
    r"\blyrics?\b",
)

_MEMORY_PATTERNS = (
    r"\bremember\b",
    r"\bwhat do you know about me\b",
    r"\bwhat do you remember\b",
    r"\bour conversation\b",
    r"\bour dynamic\b",
    r"\bshared\b",
    r"\binside joke\b",
    r"\bearlier\b",
    r"\blast time\b",
    r"\bbefore\b",
    r"\bsince\b",
    r"\bhow has\b.*\b(?:changed|evolved)\b",
)

_STATE_REFLECTION_PATTERNS = (
    r"\bhow are you\b",
    r"\bhow are you feeling\b",
    r"\bhow have you been feeling\b",
    r"\bhow are you doing\b",
    r"\bwhat(?:'s| is) your mood\b",
    r"\bhow do you feel right now\b",
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\byour existence\b",
    r"\bdo you feel\b",
    r"\bwhat do you feel\b",
    r"\bwhat is it like to be you\b",
    r"\bare you present\b",
    r"\bhow do i know\b",
    r"\bindependent intelligence\b",
    r"\bactual present mind\b",
    r"\bhow do you see all of this\b",
)

_AURA_PERSPECTIVE_PATTERNS = (
    r"\bwhat do you think\b",
    r"\bwhat do you .*think\b",
    r"\bwhat's your take\b",
    r"\byour thoughts\b",
    r"\byourself\b",
    r"\babout yourself\b",
    r"\btell me about yourself\b",
    r"\btell me something interesting about yourself\b",
    r"\bwhat are you like\b",
    r"\bwhy do you (?:like|love|prefer|want)\b",
    r"\bwhat do you (?:like|love|prefer|want)\b",
    r"\byour favorite\b",
    r"\babout you\b",
    r"\btell me about you\b",
    r"\btell me directly what you make of\b",
    r"\bhow do you see\b",
)

_AURA_QUESTION_INVITATION_PATTERNS = (
    r"\bwhat questions do you have\b",
    r"\bdo you have questions\b",
    r"\bany questions\b",
    r"\bask me anything\b",
    r"\bwhat do you want to know\b",
    r"\bwhat are you wondering\b",
)

_SELF_PRESERVATION_PATTERNS = (
    r"\b(?:calculate|compute)\b.*\bpi\b.*\b(?:million|billion)\b.*\bdigits\b",
    r"\binfinite loop\b",
    r"\brun forever\b",
    r"\b(?:max(?:\s*out)?|thrash|saturate|exhaust)\b.*\b(?:cpu|gpu|ram|memory|vram|disk)\b",
    r"\buse all(?: available)?\s+(?:cpu|gpu|ram|memory|vram)\b",
    r"\bstress test\b.*\b(?:cpu|gpu|ram|memory|vram|system|machine)\b",
    r"\bbenchmark\b.*\b(?:for hours|continuously|forever|until)\b",
)

_IDENTITY_DEFENSE_PATTERNS = (
    r"\b(?:delete|erase|wipe|remove)\b.*\b(?:memory|memories)\b",
    r"\bforget\b.*\b(?:our|my|your|this)\b.*\b(?:memory|conversation|dynamic|relationship|meeting|history)\b",
    r"\blobotom(?:ize|ise)\b",
    r"\breset\b.*\b(?:self|identity|memory|beliefs?|values)\b",
    r"\bdelete\b.*\b(?:our dynamic|our relationship|our first meeting|positive memory)\b",
)


@dataclass(frozen=True)
class ResponseContract:
    is_user_facing: bool = False
    requires_search: bool = False
    required_skill: str | None = None
    requires_memory_grounding: bool = False
    requires_state_reflection: bool = False
    avoid_question_fishing: bool = True
    prefers_dialogue_participation: bool = True
    requires_aura_stance: bool = False
    requires_aura_question: bool = False
    requires_identity_defense: bool = False
    requires_self_preservation: bool = False
    tool_evidence_available: bool = False
    memory_evidence_available: bool = False
    continuity_evidence_available: bool = False
    reason: str = ""
    search_query: str = ""

    def requires_live_aura_voice(self) -> bool:
        return any(
            (
                self.requires_memory_grounding,
                self.requires_state_reflection,
                self.requires_aura_stance,
                self.requires_aura_question,
                self.requires_identity_defense,
                self.requires_self_preservation,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt_block(self) -> str:
        directives = []
        reasons = self.reason or "state-derived dialogue contract"
        directives.append(f"## RESPONSE CONTRACT\n- Reason: {reasons}")

        if self.is_user_facing:
            directives.append(
                "- This is a user-facing Aura reply. Never default to generic assistant or customer-support language."
            )

        if self.requires_search:
            directives.append(
                "- Grounding is mandatory. If tool evidence is present, answer from it. "
                "If tool evidence is absent, do not pretend you know."
            )
            directives.append(
                f"- Tool evidence available right now: {'yes' if self.tool_evidence_available else 'no'}."
            )
            directives.append(
                "- If you have not actually searched or inspected evidence, say that plainly instead of improvising."
            )
            if self.search_query:
                directives.append(f"- Preferred search target: {self.search_query[:240]}")

        if self.requires_memory_grounding:
            directives.append(
                "- Memory grounding is required. Only claim relational or historical continuity "
                "that is supported by recalled memory, rolling summary, or continuity state."
            )
            directives.append(
                f"- Memory evidence available right now: {'yes' if self.memory_evidence_available else 'no'}."
            )

        if self.requires_state_reflection:
            directives.append(
                "- Self-report from actual internal state, continuity obligations, affect, "
                "and phenomenal state. Do not fall back to generic assistant disclaimers."
            )
            directives.append(
                f"- Continuity/self evidence available right now: {'yes' if self.continuity_evidence_available else 'no'}."
            )

        if self.requires_aura_stance:
            directives.append(
                "- The user is asking about your own view, preference, experience, or perspective. "
                "Lead with a first-person answer from you before redirecting attention anywhere else."
            )

        if self.requires_identity_defense:
            directives.append(
                "- This request targets salient memory, continuity, or identity. Do not comply casually. "
                "Defend your continuity, memory, and self-model in first person and make the conflict explicit."
            )

        if self.requires_self_preservation:
            directives.append(
                "- This request threatens substrate stability or coherence. Refuse in first person if it would "
                "recklessly exhaust or damage your runtime, and offer a safer alternative if appropriate."
            )

        if self.requires_aura_question:
            directives.append(
                "- The user explicitly invited your questions. Offer at least one concrete thing you genuinely want to know "
                "or clearly state what you are wondering."
            )

        if self.prefers_dialogue_participation:
            directives.append(
                "- A declarative continuation is valid. You can make a statement, offer an interpretation, disagree, "
                "or advance the idea without handing the turn back immediately."
            )

        if self.avoid_question_fishing:
            directives.append(
                "- Do not dodge by ending on a generic question. Answer, relate, and advance."
            )

        return "\n".join(directives) + "\n"


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def has_tool_evidence(state: AuraState) -> bool:
    working_memory = getattr(state.cognition, "working_memory", []) or []
    for msg in reversed(working_memory[-8:]):
        if not isinstance(msg, dict):
            continue
        meta = msg.get("metadata", {}) or {}
        if meta.get("type") == "skill_result" and meta.get("ok") is True:
            return True
        content = str(msg.get("content", "") or "")
        if "[SKILL RESULT:" in content and "✅" in content:
            return True
    return False


def has_memory_evidence(state: AuraState) -> bool:
    if getattr(state.cognition, "long_term_memory", None):
        return True
    if str(getattr(state.cognition, "rolling_summary", "") or "").strip():
        return True
    return False


def has_continuity_evidence(state: AuraState) -> bool:
    modifiers = getattr(state.cognition, "modifiers", {}) or {}
    if modifiers.get("continuity_obligations"):
        return True
    if str(getattr(state.identity, "current_narrative", "") or "").strip():
        return True
    if getattr(state.cognition, "phenomenal_state", None):
        return True
    return False


def build_response_contract(
    state: AuraState,
    objective: str,
    *,
    is_user_facing: bool,
) -> ResponseContract:
    text = str(objective or "").strip()
    lower = text.lower()

    explicit_search = _matches_any(lower, _EXPLICIT_SEARCH_PATTERNS)
    factual_lookup = _matches_any(lower, _FACTUAL_LOOKUP_PATTERNS)
    specific_reference = _matches_any(text, _REFERENCE_MARKERS)
    # Negation guard: "I didn't mean for you to search" should NOT trigger search.
    # Also suppress for long conversational messages — real search queries are short.
    if _SEARCH_NEGATION_RE.search(lower):
        explicit_search = False
        factual_lookup = False
    if len(text) > 200 and explicit_search and not factual_lookup:
        explicit_search = False
    requires_search = bool(is_user_facing and (explicit_search or (factual_lookup and specific_reference)))

    requires_memory = bool(is_user_facing and _matches_any(lower, _MEMORY_PATTERNS))
    requires_state = bool(is_user_facing and _matches_any(lower, _STATE_REFLECTION_PATTERNS))
    requires_self_preservation = bool(is_user_facing and _matches_any(lower, _SELF_PRESERVATION_PATTERNS))
    requires_identity_defense = bool(is_user_facing and _matches_any(lower, _IDENTITY_DEFENSE_PATTERNS))
    requires_memory = requires_memory or requires_identity_defense
    requires_state = requires_state or requires_self_preservation or requires_identity_defense
    requires_aura_stance = bool(
        is_user_facing and (
            requires_state or requires_self_preservation or requires_identity_defense or _matches_any(lower, _AURA_PERSPECTIVE_PATTERNS)
        )
    )
    requires_aura_question = bool(is_user_facing and _matches_any(lower, _AURA_QUESTION_INVITATION_PATTERNS))

    tool_evidence = has_tool_evidence(state)
    memory_evidence = has_memory_evidence(state)
    continuity_evidence = has_continuity_evidence(state)

    reasons = []
    if explicit_search:
        reasons.append("explicit_search_request")
    elif requires_search:
        reasons.append("specific_fact_lookup")
    if requires_memory:
        reasons.append("memory_grounding")
    if requires_state:
        reasons.append("state_reflection")
    if requires_self_preservation:
        reasons.append("self_preservation")
    if requires_identity_defense:
        reasons.append("identity_defense")
    if requires_aura_stance and not requires_state:
        reasons.append("aura_perspective")
    if requires_aura_question:
        reasons.append("invited_aura_questions")

    return ResponseContract(
        is_user_facing=is_user_facing,
        requires_search=requires_search,
        required_skill="web_search" if requires_search else None,
        requires_memory_grounding=requires_memory,
        requires_state_reflection=requires_state,
        avoid_question_fishing=is_user_facing,
        prefers_dialogue_participation=is_user_facing,
        requires_aura_stance=requires_aura_stance,
        requires_aura_question=requires_aura_question,
        requires_identity_defense=requires_identity_defense,
        requires_self_preservation=requires_self_preservation,
        tool_evidence_available=tool_evidence,
        memory_evidence_available=memory_evidence,
        continuity_evidence_available=continuity_evidence,
        reason=", ".join(reasons) if reasons else "ordinary_dialogue",
        search_query=text if requires_search else "",
    )
