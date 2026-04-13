from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
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
    r"\bread (?:this|that|the)\b",
    r"\bfind (?:this|that|the)\b.*\b(?:story|article|post|page)\b",
    r"https?://[^\s]+",                         # Any URL in the message
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

_TEMPORAL_CURRENTNESS_PATTERNS = (
    r"\blatest\b",
    r"\bmost recent\b",
    r"\bcurrent\b",
    r"\bcurrently\b",
    r"\brecent\b",
    r"\brecently\b",
    r"\bup[- ]to[- ]date\b",
    r"\bas of\b",
    r"\bright now\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\btomorrow\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\bthis year\b",
)

_LIVE_FACT_PATTERNS = (
    r"\bnews\b",
    r"\bheadline\b",
    r"\bprice\b",
    r"\bstock\b",
    r"\bscore\b",
    r"\bschedule\b",
    r"\bversion\b",
    r"\brelease\b",
    r"\bapi\b",
    r"\bdocs?\b",
    r"\bdocumentation\b",
    r"\bmodel\b",
    r"\bceo\b",
    r"\bpresident\b",
    r"\belection\b",
    r"\blaw\b",
    r"\bpolicy\b",
    r"\brule\b",
    r"\bregulation\b",
    r"\bavailability\b",
)

_TIME_UTILITY_PATTERNS = (
    r"\bwhat time\b",
    r"\bcurrent time\b",
    r"\bdate\b",
    r"\bday is it\b",
    r"\btimezone\b",
    r"\bclock\b",
)

_GROUNDED_FOLLOWUP_SUMMARY_PATTERNS = (
    r"\bwhat happens\b",
    r"\bsummar(?:y|ize|ise)\b",
    r"\brecap\b",
    r"\bstory beats?\b",
    r"\bplot beats?\b",
    r"\bhow does it end\b",
    r"\bwhat(?:'s| is) the ending\b",
    r"\bin full\b",
    r"\bread (?:it|this|that|the)\b",
)

_GROUNDED_FOLLOWUP_PRECISION_PATTERNS = (
    r"\bspecifically\b",
    r"\bwhat specific\b",
    r"\bwhich one\b",
    r"\bexactly\b",
    r"\bwhat does it say\b",
    r"\bwhat does (?:it|the page|the post|the article|the story|the document) say\b",
)

_GROUNDED_FOLLOWUP_DOCUMENT_PATTERNS = (
    r"\b(?:story|article|post|page|thread|document|source|text|link|site|paper|report|policy|guide|log|logs|journal|journals)\b",
    r"\b(?:this|that|it)\b",
)

_GROUNDED_FOLLOWUP_OPENING_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|right|so|well|wait|question)?[\s,.:;-]*(?:but\s+)?(?:what|who|when|where|why|how|which)\b",
    re.IGNORECASE,
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

_ANCHOR_STOPWORDS = frozenset({
    "about", "after", "again", "article", "being", "below", "could", "document",
    "from", "have", "into", "its", "journal", "journals", "just", "link", "page",
    "paper", "post", "report", "said", "says", "search", "source", "story",
    "text", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "thread", "what", "when", "where", "which", "while", "with", "would",
})

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
    requires_exact_dates: bool = False
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
    max_tool_turns: int = 1
    max_tools: int = 4
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

        if self.requires_search or self.requires_exact_dates:
            now = datetime.now().astimezone()
            directives.append(f"- Current local date: {now.strftime('%A, %B %d, %Y')}.")
            directives.append(f"- Current local time: {now.strftime('%I:%M %p %Z')}.")

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

        if self.requires_exact_dates:
            directives.append(
                "- If the user says today, tomorrow, yesterday, latest, current, or recent, anchor the answer with exact dates."
            )

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

        directives.append(
            f"- Tool/function-call budget for this reply: at most {max(1, int(self.max_tool_turns or 1))} tool turns."
        )
        directives.append(
            f"- Keep the active tool catalog narrow: prefer {max(1, int(self.max_tools or 1))} relevant tools or fewer."
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


def has_grounding_tool_evidence(state: AuraState) -> bool:
    modifiers = getattr(state, "response_modifiers", {}) or {}
    last_skill = str(modifiers.get("last_skill_run", "") or "").strip()
    if (
        last_skill in {"web_search", "sovereign_browser"}
        and modifiers.get("last_skill_ok")
        and isinstance(modifiers.get("last_skill_result_payload"), dict)
    ):
        return True

    working_memory = getattr(state.cognition, "working_memory", []) or []
    for msg in reversed(working_memory[-8:]):
        if not isinstance(msg, dict):
            continue
        meta = msg.get("metadata", {}) or {}
        if (
            meta.get("type") == "skill_result"
            and meta.get("ok") is True
            and str(meta.get("skill", "") or "").strip() in {"web_search", "sovereign_browser"}
        ):
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


def _recent_grounding_anchor_terms(state: AuraState) -> set[str]:
    texts: list[str] = []
    modifiers = getattr(state, "response_modifiers", {}) or {}
    last_skill = str(modifiers.get("last_skill_run", "") or "").strip()
    payload = modifiers.get("last_skill_result_payload")
    if last_skill in {"web_search", "sovereign_browser"} and modifiers.get("last_skill_ok") and isinstance(payload, dict):
        for key in ("title", "query", "answer", "summary", "source", "url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value[:320])
        content = str(payload.get("content", "") or payload.get("result", "") or "").strip()
        if content:
            texts.append(content.splitlines()[0][:240])

    working_memory = getattr(state.cognition, "working_memory", []) or []
    for msg in reversed(working_memory[-6:]):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or "").strip().lower()
        content = str(msg.get("content", "") or "").strip()
        if not content:
            continue
        if role == "system" and "[FETCHED PAGE CONTENT]" in content:
            texts.append("\n".join(content.splitlines()[:3])[:320])
        elif role in {"user", "assistant"}:
            texts.append(content[:220])

    anchors: set[str] = set()
    for raw in texts:
        for token in re.findall(r"[a-z0-9']+", raw.lower()):
            normalized = token.strip("'")
            if len(normalized) < 3 or normalized in _ANCHOR_STOPWORDS:
                continue
            anchors.add(normalized)
    return anchors


def _looks_like_grounded_followup(state: AuraState, text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered or not has_grounding_tool_evidence(state):
        return False

    summary_followup = _matches_any(lowered, _GROUNDED_FOLLOWUP_SUMMARY_PATTERNS)
    precision_followup = _matches_any(lowered, _GROUNDED_FOLLOWUP_PRECISION_PATTERNS)
    document_reference = _matches_any(lowered, _GROUNDED_FOLLOWUP_DOCUMENT_PATTERNS)
    wh_opening = bool(_GROUNDED_FOLLOWUP_OPENING_RE.match(lowered))
    anchors = _recent_grounding_anchor_terms(state)
    anchor_overlap = bool(anchors and any(anchor in lowered for anchor in anchors))
    short_followup = len(lowered.split()) <= 8

    if summary_followup:
        return True
    if precision_followup and (document_reference or anchor_overlap or short_followup):
        return True
    if wh_opening and (document_reference or anchor_overlap):
        return True
    return False


def build_response_contract(
    state: AuraState,
    objective: str,
    *,
    is_user_facing: bool,
) -> ResponseContract:
    from core.runtime.turn_analysis import analyze_turn

    text = str(objective or "").strip()
    lower = text.lower()

    explicit_search = _matches_any(lower, _EXPLICIT_SEARCH_PATTERNS)
    factual_lookup = _matches_any(lower, _FACTUAL_LOOKUP_PATTERNS)
    specific_reference = _matches_any(text, _REFERENCE_MARKERS)
    factual_followup = _looks_like_grounded_followup(state, text)
    temporal_currentness = _matches_any(lower, _TEMPORAL_CURRENTNESS_PATTERNS)
    live_fact_lookup = _matches_any(lower, _LIVE_FACT_PATTERNS)
    time_utility_lookup = _matches_any(lower, _TIME_UTILITY_PATTERNS)
    search_negated = bool(_SEARCH_NEGATION_RE.search(lower))

    requires_memory = bool(is_user_facing and _matches_any(lower, _MEMORY_PATTERNS))
    requires_state = bool(is_user_facing and _matches_any(lower, _STATE_REFLECTION_PATTERNS))
    requires_self_preservation = bool(is_user_facing and _matches_any(lower, _SELF_PRESERVATION_PATTERNS))
    requires_identity_defense = bool(is_user_facing and _matches_any(lower, _IDENTITY_DEFENSE_PATTERNS))
    requires_memory = requires_memory or requires_identity_defense
    requires_state = requires_state or requires_self_preservation or requires_identity_defense

    temporal_live_lookup = bool(
        is_user_facing
        and temporal_currentness
        and live_fact_lookup
        and not any(
            (
                requires_memory,
                requires_state,
                requires_self_preservation,
                requires_identity_defense,
                time_utility_lookup,
            )
        )
    )
    # Negation guard: "I didn't mean for you to search" should NOT trigger search.
    if search_negated:
        explicit_search = False
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
    # URL presence always forces search — user expects content to be fetched
    has_url = bool(re.search(r'https?://[^\s]+', text))
    requires_search = bool(
        is_user_facing
        and (
            explicit_search
            or has_url
            or (factual_lookup and specific_reference)
            or factual_followup
            or temporal_live_lookup
        )
    )
    requires_exact_dates = bool(
        is_user_facing
        and temporal_currentness
        and not any((requires_memory, requires_state, requires_self_preservation, requires_identity_defense))
    )
    requires_aura_stance = bool(
        is_user_facing and (
            requires_state or requires_self_preservation or requires_identity_defense or _matches_any(lower, _AURA_PERSPECTIVE_PATTERNS)
        )
    )
    requires_exact_dates = bool(requires_exact_dates and not requires_aura_stance)
    requires_aura_question = bool(is_user_facing and _matches_any(lower, _AURA_QUESTION_INVITATION_PATTERNS))

    tool_evidence = has_tool_evidence(state)
    memory_evidence = has_memory_evidence(state)
    continuity_evidence = has_continuity_evidence(state)
    turn_analysis = analyze_turn(text)

    max_tool_turns = 1
    max_tools = 4
    if turn_analysis.suggests_deliberate_mode or turn_analysis.intent_type in {"TASK", "SKILL"}:
        max_tool_turns = 4 if requires_search else 3
        max_tools = 8 if requires_search else 6
    elif requires_search:
        max_tool_turns = 3
        max_tools = 4
    elif requires_memory or requires_state or requires_aura_stance:
        max_tool_turns = 1
        max_tools = 3

    reasons = []
    if explicit_search:
        reasons.append("explicit_search_request")
    elif temporal_live_lookup:
        reasons.append("temporal_live_lookup")
    elif factual_followup:
        reasons.append("grounded_followup")
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
        requires_exact_dates=requires_exact_dates,
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
        max_tool_turns=max_tool_turns,
        max_tools=max_tools,
        reason=", ".join(reasons) if reasons else "ordinary_dialogue",
        search_query=text if requires_search else "",
    )
