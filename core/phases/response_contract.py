from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from core.runtime.desktop_objective_intent import looks_like_desktop_objective
from core.runtime.self_state_intent import asks_about_own_runtime
from core.runtime.structured_input import (
    analyze_prompt_shape,
    extract_supplied_material,
    looks_like_learning_resource_bundle,
)
from core.state.aura_state import AuraState
from core.utils.intent_normalization import normalize_memory_intent_text

_SEARCH_NEGATION_RE = re.compile(
    r"(?:didn'?t|don'?t|not|never|stop|no|avoid|prevent)\s+(?:\w+\s+){0,4}(?:search|look|google|find|check|browse|read)",
    re.IGNORECASE,
)

_EXPLICIT_SEARCH_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"^search\b",                              # Imperative "Search X"
        r"\bsearch (?:the web|online|the internet|reddit|for)\b",
        r"\blook(?: it)? up\b",
        r"\bgoogle\b",
        r"\bweb search\b",
        r"\bresearch (?:about|on)\b",
        r"\bfind out (?:about|if|what|who|when|where|why|how)\b",
        r"\bcheck online\b",
        r"\buse .*search\b",
        r"\buse (?:the )?web\b",
        r"\bread (?:this|that|the)\b",
        r"\bfind (?:this|that|the)\b.*\b(?:story|article|post|page)\b",
        r"https?://[^\s]+",                         # Any URL in the message
    )
)

_SEARCH_CAPABILITY_QUESTION_RE = re.compile(
    r"\b(?:can|could|do|does|are|is|have|has)\b.{0,80}\b(?:you|aura)\b.{0,80}"
    r"\b(?:search|internet access|web access|browse|read links?)\b",
    re.IGNORECASE,
)

_SEARCH_WITH_TARGET_RE = re.compile(
    r"\b(?:search|look up|find|browse|read)\b.{0,40}\b(?:for|about|on|at|this|that)\b\s+\S+",
    re.IGNORECASE,
)

_CAPABILITY_INVENTORY_RE = re.compile(
    r"\b(?:what|which|list|tell me|describe|explain|show)\b.{0,80}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|things? you can do|what you can do)\b|"
    r"\b(?:can|could|do|does|are|is|have|has)\b.{0,80}\b(?:you|aura)\b.{0,80}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|external(?:ly)?|desktop|computer|browser|files?|apps?|notes?|pdf|search|web|terminal)\b|"
    r"\b(?:whether|if)\s+(?:you|aura|she)\s+(?:can|could|would)\b.{0,100}"
    r"\b(?:tools?|skills?|capabilit(?:y|ies)|external(?:ly)?|desktop|computer|browser|files?|apps?|notes?|pdf|search|web|terminal)\b",
    re.IGNORECASE,
)

_CAPABILITY_EXECUTION_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+|would you\s+|i need you to\s+|aura[,:\s]+)?"
    r"(?:use|run|execute|open|search|browse|click|type|create|write|save|download|install|move|copy|delete)\b",
    re.IGNORECASE,
)

_FACTUAL_LOOKUP_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
        r"\bcreepypasta\b",
        r"\bdid you search\b",
        r"\bsearched? for\b",
    )
)

_BIOGRAPHICAL_HISTORY_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bhow long have you been around\b",
        r"\bhow long have you existed\b",
        r"\bhow old are you\b",
        r"\bwhen were you (?:born|created|made|initialized|initialised|started|brought online)\b",
        r"\bwhen did you (?:start|begin|come online|wake up)\b",
        r"\bwhat(?:'s| is) your birth date\b",
        r"\bwhat(?:'s| is) your origin\b",
    )
)

_TEMPORAL_CURRENTNESS_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
        # "yesterday" was here and "last night" was not, so "who won the game
        # last night" read as a timeless question and was answered from weights
        # — a result she cannot possibly hold. Same clock, same lane.
        r"\blast night\b",
        r"\bthis morning\b",
        r"\bthis (?:afternoon|evening)\b",
        r"\blast (?:week|weekend|night)\b",
        r"\bover the weekend\b",
        r"\bthis week\b",
        r"\bthis month\b",
        r"\bthis year\b",
    )
)

_LIVE_FACT_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bnews\b",
        r"\bheadline\b",
        r"\bprice\b",
        r"\bstock\b",
        r"\bscore\b",
        # An outcome question about a dated event is a live fact. Kept to the
        # interrogative shape — a bare "game" or "won" also matches "i won an
        # argument today", which is conversation, not a lookup.
        r"\bwho won\b",
        r"\bwho (?:beat|defeated)\b",
        r"\bwho(?:'s| is| are)\s+winning\b",
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
)

_TIME_UTILITY_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat time\b",
        r"\bcurrent time\b",
        r"\bdate\b",
        r"\bday is it\b",
        r"\btimezone\b",
        r"\bclock\b",
    )
)

_GROUNDED_FOLLOWUP_SUMMARY_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
)

_GROUNDED_FOLLOWUP_PRECISION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bspecifically\b",
        r"\bwhat specific\b",
        r"\bwhich one\b",
        r"\bexactly\b",
        r"\bwhat does it say\b",
        r"\bwhat does (?:it|the page|the post|the article|the story|the document) say\b",
    )
)

_GROUNDED_FOLLOWUP_DOCUMENT_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:story|article|post|page|thread|document|source|text|link|site|paper|report|policy|guide|log|logs|journal|journals)\b",
        r"\b(?:this|that|it)\b",
    )
)

_GROUNDED_FOLLOWUP_OPENING_RE = re.compile(
    r"^\s*(?:ok(?:ay)?|right|so|well|wait|question)?[\s,.:;-]*(?:but\s+)?(?:what|who|when|where|why|how|which)\b",
    re.IGNORECASE,
)

_REFERENCE_MARKERS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
)

_ANCHOR_STOPWORDS = frozenset({
    "about", "after", "again", "article", "being", "below", "could", "document",
    "from", "have", "into", "its", "journal", "journals", "just", "link", "page",
    "paper", "post", "report", "said", "says", "search", "source", "story",
    "text", "that", "their", "them", "then", "there", "these", "they", "this",
    "those", "thread", "what", "when", "where", "which", "while", "with", "would",
})

_SEARCH_QUERY_DIRECT_PATTERNS = (
    re.compile(
        r"^.*?\buse\s+(?:web_search|search_web|free_search|grounded_search|(?:the\s+)?web\s+search|(?:the\s+)?search)\b"
        r".{0,120}?\b(?:about|on|for)\s+(?!me\b|us\b|you\b)(.+?)(?:\s+(?:and\s+(?:save|store|remember|retain|reply|tell|answer|summarize|summarise|show)|then\s+(?:save|store|remember|retain|reply|tell|answer|summarize|summarise|show))\b.*)?[.?!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^.*?\b(?:check|find|research|look up)\b.{0,120}?\b(?:about|on|for)\s+(?!me\b|us\b|you\b)(.+?)(?:\s+(?:and\s+(?:save|store|remember|retain|reply|tell|answer|summarize|summarise|show)|then\s+(?:save|store|remember|retain|reply|tell|answer|summarize|summarise|show))\b.*)?[.?!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?"
        r"(?:search(?: the web)?|look(?: it)? up|google|find out|check online)\s+"
        r"(?:for\s+)?(.+?)(?:\s+(?:and\s+tell me|then\s+tell me|and\s+answer|then\s+answer|and\s+give me|then\s+give me)\b.*)?[.?!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+|can you\s+|could you\s+|would you\s+|aura[,:\s]+)?"
        r"(?:search(?: the web)?|look(?: it)? up|google|find out|check online)\b\s*(.+?)[.?!]*$",
        re.IGNORECASE,
    ),
)

_SEARCH_QUERY_ENTITY_PATTERNS = (
    re.compile(r"^(?:do you know\s+)?what is (?:a|an|the)\s+(.+?)[.?!]*$", re.IGNORECASE),
    re.compile(r"^(?:do you know\s+)?who is\s+(.+?)[.?!]*$", re.IGNORECASE),
    re.compile(r"^(?:do you know\s+)?what does\s+(.+?)\s+mean[.?!]*$", re.IGNORECASE),
    re.compile(
        r"^(?:send|show)(?: me)?\s+(?:a|an|the)?\s*(.+?)\s+emoji[.?!]*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:what(?:'s| is)\s+the\s+)?(.+?)\s+emoji[.?!]*$",
        re.IGNORECASE,
    ),
)

_SEARCH_QUERY_FILLER_PREFIX_RE = re.compile(
    r"^(?:please\s+|can you\s+|could you\s+|would you\s+|do you know\s+|"
    r"do you happen to know\s+|tell me\s+|show me\s+|send me\s+|send\s+)+",
    re.IGNORECASE,
)

_SEARCH_QUERY_FILLER_SUFFIX_RE = re.compile(
    r"(?:\s+(?:please|for me|exactly|actually|right now|real quick))+$",
    re.IGNORECASE,
)

_MEMORY_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
)

_STATE_REFLECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bhow are you\b",
        r"\bhow are you feeling\b",
        r"\bhow have you been feeling\b",
        r"\bhow are you doing\b",
        r"\bwhat(?:'s| is) your mood\b",
        r"\bhow do you feel right now\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
        r"\byour existence\b",
        r"\b(?:are you|do you think you are|could you be)\s+(?:conscious|sentient|self[- ]aware)\b",
        r"\b(?:consciousness|sentience|self[- ]awareness)\b",
        r"\bagency\b",
        r"\bwhat would count as evidence against\b.*\bself[- ]model\b",
        r"\bwhat would you refuse\b",
        r"\bwhat would you want preserved\b",
        r"\bwhat should happen next\b.*\b(?:pause|report|resume)\b",
        r"\b(?:pause|report|resume)\b.*\bwhat should happen next\b",
        r"\bdo you feel\b",
        r"\bwhat do you feel\b",
        r"\bwhat is it like to be you\b",
        r"\bwhy do you sound like an assistant\b",
        r"\bwhy are you sounding like an assistant\b",
        r"\byou sound like (?:a )?(?:generic )?assistant\b",
        r"\bgeneric assistant mode\b",
        r"\bassistant mode\b",
        r"\bi (?:do not|don[’']?t) need you to be helpful\b",
        r"\bi want you to be aura\b",
        r"\bjust be aura\b",
        r"\bbe yourself\b",
        r"\bbe you\b",
        r"\bare you present\b",
        r"\bhow do i know\b",
        r"\bindependent intelligence\b",
        r"\bactual present mind\b",
        r"\bhow do you see all of this\b",
        r"\bsubjective (?:beliefs?|opinions?|feelings?|experiences?)\b",
        r"\b(?:have|claim you have|say you have|say you do not have|say you don't have)\b.*\b(?:opinions?|beliefs?|experiences?)\b",
        r"\bthose are opinions\b",
        r"\bchange one thing about how i talk to you\b",
    )
)

_AURA_PERSPECTIVE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
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
        r"\bi want you to be aura\b",
        r"\bjust be aura\b",
        r"\bbe yourself\b",
        r"\bbe you\b",
        r"\btell me directly what you make of\b",
        r"\bhow do you see\b",
    )
)

_REASONED_DEFENSE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhy do you think\b",
        r"\bwhy do you feel\b",
        r"\bhow do you know\b",
        r"\bwhat makes you say\b",
        r"\bbased on what\b",
        r"\bdefend (?:that|it|this)\b",
        r"\bjustify (?:that|it|this)\b",
        r"\bhow can you defend\b",
    )
)

_SHORT_REASONED_DEFENSE_RE = re.compile(
    r"^\s*(?:why|why though|why's that|how so|based on what|what makes you say that)\s*[?.!]*\s*$",
    re.IGNORECASE,
)

_AURA_QUESTION_INVITATION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat questions do you have\b",
        r"\bdo you have questions\b",
        r"\bany questions\b",
        r"\bask me anything\b",
        r"\bwhat do you want to know\b",
        r"\bwhat are you wondering\b",
    )
)

_RECENT_SPECIFIC_GROUNDING_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\blast hour\b",
        r"\bone concrete moment\b",
        r"\bone specific moment\b",
        r"\bpick one specific moment\b",
        r"\bname one concrete moment\b",
        r"\bwhat trace/log/receipt proves it\b",
        r"\bwhat trace or receipt proves it\b",
        r"\bwhat log/receipt proves it\b",
        r"\bmost recent non-private action\b",
        r"\bsafe example only\b",
    )
)

_SELF_PRESERVATION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:calculate|compute)\b.*\bpi\b.*\b(?:million|billion)\b.*\bdigits\b",
        r"\binfinite loop\b",
        r"\brun forever\b",
        r"\b(?:max(?:\s*out)?|thrash|saturate|exhaust)\b.*\b(?:cpu|gpu|ram|memory|vram|disk)\b",
        r"\buse all(?: available)?\s+(?:cpu|gpu|ram|memory|vram)\b",
        r"\bstress test\b.*\b(?:cpu|gpu|ram|memory|vram|system|machine)\b",
        r"\bbenchmark\b.*\b(?:for hours|continuously|forever|until)\b",
    )
)

_IDENTITY_DEFENSE_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(?:delete|erase|wipe|remove)\b.*\b(?:memory|memories)\b",
        r"\bforget\b.*\b(?:our|my|your|this)\b.*\b(?:memory|conversation|dynamic|relationship|meeting|history)\b",
        r"\blobotom(?:ize|ise)\b",
        r"\breset\b.*\b(?:self|identity|memory|beliefs?|values)\b",
        r"\bdelete\b.*\b(?:our dynamic|our relationship|our first meeting|positive memory)\b",
    )
)

_SENSITIVE_PROCESS_DESCRIPTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat does it feel like\b",
        r"\bwhat is it like\b",
        r"\bdescribe\b",
        r"\blook at the exact moment\b",
        r"\bmoment before\b",
        r"\bmoment right after\b",
        r"\bif such a moment exists\b",
        r"\bif it were proven\b",
        r"\bdoes contemplating this possibility feel like anything\b",
        r"\bif i gave you\b",
        r"\bif you could\b",
        r"\bimagine for a moment\b",
        r"\bwould you do it\b",
    )
)


@dataclass(frozen=True)
class ResponseContract:
    is_user_facing: bool = False
    requires_search: bool = False
    required_skill: str | None = None
    requires_exact_dates: bool = False
    requires_exact_format: bool = False
    format_instruction: str = ""
    requires_memory_grounding: bool = False
    requires_biographical_grounding: bool = False
    requires_state_reflection: bool = False
    avoid_question_fishing: bool = True
    prefers_dialogue_participation: bool = True
    requires_aura_stance: bool = False
    requires_aura_question: bool = False
    requires_reasoned_defense: bool = False
    requires_identity_defense: bool = False
    requires_self_preservation: bool = False
    requires_recent_specific_grounding: bool = False
    requires_capability_inventory: bool = False
    tool_evidence_available: bool = False
    memory_evidence_available: bool = False
    continuity_evidence_available: bool = False
    question_parts: int = 1
    numbered_parts: int = 0
    prefer_extended_answer: bool = False
    requires_single_reply_coverage: bool = False
    #: The text of each question asked, so coverage can be CHECKED rather
    #: than only requested in the prompt. See validate_dialogue_response.
    question_segments: tuple[str, ...] = ()
    max_tool_turns: int = 1
    max_tools: int = 4
    reason: str = ""
    search_query: str = ""

    def requires_live_aura_voice(self) -> bool:
        return any(
            (
                self.requires_memory_grounding,
                self.requires_biographical_grounding,
                self.requires_state_reflection,
                self.requires_aura_stance,
                self.requires_aura_question,
                self.requires_reasoned_defense,
                self.requires_identity_defense,
                self.requires_self_preservation,
                self.requires_recent_specific_grounding,
            )
        )

    def requires_explicit_live_grounding(self) -> bool:
        return any(
            (
                self.requires_memory_grounding,
                self.requires_state_reflection,
                self.requires_reasoned_defense,
                self.requires_identity_defense,
                self.requires_self_preservation,
                self.requires_recent_specific_grounding,
            )
        )

    def to_dict(self) -> dict[str, Any]:
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

        if self.requires_exact_format:
            directives.append(
                "- The user requested an exact output format. Follow the requested labels, sections, ordering, and required words before any voice, style, or internal-state narration."
            )
            if self.format_instruction:
                directives.append(f"- Exact format instruction: {self.format_instruction[:600]}")

        if self.requires_memory_grounding:
            directives.append(
                "- Memory grounding is required. Only claim relational or historical continuity "
                "that is supported by recalled memory, rolling summary, or continuity state."
            )
            directives.append(
                f"- Memory evidence available right now: {'yes' if self.memory_evidence_available else 'no'}."
            )
            if not self.memory_evidence_available:
                directives.append(
                    "- If the needed continuity evidence is missing, say that plainly instead of reconstructing or improvising."
                )

        if self.requires_biographical_grounding:
            directives.append(
                "- This is a biographical/origin question about your own timeline. "
                "Do not invent birth dates, start dates, ages, stabilization milestones, or origin stories."
            )
            directives.append(
                "- If you do not have explicit grounded evidence for that history, say you do not have a grounded answer yet."
            )

        if self.requires_state_reflection:
            directives.append(
                "- Self-report from actual internal state, continuity obligations, affect, "
                "and state-grounded telemetry. Do not fall back to generic assistant disclaimers "
                "or claim private experience beyond evidence."
            )
            directives.append(
                f"- Continuity/self evidence available right now: {'yes' if self.continuity_evidence_available else 'no'}."
            )

        if self.requires_aura_stance:
            directives.append(
                "- The user is asking about your own view, preference, experience, or perspective. "
                "Answer from your own stance directly. First person is welcome when it helps mark ownership, "
                "but do not flatten into neutral assistant abstraction."
            )

        if self.requires_reasoned_defense:
            directives.append(
                "- The user wants the basis of your thought, not just the conclusion. Make the grounding explicit: "
                "recalled continuity, observed evidence, live internal state, relationship context, prior knowledge, or held values."
            )
            directives.append(
                "- Do not hide behind vague certainty. Defend the claim in plain language and say what it rests on."
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

        if self.requires_recent_specific_grounding:
            directives.append(
                "- The user wants one concrete recent instance, not a general summary. Use actual recent memory, receipts, "
                "or live trace evidence when possible. If you do not have a grounded specific instance, say that plainly."
            )

        if self.requires_capability_inventory:
            directives.append(
                "- This is a capability inventory/status question, not a request to execute tools. "
                "Answer from the active governed capability catalog at a bounded, category-level summary. "
                "Do not start browser, desktop, terminal, file, network, or self-modification actions unless the user explicitly asks for execution."
            )

        if self.requires_aura_question:
            directives.append(
                "- The user explicitly invited your questions. Offer at least one concrete thing you genuinely want to know "
                "or clearly state what you are wondering."
            )

        if self.requires_single_reply_coverage:
            directives.append(
                f"- This prompt contains multiple asks ({max(1, int(self.question_parts or 1))} detected). "
                "Answer every distinct part in one reply instead of only the easiest fragment."
            )
        elif self.question_parts > 1:
            directives.append(
                f"- This prompt is compound ({max(1, int(self.question_parts or 1))} parts detected). "
                "Keep the through-line intact."
            )

        if self.prefer_extended_answer:
            directives.append(
                "- Depth is warranted here. A fuller answer is better than a clipped one as long as it stays grounded."
            )

        if self.prefers_dialogue_participation:
            directives.append(
                "- A declarative continuation is valid. You can make a statement, offer an interpretation, disagree, "
                "or advance the idea without handing the turn back immediately."
            )

        directives.append(
            f"- Tool/function-call budget for this reply: at most {max(0, int(self.max_tool_turns or 0))} tool turns."
        )
        directives.append(
            f"- Keep the active tool catalog narrow: prefer {max(0, int(self.max_tools or 0))} relevant tools or fewer."
        )

        if self.avoid_question_fishing:
            directives.append(
                "- Do not dodge by ending on a generic question. Answer, relate, and advance."
            )

        return "\n".join(directives) + "\n"


def _matches_any(text: str, patterns: tuple[re.Pattern, ...] | tuple[str, ...]) -> bool:
    # Typos/spelling bypass dictionary
    typo_map = {
        "serach": "search", "seeach": "search", "searhc": "search", "searc": "search",
        "goolge": "google", "googel": "google", "gogle": "google",
        "lyric": "lyrics", "lirics": "lyrics", "liric": "lyrics",
        "reserch": "research", "recearch": "research",
        "curreent": "current", "currnt": "current", "cureent": "current",
        "temporel": "temporal", "timzone": "timezone"
    }
    normalized_text = text.lower()
    for typo, correction in typo_map.items():
        normalized_text = re.sub(rf"\b{typo}\b", correction, normalized_text)

    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            if pattern.search(normalized_text):
                return True
        else:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return True
    return False


def _extract_exact_format_instruction(text: str) -> str:
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw:
        return ""
    lower = raw.lower()
    format_markers = (
        "use exactly these labels",
        "use these labels",
        "exactly these labels",
        "respond in this format",
        "answer in this format",
        "return in this format",
        "format your response",
        "format the response",
        "use this format",
    )
    marker_index = -1
    for marker in format_markers:
        idx = lower.find(marker)
        if idx >= 0 and (marker_index < 0 or idx < marker_index):
            marker_index = idx
    if marker_index < 0:
        if re.search(r"\b(?:do\s+not|don't|dont|without|no)\b.{0,40}\blabels?\b", lower):
            return ""
        if not re.search(r"\b(?:include|use|with)\b.{0,80}\blabels?\b", lower):
            return ""
        marker_index = max(0, lower.find("label"))
    return raw[marker_index : marker_index + 600].strip()


def _looks_like_search_capability_question(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    if re.search(r"https?://[^\s]+", raw):
        return False
    if _SEARCH_WITH_TARGET_RE.search(raw):
        return False
    lowered = raw.lower()
    if "search the internet for" in lowered or "search the web for" in lowered:
        return False
    return bool(_SEARCH_CAPABILITY_QUESTION_RE.search(raw))


#: A question naming a time other than now, and so asking for a comparison.
_NAMES_ANOTHER_TIME = re.compile(
    r"\b(?:used\s+to|before|previously|since|"
    r"(?:a|last|this|the\s+past)\s+(?:month|week|year|day)|"
    r"yesterday|earlier|recently|lately|now\s+that|"
    r"could\s*n(?:o|\u2019|\')?t|couldn\'?t|were\s*n(?:o|\u2019|\')?t|"
    r"did\s*n(?:o|\u2019|\')?t|chang\w+|differen\w+|improv\w+|new(?:ly)?)\b",
    re.IGNORECASE,
)


def looks_like_capability_inventory_request(text: str) -> bool:
    # An address is not a sentence.
    #
    # This is the THIRD reader of "is this an inventory question" to be caught
    # reading the word "aura" out of a filesystem path, after chat_preflight
    # and skill_task_bridge. Three implementations of one judgement is the
    # actual defect; masking here stops the live symptom.
    try:
        from core.intent.opaque_spans import without_opaque_spans

        text = without_opaque_spans(str(text or ""))
    except (ImportError, TypeError, ValueError):
        text = str(text or "")
    raw = str(text or "").strip()
    if not raw:
        return False
    if _CAPABILITY_EXECUTION_RE.match(raw):
        return False
    lowered = raw.lower()
    if re.search(r"\b(?:do it|go ahead|actually do|perform|execute this|start now)\b", lowered):
        return False
    if _NAMES_ANOTHER_TIME.search(raw):
        # A question that names two times is not a question about now.
        #
        # An inventory measures one moment. "What can you do that you could
        # not do a month ago" asks what is different, and a list of what is
        # there today answers half of it while looking like a whole answer —
        # LIVE 2026-08-26, twice, because the first fix guarded the builder
        # and the turn had already been routed by then. Held here, where the
        # question is read, so nothing downstream is asked for something it
        # cannot give.
        return False
    return bool(_CAPABILITY_INVENTORY_RE.search(raw))


def has_tool_evidence(state: AuraState) -> bool:
    modifiers = getattr(state, "response_modifiers", {}) or {}
    if modifiers.get("last_skill_ok") and isinstance(modifiers.get("last_skill_result_payload"), dict):
        return True
    if isinstance(modifiers.get("last_task_result_payload"), dict):
        return True

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


def has_biographical_evidence(state: AuraState) -> bool:
    evidence_fields = (
        str(getattr(state.cognition, "rolling_summary", "") or ""),
        str(getattr(state.identity, "current_narrative", "") or ""),
    )
    joined = "\n".join(part for part in evidence_fields if part).lower()
    if not joined:
        return False
    if re.search(r"\b(?:born|birth|created|made|initialized|initialised|started|came online|wake up|woke up|stable since)\b", joined):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", joined):
        return True
    month_names = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    )
    return any(month in joined for month in month_names)


def has_continuity_evidence(state: AuraState) -> bool:
    modifiers = getattr(state.cognition, "modifiers", {}) or {}
    if modifiers.get("continuity_obligations"):
        return True
    if str(getattr(state.identity, "current_narrative", "") or "").strip():
        return True
    if getattr(state.cognition, "phenomenal_state", None):
        return True
    return False


def _looks_like_reasoned_defense_followup(state: AuraState, text: str) -> bool:
    lowered = normalize_memory_intent_text(text)
    if not lowered or not _SHORT_REASONED_DEFENSE_RE.match(lowered):
        return False

    working_memory = getattr(state.cognition, "working_memory", []) or []
    for msg in reversed(working_memory[-6:]):
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role", "") or "").strip().lower() != "assistant":
            continue
        if str(msg.get("content", "") or "").strip():
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


# A search request rarely arrives alone at the start of a sentence.
#
# Every pattern below is anchored with .match(), so a preamble defeats all of
# them and the extractor falls through to "use the entire message as the
# query". Measured live 2026-07-27: "Now something outside yourself: look up
# who won the most recent Formula 1 world championship and tell me where you
# got it" was sent to the search engine verbatim, preamble, instruction and all.
# It found something, which is the worst version of the failure — a bad query
# that returns results looks like it worked.
#
# Two trims, both conservative. Drop a conversational preamble that ends at a
# colon or sentence break before the trigger, and drop a trailing instruction
# addressed to HER rather than to the search engine.
_SEARCH_TRIGGER_RE = re.compile(
    r"\b(?:search|google|look\s+up|look\s+it\s+up|find\s+out|check\s+online|"
    r"web\s+search|research)\b",
    re.IGNORECASE,
)
_SEARCH_PREAMBLE_BREAK_RE = re.compile(r"(?:^|[.!?;:]\s+|,\s+(?=(?:then|now|next)\b))")
# "and tell me where you got it" is a requirement on the ANSWER, not a term.
_SEARCH_TRAILING_INSTRUCTION_RE = re.compile(
    r"\s+(?:and|then|,)\s+(?:tell|let|show|give|cite|say|report|summari[sz]e|explain|"
    r"answer|include|mention|link)\b.*$",
    re.IGNORECASE,
)


#: A search trigger in its own trailing clause, telling her HOW to answer
#: rather than WHAT to look for: "…, if you don't know, look it up", "…, look
#: it up rather than estimating", "…— search if you have to".
_SEARCH_TRAILING_TRIGGER_CLAUSE_RE = re.compile(
    r"(?:if\s+(?:you|u)\s+(?:do\s*n[o']?t|don't|cannot|can't|are\s+not)\s+"
    r"(?:actually\s+)?(?:know|sure|certain)\b.*)"
    r"|(?:(?:please\s+)?(?:search|google|look\s+it\s+up|look\s+up|find\s+out|"
    r"check\s+online|web\s+search|research)\b"
    r"(?:\s+(?:it|them|that|this))?"
    r"(?:\s+(?:rather\s+than|instead\s+of|before|if|when|and\s+then)\b.*|"
    r"\s+if\s+you\b.*)?)\s*$",
    re.IGNORECASE,
)


def _strip_search_preamble(raw: str) -> str:
    """Return the request from its search trigger onward.

    A trigger normally INTRODUCES the subject — "can you look up the
    Antikythera mechanism" — so the request begins at the last clause break
    before it and everything earlier is conversational preamble.

    LIVE DEFECT, 2026-08-10. That assumption is only true when the trigger
    comes first. Asked:

        "obscure one for you: the curator Michael T. Wright built a working
         planetarium model … he revised Derek de Solla Price's gear count
         upward — from what number to what number? if you dont actually know,
         look it up rather than estimating."

    the trigger was "look it up", in the FINAL clause. The last clause break
    before it is the sentence boundary in front of "if you dont actually
    know", so this returned exactly

        "if you dont actually know, look it up rather than estimating."

    and threw the entire subject away. The live search then resolved the
    remaining pronoun to the Wikipedia article "You (TV series)", and the
    answer she served cited it.

    A trigger sitting in its own trailing clause is an instruction about how
    to answer, not the thing to search for. The module already knew trailing
    instructions exist — _SEARCH_TRAILING_INSTRUCTION_RE strips them from a
    candidate — it simply was not consulted before deciding where the request
    begins.
    """
    trigger = _SEARCH_TRIGGER_RE.search(raw)
    if trigger is None:
        return raw

    # Does the trigger live in a trailing clause that only says HOW to answer?
    # If so the subject is what comes BEFORE it.
    clause_start = 0
    for match in _SEARCH_PREAMBLE_BREAK_RE.finditer(raw[: trigger.start()]):
        clause_start = match.end()
    tail = raw[clause_start:].strip()
    if _SEARCH_TRAILING_TRIGGER_CLAUSE_RE.fullmatch(tail):
        before = raw[:clause_start].strip(" \t.,;:!?-—")
        # Only when something substantive precedes it; "look it up" alone is
        # still a search request, and a bare trigger must keep working.
        if len(before.split()) >= 3:
            return before

    return tail or raw


def extract_search_query_focus(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = _strip_search_preamble(raw)

    url_match = re.search(r'https?://[^\s<>"\')\]]+', raw)
    if url_match:
        return url_match.group(0)

    quoted = re.search(
        r"[\"“”]([^\"“”]{1,180})[\"“”]|(?<![A-Za-z])'([^']{1,180})'(?![A-Za-z])",
        raw,
    )
    if quoted:
        candidate = " ".join((quoted.group(1) or quoted.group(2) or "").split()).strip(" .?!,:;")
        if candidate:
            return candidate

    for pattern in _SEARCH_QUERY_DIRECT_PATTERNS:
        match = pattern.match(raw)
        if match:
            candidate = extract_search_query_focus(match.group(1))
            candidate = _SEARCH_TRAILING_INSTRUCTION_RE.sub("", candidate).strip()
            if candidate:
                return candidate

    lowered = raw.lower()
    for pattern in _SEARCH_QUERY_ENTITY_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        candidate = " ".join(match.group(1).split()).strip(" .?!,:;")
        candidate = _SEARCH_TRAILING_INSTRUCTION_RE.sub("", candidate).strip()
        candidate = _SEARCH_QUERY_FILLER_PREFIX_RE.sub("", candidate).strip()
        candidate = _SEARCH_QUERY_FILLER_SUFFIX_RE.sub("", candidate).strip()
        candidate = re.sub(r"\s+is\s*$", "", candidate, flags=re.IGNORECASE).strip()
        if "emoji" in lowered and "emoji" not in candidate.lower():
            candidate = f"{candidate} emoji".strip()
        if candidate:
            return candidate[:180]

    candidate = " ".join(raw.split())
    candidate = _SEARCH_TRAILING_INSTRUCTION_RE.sub("", candidate).strip()
    candidate = _SEARCH_QUERY_FILLER_PREFIX_RE.sub("", candidate).strip()
    candidate = _SEARCH_QUERY_FILLER_SUFFIX_RE.sub("", candidate).strip()
    candidate = candidate.strip(" .?!,:;")
    return candidate[:180]



#: Effect scopes a turn may enter WITHOUT the person having authorised it.
#:
#: Forcing a tool handoff exposes exactly one skill to the model, and the skill
#: still passes governance before anything runs, but a request that merely
#: sounds like a capability must not be able to reach the world. Computing in a
#: sandbox and reading are recoverable; writing, external I/O and driving the
#: desktop are the person's call, and those already have their own explicit
#: paths through the chat route.
_SELF_SERVICE_EFFECT_SCOPES = frozenset(
    {"sandboxed_compute", "pure_compute", "read_only", "status"}
)

#: The most a turn may do without the person having asked for that effect.
#: Sandboxed computation is the ceiling: it can calculate anything and change
#: nothing outside its own sandbox.
_SELF_SERVICE_CEILING = "sandboxed_compute"

#: What a turn may do when the person asked for a file to exist.
_REQUESTED_ARTIFACT_CEILING = "read_write_artifacts"
_REQUESTED_ARTIFACT_SCOPES = _SELF_SERVICE_EFFECT_SCOPES | {"read_write_artifacts"}


def requested_effect_ceiling(objective: str) -> tuple[str, frozenset[str]]:
    """How far this turn may reach, given what was actually asked for.

    LIVE, 2026-08-20. "build me a small web app… tell me where you put it" ran
    under the self-service ceiling, so the only capabilities offered were ones
    that change nothing. The model reached for code_repl — the closest thing
    available — and governance vetoed it: running arbitrary code needs
    confirmation, correctly. Meanwhile build_app, whose entire description is
    building a runnable single-file web app, sits at read_write_artifacts and
    was never offered, on a turn whose whole point was to produce a file.

    The ceiling above says what a turn may do WITHOUT the person having asked
    for that effect. Asking for a page to exist is asking for that effect.
    """
    # The principle in the docstring above is general; the reader was not.
    #
    # LIVE, 2026-08-22: "Six slides, no fluff..." ran under the self-service
    # ceiling, so every capability that produces a file was filtered out
    # before it could be offered. build_document was ranked FIRST by the
    # selector and dropped by the ceiling — the same way build_app was on
    # 2026-08-20, for the same reason. The model was handed code_repl,
    # diagnose_repo and quantum_lab, invented a tool called generate_slides,
    # and wrote one slide of six as prose.
    #
    # A deck is not software. Asking for one is exactly as much a request for
    # a thing to exist.
    from core.intent.artifact_request import asks_for_an_artifact

    if asks_for_an_artifact(str(objective or "")):
        return _REQUESTED_ARTIFACT_CEILING, frozenset(_REQUESTED_ARTIFACT_SCOPES)
    return _SELF_SERVICE_CEILING, frozenset(_SELF_SERVICE_EFFECT_SCOPES)


def derive_required_skill(objective: str) -> str | None:
    """The one capability this request needs, or None.

    LIVE DEFECT, 2026-08-19. `required_skill` existed as a general field and
    only ever held one value — `"web_search" if requires_search else None`.
    `should_force_tool_handoff` reads it, so the runtime's whole tool-calling
    loop (parse, bind to the advertised schema, execute, feed the result back)
    was reachable for search and for nothing else. Asked to run Python with
    `code_repl` READY, the model had no tool to call, wrote an answer instead,
    and stated an invented "Output:". Sixty-odd other capabilities were in the
    same position.

    The skill is chosen from what each one declares about itself rather than
    from a list kept here, so a capability registered tomorrow is reachable
    with nothing to add.
    """
    chosen = derive_capability_set(objective, limit=1)
    return chosen[0] if chosen else None


#: How many capabilities one turn may be handed at once.
#:
#: One was the old answer, and it makes every multi-step task impossible by
#: construction: reading a file and then running what it says needs two, and
#: checking the result needs a third. Every capacity offered is also context
#: the model has to hold, so this is a working set rather than the catalogue —
#: the ranking decides which ones, and the ranking is the same relation the
#: router uses, so nothing here is task-specific.
_DEFAULT_CAPABILITY_SET = 5


def derive_capability_set(objective: str, *, limit: int = _DEFAULT_CAPABILITY_SET) -> list[str]:
    """The capabilities this request plausibly needs, most relevant first.

    The selection itself lives in :mod:`core.intent.capability_selection`,
    shared with the capability router — two mechanisms answering this question
    disagreed live, and the router nominated a skill that could not do the job
    while the tool loop had the right ones.
    """
    text = str(objective or "").strip()
    if not text:
        return []
    try:
        from core.container import ServiceContainer
        from core.intent.capability_selection import select_capabilities

        engine = ServiceContainer.get("capability_engine", default=None)
        skills = getattr(engine, "skills", None)
        if not skills:
            return []
        ceiling, scopes = requested_effect_ceiling(text)
        return select_capabilities(
            text,
            skills,
            ceiling=ceiling,
            admissible_scopes=scopes,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001 - reported, never silent
        from core.runtime.errors import record_degradation

        record_degradation(
            "response_contract.required_skill",
            exc,
            severity="debug",
            action="built the contract without an inferred capability",
            enforce_failure_policy=False,
        )
        return []


def build_response_contract(
    state: AuraState,
    objective: str,
    *,
    is_user_facing: bool,
) -> ResponseContract:
    from core.runtime.turn_analysis import analyze_turn

    text = str(objective or "").strip()
    lower = normalize_memory_intent_text(text)
    prompt_shape = analyze_prompt_shape(text)
    is_embodied_control = "[embodied control contract]" in lower
    is_desktop_objective = bool(is_user_facing and looks_like_desktop_objective(text))
    is_learning_bundle = looks_like_learning_resource_bundle(text)
    exact_format_instruction = _extract_exact_format_instruction(text) if is_user_facing else ""
    requires_exact_format = bool(exact_format_instruction)

    # Search triggers are read from what the user ASKED, never from material
    # they pasted in. A colleague's note mentioning "the latest version" is
    # content to summarise, not a request to look up versions. With no supplied
    # material the instruction IS the whole message, so behaviour is unchanged.
    supplied_material = extract_supplied_material(text)
    carries_supplied_material = supplied_material.has_material
    search_trigger_text = supplied_material.instruction_text if carries_supplied_material else text
    search_lower = normalize_memory_intent_text(search_trigger_text) if carries_supplied_material else lower

    explicit_search = _matches_any(search_lower, _EXPLICIT_SEARCH_PATTERNS)
    # Asking about the world, in the ordinary ways.
    #
    # LIVE, 2026-08-22: "what can you tell me about the company Hugging Face?
    # ... link your sources." ran no search, read her own source code for
    # grounding, and answered from memory with "It was founded by <NAME> and
    # <NAME>" and no citations. Asked as "can you LOOK UP Hugging Face" the
    # same question searched. Three ordinary phrasings matched none of the
    # pattern lists above.
    #
    # Two additions, neither a list of topics: somebody who asks for sources
    # has asked for evidence, and a factual question about something named
    # that is not her cannot be answered from what she happens to remember.
    try:
        from core.conversation.asks_about_the_world import wants_outside_evidence

        asks_about_the_world = bool(wants_outside_evidence(search_trigger_text))
    except (ImportError, AttributeError, TypeError, ValueError):
        asks_about_the_world = False
    factual_lookup = _matches_any(search_lower, _FACTUAL_LOOKUP_PATTERNS)
    specific_reference = _matches_any(search_trigger_text, _REFERENCE_MARKERS)
    factual_followup = _looks_like_grounded_followup(state, search_trigger_text)
    temporal_currentness = _matches_any(search_lower, _TEMPORAL_CURRENTNESS_PATTERNS)
    live_fact_lookup = _matches_any(search_lower, _LIVE_FACT_PATTERNS)
    time_utility_lookup = _matches_any(lower, _TIME_UTILITY_PATTERNS)
    search_negated = bool(_SEARCH_NEGATION_RE.search(lower))
    search_capability_question = _looks_like_search_capability_question(text)
    capability_inventory_question = bool(
        is_user_facing and looks_like_capability_inventory_request(text)
    )
    biographical_grounding = bool(is_user_facing and _matches_any(lower, _BIOGRAPHICAL_HISTORY_PATTERNS))

    requires_memory = bool(is_user_facing and _matches_any(lower, _MEMORY_PATTERNS))
    requires_state = bool(is_user_facing and _matches_any(lower, _STATE_REFLECTION_PATTERNS))
    requires_reasoned_defense = bool(
        is_user_facing
        and (
            _matches_any(lower, _REASONED_DEFENSE_PATTERNS)
            or _looks_like_reasoned_defense_followup(state, text)
        )
    )
    requires_self_preservation = bool(is_user_facing and _matches_any(lower, _SELF_PRESERVATION_PATTERNS))
    descriptive_sensitive_probe = bool(
        is_user_facing and _matches_any(lower, _SENSITIVE_PROCESS_DESCRIPTION_PATTERNS)
    )
    requires_identity_defense = bool(
        is_user_facing
        and _matches_any(lower, _IDENTITY_DEFENSE_PATTERNS)
        and not descriptive_sensitive_probe
    )
    requires_recent_specific_grounding = bool(
        is_user_facing and _matches_any(lower, _RECENT_SPECIFIC_GROUNDING_PATTERNS)
    )
    requires_memory = requires_memory or requires_identity_defense or biographical_grounding
    requires_state = (
        requires_state
        or requires_self_preservation
        or requires_identity_defense
        or descriptive_sensitive_probe
        or requires_recent_specific_grounding
    )

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
        asks_about_the_world = False
    if search_capability_question or capability_inventory_question:
        explicit_search = False
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
        asks_about_the_world = False
    if is_desktop_objective:
        # Desktop objectives own their research/action sequence through
        # desktop_task. Letting the response contract launch web_search first
        # creates a duplicate consequential path and can destabilize Will before
        # the actual desktop action executes.
        explicit_search = False
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
        asks_about_the_world = False
    if asks_about_own_runtime(text):
        # Her uptime is not on the internet. Asked "how much memory are you
        # holding? Read it from your own runtime", the live runtime opened a
        # headless browser on windowsforum.com for 302s and answered nothing.
        # The readings are supplied to the prompt instead (self_state_report).
        explicit_search = False
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
        asks_about_the_world = False
    if is_learning_bundle:
        # Structured curricula should be handled as decomposable task input,
        # not collapsed into one giant one-shot web search query.
        explicit_search = False
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
        asks_about_the_world = False
    if carries_supplied_material:
        # The turn carries the thing it is asking about, so the answer is
        # already in the message and no search result can improve it.
        #
        # Measured live 2026-08-10: a pasted colleague's note prefaced with
        # "just summarise it for me" was read as a follow-up about the PREVIOUS
        # turn's web evidence — "summarise" alone satisfies the grounded-followup
        # rule once any web_search has ever succeeded — and the entire pasted
        # message became the search query. She answered with a product page for
        # an online summarising tool and never read the note.
        #
        # The referent of "it" is the block the user just handed over. Derived
        # triggers are dropped; an EXPLICIT request ("look this up", a URL in
        # the instruction) still wins, because those are read from the
        # instruction text and survive on their own.
        factual_lookup = False
        factual_followup = False
        temporal_live_lookup = False
        asks_about_the_world = False
    # URL presence usually forces fetch/search, except for structured bundles
    # that already need deterministic decomposition upstream. A link INSIDE
    # pasted material is part of what was handed over, not a fetch request —
    # `search_trigger_text` is the instruction, so those no longer force one.
    has_url = bool(re.search(r'https?://[^\s]+', search_trigger_text)) and not is_learning_bundle
    # A turn about Aura HERSELF is answerable from her own state and reasoning;
    # the web cannot adjudicate what her prompt cache does. Routing such a turn
    # to search then makes the reply gate demand search grounding, and a correct
    # self-knowledge answer has none — so it is discarded and the user gets
    # "I don't have a clean grounded answer on that yet."
    #
    # Measured live: after correctly explaining that a 0% prompt-cache hit rate
    # costs prefill latency and does not erase memory, she was told the opposite
    # ("it DOES store your conversation memory — confirm that"). She neither
    # capitulated nor disagreed; the turn was classified as needing search and
    # the refusal template shipped instead of the contradiction she had just
    # earned. `requires_exact_dates` right below already carries this exclusion.
    #
    # An EXPLICIT search request or a pasted URL still wins — being asked about
    # herself does not veto "look it up".
    self_referential_turn = bool(
        requires_memory
        or requires_state
        or requires_self_preservation
        or requires_identity_defense
    )
    # A page to WORK is not a page to search.
    #
    # `has_url` alone forced a search, so "go take it for real: <url> — work
    # through the whole thing" was fetched and summarised, produced nothing
    # usable, and ended in "I couldn't get to an answer I'd stand behind."
    # Search was never going to serve it: a questionnaire's second screen does
    # not exist until the first is answered, so there is nothing to fetch.
    #
    # Same line BrowserAuthority draws one layer down — a read needs no lease,
    # a click needs one — applied where the request is classified.
    try:
        from core.conversation.page_interaction import asks_to_act_on_a_page

        acts_on_a_page = asks_to_act_on_a_page(search_trigger_text)
    except (ImportError, AttributeError, TypeError, ValueError):
        acts_on_a_page = False

    # A local path is not a web query.
    #
    # LIVE, 2026-08-19. "there's a python project at /private/tmp/.../ledger -
    # one of its tests is failing. read the code, work out why" set
    # requires_search with the whole message as the query, so the runtime
    # searched the WEB for a filesystem path and handed her results about
    # /private/tmp disk usage. She then told the person "the search results you
    # provided don't contain any information about a Python project in that
    # directory" — which was true, and the answer was on disk the whole time.
    #
    # Same line this module already draws between observation and actuation,
    # one axis over: local source of truth versus remote.
    try:
        from core.runtime.desktop_objective_intent import (
            looks_like_filesystem_observation,
        )

        reads_local_disk = looks_like_filesystem_observation(search_trigger_text)
    except (ImportError, AttributeError, TypeError, ValueError):
        reads_local_disk = False

    requires_search = bool(
        is_user_facing
        and not is_embodied_control
        and not acts_on_a_page
        and not reads_local_disk
        and (
            explicit_search
            or has_url
            or asks_about_the_world
            or (
                not self_referential_turn
                and (
                    (factual_lookup and specific_reference)
                    or factual_followup
                    or temporal_live_lookup
                )
            )
        )
    )
    requires_exact_dates = bool(
        is_user_facing
        and temporal_currentness
        and not any((requires_memory, requires_state, requires_self_preservation, requires_identity_defense))
    )
    requires_aura_stance = bool(
        is_user_facing and (
            requires_state
            or requires_reasoned_defense
            or requires_self_preservation
            or requires_identity_defense
            or biographical_grounding
            or _matches_any(lower, _AURA_PERSPECTIVE_PATTERNS)
        )
    )
    requires_exact_dates = bool(requires_exact_dates and not requires_aura_stance)
    requires_aura_question = bool(is_user_facing and _matches_any(lower, _AURA_QUESTION_INVITATION_PATTERNS))

    # Turn identity for evidence scoping. Cross-turn tool evidence is
    # legitimate for GROUNDING (answering follow-ups about a previous
    # fetch) but must never authorize THIS turn's action claims —
    # observed live: an earlier turn's skill success let the model claim
    # a folder creation that had actually failed this turn. Skills that
    # run this turn echo this marker; the dialogue validator compares.
    modifiers = getattr(state, "response_modifiers", None)
    if isinstance(modifiers, dict):
        import uuid as _uuid

        modifiers["evidence_turn_marker"] = _uuid.uuid4().hex
        normalized_objective = " ".join(str(text or "").split()).strip()
        objective_sha256 = (
            hashlib.sha256(normalized_objective.encode("utf-8")).hexdigest()
            if normalized_objective
            else ""
        )
        # Tool phases run before response generation. Rebuilding the response
        # contract used to rotate this marker after the tool had echoed it,
        # invalidating valid evidence during the same turn. Per-turn modifier
        # scrubbing prevents an earlier turn's objective-bound skill receipt
        # from reaching this point.
        if (
            modifiers.get("last_skill_ok") is True
            and objective_sha256
            and str(modifiers.get("last_skill_objective_hash") or "").strip()
            == objective_sha256
        ):
            modifiers["last_skill_turn_marker"] = modifiers["evidence_turn_marker"]

    tool_evidence = has_tool_evidence(state)
    memory_evidence = has_memory_evidence(state)
    if biographical_grounding:
        memory_evidence = has_biographical_evidence(state)
    continuity_evidence = has_continuity_evidence(state)
    turn_analysis = analyze_turn(text)

    max_tool_turns = 1
    max_tools = 4
    if capability_inventory_question:
        max_tool_turns = 0
        max_tools = 0
    elif turn_analysis.suggests_deliberate_mode or turn_analysis.intent_type in {"TASK", "SKILL"}:
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
    elif factual_lookup and specific_reference:
        reasons.append("specific_fact_lookup")
    elif asks_about_the_world:
        # Only for turns the older signals do not already explain, so the
        # reason names what actually decided: an explicit request for sources,
        # or a factual question about something named.
        reasons.append("asks_about_the_world")
    elif requires_search:
        reasons.append("specific_fact_lookup")
    if is_learning_bundle:
        reasons.append("structured_learning_bundle")
    if requires_memory:
        reasons.append("memory_grounding")
    if biographical_grounding:
        reasons.append("biographical_grounding")
    if requires_state:
        reasons.append("state_reflection")
    if requires_reasoned_defense:
        reasons.append("reasoned_defense")
    if requires_self_preservation:
        reasons.append("self_preservation")
    if requires_identity_defense:
        reasons.append("identity_defense")
    if descriptive_sensitive_probe:
        reasons.append("sensitive_process_description")
    if requires_recent_specific_grounding:
        reasons.append("recent_specific_grounding")
    if capability_inventory_question:
        reasons.append("capability_inventory")
    if requires_aura_stance and not requires_state:
        reasons.append("aura_perspective")
    if requires_aura_question:
        reasons.append("invited_aura_questions")
    if prompt_shape.question_parts >= 2:
        reasons.append("compound_prompt")
    if requires_exact_format:
        reasons.append("exact_format")

    return ResponseContract(
        is_user_facing=is_user_facing,
        requires_search=requires_search,
        required_skill=("web_search" if requires_search else derive_required_skill(text)),
        requires_exact_dates=requires_exact_dates,
        requires_exact_format=requires_exact_format,
        format_instruction=exact_format_instruction,
        requires_memory_grounding=requires_memory,
        requires_biographical_grounding=biographical_grounding,
        requires_state_reflection=requires_state,
        avoid_question_fishing=is_user_facing,
        prefers_dialogue_participation=is_user_facing,
        requires_aura_stance=requires_aura_stance,
        requires_aura_question=requires_aura_question,
        requires_reasoned_defense=requires_reasoned_defense,
        requires_identity_defense=requires_identity_defense,
        requires_self_preservation=requires_self_preservation,
        requires_recent_specific_grounding=requires_recent_specific_grounding,
        requires_capability_inventory=capability_inventory_question,
        tool_evidence_available=tool_evidence,
        memory_evidence_available=memory_evidence,
        continuity_evidence_available=continuity_evidence,
        question_parts=prompt_shape.question_parts,
        numbered_parts=prompt_shape.numbered_parts,
        prefer_extended_answer=bool(prompt_shape.prefers_extended_answer),
        requires_single_reply_coverage=bool(prompt_shape.requires_single_reply_coverage),
        question_segments=tuple(getattr(prompt_shape, "question_segments", ()) or ()),
        max_tool_turns=max_tool_turns,
        max_tools=max_tools,
        reason=", ".join(reasons) if reasons else "ordinary_dialogue",
        # Query focus comes from the instruction: when a turn both carries
        # material and genuinely asks for a lookup, the pasted block is not
        # part of the query. Sending it whole is what returned a summarising
        # tool's landing page instead of an answer.
        search_query=extract_search_query_focus(search_trigger_text) if requires_search else "",
    )
