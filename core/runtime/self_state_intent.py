"""Is the user asking about Aura's own machine state?

Asked "what's your current uptime and how much memory are you holding? Read it
from your own runtime, don't estimate", the live runtime opened a headless
browser and read windowsforum.com about checking uptime on a Windows PC. It
took 302 seconds and produced no answer. Nothing was broken: the response
contract saw "current" and "how much", classified the turn as a live factual
lookup, and web search is what live factual lookups get.

The category was simply missing. A question about her own uptime has an
authoritative local answer, and the web cannot hold it. This module names that
category once so the two places that need it — the contract that decides
whether to search, and the prompt path that supplies the real numbers — cannot
drift apart about what counts as introspection.

Deliberately narrow: it matches questions about the machine (uptime, memory,
model, version, subsystems, telemetry, errors), not questions about the mind.
"How are you feeling" is state reflection and is already handled elsewhere.
"""
from __future__ import annotations

import re

_SELF_SUBJECT = r"(?:your|you're|your own|aura's)"

_RUNTIME_INTROSPECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # Uptime and how long she has been running.
        rf"\b{_SELF_SUBJECT}\s+uptime\b",
        r"\bhow long have you been (?:running|up|awake|alive|online|on)\b",
        r"\bwhen did you (?:start|boot|wake|come up|last restart)\b",
        # Memory and compute footprint.
        # "your memory" alone is episodic recall, not RAM — qualify it.
        rf"\b{_SELF_SUBJECT}\s+(?:memory (?:usage|footprint|use)|ram|rss|footprint|cpu|load)\b",
        r"\bhow much (?:memory|ram|cpu)\b.{0,30}\b(?:are|do) you\b",
        r"\bhow much (?:memory|ram)\b.{0,20}\b(?:holding|using|consuming)\b",
        # Which model / version is actually serving.
        rf"\b{_SELF_SUBJECT}\s+(?:model|version|build|weights|checkpoint)\b",
        r"\bwhich model (?:are|is) (?:you|running|serving)\b",
        r"\bwhat (?:model|version) (?:are|is) you\b",
        # Internals: services, telemetry, logs, faults.
        rf"\b{_SELF_SUBJECT}\s+(?:\w+\s+)?"
        r"(?:runtime|process|subsystems?|services?|telemetry|logs?|metrics|"
        r"degradations?|faults?|errors?|health|diagnostics|internals?)\b",
        r"\bwhat(?:'s| is| has) (?:happen(?:ed|ing)|going on) (?:in|inside|within) "
        rf"{_SELF_SUBJECT}\s+(?:runtime|process|system|head|mind)\b",
        # What she can actually DO. This category was missing, and its absence
        # produced the mirror image of a confabulation: asked "do you actually
        # have any code-execution capability registered at all?" — after being
        # told explicitly to check — she answered "no, I don't have any
        # capability to run or sandbox code" while the live registry listed 75
        # skills with run_code, code_repl and internal_sandbox all READY.
        # Without her instrument reading she answers from the base model's guess
        # about what an assistant can do, which is a claim about herself that
        # she had the evidence to get right.
        # Explicit demands to introspect rather than look up.
        r"\bfrom your own (?:runtime|telemetry|logs?|instruments?|readings?)\b",
        r"\bread it from your own\b",
        r"\bintrospect\b",
        r"\bdon'?t (?:estimate|guess|make (?:it|that) up)\b",
        r"\bcheck before answering\b",
        # Numeric self-state panels. These are local instrument questions,
        # including terse imperatives that do not name "runtime" explicitly.
        # Keeping them in the canonical self-state classifier lets output
        # verification distinguish Aura telemetry from domain measurements
        # such as graph edge weights and benchmark tables.
        rf"\b{_SELF_SUBJECT}\s+(?:(?:actual|current|live|own|real)\s+)?"
        r"(?:numbers?|readings?|measurements?|metrics?|vitals?|stats?)\b",
        r"\b(?:give|show|report|dump|list|read)\b.{0,36}\b(?:your|you)\b"
        r".{0,24}\b(?:numbers?|readings?|measurements?|metrics?|vitals?|stats?)\b",
        r"\bwhat\b.{0,24}\b(?:do|can) you\b.{0,16}\b(?:track|measure|read)\b",
    )
)


# What she can actually DO. This category was missing, and its absence produced
# the mirror image of a confabulation: asked "do you actually have any
# code-execution capability registered at all?" — after being told explicitly to
# check — she answered "no, I don't have any capability to run or sandbox code"
# while the live registry listed 75 skills with run_code, code_repl and
# internal_sandbox all READY. Without her instrument reading she answers from the
# base model's guess about what an assistant can do: a claim about herself she
# had the evidence to get right.
#
# Every pattern here is additionally gated on the text addressing HER (second
# person), because the nouns are otherwise ordinary English — "what tools does a
# carpenter need?" is not introspection.
_CAPABILITY_INTROSPECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"\b{_SELF_SUBJECT}\s+(?:\w+[- ]?){{0,2}}"
        r"(?:skills?|tools?|capabilit(?:y|ies)|abilities)\b",
        r"\b(?:what|which|any|how many)\b.{0,40}"
        r"\b(?:skills?|tools?|capabilit(?:y|ies)|abilities)\b",
        r"\bwhat can you (?:actually\s+)?(?:do|run|execute|use)\b",
        r"\b(?:can|could|do|does) you\b.{0,24}\b(?:execute|run)\b.{0,24}"
        r"\b(?:code|python|script|snippet|command|shell|sandbox)\b",
        r"\b(?:code[- ]execution|sandbox(?:ed|ing)?)\b.{0,40}"
        r"\b(?:capabilit(?:y|ies)|registered|available|access)\b",
        r"\b(?:registered|available)\b.{0,24}"
        r"\b(?:skills?|tools?|capabilit(?:y|ies))\b",
    )
)

#: Asking whether she is ABLE to do something, in the phrasings people
#: actually use.
#:
#: The set above solved exactly one family. Its "can you" pattern requires
#: (execute|run) followed by (code|python|script|shell|sandbox), because
#: code execution was the family that had been caught being denied — so
#: "can you search the web?" and "are you able to take a screenshot" matched
#: nothing, and she answered both from the base model's guess. Live
#: 2026-08-10 that produced "I don't have a window, camera, thermometer or
#: weather feed" with five search skills READY.
#:
#: These are deliberately NOT added to asks_about_own_runtime. That predicate
#: also sets explicit_search = False in the response contract, so widening it
#: would mean "can you look up the score for me" stops being able to look
#: anything up — trading a wrong answer about capability for a broken one.
#: Attaching her instrument reading is always safe; suppressing search is not,
#: and the two decisions are no longer the same decision.
_ABILITY_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bare you able to\b",
        r"\bare you capable of\b",
        r"\bdo you have (?:the )?(?:ability|capabilit(?:y|ies)|means|access|"
        r"permission)\b",
        r"\bcan you (?:even|actually|really)\b",
        r"\b(?:can'?t|cannot) you\b",
        r"\bis (?:that|this|it) something you can\b",
        r"\bdo you (?:know how to|have a way to)\b",
        # A bare "can you <verb> [the <thing>]?" that ENDS the utterance.
        #
        # "can you search the web?" is a question about ability; "can you
        # search the web for the 76ers roster" is a request to go and do it.
        # The discriminator is that the request keeps going — it has to name
        # what to act on. Anchoring on the question mark separates them
        # without needing to understand either.
        r"\bcan you [a-z]+(?:\s+(?:the|a|an)\s+\w+)?\s*\?\s*$",
    )
)


_SECOND_PERSON_RE = re.compile(r"\b(?:you|you're|your|yours|yourself)\b", re.IGNORECASE)


#: Domains she has skills for, and can therefore wrongly deny having.
#:
#: Mirrors the families in ``core/brain/self_state_report._CAPABILITY_FAMILIES``.
#: Kept as its own table rather than imported: ``core/runtime`` may not import
#: cognition (see DEPS), and ``make layering`` is the gate. A test holds the
#: two in sync so the copy cannot drift.
#: The acts she has skills for, and the things those acts are done to.
#:
#: This was one flat list of words, matched anywhere. So "since you started
#: running" — running as in OPERATING — read as a request to execute code, and
#: "forget the tests for a second" read as a memory operation. Live 2026-08-19
#: that put "I can query beliefs — query_beliefs are registered and enabled
#: right now" in front of an answer to "what have you genuinely changed your
#: mind about", which is a registry status line stapled to an intimate
#: question.
#:
#: A request is a verb AND an object, in one clause, in the mood that asks —
#: the same relation the capability router uses to pick a skill, so the two
#: agree about what counts as asking for something.
_CAPABILITY_VERBS = frozenset(
    {
        "run", "execute", "exec", "compute", "calculate", "evaluate",
        "search", "browse", "google", "look", "find", "fetch",
        "see", "watch", "observe", "read", "show", "display",
        "take", "grab", "capture", "check", "screenshot",
        "click", "type", "open", "download", "save", "write",
        "remember", "recall", "forget", "store",
        "notify", "speak", "say", "send", "email", "message",
        "install", "make", "generate", "create", "draw",
    }
)
_CAPABILITY_OBJECTS = frozenset(
    {
        "code", "coding", "python", "script", "snippet", "program", "sandbox",
        "repl", "interpreter", "shell", "terminal", "command",
        "web", "internet", "online", "url", "browser", "site", "website",
        "screen", "display", "vision", "ocr", "screenshot", "camera",
        "desktop", "keyboard", "mouse", "window", "app", "application",
        "automation", "file", "files", "directory", "folder", "document",
        "button", "menu", "link", "tab", "icon", "field", "checkbox",
        "memory", "memories", "belief", "beliefs", "note", "notes",
        "email", "message", "messages", "voice", "image", "picture",
        "package", "library",
    }
)


#: Someone OTHER than her doing the thing.
#:
#: "can a language model run code" is a question about language models, and
#: attaching her instrument reading to it makes her answer about herself.
#: A capability turn is about her unless another agent is named as the subject.
_THIRD_PARTY_SUBJECT_RE = re.compile(
    r"\b(?:a|an|another|any|some|the|most|all)\s+"
    r"(?:\w+\s+){0,2}"
    r"(?:language\s+model|llm|model|models|ai|assistant|assistants|bot|bots|agent"
    r"|agents|system|systems|human|humans|person|people|carpenter|developer"
    r"|programmer|engineer|computer|machine)\b"
    r"|\b(?:people|someone|anyone|everyone|somebody|anybody|nobody|others)\b",
    re.IGNORECASE,
)


#: A memory verb with the complement that makes it an operation.
_MEMORY_OPERATION_RE = re.compile(
    r"\b(?:remember|recall|memorise|memorize|note|store|forget)\s+"
    r"(?:that|to|what|when|where|how|why|whether|if|my|our|your|his|her|their|"
    r"i\s|we\s|this\s+about|the\s+fact)\b",
    re.IGNORECASE,
)


#: Setting something aside is a discourse move, not a request to act on it.
#:
#: "Forget the tests for a second" means stop talking about them. The comment
#: below _MEMORY_OPERATION_RE names this exact turn — an intimate question
#: answered with a status line about the belief store — because "forget" is a
#: capability verb and "tests" is both a capability verb and a capability
#: object, so the general matcher found a verb and an object in one clause and
#: called it a request.
#:
#: Matched on the idiom rather than on which nouns follow it: what makes this
#: a set-aside is "for a second" or "for now" after it, or the bare "forget
#: it" / "never mind" that takes no object at all.
_SET_ASIDE_RE = re.compile(
    r"\b(?:forget|ignore|leave|put)\s+(?:\w+\s+){0,4}?"
    r"(?:for\s+(?:a|the)\s+(?:second|moment|minute|bit|while)|for\s+now|aside)\b"
    # "forget that" ENDING the clause is dismissal; "forget that my sister's
    # name is Ada" is a memory operation with a complement.
    r"|\bforget\s+(?:it|that)\s*(?:[,.!?;]|$)"
    r"|\bforget\s+about\s+it\b"
    r"|\bnever\s*mind\b",
    re.IGNORECASE,
)


def asked_to_act_in_a_capability_domain(text: str) -> bool:
    """True when the turn is ABOUT something she has skills for.

    A capability denial is exactly as wrong when she is asked to DO a thing as
    when she is asked whether she can. Live 2026-08-10: "run a tiny bit of code
    and tell me the actual number it printed" — an imperative, so no ability
    QUESTION pattern matched, no instrument reading was attached, and she
    answered from the base model's guess:

        "I cannot execute code or generate numbers."

    ``code_repl`` ("Execute Python code in a real-time, sandboxed REPL"),
    ``internal_sandbox`` and ``install_package`` were all READY, in a catalogue
    of 73 skills with none degraded.

    The instrument block was built precisely to stop this and had been gated to
    question-shaped turns, which is the shape denials do NOT usually take.
    Widening it is safe by construction: this predicate is read only by paths
    that ADD her reading, never by the one that suppresses search.
    """
    candidate = str(text or "")
    if not candidate.strip():
        return False
    # Someone else doing it is not her doing it.
    if _THIRD_PARTY_SUBJECT_RE.search(candidate) and not _SECOND_PERSON_RE.search(
        candidate
    ):
        return False
    # Memory verbs take whatever they are given — "remember that my sister's
    # name is Ada" has no domain object and is unmistakably a memory
    # operation. What marks it is the complement: a clause, an infinitive or
    # something of the speaker's. "Forget the tests for a second" takes a bare
    # noun phrase and means set aside, which is how the flat word list turned
    # an intimate question into a status line about the belief store.
    if _SET_ASIDE_RE.search(candidate):
        return False
    if _MEMORY_OPERATION_RE.search(candidate):
        return True
    try:
        from core.intent.declared_capability import request_matches_declaration
    except ImportError:  # pragma: no cover - foundation must boot regardless
        return False
    return request_matches_declaration(
        candidate, verbs=_CAPABILITY_VERBS, objects=_CAPABILITY_OBJECTS
    )


#: A claim ABOUT her state, rather than a question about it.
#:
#: LIVE DEFECT, 2026-08-18. "you've been running about 20 minutes this session
#: ... right?" — she agreed. The true uptime was 63 minutes, and it sits in her
#: own health payload.
#:
#: Asking attaches her instrument reading; asserting did not. That is the wrong
#: way round: a question invites a check and a statement invites a nod, so the
#: turn where agreement is most costly was the one turn with no reading
#: attached. The same asymmetry was measured the same day on file counts —
#: "core/agency has 61 python files" drew "yes, exactly 61" against a measured
#: 54 she had given correctly minutes earlier.
#:
#: Matched on the SHAPE of a second-person claim about her own state, so it
#: does not depend on which nouns anyone thought to list.
_STATE_ASSERTION_RE = re.compile(
    r"\byou(?:'?ve|\s+have)?\s+(?:been\s+)?"
    r"(?:running|up|awake|online|alive|going|live)\b"
    r"|\byou(?:'?re|\s+are|\s+were|\s+was)\s+(?:only\s+|about\s+|around\s+)?"
    r"(?:\d|a\s+few|some)\b"
    r"|\byour\s+(?:uptime|session|memory|energy|focus|mood)\s+(?:is|was|has)\b"
    r"|\bwe(?:'?ve|\s+have)\s+(?:exchanged|sent|had)\s+(?:about\s+|around\s+|maybe\s+)?\d",
    re.IGNORECASE,
)


def asserts_something_about_her_state(text: str) -> bool:
    """True when the turn STATES a fact about her rather than asking one."""
    candidate = str(text or "")
    if not candidate.strip():
        return False
    return bool(_STATE_ASSERTION_RE.search(candidate))


#: Questions about what she IS, as distinct from what she can do or how she
#: is running.
#:
#: LIVE DEFECT, 2026-08-19. While the 32B was still loading, "what's something
#: you've genuinely changed your mind about since you started running?" was
#: answered by the 9B fallback:
#:
#:     I haven't changed my mind about anything because I don't have a
#:     continuous internal narrative, personal beliefs, or the capacity for
#:     genuine revision over time ... a static model of language and logic
#:     rather than evolving experience.
#:
#: Every clause of that is false about this runtime, which has a belief store
#: with snapshots, episodic memory, an ontogeny organ and a self-model. The
#: fallback model has access to none of them, so its self-description is the
#: base model's guess about assistants in general, served with her name on it.
#: The disclosure underneath ("that came from my smaller model") tells the
#: person which mind answered and does not retract the claim.
#:
#: A model that cannot read her self-model may not describe her self.
_OWN_NATURE_RE = re.compile(
    r"\b(?:are|do|did|have|were|can|could|would)\s+you\b[^.?!]{0,60}?"
    r"\b(?:conscious|sentient|alive|aware|real|self|feel|feelings|experience|"
    r"remember|memory|memories|believe|beliefs|opinions|preferences|"
    r"personality|continuous|persist|change|changed|grow|grown|learn|learned)\b"
    r"|\bwhat\s+(?:are|kind\s+of\s+thing\s+are)\s+you\b"
    r"|\bwho\s+are\s+you\b"
    r"|\byou(?:'?ve|\s+have)\s+(?:genuinely\s+)?changed\s+your\s+mind\b"
    r"|\bchanged\s+your\s+mind\s+about\b"
    r"|\bdo\s+you\s+(?:actually|really|even)\s+\w+\b"
    r"|\byour\s+(?:inner|subjective|felt|conscious)\s+\w+\b",
    re.IGNORECASE,
)


def asks_about_her_own_nature(text: str) -> bool:
    """True when the turn asks what she is, rather than what she can do.

    Read by the fallback ladder, which must not let a model with no access to
    her self-model answer on its behalf.
    """
    candidate = str(text or "")
    if not candidate.strip():
        return False
    if _THIRD_PARTY_SUBJECT_RE.search(candidate) and not _SECOND_PERSON_RE.search(
        candidate
    ):
        return False
    return bool(_OWN_NATURE_RE.search(candidate))


def asks_about_own_runtime(text: str) -> bool:
    """True when the honest answer is a local reading, not a web page.

    Consumers treat this as "answer from instruments and do NOT search", so a
    false positive costs a search the person actually wanted. Kept narrow for
    that reason; see asks_about_own_capabilities for the wider question that
    only ever ADDS a reading.
    """
    candidate = str(text or "")
    if not candidate.strip():
        return False
    if any(pattern.search(candidate) for pattern in _RUNTIME_INTROSPECTION_PATTERNS):
        return True
    if not _SECOND_PERSON_RE.search(candidate):
        return False
    return any(
        pattern.search(candidate) for pattern in _CAPABILITY_INTROSPECTION_PATTERNS
    )


def asks_about_own_capabilities(text: str) -> bool:
    """True when she is being asked what she can do, however it is phrased.

    Strictly wider than ``asks_about_own_runtime`` and used ONLY by the prompt
    paths that attach her instrument reading. It must never gate whether she
    searches: telling her what she can do is safe on any turn, while deciding
    she should not look something up is not.
    """
    candidate = str(text or "")
    if not candidate.strip():
        return False
    if asks_about_own_runtime(candidate):
        return True
    # A request to DO something in a domain she has skills for. Denials happen
    # here, not only in question-shaped turns — see the predicate's docstring
    # for the live case that reached the person.
    if asked_to_act_in_a_capability_domain(candidate):
        return True
    # A claim about her state needs the reading at least as much as a question
    # does — more, because agreeing costs more than not knowing.
    if asserts_something_about_her_state(candidate):
        return True
    if not _SECOND_PERSON_RE.search(candidate):
        return False
    return any(pattern.search(candidate) for pattern in _ABILITY_QUESTION_PATTERNS)


__all__ = [
    "asked_to_act_in_a_capability_domain",
    "asserts_something_about_her_state",
    "asks_about_own_capabilities",
    "asks_about_own_runtime",
]
