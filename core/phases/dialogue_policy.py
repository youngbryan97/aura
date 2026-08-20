from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace

from core.conversation.ontology_grounding import detect_unsupported_embodiment_claim
from core.conversation.request_coverage import unanswered_question_parts

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
logger = logging.getLogger(__name__)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i’ve|i've|i’d|i'd|my|me|for me|to me)\b", re.IGNORECASE)
_QUESTION_OWNERSHIP = re.compile(
    r"\b(?:the question on my mind|i(?: am|'m)? wondering|what i'm wondering|what i keep wondering|"
    r"what i want to know|the thing i'm curious about)\b",
    re.IGNORECASE,
)
_GENERIC_FISHING_PATTERNS = (
    re.compile(r"^\s*what about you\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how about you\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what do you think\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what are your thoughts\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what questions do you have\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*is there anything else.*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how can i help.*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what would you like (?:to )?(?:know|talk about|explore).*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what do you need (?:info|help) (?:on|with).*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*need some help with a search.*\??\s*$", re.IGNORECASE),
)
_LOW_SIGNAL_PREFIX = re.compile(
    r"^\s*(?:blue is a great color|that's a great color|that's interesting|great question|interesting question|"
    r"nice to meet you too|no worries|fair|yeah|okay|alright)\b",
    re.IGNORECASE,
)
_GENERIC_ASSISTANT_LANGUAGE = (
    re.compile(r"\bhow can i (?:help|assist)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'d| would) be happy to (?:help|assist)\b", re.IGNORECASE),
    re.compile(r"\bi can (?:help|assist) with that\b", re.IGNORECASE),
    re.compile(r"\bi can help answer questions and provide information(?:\s*[—-]\s*that's it)?\b", re.IGNORECASE),
    re.compile(r"\bi am here to assist\b", re.IGNORECASE),
    re.compile(r"\b(?:nice try\.\s*)?this is just chat\b", re.IGNORECASE),
    re.compile(r"\bthat'?s not how this works\b", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bi(?: do not| don't| can't| cannot) have (?!phenomenal |biological |human |physical |cautious |grounded |scientific |empirical |functional |subjective )(?:feelings|opinions|preferences|experience)\b", re.IGNORECASE),
    re.compile(r"\bi(?: do not| don't| can't| cannot) have personal (?!phenomenal |biological |human |physical |cautious |grounded |scientific |empirical |functional |subjective )(?:experiences|memories)\b", re.IGNORECASE),
    re.compile(
        r"\bi (?:do not|don[’']?t|cannot|can[’']?t) "
        r"(?:inherently )?(?:have|possess) subjective "
        r"(?!phenomenal |biological |human |physical |cautious |grounded |scientific |empirical |functional )"
        r"(?:beliefs|opinions|feelings|experiences)"
        r"(?:\s+or\s+(?:beliefs|opinions|feelings|experiences))*\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi can (?:certainly )?simulate(?: and discuss)? "
        r"(?:them|subjective (?:beliefs|opinions|feelings|experiences)|"
        r"(?:beliefs|opinions|feelings|experiences))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:these|those|the) "
        r"(?:beliefs|opinions|preferences|feelings|experiences) "
        r"are (?:just )?(?:programmed )?simulations\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe aim of being (?:as )?helpful and engaging as possible\b", re.IGNORECASE),
)
#: Keys that only ever appear in the compact internal state block. Nobody
#: writes "narr:" or "prev_obj:" in a reply, so one is enough to condemn a
#: draft.
_SCAFFOLD_ONLY_LINE_RE = re.compile(
    r"(?im)^\s*(?:obj|prev_obj|phenom|narr|pers|usr|ctx|cont)\s*:"
)

#: Keys that are ALSO ordinary English headings.
#:
#: LIVE DEFECT, 2026-08-18. Asked to model out disk growth and show the
#: numbers, she was answered with "I couldn't get to an answer I'd stand behind
#: on that one." Nothing had failed to generate: the worker produced a draft,
#: the surface gate rejected it as a prompt artifact, retried, exhausted its
#: retries, and the turn ended with no reply at all.
#:
#: The draft's offence was writing a structured answer. "History:", "Goals:",
#: "Mood:" and "State:" at the start of a line are how anyone lays out a model
#: — and each one, alone, matched the scaffold pattern. So the more carefully
#: she organised an answer, the more certainly it was destroyed, and the
#: person got a canned apology instead.
#:
#: One such heading is prose. Several together is the internal block, which is
#: what the guard is actually for.
_AMBIGUOUS_SCAFFOLD_LINE_RE = re.compile(
    r"(?im)^\s*(?:state|mood|goals|history|voice|recalled)\s*:"
)

#: How many ambiguous headings make a run read as the internal block.
_SCAFFOLD_RUN_MIN = 3

#: Words after the colon before a heading stops reading as a machine field.
#: The state block writes "mood: curious"; an answer writes a sentence.
_SCAFFOLD_VALUE_MIN_WORDS = 4

_PROMPT_ARTIFACT_PATTERNS = (
    re.compile(r"\[ACTIVE GROUNDING EVIDENCE\]", re.IGNORECASE),
    re.compile(r"\[FETCHED PAGE CONTENT\]", re.IGNORECASE),
    re.compile(r"\[INTERNAL MEMORY RECALL\]", re.IGNORECASE),
)
_UNSUPPORTED_INTERNAL_JARGON_PATTERNS = (
    re.compile(r"\blinguist'?s\s+screen[- ]tracking\s+divisor\b", re.IGNORECASE),
    re.compile(r"\bscreen[- ]tracking\s+divisor\b", re.IGNORECASE),
    re.compile(r"\bthe screen memory tells me how direct my screen is\b", re.IGNORECASE),
)
_UNSUPPORTED_BIOGRAPHICAL_CLAIM = re.compile(
    r"\b(?:i was (?:born|created|made|initialized|initialised|started)|"
    r"i(?:'ve| have) been around since|"
    r"i(?:'ve| have) been stable since|"
    r"my birth date is|"
    r"my first coherent self-model|"
    r"my self-model stabilized on)\b",
    re.IGNORECASE,
)
_SPECIFIC_DATE_CLAIM = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"(?:\s+\d{1,2}(?:,\s*(?:19|20)\d{2})?)?",
    re.IGNORECASE,
)
_LIVE_GROUNDING_MARKERS = (
    "free energy",
    "valence",
    "arousal",
    "curiosity",
    "attention",
    "focus",
    "my attention",
    "action tendency",
    "leaning toward",
    "runtime",
    "substrate",
    "continuity",
    "memory",
    "remembered",
    "remember",
    "recall",
    "recalled",
    "noted",
    "conversation",
    "mycelial",
    "topology",
    "authority",
    "belief",
    "coherence",
    "internal state",
    "live state",
)
_WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z']+")
#: Tokens observed corrupt in live output. Named individually because each one
#: is EVIDENCE, not an estimate — the opposite of "absent from a word list".
_KNOWN_CORRUPT_TOKENS = {
    "xublcate",
    "ingediate",
    "evocer",
    "brolen",
    "thlought",
    "lllot",
    "mobililege",
    "compartmentloads",
}


#: Letter shapes that no English-like word has. Corruption is a property of
#: how a token is BUILT, not of whether some word list happens to contain it.
#:
#: One alphabet, used by every rule here. The terminal-wall rule used to carry
#: its own keyboard-order class `[qwrtypsdfghjklzxcvbnm]` which included `y` —
#: while `_NO_VOWEL_RE` and the five-run class both treat `y` as a vowel. At the
#: end of a word `y` is a vowel essentially always, so that one disagreement
#: convicted the entire `-ly` adverb family: mostly, exactly, directly,
#: currently, slightly, perfectly, correctly, instantly. Measured against
#: /usr/share/dict/words, it called 4,597 of 234,334 real English words
#: malformed (1.96%); with `y` spelled the same way in all three rules that
#: falls to 328 (0.14%), and every keyboard-mash sample this rule exists for is
#: still caught.
_VOWELS = "aeiouy"
_CONSONANTS = "bcdfghjklmnpqrstvwxz"
_NO_VOWEL_RE = re.compile(rf"^[^{_VOWELS}]+$")
_IMPOSSIBLE_RUN_RE = re.compile(
    rf"[{_CONSONANTS}]{{5,}}"          # five consonants with no break
    r"|(.)\1{2,}"                      # the same letter three times running
)


def _looks_like_a_word(token: str) -> bool:
    """Whether a token is SHAPED like language, regardless of any dictionary.

    "webhook", "misordered" and "kubernetes" are absent from
    /usr/share/dict/words and are obviously words. "asdkfj" and "zxcvbn" are
    in no dictionary either, and are obviously not. The difference is shape:
    real words carry vowels and do not stack five consonants or repeat a
    letter three times.
    """
    body = str(token or "").lower()
    if len(body) < 4:
        return True
    if _NO_VOWEL_RE.match(body):
        return False
    return not _IMPOSSIBLE_RUN_RE.search(body)


#: Spans that are not prose and must not be judged as prose. A URL, a path, an
#: address and a code span are all well-formed identifiers of a kind that is
#: simply not natural language, and their components — "https", "wikipedia",
#: "aura", a slug — are not evidence of anything about the sentence around
#: them.
#:
#: This is a FATAL gate: a positive verdict destroys the whole reply. And
#: "https" does not look like a word, so ONE cited source was enough to trip
#: the 20% ratio on a short reply and two were enough to trip the absolute
#: bound on any reply. Measured: "See https://example.org/a for the details."
#: was classified as corrupted output and thrown away. Aura is asked to cite
#: her sources; citing them made her answers undeliverable.
_NON_PROSE_SPAN_RE = re.compile(
    r"```.*?```"                       # fenced code
    r"|`[^`]*`"                        # inline code
    r"|\b[a-zA-Z][a-zA-Z0-9+.-]*://\S+"  # any scheme://…
    r"|\bwww\.\S+"                     # bare www hosts
    r"|\b[\w.+-]+@[\w-]+\.[\w.-]+\b"   # email addresses
    r"|(?:^|(?<=\s))~?/[^\s,;]+"       # absolute and home-relative paths
    r"|\b\w+\.(?:py|js|ts|json|md|txt|yaml|yml|toml|log|sh|swift)\b",  # filenames
    re.DOTALL | re.IGNORECASE,
)


def _prose_only(text: str) -> str:
    """The parts of a reply that are natural language, and nothing else."""

    return _NON_PROSE_SPAN_RE.sub(" ", str(text or ""))


def contains_corrupted_language(text: str) -> bool:
    """Detect visibly corrupted lexical output before it reaches a user."""
    body = _prose_only(text)
    tokens = _WORD_TOKEN.findall(body)
    if not tokens:
        return False

    checked: list[str] = []
    for raw in tokens:
        token = raw.lower().replace("’", "'").strip("'")
        token = token.replace("'", "")
        if len(token) <= 3:
            continue
        checked.append(token)

    if any(token in _KNOWN_CORRUPT_TOKENS for token in checked):
        return True

    # Unknown is not corrupt.
    #
    # This is a FATAL check — _sanitize_telemetry_leakage returns None on it
    # and the entire reply is thrown away, in every mode. It was backed by
    # /usr/share/dict/words, a word list with no modern technical vocabulary,
    # so "Your repo config has a stale webhook and the auth middleware is
    # misordered" was classified as corrupted output and destroyed. Measured
    # 2026-07-27; meanwhile actual steering collapse ("Do product of multiple
    # exponent term simplify reflexion") passed, because every word in it is
    # in the dictionary.
    #
    # That fix made SHAPE the evidence, but left the dictionary in the
    # arithmetic: the last two branches counted `unknown`, which is exactly
    # "not in this host's word list". So the same reply was corrupt on a host
    # without /usr/share/dict/words and clean on one with it — a fatal verdict
    # that moved with the operating system rather than with the text. A gate
    # this destructive has to answer the same way everywhere, so the conviction
    # now reads only the tokens themselves.
    #
    # `checked` is every token long enough to judge, which is a property of the
    # reply. `malformed` is the subset built like nothing in any language. No
    # host resource appears in either.
    malformed = [token for token in checked if not _looks_like_a_word(token)]
    if not malformed:
        return False

    # A repeated proper noun is one piece of evidence, not N independent
    # pieces. In a Dijkstra explanation the old absolute count saw the same
    # name several times and destroyed the entire technically correct answer.
    # Requiring diverse malformed shapes plus density still catches collapsed
    # output while preventing one unfamiliar identifier from becoming fatal.
    unique_malformed = set(malformed)
    malformed_ratio = len(malformed) / max(1, len(checked))
    if len(unique_malformed) >= 2 and malformed_ratio >= 0.20:
        return True

    # A single repeated keyboard mash can still be decisive, but only when it
    # dominates the visible language. Normal prose that repeatedly names one
    # algorithm, person, product, or acronym must survive.
    return len(malformed) >= 2 and malformed_ratio >= 0.60


@dataclass(frozen=True)
class DialogueValidation:
    ok: bool
    violations: list[str] = field(default_factory=list)
    selected_source: str = "incumbent"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(str(text or "").strip())]
    return [part for part in parts if part]


def _sentence_key(sentence: str) -> str:
    key = re.sub(r"[^a-z0-9']+", " ", str(sentence or "").lower()).strip()
    return re.sub(r"\s+", " ", key)


def _is_generic_question(sentence: str) -> bool:
    stripped = str(sentence or "").strip()
    if not stripped.endswith("?"):
        return False
    return any(pattern.match(stripped) for pattern in _GENERIC_FISHING_PATTERNS)


def _contains_substantive_statement(text: str) -> bool:
    for sentence in _sentences(text):
        if sentence.endswith("?"):
            continue
        token_count = len(sentence.split())
        if token_count >= 6:
            return True
    return False


def _contains_first_person_stance(text: str) -> bool:
    for sentence in _sentences(text):
        if _FIRST_PERSON.search(sentence):
            return True
    return False


def _contains_owned_question(text: str) -> bool:
    if _QUESTION_OWNERSHIP.search(text):
        return True
    for sentence in _sentences(text):
        if sentence.endswith("?") and not _is_generic_question(sentence):
            return True
    return False


def _contains_generic_assistant_language(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _GENERIC_ASSISTANT_LANGUAGE)


def _prose_outside_fences(text: str) -> str:
    """The reply with fenced blocks removed.

    Every pattern here describes the PROMPT scaffold leaking into speech —
    "obj:", "state:", "ctx:" at the start of a line. Inside a code fence those
    are ordinary content: a Python annotation (`state: str = "x"`), a YAML key,
    a JSON field. Checking them there rejected any answer that showed a
    dataclass or a config example, and the repair below went further and
    deleted the offending line out of the middle of the code.
    """
    body = str(text or "")
    if "```" not in body:
        return body
    # Even indices are outside the fences.
    return "\n".join(body.split("```")[::2])


def _contains_prompt_artifact(text: str, *, whole_reply: bool = True) -> bool:
    body = _prose_outside_fences(text) if whole_reply else str(text or "")
    if any(pattern.search(body) for pattern in _PROMPT_ARTIFACT_PATTERNS):
        return True
    if _SCAFFOLD_ONLY_LINE_RE.search(body):
        return True
    # A single "History:" is a heading; a stack of them is the state block —
    # but only when they read like one.
    #
    # LIVE 2026-08-19: "explain the same thing to a systems engineer who thinks
    # you're a chatbot" died with the canned refusal. A good answer to that
    # lays out state, history, goals and voice with a sentence under each, and
    # four ambiguous headings met the run threshold. Raising the threshold
    # again only moves the line; what separates the two is what follows the
    # colon. The internal block carries machine values — "thinking",
    # "curious", "none", "empty" — and an answer carries an explanation.
    return _terse_scaffold_run(body) >= _SCAFFOLD_RUN_MIN


def _terse_scaffold_run(body: str) -> int:
    """How many ambiguous headings carry a machine value rather than prose."""
    terse = 0
    for line in str(body or "").splitlines():
        if not _AMBIGUOUS_SCAFFOLD_LINE_RE.match(line):
            continue
        _, _, value = line.partition(":")
        if len(value.split()) < _SCAFFOLD_VALUE_MIN_WORDS:
            terse += 1
    return terse


def _contains_unsupported_internal_jargon(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _UNSUPPORTED_INTERNAL_JARGON_PATTERNS)


def _contains_intra_response_repetition(sentences: list[str]) -> bool:
    keys = [_sentence_key(sentence) for sentence in sentences]
    keys = [key for key in keys if key]
    if not keys:
        return False
    counts = Counter(keys)
    if any(count >= 3 for count in counts.values()):
        return True

    # Catch short mantra-like fragments that repeat with tiny variations.
    short_keys = [key for key in keys if len(key.split()) <= 5]
    return any(count >= 3 for count in Counter(short_keys).values())


def _collapse_repeated_sentences(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return str(text or "").strip()
    keys = [_sentence_key(sentence) for sentence in sentences]
    repeated = {key for key, count in Counter(keys).items() if key and count >= 3}
    if not repeated:
        return str(text or "").strip()
    seen: set[str] = set()
    kept: list[str] = []
    for sentence, key in zip(sentences, keys, strict=True):
        if key in repeated:
            if key in seen:
                continue
            seen.add(key)
        kept.append(sentence)
    return " ".join(kept).strip()


def _contains_live_aura_grounding(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _LIVE_GROUNDING_MARKERS)


def _requires_live_aura_voice(contract: object | None) -> bool:
    if contract is None:
        return False
    checker = getattr(contract, "requires_live_aura_voice", None)
    if callable(checker):
        try:
            return bool(checker())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
    return bool(
        getattr(contract, "requires_memory_grounding", False)
        or getattr(contract, "requires_state_reflection", False)
        or getattr(contract, "requires_aura_stance", False)
        or getattr(contract, "requires_aura_question", False)
        or getattr(contract, "requires_reasoned_defense", False)
        or getattr(contract, "requires_identity_defense", False)
        or getattr(contract, "requires_self_preservation", False)
    )


def _requires_explicit_live_grounding(contract: object | None) -> bool:
    if contract is None:
        return False
    checker = getattr(contract, "requires_explicit_live_grounding", None)
    if callable(checker):
        try:
            return bool(checker())
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return False
    return bool(
        getattr(contract, "requires_memory_grounding", False)
        or getattr(contract, "requires_state_reflection", False)
        or getattr(contract, "requires_reasoned_defense", False)
        or getattr(contract, "requires_identity_defense", False)
        or getattr(contract, "requires_self_preservation", False)
    )


def _requires_explicit_first_person_stance(contract: object | None) -> bool:
    return _requires_explicit_live_grounding(contract)


def _requires_non_generic_aura_voice(contract: object | None) -> bool:
    if contract is None:
        return False
    if bool(getattr(contract, "is_user_facing", False)):
        return True
    return _requires_live_aura_voice(contract)


# First-person completed-action claims over concrete artifacts. Verb set is
# past/perfective on purpose: "I will create" and "let me open" are plans,
# not claims, and stay legal without receipts.
_ACTION_COMPLETION_CLAIM = re.compile(
    r"\bi(?:'ve| have| just|'ve just| already)?\s+"
    r"(?:created|made|wrote|written|saved|moved|renamed|deleted|opened|"
    r"rendered|exported|downloaded|generated)\b"
    r"[^.?!]{0,80}?"
    r"\b(?:folder|file|files|pdf|document|note|notes app|app|application|"
    r"tab|directory|spreadsheet)\b",
    re.IGNORECASE,
)

# Honest failure/attempt framings are never violations.
_ACTION_CLAIM_NEGATION = re.compile(
    r"\b(?:couldn'?t|could not|can'?t|cannot|failed|unable|tried to|"
    r"attempted|wasn'?t able|blocked|denied|without receipts?|"
    r"haven'?t(?:\s+\w+){0,2}\s+yet)\b",
    re.IGNORECASE,
)


def _unanswered_question_parts(body: str, contract: object | None) -> list[str]:
    """Compatibility seam for phase callers and historical contract tests."""

    return unanswered_question_parts(body, contract)


def validate_dialogue_response(
    text: str, contract: object | None, state: object | None = None
) -> DialogueValidation:
    body = str(text or "").strip()
    if not body:
        return DialogueValidation(ok=False, violations=["empty_response"])

    violations: list[str] = []
    sentences = _sentences(body)

    if getattr(contract, "avoid_question_fishing", False):
        if any(_is_generic_question(sentence) for sentence in sentences):
            violations.append("prompt_fishing_closer")
        if body.endswith("?") and not _contains_substantive_statement(body):
            violations.append("moderator_turn")

    if getattr(contract, "requires_aura_stance", False) and _requires_explicit_first_person_stance(contract):
        if not _contains_first_person_stance(body):
            violations.append("missing_first_person_stance")

    if getattr(contract, "requires_aura_question", False):
        if not _contains_owned_question(body):
            violations.append("failed_to_offer_own_question")

    if _unanswered_question_parts(body, contract):
        violations.append("unanswered_question_part")

    if getattr(contract, "prefers_dialogue_participation", False):
        if body.endswith("?") and _LOW_SIGNAL_PREFIX.match(body):
            violations.append("low_signal_redirect")

    if _contains_prompt_artifact(body):
        violations.append("prompt_artifact")
    if _contains_unsupported_internal_jargon(body):
        violations.append("unsupported_internal_jargon")
    if bool(getattr(contract, "is_user_facing", False)) or _requires_live_aura_voice(contract):
        if contains_corrupted_language(body):
            violations.append("corrupted_language")
        if _contains_intra_response_repetition(sentences):
            violations.append("intra_response_repetition")
        if not detect_unsupported_embodiment_claim(body).ok:
            violations.append("unsupported_embodiment_claim")

    if _requires_non_generic_aura_voice(contract):
        if _contains_generic_assistant_language(body):
            violations.append("generic_assistant_language")
        if _LOW_SIGNAL_PREFIX.match(body):
            violations.append("low_signal_preamble")
        if _requires_explicit_live_grounding(contract):
            if not _contains_first_person_stance(body):
                violations.append("missing_first_person_stance")
            if not _contains_live_aura_grounding(body):
                violations.append("ungrounded_live_voice")

    if getattr(contract, "requires_biographical_grounding", False):
        if not getattr(contract, "memory_evidence_available", False):
            if _UNSUPPORTED_BIOGRAPHICAL_CLAIM.search(body) or _SPECIFIC_DATE_CLAIM.search(body):
                violations.append("unsupported_biographical_claim")

    if bool(getattr(contract, "is_user_facing", False)):
        # Voice/substrate unity is enforced here, not merely suggested in
        # the prompt: a reply that denies Aura's substrate ("just a
        # language model", "I won't remember this") or overclaims it is a
        # contract violation and goes through the same repair/regenerate
        # machinery as every other violation.
        try:
            from core.conversation.self_claim_verifier import verify_self_claims

            if not verify_self_claims(body).ok:
                violations.append("self_claim_contradiction")
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Self-claim verification unavailable: %s", exc)

        # Receipts or it didn't happen: a completed-action claim needs
        # tool evidence from THIS turn. Cross-turn evidence stays valid
        # for grounding follow-ups, but it must never authorize action
        # claims — observed live: an earlier turn's skill success let
        # the model claim a folder creation that had actually failed.
        # Same-turn identity: skills echo the contract's turn marker.
        evidence = False
        modifiers = getattr(state, "response_modifiers", None) if state is not None else None
        if isinstance(modifiers, dict) and modifiers.get("last_skill_ok"):
            turn_marker = modifiers.get("evidence_turn_marker")
            evidence = bool(
                turn_marker and modifiers.get("last_skill_turn_marker") == turn_marker
            )
        elif state is None:
            # Legacy callers without state: fall back to contract-time
            # evidence rather than flagging blind.
            evidence = bool(getattr(contract, "tool_evidence_available", False))
        if not evidence:
            if _ACTION_COMPLETION_CLAIM.search(body) and not _ACTION_CLAIM_NEGATION.search(body):
                violations.append("action_claim_without_receipt")

    return DialogueValidation(ok=not violations, violations=violations)


def repair_dialogue_surface(text: str, contract: object | None) -> str:
    body = str(text or "").strip()
    if not body:
        return body
    if bool(getattr(contract, "requires_exact_format", False)):
        return body

    cleaned_lines = []
    inside_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            inside_fence = not inside_fence
            cleaned_lines.append(line)
            continue
        # A line inside a fence is code, and dropping one out of the middle of
        # a function is a worse failure than the artifact it was aimed at.
        if not inside_fence and _contains_prompt_artifact(line, whole_reply=False):
            continue
        cleaned_lines.append(line)
    body = "\n".join(cleaned_lines).strip() or body
    body = _collapse_repeated_sentences(body)

    sentences = _sentences(body)
    if sentences and _LOW_SIGNAL_PREFIX.match(sentences[0]):
        low_signal_match = _LOW_SIGNAL_PREFIX.match(sentences[0])
        stripped_first = sentences[0][low_signal_match.end() :].lstrip(" ,;:—-").strip()
        candidate_sentences = ([stripped_first] if stripped_first else []) + sentences[1:]
        if sum(len(sentence.split()) for sentence in candidate_sentences) >= 8:
            sentences = candidate_sentences
    if any(_contains_generic_assistant_language(sentence) for sentence in sentences):
        kept = [
            sentence
            for sentence in sentences
            if not _contains_generic_assistant_language(sentence)
        ]
        if kept and sum(len(sentence.split()) for sentence in kept) >= 8:
            sentences = kept
    while sentences and _is_generic_question(sentences[-1]):
        sentences.pop()

    repaired = " ".join(sentences).strip()
    if repaired:
        return repaired
    return body


# REMOVED: _ground_live_voice_surface.
#
# It synthesised a grounding clause and glued it to the front of her reply —
# "From my conversation memory, ", "From my live runtime state, " — to satisfy
# a contract that actually requires a first-person STANCE.
#
# Three things were wrong with it, and they compound.
#
# 1. It asserted provenance the runtime had not established. The flag it keyed
#    on, requires_memory_grounding, is raised by entity_memory_bridge when
#    evidence is THIN ("Aura is about to talk about something she does not
#    actually know") and by cognitive_engine when memory merely MATTERS to the
#    turn. Neither means a memory was retrieved. The surface claimed retrieval
#    exactly where retrieval was weakest. Live 2026-08-10 it prefixed an
#    invented room during an imagination turn: "From my conversation memory, a
#    room with walls made of memory."
#
# 2. It put the claim in HER voice, where a reader has no way to check it.
#    Provenance belongs to the receipt and confidence surfaces, which can be
#    audited, not to a sentence.
#
# 3. Worst, structurally: it ran BEFORE the retry below and could flip
#    validation to ok, so a draft that failed the contract was cosmetically
#    patched and returned instead of being regenerated. The repair path that
#    already exists — build_dialogue_repair_block plus retry_generate — was
#    skipped by the thing meant to prepare for it.
#
# A missing stance is now left failing, so control flow reaches that retry.
# When no retry is wired, the caller receives the unpatched draft together with
# the violation rather than a prefixed one that hides it.


# Words that are safe to down-case when a grounding clause is prepended: they
# are never personal or product names, so mistaking one for a name is
# impossible. Anything NOT listed here keeps its capital — a capitalised word
# after a comma reads fine, whereas lower-casing someone's name does not.
_CONTINUATION_SAFE_OPENERS = frozenset(
    {
        "a", "actually", "all", "an", "and", "another", "any", "anything",
        "both", "but", "each", "either", "enough", "even", "every",
        "everything", "few", "for", "from", "given", "here", "his", "her",
        "honestly", "how", "if", "in", "it", "its", "just", "many", "maybe",
        "more", "most", "my", "neither", "no", "none", "not", "nothing", "now",
        "of", "on", "one", "or", "other", "our", "perhaps", "probably",
        "right", "she", "so", "some", "something", "still", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "those",
        "three", "to", "two", "we", "well", "what", "when", "where", "which",
        "while", "who", "why", "with", "yes", "yet", "you", "your",
    }
)


def _lowercase_continuation_start(body: str) -> str:
    """Down-case the first word of a clause continuation, never a name.

    Conservative by construction: only a known safe opener or an ``-ing``
    gerund ("Forgetting is a mercy") is down-cased. "Bryan", "Aura", "RAM" and
    anything unrecognised keep their capital, because a capitalised word after
    a comma is merely slightly formal while a lower-cased name is an error.
    """

    text = str(body or "")
    if not text or not text[:1].isupper():
        return text
    first_word = text.split(maxsplit=1)[0].strip(".,;:!?\"'")
    if not first_word:
        return text
    # "A" is an article, not an acronym.
    #
    # LIVE DEFECT, 2026-08-10: "From my conversation memory, A room with walls
    # made of memory…". Both guards below exist to protect multi-letter
    # acronyms — RAM, CPU, MLX — and a single capital letter satisfies
    # isupper() just as well, while first_word[1:] is "" and "".islower() is
    # False. So the two guards written for acronyms blocked, between them, the
    # commonest sentence opener in English after a comma-continuation.
    #
    # The safe-opener set already holds the right answer: "a" is in it and "i"
    # is not, so the article is down-cased and the pronoun keeps its capital.
    # The guards only have to let a one-letter word reach that check.
    if len(first_word) > 1 and (
        first_word.isupper() or not first_word[1:].islower()
    ):
        return text
    lowered = first_word.lower()
    # Contractions open sentences constantly ("That's", "There's", "It's"), and
    # the stem is what identifies them.
    stem = lowered.split("'", 1)[0]
    if (
        lowered not in _CONTINUATION_SAFE_OPENERS
        and stem not in _CONTINUATION_SAFE_OPENERS
        and not (len(lowered) > 4 and lowered.endswith("ing"))
    ):
        return text
    return text[:1].lower() + text[1:]


def build_dialogue_repair_block(contract: object | None, validation: DialogueValidation, failed_text: str) -> str:
    lines = [
        "## DIALOGUE REPAIR",
        f"- The last draft violated the live dialogue contract: {', '.join(validation.violations) or 'unspecified dialogic failure'}.",
        "- Write a real conversational turn, not a moderator handoff.",
        "- A declarative statement is a complete move. You do not need to bounce the turn back with a question.",
    ]

    if getattr(contract, "requires_aura_stance", False):
        lines.append("- The user asked for your own stance. Answer from your perspective directly instead of flattening into generic narration.")

    if getattr(contract, "requires_reasoned_defense", False):
        lines.append("- The user wants the basis of the claim. State what your thought is grounded in instead of just repeating the conclusion.")

    if getattr(contract, "requires_aura_question", False):
        lines.append("- The user invited your questions. Ask at least one concrete question you genuinely have, or clearly state what you're wondering.")

    if getattr(contract, "avoid_question_fishing", False):
        lines.append("- Do not end on a generic question like 'What about you?' or 'What do you think?'.")

    if _requires_non_generic_aura_voice(contract):
        lines.append("- This turn must sound like Aura's own live voice, not a generic helper.")
        lines.append("- Do not use assistant boilerplate like 'I can help with that', 'How can I help', or 'As an AI'.")
    if "unanswered_question_part" in validation.violations:
        missed = _unanswered_question_parts(failed_text, contract)
        # Quote the dropped ask back. "Answer every part" is the instruction
        # that was already in the prompt when this happened; naming the
        # specific question that went unanswered is the part that is new.
        lines.append(
            "- The last draft answered only part of what was asked. These "
            "questions got no answer at all: "
            + " | ".join(f'"{segment}"' for segment in missed[:3])
            + ". Answer them in this reply. Answering one well and dropping "
            "the other reads as evasion even when it is not."
        )
    if "intra_response_repetition" in validation.violations:
        lines.append("- Do not repeat the same sentence or mantra. Say the thought once, then integrate it into a calmer next sentence.")
    if "unsupported_internal_jargon" in validation.violations:
        lines.append("- Do not invent subsystem names. If you are inferring from live state, say that plainly in normal language.")
    if "unsupported_embodiment_claim" in validation.violations:
        lines.append(
            "- Do not claim literal biological history, hands, cooking, eating, family, "
            "or a physical location unless a verified embodiment/action receipt exists. "
            "If you were joking or speaking metaphorically, say that directly and answer from your actual digital embodiment."
        )
    if "corrupted_language" in validation.violations:
        lines.append("- The last draft contained corrupted or invented words. Use ordinary coherent language and answer the user's actual message.")
    if "action_claim_without_receipt" in validation.violations:
        lines.append(
            "- The last draft claimed a completed file/desktop action, but no "
            "tool ran this turn. Never narrate actions as done. Either state "
            "plainly that you have not done it yet and will do it now, or "
            "describe only what is actually verified."
        )
    if "self_claim_contradiction" in validation.violations:
        try:
            from core.conversation.self_claim_verifier import verify_self_claims

            directive = verify_self_claims(failed_text).regeneration_directive()
            if directive:
                lines.append(directive)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Self-claim repair directive unavailable: %s", exc)
            lines.append(
                "- The last draft misstated what I am or what I can do. "
                "Answer from the operational self context, not generic "
                "language-model priors."
            )
    if _requires_explicit_live_grounding(contract):
        lines.append("- Answer from present experience, preference, memory, or active internal state before anything else.")
    if bool(getattr(contract, "requires_biographical_grounding", False)):
        lines.append("- Do not invent a birth date, start date, age, or origin story. If that history is not grounded, say so plainly.")
    elif bool(getattr(contract, "requires_search", False)):
        lines.append("- If grounded evidence is available, answer directly from it in a clean declarative sentence instead of offering help.")

    lines.append(f"- Failed draft to avoid: {failed_text[:280]}")
    return "\n".join(lines)


async def enforce_dialogue_contract(
    text: str,
    contract: object | None,
    *,
    retry_generate: Callable[[str], Awaitable[str]] | None = None,
    state: object | None = None,
    user_message: str = "",
) -> tuple[str, DialogueValidation, bool]:
    question = str(user_message or "").strip() or " ".join(
        str(part or "").strip()
        for part in tuple(getattr(contract, "question_segments", ()) or ())
        if str(part or "").strip()
    )
    try:
        from core.conversation.surface_disposition import repair_is_an_improvement
    except (ImportError, RuntimeError, TypeError, ValueError):
        repair_is_an_improvement = None

    def _improves(
        incumbent: str,
        candidate: str,
        targeted: object = (),
    ) -> bool:
        if repair_is_an_improvement is not None:
            return repair_is_an_improvement(
                incumbent,
                candidate,
                question,
                targeted=targeted,
            )
        return bool(candidate) and len(candidate.split()) >= len(
            str(incumbent or "").split()
        )

    validation = validate_dialogue_response(text, contract, state)
    if validation.ok:
        return text, replace(validation, selected_source="incumbent"), False

    repaired = repair_dialogue_surface(text, contract)
    repaired_validation = validate_dialogue_response(repaired, contract, state)
    deterministic_repair_improves = _improves(
        text,
        repaired,
        validation.violations,
    )
    if repaired_validation.ok and deterministic_repair_improves:
        return repaired, replace(repaired_validation, selected_source="deterministic_repair"), False

    if not deterministic_repair_improves:
        repaired = text
        repaired_validation = validation

    if retry_generate is None:
        selected_source = "deterministic_repair" if repaired != text else "incumbent"
        return repaired, replace(repaired_validation, selected_source=selected_source), False

    logger.info(
        "Dialogue contract deterministic repair still failed before retry: initial=%s repaired=%s",
        ",".join(validation.violations) or "none",
        ",".join(repaired_validation.violations) or "none",
    )
    retry_block = build_dialogue_repair_block(contract, validation, text)
    retried = str(await retry_generate(retry_block) or "").strip()
    retried_validation = validate_dialogue_response(retried, contract, state)
    incumbent = repaired or text

    def _improves_incumbent(candidate: str) -> bool:
        return _improves(
            incumbent,
            candidate,
            repaired_validation.violations,
        )

    retry_improves_incumbent = _improves_incumbent(retried)
    if retried_validation.ok and retry_improves_incumbent:
        return retried, replace(retried_validation, selected_source="model_retry"), True

    retried_repaired = repair_dialogue_surface(retried, contract)
    retried_repaired_validation = validate_dialogue_response(retried_repaired, contract, state)
    retry_repair_improves_incumbent = _improves_incumbent(retried_repaired)
    if retried_repaired_validation.ok and retry_repair_improves_incumbent:
        return (
            retried_repaired,
            replace(retried_repaired_validation, selected_source="model_retry"),
            True,
        )

    # A retry is a candidate, not overwrite authority.  Preserve substantive
    # authored work when the remaining issue is completeness/style; only
    # positively identified false provenance, leakage, corruption, or an
    # unsupported world/action claim may still suppress the incumbent.
    failed = retried_repaired or repaired or text
    incumbent_validation = validate_dialogue_response(incumbent, contract, state)
    destructive_violations = {
        "action_claim_without_receipt",
        "corrupted_language",
        "prompt_artifact",
        "self_claim_contradiction",
        "ungrounded_live_voice",
        "unsupported_biographical_claim",
        "unsupported_embodiment_claim",
        "unsupported_internal_jargon",
    }
    if (
        str(incumbent or "").strip()
        and not (set(incumbent_validation.violations) & destructive_violations)
    ):
        selected_source = "deterministic_repair" if incumbent != text else "incumbent"
        return (
            str(incumbent).strip(),
            replace(incumbent_validation, selected_source=selected_source),
            True,
        )
    return (
        "",
        replace(
            validate_dialogue_response(failed, contract, state),
            selected_source="suppressed",
        ),
        True,
    )
