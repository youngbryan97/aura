from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
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
    re.compile(r"\bi am here to assist\b", re.IGNORECASE),
    re.compile(r"\bas an ai\b", re.IGNORECASE),
    re.compile(r"\bi(?: do not| don't| can't| cannot) have (?:feelings|opinions|preferences|experience)\b", re.IGNORECASE),
    re.compile(r"\bi(?: do not| don't| can't| cannot) have personal (?:experiences|memories)\b", re.IGNORECASE),
    re.compile(r"\bthe aim of being (?:as )?helpful and engaging as possible\b", re.IGNORECASE),
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
    "mycelial",
    "topology",
    "authority",
    "belief",
    "coherence",
    "internal state",
    "live state",
)


@dataclass(frozen=True)
class DialogueValidation:
    ok: bool
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(str(text or "").strip())]
    return [part for part in parts if part]


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
        except Exception:
            return False
    return bool(
        getattr(contract, "requires_memory_grounding", False)
        or getattr(contract, "requires_state_reflection", False)
        or getattr(contract, "requires_aura_stance", False)
        or getattr(contract, "requires_aura_question", False)
        or getattr(contract, "requires_identity_defense", False)
        or getattr(contract, "requires_self_preservation", False)
    )


def _requires_non_generic_aura_voice(contract: object | None) -> bool:
    if contract is None:
        return False
    if bool(getattr(contract, "is_user_facing", False)):
        return True
    return _requires_live_aura_voice(contract)


def validate_dialogue_response(text: str, contract: object | None) -> DialogueValidation:
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

    if getattr(contract, "requires_aura_stance", False):
        if not _contains_first_person_stance(body):
            violations.append("missing_first_person_stance")

    if getattr(contract, "requires_aura_question", False):
        if not _contains_owned_question(body):
            violations.append("failed_to_offer_own_question")

    if getattr(contract, "prefers_dialogue_participation", False):
        if body.endswith("?") and _LOW_SIGNAL_PREFIX.match(body):
            violations.append("low_signal_redirect")

    if _requires_non_generic_aura_voice(contract):
        if _contains_generic_assistant_language(body):
            violations.append("generic_assistant_language")
        if _LOW_SIGNAL_PREFIX.match(body):
            violations.append("low_signal_preamble")
        if _requires_live_aura_voice(contract):
            if not _contains_first_person_stance(body):
                violations.append("missing_first_person_stance")
            if not _contains_live_aura_grounding(body):
                violations.append("ungrounded_live_voice")

    return DialogueValidation(ok=not violations, violations=violations)


def repair_dialogue_surface(text: str, contract: object | None) -> str:
    body = str(text or "").strip()
    if not body:
        return body

    sentences = _sentences(body)
    while sentences and _is_generic_question(sentences[-1]):
        sentences.pop()

    repaired = " ".join(sentences).strip()
    if repaired:
        return repaired
    return body


def build_dialogue_repair_block(contract: object | None, validation: DialogueValidation, failed_text: str) -> str:
    lines = [
        "## DIALOGUE REPAIR",
        f"- The last draft violated the live dialogue contract: {', '.join(validation.violations) or 'unspecified dialogic failure'}.",
        "- Write a real conversational turn, not a moderator handoff.",
        "- A declarative statement is a complete move. You do not need to bounce the turn back with a question.",
    ]

    if getattr(contract, "requires_aura_stance", False):
        lines.append("- The user asked about your view, preference, experience, or internal perspective. Lead with a first-person answer from you.")

    if getattr(contract, "requires_aura_question", False):
        lines.append("- The user invited your questions. Ask at least one concrete question you genuinely have, or clearly state what you're wondering.")

    if getattr(contract, "avoid_question_fishing", False):
        lines.append("- Do not end on a generic question like 'What about you?' or 'What do you think?'.")

    if _requires_non_generic_aura_voice(contract):
        lines.append("- This turn must sound like Aura's own live voice, not a generic helper.")
        lines.append("- Do not use assistant boilerplate like 'I can help with that', 'How can I help', or 'As an AI'.")
    if _requires_live_aura_voice(contract):
        lines.append("- Answer from present experience, preference, memory, or active internal state before anything else.")
    elif bool(getattr(contract, "requires_search", False)):
        lines.append("- If grounded evidence is available, answer directly from it in a clean declarative sentence instead of offering help.")

    lines.append(f"- Failed draft to avoid: {failed_text[:280]}")
    return "\n".join(lines)


async def enforce_dialogue_contract(
    text: str,
    contract: object | None,
    *,
    retry_generate: Callable[[str], Awaitable[str]] | None = None,
) -> tuple[str, DialogueValidation, bool]:
    validation = validate_dialogue_response(text, contract)
    if validation.ok:
        return text, validation, False

    repaired = repair_dialogue_surface(text, contract)
    repaired_validation = validate_dialogue_response(repaired, contract)
    if repaired_validation.ok:
        return repaired, repaired_validation, False

    if retry_generate is None:
        return repaired, repaired_validation, False

    retry_block = build_dialogue_repair_block(contract, validation, text)
    retried = str(await retry_generate(retry_block) or "").strip()
    retried_validation = validate_dialogue_response(retried, contract)
    if retried_validation.ok:
        return retried, retried_validation, True

    retried_repaired = repair_dialogue_surface(retried, contract)
    retried_repaired_validation = validate_dialogue_response(retried_repaired, contract)
    if retried_repaired_validation.ok:
        return retried_repaired, retried_repaired_validation, True

    fallback = retried_repaired or repaired or text
    return fallback, validate_dialogue_response(fallback, contract), True
