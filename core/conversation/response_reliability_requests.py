"""What the person asked for: a count, a length, a shape, the phrases the reply must carry.

Lifted whole out of `response_reliability`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .response_reliability import (
        RequestedOutputContract,
    )


def _word_count(text: Any) -> int:
    from .response_reliability import (
        _WORD_RE,
    )

    return len(_WORD_RE.findall(str(text or "")))


def _count_token_to_int(value: str | None) -> int | None:
    from .response_reliability import (
        _NUMBER_WORDS,
    )

    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        count = int(token)
    else:
        count = _NUMBER_WORDS.get(token)
    if count is None or count < 1 or count > 20:
        return None
    return count


def _word_count_token_to_int(value: str | None) -> int | None:
    """Parse explicit word limits without imposing list-count's 20-item cap."""

    from .response_reliability import (
        _NUMBER_WORDS,
    )

    token = str(value or "").strip().lower()
    if not token:
        return None
    if token.isdigit():
        count = int(token)
    else:
        count = _NUMBER_WORDS.get(token)
    if count is None or count < 1 or count > 4096:
        return None
    return count


def _is_escaped_character(text: str, index: int) -> bool:
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return bool(backslashes % 2)


def _text_index_is_unquoted(text: str, index: int) -> bool:
    """Return whether ``index`` is outside quoted and code-literal spans."""

    from .response_reliability import (
        _EXACT_REPLY_QUOTE_PAIRS,
    )

    quote_close = ""
    fenced_code = False
    inline_code = False
    cursor = 0
    limit = max(0, min(len(text), int(index)))
    while cursor < limit:
        if not quote_close and not inline_code and text.startswith("```", cursor):
            fenced_code = not fenced_code
            cursor += 3
            continue
        if fenced_code:
            cursor += 1
            continue

        char = text[cursor]
        if not quote_close and char == "`" and not _is_escaped_character(text, cursor):
            inline_code = not inline_code
            cursor += 1
            continue
        if inline_code:
            cursor += 1
            continue

        if quote_close:
            if char == quote_close and not _is_escaped_character(text, cursor):
                quote_close = ""
            cursor += 1
            continue

        if char in _EXACT_REPLY_QUOTE_PAIRS and not _is_escaped_character(text, cursor):
            is_apostrophe = (
                char == "'"
                and cursor > 0
                and cursor + 1 < len(text)
                and text[cursor - 1].isalnum()
                and text[cursor + 1].isalnum()
            )
            if not is_apostrophe:
                quote_close = _EXACT_REPLY_QUOTE_PAIRS[char]
        cursor += 1
    return not (quote_close or fenced_code or inline_code)


def _constraint_match_is_actionable(text: str, match: re.Match[str]) -> bool:
    """Reject quoted, code-sample, and explicitly negated length language."""

    before = text[: match.start()]
    if not _text_index_is_unquoted(text, match.start()):
        return False
    prefix = (
        before[-192:]
        .lower()
        .replace("‘", "'")
        .replace("’", "'")
    )
    # Negation applies to its grammatical clause, not an unrelated command
    # after punctuation or a coordinating transition.
    prefix = re.split(r"[.!?;,\n]", prefix)[-1]
    prefix = re.split(
        r"\b(?:then|but|however|instead|otherwise|next|now)\b",
        prefix,
    )[-1]
    prefix = re.split(
        r"\b(?:and|or)\s+(?=(?:then\s+)?(?:answer|reply|respond|say|output|return|print)\b)",
        prefix,
    )[-1]
    # Some command regexes include the command verb in the match itself. In
    # that case the prefix ends at the coordinator, so the lookahead above
    # cannot see the fresh predicate even though it starts at ``match``.
    if re.search(r"\b(?:and|or)\s*$", prefix) and re.match(
        r"\s*(?:answer|reply|respond|say|output|return|print)\b",
        match.group(0),
        re.IGNORECASE,
    ):
        prefix = ""
    return not bool(
        re.search(
            r"\b(?:do\s+not|don't|never|ignore|disregard|avoid|rather\s+than|instead\s+of|"
            r"no\s+need\s+to|without|not\s+(?:limited|restricted|confined)\s+to|"
            r"(?:do(?:es)?\s+not|don't|doesn't)\s+have\s+to|"
            r"(?:old|previous|example|sample)\s+(?:instruction|prompt|command|text)\s+"
            r"(?:was|said|says|contained)|"
            r"(?:(?:i(?:'m|\s+am)|we(?:'re|\s+are)|you(?:'re|\s+are)|"
            r"they(?:'re|\s+are))\s+)?not\s+asking(?:\s+you)?\s+to)\b"
            r"[^.!?;\n]{0,72}$",
            prefix,
        )
    )


def _requested_count(pattern: re.Pattern[str], user_message: Any) -> int | None:
    match = pattern.search(str(user_message or ""))
    if not match:
        return None
    return _count_token_to_int(match.groupdict().get("count"))


def _requested_word_count_range(user_message: Any) -> tuple[int, int] | None:
    from .response_reliability import (
        _ACTION_WORD_COUNT_REQUEST_RE,
        _LIMIT_WORD_COUNT_REQUEST_RE,
    )

    text = str(user_message or "")
    candidates: list[tuple[int, int, int, int]] = []
    for pattern in (_ACTION_WORD_COUNT_REQUEST_RE, _LIMIT_WORD_COUNT_REQUEST_RE):
        for match in pattern.finditer(text):
            if not _constraint_match_is_actionable(text, match):
                continue
            minimum = _word_count_token_to_int(match.groupdict().get("count"))
            maximum = _word_count_token_to_int(match.groupdict().get("count_max"))
            if minimum is None:
                continue
            if maximum is None:
                maximum = minimum
            candidates.append(
                (match.start(), match.end(), min(minimum, maximum), max(minimum, maximum))
            )
    if not candidates:
        return None
    _start, _end, minimum, maximum = max(candidates, key=lambda item: (item[0], item[1]))
    return minimum, maximum


def _requested_sentence_count(user_message: Any) -> int | None:
    from .response_reliability import (
        _ACTION_SENTENCE_COUNT_REQUEST_RE,
        _LIMIT_SENTENCE_COUNT_REQUEST_RE,
    )

    text = str(user_message or "")
    candidates: list[tuple[int, int, int]] = []
    for pattern in (
        _ACTION_SENTENCE_COUNT_REQUEST_RE,
        _LIMIT_SENTENCE_COUNT_REQUEST_RE,
    ):
        for match in pattern.finditer(text):
            if not _constraint_match_is_actionable(text, match):
                continue
            requested = _count_token_to_int(match.groupdict().get("count"))
            if requested is not None:
                candidates.append((match.start(), match.end(), requested))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _requested_line_count(user_message: Any) -> int | None:
    """The number of LINES the turn explicitly asked for, or None."""
    from .response_reliability import (
        _LINE_COUNT_REQUEST_RE,
    )

    text = str(user_message or "")
    candidates: list[tuple[int, int, int]] = []
    for match in _LINE_COUNT_REQUEST_RE.finditer(text):
        if not _constraint_match_is_actionable(text, match):
            continue
        requested = _count_token_to_int(match.groupdict().get("count"))
        if requested is not None:
            candidates.append((match.start(), match.end(), requested))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def requested_count(user_message: Any, *units: str) -> int | None:
    """How many of a named thing the turn asked for, or None.

    The counted units in this module each carry their own pattern — lines,
    sentences, paragraphs, bullets, facts — and every one of them is the same
    shape: a count token followed by the unit. A sixth unit should not need a
    sixth regex, and on 2026-08-22 it got one: a deck builder arrived with its
    own copy of the number words beside this one.

    Takes the units so a caller can ask about slides, steps, examples or
    anything else without this module knowing what those are.
    """
    from .response_reliability import (
        _COUNT_TOKEN_RE,
    )

    text = str(user_message or "")
    if not text.strip() or not units:
        return None
    spellings = "|".join(
        re.escape(unit.strip().lower()) for unit in units if str(unit or "").strip()
    )
    if not spellings:
        return None
    pattern = re.compile(
        rf"\b{_COUNT_TOKEN_RE}[\s-]*(?:concise\s+|short\s+|brief\s+|clear\s+)?"
        rf"(?:{spellings})s?\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int]] = []
    for match in pattern.finditer(text):
        if not _constraint_match_is_actionable(text, match):
            continue
        found = _count_token_to_int(match.groupdict().get("count"))
        if found is not None:
            candidates.append((match.start(), found))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def requested_line_count(user_message: Any) -> int | None:
    """Return the explicit line-count contract requested by the user."""

    return _requested_line_count(user_message)


def requested_sentence_count(user_message: Any) -> int | None:
    """Return the exact sentence-count contract explicitly requested by the user."""

    return _requested_sentence_count(user_message)


def requested_exact_reply_target(user_message: Any) -> str:
    """Return the last actionable exact-reply target.

    Surrounding transport whitespace is not part of the contract. Target case
    and punctuation are preserved for both quoted and unquoted commands.
    """

    from .response_reliability import (
        _EXACT_REPLY_ADDITIONAL_ACTION_TAIL_RE,
        _EXACT_REPLY_COMMAND_RE,
        _EXACT_REPLY_CONDITIONAL_TAIL_RE,
        _EXACT_REPLY_INTRODUCER_RE,
        _EXACT_REPLY_QUOTE_PAIRS,
        _EXACT_REPLY_UNQUOTED_SUFFIX_RE,
    )

    raw = str(user_message or "").strip()
    if not raw:
        return ""
    commands = [
        match
        for match in _EXACT_REPLY_COMMAND_RE.finditer(raw)
        if _text_index_is_unquoted(raw, match.start())
    ]
    candidates: list[tuple[int, str]] = []
    for index, match in enumerate(commands):
        if not _constraint_match_is_actionable(raw, match):
            continue
        end = commands[index + 1].start() if index + 1 < len(commands) else len(raw)
        remainder = raw[match.end() : end].lstrip()
        remainder = _EXACT_REPLY_INTRODUCER_RE.sub("", remainder, count=1).lstrip()
        if not remainder:
            continue
        quote = remainder[0]
        if quote in _EXACT_REPLY_QUOTE_PAIRS:
            closing = _EXACT_REPLY_QUOTE_PAIRS[quote]
            target_chars: list[str] = []
            close_index = -1
            cursor = 1
            while cursor < len(remainder):
                char = remainder[cursor]
                apostrophe = bool(
                    quote == "'"
                    and char == "'"
                    and cursor > 0
                    and cursor + 1 < len(remainder)
                    and remainder[cursor - 1].isalnum()
                    and remainder[cursor + 1].isalnum()
                )
                if (
                    char == closing
                    and not apostrophe
                    and not _is_escaped_character(remainder, cursor)
                ):
                    close_index = cursor
                    break
                if (
                    char == "\\"
                    and cursor + 1 < len(remainder)
                    and remainder[cursor + 1] in {"\\", quote, closing}
                ):
                    target_chars.append(remainder[cursor + 1])
                    cursor += 2
                    continue
                target_chars.append(char)
                cursor += 1
            if close_index <= 1:
                continue
            trailing_meta = _EXACT_REPLY_UNQUOTED_SUFFIX_RE.sub(
                "",
                remainder[close_index + 1 :],
            ).strip()
            if trailing_meta.strip(".!?;:, "):
                continue
            target = "".join(target_chars).strip()
        else:
            target = remainder.strip()
            if _EXACT_REPLY_ADDITIONAL_ACTION_TAIL_RE.search(target):
                continue
            target = _EXACT_REPLY_UNQUOTED_SUFFIX_RE.sub("", target).rstrip()
            if _EXACT_REPLY_CONDITIONAL_TAIL_RE.search(target):
                continue
            target = re.split(
                r"(?<=[.!?])\s+(?=(?:now|then|after|before|also|next|instead|please)\b)",
                target,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            target = target.strip()
        if target:
            candidates.append((match.start(), target))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1]


def _compact_output_style_requested(user_message: Any) -> bool:
    from .response_reliability import (
        _normalize,
    )

    text = _normalize(user_message)
    if not text:
        return False
    pattern = re.compile(
        r"\b(?:briefly|be brief|be concise|keep (?:it|this) (?:brief|concise|short)|"
        r"(?:brief|concise|short) (?:answer|reply|response|sentence)|"
        r"(?:brief|concise|short) sentences?|"
        r"in (?:a|one) (?:brief|concise|short) sentence|"
        r"include nothing else|nothing else)\b"
    )
    return any(
        _constraint_match_is_actionable(text, match)
        for match in pattern.finditer(text)
    )


def requested_output_contract(user_message: Any) -> RequestedOutputContract:
    """Return a conservative token ceiling derived from visible user intent.

    The semantic cap is a planning target. The hard ceiling includes enough
    tokenizer and punctuation headroom to satisfy the requested shape, while
    remaining an absolute upper bound after affective and pressure modulation.
    """

    from .response_reliability import (
        RequestedOutputContract,
        _explicit_brevity_requested,
    )

    raw = str(user_message or "").strip()
    if not raw:
        return RequestedOutputContract()

    exact_target = requested_exact_reply_target(raw)
    if exact_target:
        utf8_bytes = len(exact_target.encode("utf-8"))
        estimated_tokens = (
            max(1, (len(exact_target) + 2) // 3)
            if exact_target.isascii()
            else max(1, utf8_bytes)
        )
        semantic_cap = min(8192, max(8, estimated_tokens + 4))
        # Byte-fallback tokenizers cannot require more content tokens than the
        # UTF-8 byte count. Keep protocol/EOS headroom above that real bound;
        # the selected worker tokenizer records and verifies the exact count.
        hard_ceiling = max(16, utf8_bytes + 16)
        return RequestedOutputContract(
            kind="exact_reply",
            explicit_brevity=True,
            exact_reply=True,
            exact_reply_chars=len(exact_target),
            exact_reply_utf8_bytes=utf8_bytes,
            semantic_token_cap=semantic_cap,
            hard_token_ceiling=hard_ceiling,
            confidence=1.0,
        )

    word_range = _requested_word_count_range(raw)
    sentence_count = _requested_sentence_count(raw)
    explicit_brevity = _explicit_brevity_requested(raw)
    compact_style = word_range is not None or _compact_output_style_requested(raw)
    if word_range is None and sentence_count is None and not explicit_brevity:
        return RequestedOutputContract()

    semantic_candidates: list[int] = []
    hard_candidates: list[int] = []
    kinds: list[str] = []
    if word_range is not None:
        _minimum_words, maximum_words = word_range
        semantic_candidates.append(max(16, 8 + (2 * maximum_words)))
        hard_candidates.append(max(32, 16 + (3 * maximum_words)))
        kinds.append("word_count")
    if sentence_count is not None:
        semantic_per_sentence = 32 if compact_style else 64
        hard_per_sentence = 48 if compact_style else 96
        semantic_candidates.append(max(24, semantic_per_sentence * sentence_count))
        hard_candidates.append(max(32, hard_per_sentence * sentence_count))
        kinds.append("sentence_count")
    if explicit_brevity and not semantic_candidates:
        # "Keep it short" names no quantity, so it sets a planning target
        # and no truncation point. It used to set hard=112: measured live
        # 2026-09-15, a correct technical answer needed more, the decoder
        # was cut mid-sentence at 112 (spent_budget=True), and the turn ran
        # a continuation and a repair — three generations, five minutes —
        # before a smaller model's wrong answer was served in its place.
        # Short is the model's to keep once the request says so; a hard
        # ceiling only decides where the sentence breaks.
        semantic_candidates.append(64)
        kinds.append("brevity")

    semantic_cap = min(8192, max(semantic_candidates))
    hard_ceiling = (
        min(8192, max(semantic_cap, max(hard_candidates))) if hard_candidates else None
    )
    return RequestedOutputContract(
        kind="+".join(kinds),
        word_min=word_range[0] if word_range else None,
        word_max=word_range[1] if word_range else None,
        sentence_count=sentence_count,
        explicit_brevity=compact_style,
        semantic_token_cap=semantic_cap,
        hard_token_ceiling=hard_ceiling,
        confidence=0.98 if word_range is not None or sentence_count is not None else 0.9,
    )


def _requested_reference_values(user_message: Any) -> tuple[tuple[str, int], ...]:
    from .response_reliability import (
        _INCLUDE_GENERIC_REFERENCE_VALUE_RE,
        _INCLUDE_REFERENCE_VALUE_RE,
        _REFERENCE_LABEL_VALUE_RE,
    )

    user = str(user_message or "")
    if not user:
        return ()
    requested_kinds = {
        str(match.group("kind") or "").strip().lower()
        for match in _INCLUDE_REFERENCE_VALUE_RE.finditer(user)
        if str(match.group("kind") or "").strip()
    }
    generic_reference_requested = bool(
        _INCLUDE_GENERIC_REFERENCE_VALUE_RE.search(user)
    )
    observed = [
        (
            " ".join(str(match.group("label") or "").strip().split()).lower(),
            str(match.group("kind") or "").strip().lower(),
            int(match.group("value")),
        )
        for match in _REFERENCE_LABEL_VALUE_RE.finditer(user)
    ]
    if generic_reference_requested and len(observed) != 1:
        return ()
    requested = [
        (label, value)
        for label, kind, value in observed
        if kind in requested_kinds or generic_reference_requested
    ]
    return tuple(dict.fromkeys(requested))


def _reply_contains_reference_value(reply_text: Any, value: int) -> bool:
    from .response_reliability import (
        _NUMBER_WORDS,
        _normalize,
    )

    reply = _normalize(reply_text)
    if not reply:
        return False
    if re.search(rf"(?<!\d){int(value)}(?!\d)", reply):
        return True
    number_word = next(
        (word for word, number in _NUMBER_WORDS.items() if number == int(value)),
        "",
    )
    return bool(number_word and re.search(rf"\b{re.escape(number_word)}\b", reply))


def _compact_reference_acknowledgement(user_message: Any) -> str:
    """Return a deterministic exact-format acknowledgement when that is the task."""

    from .response_reliability import (
        _COMPACT_REFERENCE_ACK_RE,
    )

    user = str(user_message or "")
    match = _COMPACT_REFERENCE_ACK_RE.match(user)
    if not match or _requested_sentence_count(user) != 1:
        return ""
    references = _requested_reference_values(user)
    value = int(match.group("value"))
    if not references or not any(reference_value == value for _, reference_value in references):
        return ""
    label = " ".join(str(match.group("label") or "").strip().split()).lower()
    if not label:
        return ""
    return f"{label[0].upper()}{label[1:]} {value} completed."


_QUOTED_REQUIRED_PHRASE_RE = re.compile(
    r"\b(?:include|mention|use)\b[^\"'“”‘’]{0,80}[\"'“”‘’](?P<phrase>[^\"'“”‘’]{1,80})[\"'“”‘’]",
    re.IGNORECASE,
)


_INCLUDE_REQUIRED_PHRASE_RE = re.compile(
    r"\b(?:include|mention)\s+(?:the\s+)?(?:(?:exact\s+)?(?:phrase|word|term)\s+)?"
    r"(?P<phrase>[A-Za-z0-9][A-Za-z0-9 _-]{1,80})(?:[.!?;,]|$)",
    re.IGNORECASE,
)


_USE_REQUIRED_PHRASE_RE = re.compile(
    r"\buse\s+(?:the\s+)?(?:exact\s+)?(?:phrase|word|term)\s+"
    r"(?P<phrase>[A-Za-z0-9][A-Za-z0-9 _-]{1,80})(?:[.!?;,]|$)",
    re.IGNORECASE,
)


# Heads that mark a scope/brevity instruction ("include nothing else"), not a
# literal phrase the reply must contain.
_BREVITY_PSEUDO_PHRASE_HEADS = frozenset(
    {"nothing", "no", "only", "just", "anything", "everything", "none"}
)


def _requested_required_phrases(user_message: Any) -> tuple[str, ...]:
    from .response_reliability import (
        _WORD_RE,
    )

    text = str(user_message or "")
    if not text:
        return ()
    phrases: list[str] = []
    for pattern in (
        _QUOTED_REQUIRED_PHRASE_RE,
        _INCLUDE_REQUIRED_PHRASE_RE,
        _USE_REQUIRED_PHRASE_RE,
    ):
        for match in pattern.finditer(text):
            phrase = " ".join(str(match.group("phrase") or "").strip(" .,:;!?\"'“”‘’").split())
            if not phrase:
                continue
            # Avoid treating a full instruction clause as a required phrase when
            # the user wrote something like "use your own voice and include X".
            if len(_WORD_RE.findall(phrase)) > 8:
                continue
            # "include nothing else", "include only the answer" are BREVITY/scope
            # instructions, not a literal phrase to echo. Treating them as a
            # required phrase made a valid short reply fail 'missing_requested_phrase'.
            if phrase.lower().split()[0] in _BREVITY_PSEUDO_PHRASE_HEADS:
                continue
            phrases.append(phrase.lower())
    return tuple(dict.fromkeys(phrases))


def has_requested_word_count_contract(user_message: Any) -> bool:
    """Return True when the user gave an explicit word-count output contract."""
    return _requested_word_count_range(user_message) is not None


def _requested_list_item_count(user_message: Any) -> int:
    from .response_reliability import (
        _BULLET_REQUEST_RE,
        _NUMBERED_LIST_REQUEST_RE,
        _NUMBERED_SENTENCE_REQUEST_RE,
    )

    requested_bullets = _requested_count(_BULLET_REQUEST_RE, user_message)
    requested_numbered = _requested_count(_NUMBERED_LIST_REQUEST_RE, user_message)
    requested_numbered_sentences = _requested_count(_NUMBERED_SENTENCE_REQUEST_RE, user_message)
    return max(requested_bullets or 0, requested_numbered or 0, requested_numbered_sentences or 0)
