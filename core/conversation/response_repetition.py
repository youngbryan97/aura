"""Whether a reply says the same thing more than once, and what to cut.

Lifted whole out of `response_reliability`, which is 9,000 lines and the
largest single contributor to the module-size budget. Every name taken from
that module is imported at CALL time: it imports this one to re-export these
five, and a test that patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from typing import Any


def _phrase_loop_reason(user_message: Any, reply_text: Any) -> str:
    from .response_reliability import (
        _LOW_INFORMATION_LOOP_RE,
        _WORD_RE,
        _is_code_response,
        _normalize,
        _word_count,
    )

    reply = _normalize(reply_text)
    if not reply:
        return ""
    if _is_code_response(reply_text):
        return ""
    user = _normalize(user_message)
    if _LOW_INFORMATION_LOOP_RE.search(reply):
        return "low_information_loop"
    if "get it" in reply:
        reply_count = reply.count("get it")
        user_count = user.count("get it")
        if reply_count >= 2 and reply_count > user_count:
            return "repeated_get_it_loop"
        if reply_count >= 1 and _word_count(reply) <= 6:
            return "low_information_loop"
    if "i don't get it" in reply and "i get it" in reply:
        return "self_contradictory_loop"

    words = _WORD_RE.findall(reply)
    if len(words) < 8:
        return ""
    lower_words = [w.lower() for w in words]
    stop_words = {
        "i", "i'm", "am", "you", "it", "that", "this", "the", "a", "an",
        "to", "and", "but", "then", "is", "are", "was", "were", "be", "being",
        "with", "on", "in", "of", "for", "as", "so", "my", "your",
    }
    
    # Detect structured dialogue speaker names / headings to avoid false positive loops on speaker prefixes (e.g. "Mainframe", "Quantum Processor")
    speaker_labels = set()
    for line in str(reply_text or "").splitlines():
        # Match "Mainframe:", "Quantum Processor:", "[Mainframe]", "[Quantum Processor]", "Alice (excited):", etc.
        match = re.match(r"^\s*(?:\*\*|###*|[-*+]\s+)?(?:\[\s*([A-Za-z][A-Za-z0-9_'\s-]{1,30})\s*\]|([A-Za-z][A-Za-z0-9_'\s-]{1,30})\s*[:：])", line)
        if match:
            label_text = (match.group(1) or match.group(2) or "").lower()
            if label_text:
                for w in _WORD_RE.findall(label_text):
                    speaker_labels.add(w)
    if speaker_labels:
        stop_words = stop_words.union(speaker_labels)
    # Length-aware loop threshold: a genuine degeneration loop repeats a
    # phrase dozens of times, while a long technical answer legitimately
    # names its subject three or four times across 400+ words. An absolute
    # 3-repeat rule rejected a correct 350-token deep-reasoning answer live.
    required_repeats = 3 + min(2, len(lower_words) // 220)
    # An enumerated answer repeats its phrasing once per case, by necessity.
    # "The probability of drawing a red first is 3/12… the probability of
    # drawing a blue first is 4/12… the probability of drawing a green first
    # is 5/12" is parallel structure with progression, which is what a correct
    # worked derivation looks like — not a model stuck in a loop.
    #
    # LIVE DEFECT, 2026-07-26: exactly that answer to the marble question was
    # rejected as repetitive_phrase_loop, four times over, and the person got
    # "I couldn't get to an answer I'd stand behind on that one".
    #
    # A repeat that occurs at most once per enumerated item is structure. One
    # that outruns the items is a loop, and still caught.
    # Count markers wherever they are, not only at line starts: the model
    # often welds its items together, and an answer's structure should not
    # depend on whether the formatting repair has run yet.
    enumerated_items = len(
        re.findall(r"(?:^|[\n.!?:])\s*(?:[-*+]|\d{1,2}[.)])\s+\S", str(reply_text or ""))
    )
    if enumerated_items >= 2:
        required_repeats = max(required_repeats, enumerated_items + 1)
    # Question-sourced phrases are topical by definition: an answer that
    # compares "an early single-owner design with a late deduplication
    # design" MUST echo those noun phrases while comparing, choosing, and
    # describing verification. They only count as a loop at pathological
    # density (a model looping the question's own words still gets caught).
    question_content_words = {
        _topical_stem(w) for w in _WORD_RE.findall(user) if w.lower() not in stop_words
    }
    question_phrase_repeats = max(8, required_repeats * 2)
    for n in (4, 3, 2):
        counts: dict[tuple[str, ...], int] = {}
        for i in range(0, max(0, len(lower_words) - n + 1)):
            gram = tuple(lower_words[i:i + n])
            if sum(1 for part in gram if part not in stop_words) < 2:
                continue
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            # Content-word containment (not literal n-gram match): the answer
            # says "the single-owner design" where the question said "an
            # early single-owner design" — same topical phrase, different
            # articles. Recombining question vocabulary is topical; only
            # pathological density of it reads as a loop.
            question_sourced = question_content_words and all(
                _topical_stem(part) in question_content_words
                for part in gram
                if part not in stop_words
            )
            threshold = (
                question_phrase_repeats if question_sourced else required_repeats
            )
            if count >= threshold:
                # A loop is a reply that stops going anywhere. Repetition alone
                # does not establish that — a worked answer repeats its framing
                # once per case by construction, and five different correct
                # answers to the same question were rejected here on 2026-07-26,
                # each for a different n-gram.
                #
                # Progression is the thing being asked about, so measure it: if
                # the reply's statements are nearly all different, it is
                # advancing and the repeated phrase is its subject. If they are
                # not, it is stuck, and that is the loop this exists to catch.
                if _distinct_statement_ratio(reply_text) >= 0.7:
                    continue
                return "repetitive_phrase_loop"

    content_words = [
        w for w in lower_words
        if w not in {"i", "you", "it", "that", "this", "the", "a", "to", "and", "but", "then", "mean", "know"}
    ]
    if len(content_words) >= 8 and len(set(content_words)) / max(1, len(content_words)) < 0.36:
        # Lexical diversity alone cannot tell a worked derivation from a loop.
        # Measured 2026-07-26 on the live surface: a correct enumerated answer
        # to the marble question scored 0.327 and a model repeating "I want to
        # help you with that" scored 0.318. Both are enumerated, both reuse a
        # small vocabulary — because that is what enumerating cases looks like.
        #
        # What separates them is PROGRESSION. The derivation's items all differ
        # (3/12, 4/12, 5/12; red, blue, green); the loop's are verbatim repeats.
        if _distinct_statement_ratio(reply_text) >= 0.7:
            return ""
        return "low_lexical_diversity_loop"
    return ""

def _topical_stem(word: str) -> str:
    """Crude suffix strip, enough to match a word to its own inflections.

    The question-sourced exemption compares the answer's repeated phrases
    against the question's vocabulary, and it compared them literally — so
    "drawing" did not match "draw" and "probabilities" did not match
    "probability". Live 2026-07-26 that cost a correct derivation: the person
    asked "I draw two without replacement… what's the probability", the answer
    said "the probability of drawing" three times as it worked each case, and
    the repetition read as invented rather than topical.

    Deliberately shallow. Over-stemming would let genuinely unrelated words
    collide, so this only removes the endings that separate a word from its
    own forms.
    """
    lowered = str(word or "").lower()
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"  # probabilities -> probability
    for suffix in ("ingly", "ing", "edly", "ed", "ly"):
        if len(lowered) - len(suffix) >= 4 and lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    if len(lowered) > 4 and lowered.endswith("es") and lowered[-3:-2] in "sxzoh":
        return lowered[:-2]  # boxes -> box, matches -> match
    if len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]  # marbles -> marble
    return lowered

def repeated_statements(reply_text: Any) -> list[tuple[str, int]]:
    """Sentences of substance this reply says more than once, verbatim.

    The distinct-statement RATIO cannot see this. A reply that repeats three
    of its eleven sentences word for word scores 0.727 and passes a 0.7 bar,
    which is what happened live on 2026-08-19: the closing paragraph of a
    statistics answer repeated three times and the reply ran off the end
    mid-sentence.

    Raising the bar instead would re-break what the bar protects — a worked
    derivation reuses its scaffolding across items and scores low by design.
    Verbatim repetition separates them exactly: enumerated items differ from
    each other ("the probability of drawing a blue first is 4/12", "green is
    5/12"), so a correct derivation repeats no whole sentence at all.
    """
    from .response_reliability import (
        _SENTENCE_SPLIT_RE,
        _VERBATIM_REPEAT_MIN_WORDS,
    )

    body = str(reply_text or "").strip()
    if not body:
        return []
    counts: dict[str, int] = {}
    for line in body.splitlines():
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", sentence)
            cleaned = re.sub(r"[*_`#]+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            if len(cleaned.split()) < _VERBATIM_REPEAT_MIN_WORDS:
                continue
            counts[cleaned] = counts.get(cleaned, 0) + 1
    return sorted(
        ((text, count) for text, count in counts.items() if count > 1),
        key=lambda pair: -pair[1],
    )

def repair_verbatim_repeats(reply_text: Any) -> str:
    """Drop sentences this reply already said, keeping the first of each.

    Detecting the loop and rejecting the draft would cost the person a whole
    answer over a duplicated closing paragraph. The content is all there; one
    copy of it is the answer they asked for.

    Only exact repeats go. Everything else — order, wording, code, lists —
    survives byte for byte, so a repair can never be the thing that changed
    what she said.
    """
    from .response_reliability import (
        _SENTENCE_SPLIT_RE,
        _VERBATIM_REPEAT_MIN_WORDS,
    )

    body = str(reply_text or "")
    if not body.strip():
        return ""
    if not repeated_statements(body):
        return body.strip()

    seen: set[str] = set()
    kept_lines: list[str] = []
    for line in body.splitlines():
        pieces = _SENTENCE_SPLIT_RE.split(line)
        if len(pieces) <= 1:
            kept_lines.append(line)
            continue
        kept: list[str] = []
        for sentence in pieces:
            cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", sentence)
            cleaned = re.sub(r"[*_`#]+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            if len(cleaned.split()) >= _VERBATIM_REPEAT_MIN_WORDS:
                if cleaned in seen:
                    continue
                seen.add(cleaned)
            kept.append(sentence)
        kept_lines.append(" ".join(part.strip() for part in kept if part.strip()))
    repaired = "\n".join(kept_lines)
    return re.sub(r"\n{3,}", "\n\n", repaired).strip()

def _distinct_statement_ratio(reply_text: Any) -> float:
    """Share of this reply's statements that say something new.

    Statements are lines and sentences with list markers, emphasis and
    whitespace normalised away, so "2. **Both red:** …" and "3. **Both blue:**
    …" compare as the different claims they are, while three identical
    sentences under three different numbers compare as one.
    """
    from .response_reliability import (
        _SENTENCE_SPLIT_RE,
    )

    body = str(reply_text or "").strip()
    if not body:
        return 1.0
    statements: list[str] = []
    for line in body.splitlines():
        for sentence in _SENTENCE_SPLIT_RE.split(line):
            cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", sentence)
            cleaned = re.sub(r"[*_`#]+", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
            if len(cleaned) >= 12:
                statements.append(cleaned)
    if len(statements) < 3:
        return 1.0
    return len(set(statements)) / len(statements)
