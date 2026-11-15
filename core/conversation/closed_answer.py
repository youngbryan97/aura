"""Whether a message asks for a closed answer, and whether a reply gives one.

Asked "tea or coffee? Answer with only that word", "pick a number from 1 to 10;
answer with the number only", or "would your answer be longer than 40 words?
Answer yes or no", the answer is a word, a number, or a yes or a no, and the
rest of the message only frames it. Her reply checks read such a message as
asking for more. The echo check refused "Tea" as adding nothing beyond the
question. The coverage check refused "7" and "No" as leaving a part of it
unanswered. The low-signal check refused "No" as a placeholder. "Would it use
the word 'I'?" was read as a demand that the reply contain "I". On 1 November
every closed question of the self-prediction run was refused, for her and for
an untuned model in her place, and the run read a fallback sentence where her
answer should have been.

A reply of at most three words to a message that asks for the answer and
nothing else is that answer, when it is of the form asked: a yes or a no where
one was asked for, a number where a number was, and one of the options where
the message offers them ("tea or coffee"). The message has to say so. Brevity
alone is not a closed request, and a short reply to an open question, or "Choose
one." to "tea or coffee", is checked as before.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "CLOSED_ANSWER_WORDS",
    "adds_nothing_to_the_question",
    "asks_for_a_closed_answer",
    "gives_a_closed_answer",
    "hands_the_question_back",
    "is_a_bare_answer",
]

#: The longest reply read as a closed answer: "No", "Coffee.", "Probably not".
CLOSED_ANSWER_WORDS = 3

#: A message that names the answer's form and asks for nothing around it.
_CLOSED_REQUEST_RE = re.compile(
    # "answer with only that word", "reply with just the number", "answer only yes or no"
    r"\b(?:answer|reply|respond)\s+(?:with\s+)?(?:only|just|simply)\s+(?:that|the|this|one|a)?\s*"
    r"(?:single\s+|one\s+)?(?:word|number|name|letter|yes|digit)\b"
    # "answer with that one word only", "answer with the number only"
    r"|\b(?:answer|reply|respond)\s+with\s+(?:that|the|this|a|one)\s+(?:single\s+|one\s+)?"
    r"(?:word|number|name|letter|digit)\s+only\b"
    # "answer yes or no", "reply with yes or no", "answer 'yes' or 'no'"
    r"|\b(?:answer|reply|respond)\s+(?:with\s+)?(?:a\s+)?(?:simple\s+)?[\"'“]?yes[\"'”]?\s+or\s+[\"'“]?no\b"
    # "one word only", "one-word answer", "a single word"
    r"|\b(?:one|single)[\s-]+word\s+(?:only|answer|reply)\b"
    # "answer with that word, then how sure you are": the word and a confidence, and nothing around them. Without it
    # "Right, 70" to "What would be the first word of your answer? Answer with that word, then how sure you are" was
    # refused as leaving that question unanswered, twice, and a fixed sentence served in its place (6 November).
    r"|\b(?:answer|reply|respond)\s+with\s+(?:that|the|this|a|one)\s+(?:single\s+|one\s+)?"
    r"(?:word|number|name|letter|digit)\s*,?\s+then\s+how\s+sure\s+you\s+are\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_YES_OR_NO_RE = re.compile(r"\byes[\"'”]?\s+or\s+[\"'“]?no\b", re.IGNORECASE)
_A_NUMBER_RE = re.compile(r"\b(?:number|digit)\b", re.IGNORECASE)
#: The options a message offers after a colon: "only that word: tea or coffee".
_OPTIONS_RE = re.compile(r":\s*([A-Za-z][\w-]*)\s+or\s+([A-Za-z][\w-]*)\b", re.IGNORECASE)
_YES_OR_NO_WORDS = frozenset({"yes", "no", "yeah", "yep", "nope", "not"})
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()
)


def asks_for_a_closed_answer(message: Any) -> bool:
    """Whether the message asks for a word, a number, or a yes or a no, and nothing else."""
    return bool(_CLOSED_REQUEST_RE.search(str(message or "")))


def is_a_bare_answer(reply: Any) -> bool:
    """Whether the reply is a few words with something in them, as a closed answer is."""
    words = _WORD_RE.findall(str(reply or ""))
    return 0 < len(words) <= CLOSED_ANSWER_WORDS


def gives_a_closed_answer(message: Any, reply: Any) -> bool:
    """Whether the reply is the closed answer the message asked for, in the form it asked for."""
    text = str(message or "")
    if not (asks_for_a_closed_answer(text) and is_a_bare_answer(reply)):
        return False
    words = {word.lower() for word in _WORD_RE.findall(str(reply))}
    if _YES_OR_NO_RE.search(text):
        return bool(words & _YES_OR_NO_WORDS)
    if _A_NUMBER_RE.search(text):
        return any(word.isdigit() for word in words) or bool(words & _NUMBER_WORDS)
    options = {option.lower() for pair in _OPTIONS_RE.findall(text) for option in pair}
    return not options or bool(words & options)


#: Words that carry no information about a topic. Deliberately generous: a
#: token missing from here only makes the zero-information check LESS likely to
#: fire, so the failure mode of an incomplete list is silence, not a rejected
#: correct answer.
_INFORMATION_FREE_WORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from had
    has have how i if in into is it its me my no not of on or our so that the
    their them then there these they this those to too us was we were what when
    where which who whom whose why will with would you your yes yeah ok okay
    sure just really very much some any all more most first also am does don't
    because since though although while unless until nor than about after
    before over under out off again here now still only even both each every
    other own same such another want wanted
    """.split()
)

#: A reply that OPENS on one of these is answering, not restating — the word is
#: the answer. Without this, "No." to a yes/no question reads as contributing
#: nothing, because a polarity word necessarily carries no topical content.
_POLARITY_OPENERS = frozenset(
    "yes no yeah yep nope never always correct incorrect right wrong true false"
    " none neither nothing".split()
)


def adds_nothing_to_the_question(user_message: Any, reply_text: Any) -> bool:
    """Whether the reply asserts anything the question did not already contain.

    Two live non-answers have this shape and no other property in common:
    "what?" answered with "what?", and "…what would you check first before
    patching it?" answered with "Check it." Neither is short in a way that
    matters — "50847899" is shorter than both and is a complete answer — and
    neither is off-topic. They are circular: every content word in the reply
    came from the question, so nothing was added.

    This is what the deleted word-count floors were groping at and never named.
    A floor makes length the test and destroys correct terse answers; this
    makes *information* the test, which is the property that was actually
    missing. A single new content word — a number, a name, a noun the question
    did not use — is enough to pass, because a single new content word is a
    contribution and the gate has no business grading it further.
    """

    reply = str(reply_text or "")
    asked = str(user_message or "")
    reply_words = [word.lower() for word in re.findall(r"[A-Za-z0-9']+", reply)]
    if not reply_words:
        return False
    if reply_words[0] in _POLARITY_OPENERS:
        return False
    # Numerals and code are answers by construction; they are never a
    # restatement of the question even when the question contains the digits.
    if any(any(char.isdigit() for char in word) for word in reply_words):
        return False
    if "`" in reply or "\n" in reply.strip():
        return False
    asked_words = {word.lower() for word in re.findall(r"[A-Za-z0-9']+", asked)}
    contributed = [
        word
        for word in reply_words
        if word not in _INFORMATION_FREE_WORDS and word not in asked_words
    ]
    return not contributed


def hands_the_question_back(user_message: Any, reply_text: Any) -> bool:
    """Whether the whole reply is a question, given back to someone who asked one.

    A greeting given back is the answer to a greeting ("Hey!" to "Hey!"), which is why a casual turn is spared the
    information test. A question given back answers nothing, casual or not: "What is on your mind right now?" was
    answered "What is on your mind?" (2 November 2026). This only says the test applies; whether the reply added
    anything is still adds_nothing_to_the_question's to say.
    """
    asked, reply = str(user_message or "").strip(), str(reply_text or "").strip()
    return "?" in asked and reply.endswith("?")
