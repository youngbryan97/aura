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

__all__ = ["CLOSED_ANSWER_WORDS", "asks_for_a_closed_answer", "gives_a_closed_answer", "is_a_bare_answer"]

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
