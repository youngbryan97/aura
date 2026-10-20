"""What, in text her model is handed, tells it who to be or what to do.

One reading for two uses: the map of her source (tools/where_she_is_still_prompted.py) and the census of what each
dispatch hands her model (core/brain/what_she_is_given.py). Text instructs when it addresses the model in the second
person ("You are", "Your task", "you must") or orders it ("Do not", "Never", "Always", "Respond with", "Return only",
"Output only"); in source, a headed section of a prompt ("## ") counts too. Quoted material inside a fence
(core/security/prompt_fencing.py) is content, and is not read; the note a fence closes with is read.
"""
from __future__ import annotations

import re

__all__ = ["INSTRUCTS", "ORDERS", "instructing_sentences"]

#: Second person or the imperative, addressed to the model.
ORDERS = re.compile(
    r"\bYou are\b|\bYour task\b|\byou must\b|\byou should\b|\bDo not\b|\bDon't\b|\bNever\b|\bAlways\b"
    r"|\bRespond (?:with|only|in)\b|\bReturn only\b|\bOutput only\b|\bYou (?:speak|have|will|can)\b",
)
#: In source, a prompt's headed section as well.
INSTRUCTS = re.compile(ORDERS.pattern + r"|(?m:^## )")

_FENCED = re.compile(r"<UNTRUSTED id=(\w+)[^>]*>.*?</UNTRUSTED id=\1>", re.DOTALL)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def instructing_sentences(text: object) -> list[str]:
    """The sentences of ``text``, outside any fence, that address the model or order it."""
    unfenced = _FENCED.sub(" ", str(text or ""))
    return [part.strip() for part in _SENTENCES.split(unfenced) if part.strip() and ORDERS.search(part)]
