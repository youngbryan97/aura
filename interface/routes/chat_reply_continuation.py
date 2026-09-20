"""Joining a same-model continuation to the draft it continues.

Lifted whole out of `chat_reply_shaping`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import re


def _continuation_restates(head: str, tail: str) -> bool:
    """Whether the continuation says again what the head already said.

    Its subject words are the head's, in any form, and it brings no number
    the head did not have. Short tails are not judged: a few words can
    coincide by chance.
    """
    from core.conversation.thread_continuity import content_terms
    from core.language.word_forms import matching_word_forms

    tail_terms = content_terms(tail)
    if len(tail_terms) < 4:
        return False
    head_terms = content_terms(head)
    shared = (tail_terms & head_terms) | matching_word_forms(tail_terms, head_terms)
    if len(shared) / float(len(tail_terms)) < 0.7:
        return False
    numbers = lambda text: set(re.findall(r"\d[\d,]*(?:\.\d+)?", text))  # noqa: E731
    return numbers(tail) <= numbers(head)


def _merge_reply_continuation(partial: object, continuation: object) -> str:
    """Join a same-model continuation without repeating its overlap.

    The continuation model may resume at the exact next token, repeat a short
    suffix for coherence, or ignore the contract and regenerate the complete
    answer. All three are valid model outputs; this deterministic merge only
    removes byte-identical overlap and never invents prose.
    """
    head = str(partial or "").rstrip()
    tail = str(continuation or "").lstrip()
    if not head:
        return tail
    if not tail:
        return head
    if tail.startswith(head):
        return tail

    common_prefix = 0
    for left, right in zip(head, tail, strict=False):
        if left != right:
            break
        common_prefix += 1
    if common_prefix >= 24:
        # A model that regenerated despite the continuation contract may have
        # produced a complete replacement, or it may have hit an earlier
        # deadline. Never let the latter erase already-authored progress.
        tail_complete = tail.rstrip().endswith(
            (".", "!", "?", '"', "'", "”", "’", ")", "]")
        )
        if tail_complete or len(tail) >= len(head):
            return tail
        return head

    max_overlap = min(len(head), len(tail), 1200)
    overlap = 0
    for size in range(max_overlap, 2, -1):
        if head[-size:] == tail[:size]:
            overlap = size
            break
    if overlap:
        return head + tail[overlap:]

    # A model that started the answer again in other words has not
    # continued it: "**240 minutes (4 hours).** Net drain is 8 − 3 = 5 L/min,
    # so 1,200 ÷ 5 = 240." followed by "**240 min.** Net drain 8 − 3 = 5
    # L/min → 1,200 ÷ 5 = 240." went to the person as two answers (LIVE
    # 2026-09-19). A restatement replaces the head only when the head is
    # not itself complete.
    if _continuation_restates(head, tail):
        head_complete = head.rstrip().endswith(
            (".", "!", "?", '"', "'", "”", "’", ")", "]")
        )
        return head if head_complete else tail

    separator = ""
    if not head[-1].isspace() and not tail[0].isspace():
        separator = "" if tail[0] in ".,;:!?)]}" else " "
    return f"{head}{separator}{tail}"
