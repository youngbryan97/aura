"""Quoting the screen is a claim, and it needs the capture that would justify it.

MEASURED live 2026-08-04. Asked "read me the actual text you can see in the
visible part of System Settings and Chrome — quote it", she answered::

    Settings: "Show Closed Captions on supported websites"
    Chrome: "Analysis: Codebase has 15% unused imports, 8% redundant code
             blocks. Suggestion: Refactor global scope to reduce cognitive load."
    That's the visible text on those windows.

An independent ``screencapture`` taken seconds later returned an all-black
frame — min 0, max 0, mean 0.0 across 3456x2234. There was nothing on that
display to read, and no capture ran on that turn. The quotes were invented.

The turn before it was honest and correct: "Look at my screen… what else is
visible behind or beside yours?" matched ``asks_about_occluded_view``, ran
``capture_blueprint()``, and named System Settings, Chrome, Contacts, Finder
and TextEdit with visibility fractions — every one of which
``System Events`` independently confirms. That path had evidence and used it.

The difference is not that one question was harder. It is that the second one
did not match any intent predicate, so it went to free generation, and free
generation has no way to know it cannot see. So the gate has to know:

  * a reply that QUOTES on-screen text is asserting a reading happened;
  * a reading happened only if a capture produced text this turn;
  * without that, the reply is an unsupported claim and must not ship.

This does not restrict what she may say about a screen. She can describe the
layout, say she cannot read something, or refuse. What it stops is presenting
invented strings as things she read — the same standard the rest of this
runtime applies to a tool it did not run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

#: The turn asked her to READ or QUOTE what is on the screen, as opposed to
#: describing the arrangement of windows (which `occluded_view_intent` owns).
_READ_THE_SCREEN_RE = re.compile(
    r"\b(?:read|quote|transcribe|type\s+out|spell\s+out|what\s+does\s+it\s+say|"
    r"what'?s\s+written|word\s+for\s+word|verbatim|literally\s+say)\b"
    # Asking WHAT IS THERE is a request for a reading just as much as asking
    # for the words. Live 2026-08-10: "whats on my screen right now? name the
    # actual apps you can see." matched none of the verbs above, so the guard
    # stood down — and with no capture on that turn she answered with Chrome,
    # three tab titles, a document and "an email from my landlord about the
    # rent increase", all invented.
    r"|what(?:'?s)?\s+(?:is\s+)?(?:on|up\s+on|showing\s+on)\s+"
    r"(?:my|the|your)?\s*(?:screen|display|desktop|monitor)\b"
    r"|\bname\s+the\s+(?:actual\s+)?(?:apps?|applications?|windows?|tabs?)\b"
    r"|\bwhich\s+(?:app|application|window|tab)\b"
    r"|\bwhat\s+(?:apps?|applications?|windows?|tabs?)\b"
    r"|\bwhat\s+(?:do|can)\s+you\s+see\b"
    r"|\bfrontmost\b|\bin\s+front\b",
    re.IGNORECASE,
)
#: "Read me the actual text you can see in the visible part of System Settings
#: and Chrome" — the live request — names no screen and no window. It names two
#: APPLICATIONS and the word "visible", which is how a person actually asks.
#: A subject list that only knows the word "screen" misses the real question.
_SCREEN_SUBJECT_RE = re.compile(
    r"\b(?:screen|display|monitor|desktop|window|windows|tab|app|application|"
    r"page|dialog|panel|visible|on[- ]screen|showing|in\s+front\s+of\s+(?:me|you))\b",
    re.IGNORECASE,
)

#: The reply presents a specific string as something visible on screen. Quoted
#: text is the signal — an unquoted description of a window is not this claim.
# The lookarounds are load-bearing. A straight apostrophe is far more often a
# contraction than a quote mark, and without them ANY two contractions in one
# sentence read as a quotation: "I couldn't get a clear enough answer together,
# and I'd rather…" was matched as the quoted string `'t get a clear enough
# answer together, and I'`. Paired with a screen noun anywhere in the reply,
# that is a fabricated screen-reading claim built entirely out of punctuation —
# measured 2026-08-10 against the runtime's own last-resort text.
# A real quotation opens where a word does not end and closes where one does
# not begin, which is exactly what 'New Chat' does and "couldn't" does not.
_QUOTED_TEXT_RE = re.compile(r"(?<![A-Za-z])[\"“”'][^\"“”']{8,}[\"“”'](?![A-Za-z])")
_ASSERTS_A_READING_RE = re.compile(
    r"\b(?:the\s+visible\s+text|on\s+(?:the\s+)?screen\s+it\s+says|"
    r"it\s+says|i\s+can\s+see\s+the\s+text|reads?\s*:|"
    r"the\s+text\s+(?:on|in)\s+(?:it|them|the)|showing\s*:"
    # Naming what is displayed is a reading. "The tabs say 'New Chat'..." was
    # served with no capture behind it, and matched nothing above.
    r"|(?:tabs?|windows?|titles?)\s+(?:say|says|read|reads)"
    r"|i\s+can\s+see\b|i\s+see\b"
    r"|(?:is|are)\s+(?:in\s+front|frontmost|open|showing|visible)"
    r"|behind\s+it\b|partly\s+visible\b)",
    re.IGNORECASE,
)

# LIVE DEFECT 2026-08-10. Asked "if i asked you to keep an eye on something
# while i'm gone, would that mean anything to you?" — a question about memory,
# containing no screen, no window and no display — she wrote a 930-character
# answer and the gate destroyed all of it as
# `unsupported_screen_reading_claim`. The person got
# "I couldn't get a clear enough answer together".
#
# Nothing about that turn concerned a display. The gate armed itself off HER
# OWN reply, because the two halves above are satisfied by ordinary English:
#
#   * `_ASSERTS_A_READING_RE` matches `i\s+can\s+see` — and "see" in English is
#     overwhelmingly COMPREHENSION, not vision. "I can see why you'd ask that"
#     is the single most natural way to open that answer.
#   * `_SCREEN_SUBJECT_RE` then matches the bare word "visible", "app", "page"
#     or "showing" ANYWHERE in the reply — 900 characters away, in a different
#     paragraph, about a different subject.
#
# Measured on the real predicate: "I can see why you'd ask that. Nothing I hold
# right now is visible to me after this process ends." → unsupported=True. So
# does "I can see that the app you're describing would need a scheduler."
#
# The repair is not a longer exception list; the next idiom would land in the
# same place. It is that a claim to have READ THE DISPLAY is a claim with a
# grammatical shape: a perception or reading predicate bound to a display
# referent WITHIN ONE SENTENCE. "I can see" with no display in the clause is
# not a reading. "The tabs say '…'" is.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")

#: A noun phrase that refers to the user's display. `screen`/`display`/
#: `monitor` are unambiguous. `desktop` and UI objects (`window`, `tab`,
#: `dialog`) refer to the display only when grammar anchors them to something
#: present. A bare `desktop` is commonly an attributive software noun, as in
#: "desktop task lane", and is not evidence of physical perception.
_DISPLAY_REFERENT_RE = re.compile(
    r"\b(?:screens?|displays?|monitors?"
    r"|(?:your|the|those|these|that|this|my|his|her|their|other|another|each|both)\s+"
    r"(?:window|windows|tab|tabs|dialog|dialogs|title\s*bars?|browser)"
    r"|menu\s*bar|status\s*bar|dock\b|frontmost"
    r"|on[-\s]screen)\b",
    re.IGNORECASE,
)

# `desktop` is ambiguous until its syntactic role is known. This recognises it
# only as the head of a present display noun phrase or the object of a spatial
# preposition. It therefore accepts "your desktop shows" and "on my desktop"
# while rejecting attributive compounds such as "the desktop task lane".
_DESKTOP_DISPLAY_REFERENT_RE = re.compile(
    r"\b(?:"
    r"(?:your|the|those|these|that|this|my|his|her|their|other|another|each|both)\s+"
    r"desktops?\b(?=\s*(?:$|[,.!?;:]|['’]s\b|(?:says?|reads?|shows?|displays?|"
    r"contains?|is|are|has|have)\b))"
    r"|(?:on|across|over)\s+(?:(?:your|the|my|his|her|their)\s+)?desktops?\b"
    r")",
    re.IGNORECASE,
)

#: A perception or reading verb. Bound to a display referent in the same
#: sentence, this asserts a reading; on its own it asserts nothing about a
#: display. `see` is here, but only ever reaches a verdict through the binding
#: below — which is precisely what "I can see why" lacks.
_PERCEPTION_VERB_RE = r"(?:see|seeing|saw|read|reading|look(?:ing)?\s+at|make\s+out)"

#: The referent doing the showing: "the tabs SAY", "your window IS SHOWING".
_REFERENT_PREDICATE_RE = r"(?:says?|reads?|shows?|displays?|contains?|is\s+showing|are\s+showing|is\s+titled|are\s+titled)"

#: Perception → referent ("I can see the screen"), or referent → predicate
#: ("the tabs say"). `[^.!?\n]{0,60}` keeps both inside one sentence: the
#: binding is the whole point, so it may not reach across a full stop.
_DISPLAY_REFERENT_PATTERN = (
    r"(?:" + _DISPLAY_REFERENT_RE.pattern + r"|" + _DESKTOP_DISPLAY_REFERENT_RE.pattern + r")"
)
_BARE_DESKTOP_SUBJECT_RE = re.compile(
    r"\bdesktops?\b[^.!?\n]{0,20}?\b" + _REFERENT_PREDICATE_RE + r"\b",
    re.IGNORECASE,
)
_BOUND_READING_RE = re.compile(
    r"(?:\b" + _PERCEPTION_VERB_RE + r"\b[^.!?\n]{0,60}?" + _DISPLAY_REFERENT_PATTERN + r")"
    r"|(?:" + _DISPLAY_REFERENT_PATTERN + r"[^.!?\n]{0,60}?\b" + _REFERENT_PREDICATE_RE + r"\b)"
    # A bare noun can be the display subject ("Desktop shows ...") without
    # making every compound beginning with `desktop` a display reference.
    r"|(?:\bdesktops?\b[^.!?\n]{0,20}?\b" + _REFERENT_PREDICATE_RE + r"\b)",
    re.IGNORECASE,
)


def _has_display_referent(text: Any) -> bool:
    body = str(text or "")
    return bool(
        _DISPLAY_REFERENT_RE.search(body)
        or _DESKTOP_DISPLAY_REFERENT_RE.search(body)
        or _BARE_DESKTOP_SUBJECT_RE.search(body)
    )


#: Phrases that assert a reading with no referent needed, because the display
#: is already inside the phrase. These are the live confabulations verbatim.
_EXPLICIT_READING_RE = re.compile(
    r"\b(?:the\s+visible\s+text|on\s+(?:the\s+)?screen\s+it\s+says"
    r"|i\s+can\s+see\s+the\s+text|the\s+text\s+(?:on|in)\s+(?:it|them|the)"
    r"|what'?s\s+(?:currently\s+)?on\s+(?:your|the|my)\s+(?:screen|display)\s+is)\b",
    re.IGNORECASE,
)

#: Saying she could NOT see is the honest outcome and must always pass.
_ADMITS_NO_READING_RE = re.compile(
    r"\b(?:couldn'?t|could\s+not|can'?t|cannot|unable\s+to|did\s*n[o']?t)\s+"
    r"(?:actually\s+)?(?:read|see|capture|look)"
    r"|\bno\s+capture\b|\bnothing\s+(?:came\s+back|to\s+quote)\b"
    r"|\bi\s+won'?t\s+make\s+(?:one|it)\s+up\b"
    r"|\bhave\s+no\s+text\s+to\s+quote\b",
    re.IGNORECASE,
)


#: Where things sit relative to each other, which `occluded_view_intent` owns.
#: "What windows are behind yours" is answered from arrangement, not from
#: reading, and must not be dragged in by the widened content cues below.
_ARRANGEMENT_RE = re.compile(
    r"\b(?:behind|under|underneath|beneath|on\s+top\s+of|over|covering|"
    r"obscur\w*|overlap\w*|stacked)\b",
    re.IGNORECASE,
)


def asks_to_read_the_screen(user_message: Any) -> bool:
    """True when the turn asks what is on the screen."""
    text = str(user_message or "")
    if not text.strip():
        return False
    if not _SCREEN_SUBJECT_RE.search(text):
        return False
    explicit_read = bool(
        re.search(
            r"\b(?:read|quote|transcribe|type\s+out|spell\s+out|verbatim|"
            r"word\s+for\s+word)\b",
            text,
            re.IGNORECASE,
        )
    )
    if _ARRANGEMENT_RE.search(text) and not explicit_read:
        return False
    return bool(_READ_THE_SCREEN_RE.search(text))


def quotes_screen_content(reply_text: Any, *, display_binding_required: bool = False) -> bool:
    """True when the reply presents specific content as read from the screen.

    Widened from "a quoted string" to "a specific claim about what is
    displayed". Naming the frontmost application, or what the tabs say, is a
    checkable assertion about the display exactly as a quotation is, and
    requires the same evidence — the standard this runtime applies to any tool
    it did not run.

    An admission that she could not see is the honest outcome and always
    passes, so the guard can never push her toward inventing rather than
    saying so.

    ``display_binding_required`` is the tier used when the turn itself was
    never about a screen. There, a loose cue is not enough: the reply has to
    bind a perception verb to a display referent inside one sentence before it
    counts as claiming a reading. See ``_BOUND_READING_RE`` for the live
    defect that made the distinction necessary.
    """
    body = str(reply_text or "")
    if not body.strip():
        return False
    if _ADMITS_NO_READING_RE.search(body):
        return False

    if display_binding_required:
        if _EXPLICIT_READING_RE.search(body):
            return True
        if _BOUND_READING_RE.search(body):
            return True
        # A quotation still counts, but only in a sentence that is itself
        # about the display — not because the word "visible" appears in some
        # other paragraph.
        return any(
            _QUOTED_TEXT_RE.search(sentence) and _has_display_referent(sentence)
            for sentence in _SENTENCE_SPLIT_RE.split(body)
        )

    if _ASSERTS_A_READING_RE.search(body):
        return True
    return bool(_QUOTED_TEXT_RE.search(body) and _SCREEN_SUBJECT_RE.search(body))


@dataclass(frozen=True)
class ScreenReadingEvidence:
    """What a capture actually produced on this turn."""

    captured: bool = False
    text: str = ""
    source: str = ""
    unavailable_reason: str = ""
    capture_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    captured_at: float = 0.0
    entities: tuple[str, ...] = ()

    @property
    def supports_a_quotation(self) -> bool:
        """A capture that returned no text supports no quotation."""
        return bool(self.captured and self.text.strip())

    def supports_claim(self, reply_text: Any, *, max_age_seconds: float = 120.0) -> bool:
        """Whether this exact capture contains the content the reply attributes to it."""

        if not self.supports_a_quotation:
            return False
        try:
            from core.conversation.session_scope import (
                current_conversation_session,
                current_conversation_turn,
            )

            active_session = current_conversation_session()
            active_turn = current_conversation_turn()
        except (ImportError, RuntimeError):
            active_session = active_turn = ""
        if self.session_id and self.session_id != active_session:
            return False
        if self.turn_id and self.turn_id != active_turn:
            return False
        if self.captured_at and time.time() - self.captured_at > max_age_seconds:
            return False

        corpus = _normalize_screen_text(" ".join((self.text, *self.entities)))
        quoted = [
            _normalize_screen_text(match.group(0).strip('"“”\''))
            for match in _QUOTED_TEXT_RE.finditer(str(reply_text or ""))
        ]
        quoted = [fragment for fragment in quoted if fragment]
        if quoted:
            return all(fragment in corpus for fragment in quoted)

        # Unquoted display descriptions must still share substantive observed
        # entities with the capture; the mere fact that OCR returned something
        # cannot authorize an unrelated description.
        claimed = {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9._-]{2,}", str(reply_text or "").casefold())
            if token not in _SCREEN_CLAIM_STOP_WORDS
        }
        observed = set(re.findall(r"[a-z0-9][a-z0-9._-]{2,}", corpus))
        return bool(claimed and claimed & observed)

    def as_metrics(self) -> dict[str, Any]:
        return {
            "screen_captured": self.captured,
            "screen_text_chars": len(self.text.strip()),
            "screen_source": self.source,
            "screen_unavailable_reason": self.unavailable_reason,
            "screen_capture_id": self.capture_id,
            "screen_session_id": self.session_id,
            "screen_turn_id": self.turn_id,
        }


def _normalize_screen_text(value: Any) -> str:
    # OCR punctuation and whitespace are unstable; lexical order is the
    # evidence-bearing part. This remains exact after normalization, so a
    # merely related capture cannot license a different quotation.
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


_SCREEN_CLAIM_STOP_WORDS = frozenset(
    {
        "actual", "also", "another", "behind", "both", "browser", "desktop",
        "display", "front", "frontmost", "open", "screen", "showing", "that",
        "there", "these", "this", "visible", "window", "windows", "with", "your",
    }
)


def screen_reading_claim_is_unsupported(
    user_message: Any,
    reply_text: Any,
    evidence: ScreenReadingEvidence | None = None,
) -> bool:
    """True when the reply quotes the screen and nothing read it.

    Conservative in the direction this runtime always chooses: absence of
    evidence blocks only a QUOTATION, never a description. "System Settings is
    37% visible and I can't read what's under my window" needs no capture and
    is not touched.

    The two arms are deliberately asymmetric, because the ambiguity resolves
    differently on each. When the turn ASKED about the screen, "visible" and
    "I can see" mean the display, and any content report needs a capture. When
    the turn was about something else entirely, those same words carry their
    ordinary English senses, and only a sentence that actually binds a
    perception to a display counts as a claim. Treating both arms the same is
    what destroyed a correct 930-character answer about memory on 2026-08-10.
    """
    asked_about_the_screen = asks_to_read_the_screen(user_message)
    if not quotes_screen_content(reply_text, display_binding_required=not asked_about_the_screen):
        return False
    if not asked_about_the_screen and not _has_display_referent(reply_text):
        return False
    if evidence is None:
        return True
    return not evidence.supports_claim(reply_text)


def honest_unread_screen_reply(evidence: ScreenReadingEvidence | None = None) -> str:
    """What to say instead: the true state of the display."""
    reason = (evidence.unavailable_reason if evidence else "") or ""
    if reason:
        return (
            f"I couldn't actually read the screen just now ({reason}), so I have "
            "no text to quote you. I won't make one up."
        )
    return (
        "I couldn't actually read the screen just now — the capture came back "
        "with nothing on it — so I have no text to quote you. I won't make one up."
    )


__all__ = [
    "ScreenReadingEvidence",
    "asks_to_read_the_screen",
    "honest_unread_screen_reply",
    "quotes_screen_content",
    "screen_reading_claim_is_unsupported",
]
