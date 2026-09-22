"""core/senses/sight_intent.py — telling "look at this" from "tell me about cameras".

Two intents share most of their vocabulary and have nothing else in common:

    "how many fingers am I holding up"   → look through the camera, now
    "turn the camera on"                 → operate the camera control

and a third case matters more than either, because it is the common one:

    "the camera on my phone is broken"   → neither; just talk

Getting this wrong is expensive in both directions. Missing a real request
makes her blind exactly when she was asked to see. Firing on a remark about
cameras turns the webcam on in the middle of a conversation that was not
about it, which is the single most alarming thing an assistant with a camera
can do.

So, as with the addressivity gate, the rules are readable and each one says
what it is for. Sight requests need a *deictic* — "this", "here", "I", "my",
"in front of you" — because a question about the visible world is almost
always anchored to the present moment and the shared space. "What colour is
this?" is a request to look; "what colour is a stop sign" is not, and the
only difference is the pointing word.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Verbs of seeing, aimed at her. "Can you see", "look at", "watch me".
_SEEING = (
    r"\b(?:can you |could you |do you )?(?:see|look at|look|watch|check|read|"
    r"tell me what.{0,20}see|show me what.{0,20}see)\b"
)

# Words that anchor a question to the here and now. Without one of these, a
# question about the visible world is almost certainly about the world in
# general — which is a memory question, not a camera question.
_DEICTIC = re.compile(
    r"\b(?:this|these|here|right now|in front of (?:you|me|the camera)|"
    r"on (?:my|the) (?:screen|desk|face)|i'?m holding|i am holding|"
    r"i'?m wearing|am i|my hand|my face|at me|the camera)\b",
    re.IGNORECASE,
)

# Questions that only make sense about something being shown.
#
# "how many fingers" needs the person in it. On its own it is as likely to be
# general knowledge — "how many fingers does a hand have" — and a camera that
# switches on for a trivia question is a camera nobody leaves enabled.
_SHOWING = re.compile(
    r"\bhow many (?:fingers|of these)\b\s*(?:am i|i'?m|i am|are you|do you|can you)\b"
    r"|\bhow many (?:fingers|of these)\b(?=[^?]*\b(?:holding|up|showing)\b)"
    r"|\bwhat am i (?:holding|wearing|pointing at|doing)\b"
    r"|\bwhat(?:'s| is) (?:this|that|in my hand)\b"
    r"|\bwho(?:'s| is) (?:this|here|in frame|in the (?:room|shot))\b"
    r"|\bdo i look\b"
    r"|\bwhat colou?r is (?:this|it|my)\b"
    r"|\bread (?:this|the label|the screen|what.{0,15}holding)\b",
    re.IGNORECASE,
)

_SEEING_RE = re.compile(_SEEING, re.IGNORECASE)

# Operating the camera control, rather than looking through it. These are
# imperatives about the *device*, and answering them with a description of
# what the camera sees would be answering a different question.
_CAMERA_ON = re.compile(
    r"\b(?:turn|switch|power)\s+(?:on\s+)?(?:the\s+|your\s+|my\s+)?"
    r"(?:camera|webcam|video)\b(?:\s+on)?"
    r"|\b(?:enable|start|open)\s+(?:the\s+|your\s+|my\s+)?(?:camera|webcam)\b"
    r"|\bcamera\s+on\b",
    re.IGNORECASE,
)
_CAMERA_OFF = re.compile(
    r"\b(?:turn|switch|power)\s+(?:off\s+)?(?:the\s+|your\s+|my\s+)?"
    r"(?:camera|webcam|video)\b(?:\s+off)?"
    r"|\b(?:disable|stop|close)\s+(?:the\s+|your\s+|my\s+)?(?:camera|webcam)\b"
    r"|\bcamera\s+off\b",
    re.IGNORECASE,
)

# Talking *about* a camera rather than asking for one. A remark, a story, a
# question about hardware — none of which should start a webcam.
_ABOUT_CAMERAS = re.compile(
    # "my phone camera", "the security camera"
    r"\b(?:my|the|his|her|their|a)\s+(?:phone|laptop|security|dash|film|old|new)\s+camera\b"
    # "the camera on my phone" — the same remark with the words the other way
    # round, which is how people actually say it.
    r"|\bcamera\s+(?:on|in|of)\s+(?:my|the|his|her|their|that|this)\s+"
    r"(?:phone|laptop|macbook|computer|tablet|ipad|car|door|house|room)\b"
    # The camera as the subject of a statement rather than the object of a
    # request: "the camera is broken", "camera quality is terrible".
    r"|\bcamera(?:'s)?\s+(?:is|was|isn'?t|wasn'?t|broke|broken|died|quality|lens|app|roll)\b"
    r"|\bcamera\s+\w+\s+(?:is|was)\s+(?:terrible|awful|bad|great|good|broken)\b"
    r"|\b(?:what|which|how much|how do)\b.{0,30}\bcamera\b.{0,30}\b(?:cost|buy|work|mean)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class SightIntent:
    """What the turn is asking of the camera, if anything."""

    kind: str  # "look" | "camera_on" | "camera_off" | "none"
    question: str = ""
    reason: str = ""

    @property
    def wants_camera(self) -> bool:
        return self.kind != "none"


def classify(message: str) -> SightIntent:
    """Decide whether this turn is about the camera, and in which sense."""
    text = " ".join(str(message or "").split())
    if not text:
        return SightIntent("none")

    # Talking about a camera is not asking for one, and this outranks the
    # control patterns below — "my phone camera is broken" contains "camera"
    # and nothing else about it is a request.
    if _ABOUT_CAMERAS.search(text):
        return SightIntent("none", reason="the camera is the subject, not the request")

    # Off before on: "turn the camera off" contains "turn the camera", and a
    # request to stop being watched must never be read as a request to start.
    if _CAMERA_OFF.search(text) and re.search(r"\boff|disable|stop|close\b", text, re.I):
        return SightIntent("camera_off", reason="asked to switch the camera off")

    if _CAMERA_ON.search(text):
        return SightIntent("camera_on", reason="asked to switch the camera on")

    # A question that only makes sense about something being shown needs no
    # further evidence — nobody asks "how many fingers am I holding up" about
    # the world in general.
    if _SHOWING.search(text):
        return SightIntent("look", question=text, reason="a question about what is in frame")

    # A verb of seeing, anchored to the here and now. Both halves are
    # required: "can you see the difference" is a figure of speech, and "what
    # colour is a stop sign" is a memory question wearing a visual verb.
    if _SEEING_RE.search(text) and _DEICTIC.search(text):
        return SightIntent(
            "look", question=text, reason="asked to look at something present"
        )

    # Language is not a finite trigger list. Requests such as "which of your
    # senses can actually tell whether anyone else is physically here?" do not
    # need a visual verb or one of the deictics above, but their answer still
    # depends on a fresh physical observation. Route the meaning through the
    # same local semantic evidence engine used for screen and source questions.
    # The lexical rules remain a cheap, privacy-conservative floor when the
    # embedding model is unavailable; semantic routing may add a request but
    # never turns an explicit camera-off or about-camera statement into one.
    try:
        from core.cognition.evidence_relevance import (
            PHYSICAL_PERCEPTION,
            wants_evidence,
        )

        if wants_evidence(text, PHYSICAL_PERCEPTION):
            return SightIntent(
                "look",
                question=text,
                reason="the question semantically requires present physical perception",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug(
            "%s unavailable (%s: %s); the semantic reader could not be asked, so this is not classified as needing sight",
            "it",
            type(exc).__name__,
            exc,
        )

    return SightIntent("none")


__all__ = ["SightIntent", "classify"]
