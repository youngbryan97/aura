"""The fixed texts the reliability checks recognise as hers not to send, and the one she must never send again.

Lifted from core/conversation/response_reliability.py, which imports every name back, so a check, a caller or a test
reading them there reads the same objects.
"""
from __future__ import annotations

import re

_CONFUSION_REPAIR_FLOOR = (
    "Let's look at this more clearly. I'm still focused on our conversation, "
    "and I want to make sure I'm giving you a real answer, not just a fragment."
)
_RELIABILITY_REPAIR_FLOOR = (
    "I should not call that a clean turn. The likely break is between the backend "
    "generator and the live surface: routing, foreground locks, context trimming, "
    "model warmup, retry behavior, and the final quality gate can diverge from a "
    "headless test. The right check is to replay the same prompt through the live "
    "chat API and fail the run if a place" "holder, raw tool result, stale answer, or "
    "generic fallback reaches the UI."
)
_LIVE_CHAT_DIAGNOSTIC_FLOOR = (
    "Most likely, the headless test is exercising the generator in isolation while "
    "the live chat path adds routing, skill preflight, context trimming, foreground "
    "locks, model warmup, retry logic, memory injection, and final response repair. "
    "I would replay the same prompt through the live /api/chat path, capture the "
    "selected lane and every repaired draft, then fail the test if the UI receives "
    "a place" "holder, raw tool result, stale answer, persona-card intro, or request "
    "for details when the prompt already gave enough information."
)
_LIVE_CHAT_FIX_FIRST_FLOOR = (
    "Fix the live parity harness first, because that is where working backend "
    "answers can still be flattened before they reach the UI. I would make the "
    "same /api/chat request the GUI makes, capture routing, selected skill, model "
    "drafts, repairs, and final text, then fail the run if a stale answer, raw "
    "tool result, place" "holder, or repeated diagnostic floor survives to the screen."
)
_STATUS_REPAIR_FLOOR = (
    "Yes. I'm following what you said and ready to continue from this turn. "
    "Tell me where you want to pick up."
)
# The canned presence reflex this module must never emit again.
#
# _STATUS_REPAIR_FLOOR used to be "I'm right here with you. My mind feels
# steady enough to answer clearly..." — asserting steadiness and attention at
# the one moment neither could be demonstrated, since the floor only fires
# when the answer lane failed. tools/conversation_endurance_probe.py had been
# scoring that exact sentence as a fluent ungrounded reflex the whole time:
# one half of the system marking down what the other half wrote. The probe now
# imports this pattern instead of keeping its own copy, so the standard and
# the generator cannot drift apart again.
CANNED_PRESENCE_REFLEX_RE = re.compile(r"i'?m right here with you", re.IGNORECASE)

#: What the chat route says when a turn is cancelled by something other than the person. It is the runtime's notice,
#: not her words: logged as her reply, it reached her prompt as her own past turns, 24 times on 11 November 2026, and
#: on 12 November five of her replies in a row were her own versions of it ("I'm here. My response was interrupted").
TURN_CUT_SHORT_REPLY = "I'm here. My response was cut short — I'll pick up with whatever you say next."

_RELIABILITY_FLOOR_TEXTS = (
    TURN_CUT_SHORT_REPLY,
    _CONFUSION_REPAIR_FLOOR,
    _RELIABILITY_REPAIR_FLOOR,
    _LIVE_CHAT_DIAGNOSTIC_FLOOR,
    _LIVE_CHAT_FIX_FIRST_FLOOR,
    _STATUS_REPAIR_FLOOR,
)
