"""Whether the person asked her to talk while something plays: an invitation to remark, not a silence to keep.

LIVE 20 October 2026, rehearsal 8 (Mark Rober's gingerbread contest): Bryan had asked "think out loud while we
watch", and six minutes in she had said nothing. Her rule for speaking unasked (core/together/when_to_say_something.py)
weighs how welcome a remark is; with no remark of hers yet taken up or turned down, and no silence of his to measure, a
remark of the greatest interest was outweighed: "welcome 0.250 x worth 0.954 x (1 + 1.00 lost by waiting) = 0.477
against (1 - welcome) x attention held 1.000 = 0.750". Asked to think out loud, a remark is what was asked for.
"""
from __future__ import annotations

import re

__all__ = ["asks_for_remarks"]

#: Asking to hear her as it goes: "think out loud", "tell me what you think as we watch", "react", "comment",
#: "talk me through it", "narrate".
_AS_IT_GOES = re.compile(
    r"\bthink(?:ing)?\s+(?:out\s+)?loud\b|"
    r"\b(?:react|comment|narrate|commentate|riff)\b|"
    r"\btalk\s+(?:me\s+)?(?:through|along)\b|"
    r"\b(?:tell|let)\s+me\s+(?:know\s+)?what\s+you(?:'re)?\s+(?:think|thinking|notice|noticing|see)\w*\s+"
    r"(?:as|while|when)\s+(?:we|it|you)\b|"
    r"\b(?:as|while)\s+(?:we|it)\s+(?:watch|play|go)\w*[^.?!\n]{0,40}\b(?:tell|say|talk|share)\b",
    re.IGNORECASE,
)


def asks_for_remarks(request: object) -> bool:
    """Whether ``request`` asks her to say what she thinks while it plays."""
    return bool(_AS_IT_GOES.search(str(request or "")))
