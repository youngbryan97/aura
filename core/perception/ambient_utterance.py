"""Her turn at the desk, and the newest thing her screen sense read.

Whether she says something unprompted about what is on the screen is decided by the company she keeps with the
person (core/perception/ambient_presence.py, ``_consider_speaking``; core/together/remarks.py): how much what is on
screen interests her, read off her own record, and the with's expected utility of saying something now, learned from
whether her remarks are taken up. With her window up it was decided here, by a list of fault words ("traceback",
"error:", "failed:"), so a form, a draft or a conversation on the screen never reached her, and on 17 November 2026
"failed:" twice in her own window, her own earlier replies, set off a remark that described her own window. The list
is gone; what is written here is what her turn at the desk is given.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.Perception.AmbientUtterance")

#: The least time she has to decide whether to say something unprompted; a longer prompt gets what it needs.
#: Past this the screen has moved on and the remark would be about a window the person already left.
_COMPOSE_TIMEOUT_S = 25.0


async def compose_at_the_desk(observation: Any) -> str:
    """Her turn while she keeps a person company at the desk: the screen as a reading, and her desk tools offered.

    Whether to point at something there, and at what, is her call in that turn (core/brain/her_unprompted_turn.py,
    her_turn_at_the_desk). What she says is her words after it.
    """
    from core.brain.her_unprompted_turn import her_turn_at_the_desk
    from core.together.remarks import SCREEN_CHARS

    capture = str(getattr(observation, "capture", "") or "")
    source = str(getattr(observation, "source", "") or "")
    screen = "\n".join(part for part in (source, capture[:SCREEN_CHARS]) if part)
    return await her_turn_at_the_desk("WHAT IS ON THE SCREEN", screen, least_s=_COMPOSE_TIMEOUT_S)


def _latest_observation() -> Any:
    try:
        from core.perception.observation_evidence import (
            ObservationKind,
            get_observation_memory,
        )

        return get_observation_memory().latest(ObservationKind.SCREEN_TEXT)
    # not a failure: no observation memory means there is no latest screen text.
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return None


def latest_screen() -> Any:
    """The newest screen reading her screen sense took, or None."""
    return _latest_observation()


__all__ = ["compose_at_the_desk", "latest_screen"]
