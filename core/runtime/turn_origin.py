"""One answer to "is a person waiting on this turn".

The question was answered in eleven places. Between them they knew
forty-three origin names, and exactly two — "user" and "voice" — appeared in
all eleven; twenty-one were known to a single file. So the answer depended on
which layer happened to ask, and layers that disagree about whether somebody
is waiting make decisions that contradict each other.

Measured on 2026-08-29, that cost a turn: the conscience holds a skill whose
worst case looks harmful unless a person asked for it directly, and its own
list did not contain the names the desktop lane actually emits — a turn
arrives as "desktop_quick_user" and generates under
"response_generation_user". A request whose words were "use that library to
record this" was held at worst-case harm 0.80, twice, while every other layer
in the same turn treated it as user-facing.

Two things are wrong with deciding this from a name, and only one of them is
fixed by sharing a list.

A name has to be parsed, and every parser drifts. That is what a single
predicate here fixes.

The deeper one is that the turn already knows. Whether somebody is waiting is
a fact established where the turn begins, and passing the fact is better than
passing a label for other layers to interpret — so ``a_person_is_waiting``
takes a stated answer and believes it. The name is the fallback for callers
that do not carry the fact, and a label for logs.

Deliberately not the union of all eleven. Some of those sets answer different
questions — whether a turn is eligible for bonding, whether a surface is
auditable — and a name that means "foreground" to one is not an answer to
another. This is the one question, and the sets that were asking it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FOREGROUND_ORIGIN_PREFIXES",
    "USER_FACING_ORIGINS",
    "a_person_is_waiting",
    "normalise_origin",
]

#: Every name the layers that ask this question knew between them.
USER_FACING_ORIGINS: frozenset[str] = frozenset(
    {
        "admin",
        "api",
        "audit",
        "benchmark",
        "chat",
        "desktop",
        "desktop_chat",
        "desktop_task",
        "desktop_ui",
        "direct",
        "embodied",
        "embodied_motor_reflex",
        "embodied_sensory_feed",
        "external",
        "gui",
        "native_shell",
        "reflex",
        "simulate",
        "test",
        "user",
        "user_voice",
        "voice",
        "websocket",
        "ws",
    }
)

#: A lane names its entry points after itself, so a turn arriving as
#: "desktop_quick_user" is recognisable without listing every variant. A
#: lane's internal PHASE names are not — "response_generation_user" is the
#: same turn seen from inside, and no rule over the string says so without
#: guessing. That is the case the stated fact exists for, and inventing a
#: suffix rule to cover it would be the same drift in a new place.
FOREGROUND_ORIGIN_PREFIXES: tuple[str, ...] = (
    "api_",
    "chat_",
    "desktop_",
    "gui_",
    "native_",
    "owner_",
    "user_",
    "voice_",
    "websocket_",
    "ws_",
)


def normalise_origin(value: Any) -> str:
    """The origin as a comparable name.

    Hyphens and underscores are the same word: the lists carried both
    "desktop-ui" and "desktop_ui", and "native-shell" and "native_shell",
    because different callers spelled them differently and each list learned
    whichever spelling reached it first.
    """

    return str(value or "").strip().lower().replace("-", "_")


def a_person_is_waiting(origin: Any = "", *, stated: bool | None = None) -> bool:
    """Whether somebody is waiting on this turn.

    ``stated`` is the fact, from a caller that knows. It is believed in both
    directions: a caller saying nobody is waiting is believed over a
    foreground-looking name, because it knows something the name cannot say.

    Falling back to the name is for callers that do not carry the fact, and
    it is honest about its reach: it accepts the names these layers knew and
    the entry points a lane derives from its own. It does not try to recognise
    a lane's internal phase names, because nothing about the string
    "response_generation_user" says it belongs to a person's turn — the turn
    says that, and says it by passing ``stated``.
    """

    if isinstance(stated, bool):
        return stated
    name = normalise_origin(origin)
    if not name:
        return False
    if name in USER_FACING_ORIGINS:
        return True
    return name.startswith(FOREGROUND_ORIGIN_PREFIXES)
