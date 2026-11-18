"""What a person is shown when no reply of hers formed.

For months the runtime filled that gap in her first person: "I couldn't get
to an answer I'd stand behind on that one", "Ugh, the local answer path
failed before producing a coherent reply. I'm restarting it", "I hit an
error before a coherent answer formed". Each went out as her reply, her
voice read it aloud, and it was kept in her conversation as something she
had said, so her next turn read it there as her own words. Bryan's rule of
13 November is that the runtime's text never stands in her voice.

A turn that ends with no reply of hers now carries an empty reply and a
notice. The window shows the notice as the runtime's, her voice says
nothing, and the exchange keeps the person's words with no reply of hers.
What the runtime held for the message, such as a computed result or what a
tool returned, goes in the notice under its own heading.
"""

from __future__ import annotations

from typing import Any

#: The notice when nothing more specific is known.
NO_REPLY_FORMED = "No reply formed for this message. Send it again to retry."

#: Who the notice is from, in the payload's ``said_by``.
RUNTIME = "runtime"


def notice_with_what_was_held(why: str, held: str = "") -> str:
    """The notice: why no reply formed, then whatever the runtime held for the message."""

    said = " ".join(str(why or "").split()) or NO_REPLY_FORMED
    found = str(held or "").strip()
    if not found:
        return said
    return f"{said}\n\nWhat the runtime held for it:\n{found}"


def the_runtimes_notice(held: str = "") -> str:
    """The notice for a path that returns her text, declared as the runtime's for every gate downstream."""

    notice = notice_with_what_was_held(NO_REPLY_FORMED, held)
    try:
        from core.conversation.reply_provenance import ReplyProvenance, declare_provenance

        declare_provenance(notice, ReplyProvenance.HONEST_FAILURE)
    # not a failure: the notice is still recognised by its own words at the
    # boundary; the declaration only spares the gates a second look.
    except ImportError:
        pass
    return notice


def no_reply_fields(why: str, *, held: str = "") -> dict[str, str]:
    """The payload fields of a turn that ends with no reply of hers."""

    return {
        "response": "",
        "notice": notice_with_what_was_held(why, held),
        "said_by": RUNTIME,
        "response_confidence": "failed",
    }


def is_the_runtimes_notice(text: Any) -> bool:
    """Whether this text is the runtime's notice, wherever it travelled.

    The last-resort composer returns its notice down a path built to carry
    her replies; this is how the voice shaping and the delivery boundary know
    the text is not hers to shape or to send as hers.
    """
    return str(text or "").strip().startswith(NO_REPLY_FORMED)


def is_no_reply(payload: Any) -> bool:
    """Whether a turn's payload ended with the runtime's notice and no reply of hers."""

    return (
        isinstance(payload, dict)
        and not str(payload.get("response") or "").strip()
        and bool(str(payload.get("notice") or "").strip())
    )


__all__ = [
    "NO_REPLY_FORMED",
    "RUNTIME",
    "is_no_reply",
    "is_the_runtimes_notice",
    "no_reply_fields",
    "notice_with_what_was_held",
    "the_runtimes_notice",
]
