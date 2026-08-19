"""core/media/playback.py — turning "play something" into something playing.

A request to play media has three possible honest outcomes and this module's
whole job is to tell them apart:

  * **It is here.** A file on this machine matches. Nothing needs a network,
    nothing leaves the host, and it plays in the chat where it was asked for.
  * **It is not here, but the network is.** She can go and find it.
  * **It is not here and there is no network.** Then she says so — and this
    is the part that usually goes wrong.

That last case is where assistants reach for a canned line, and the canned
line is what makes them feel like appliances. "I'm unable to play music right
now" is written by a developer months in advance and read out by something
that, at that moment, knows *exactly* what happened: which folders it looked
in, how many files it searched, that the DNS probe has been failing for four
minutes, and that there are nine hundred other tracks it could play instead.
All of that is true, specific, and more useful than the apology.

So this module never composes a sentence for her. It resolves the request and
records what happened as facts, through ``failure_context``. She reads the
facts and says the thing herself — which is why the offline answer comes out
as "I can't get out to the network — that probe's been failing for a few
minutes — but you've got the whole Kind of Blue album locally, want that?"
rather than as the same sentence every time.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from core.conversation.failure_context import (
    record_capability_failure,
    record_offline_failure,
)
from core.media.library import MediaItem, get_media_library
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Media.Playback")

# "play some jazz", "put on the new Radiohead", "watch the wedding video".
# Matching the verb is how a play request is told apart from talking about
# music, which people do constantly and which must not start playback.
_PLAY_REQUEST = re.compile(
    r"\b(?:play|put on|start playing|queue up|listen to|watch)\b\s+(?P<what>.+)",
    re.IGNORECASE,
)

# Words that are about the request rather than part of the title. Applied
# repeatedly until the text stops changing, because they stack: "play me that
# song by Radiohead" has four of them in a row before the part that matters.
_STOPWORDS = re.compile(
    r"^\s*(?:me\s+|some\s+|the\s+|a\s+|an\s+|that\s+|my\s+|by\s+)+"
    # A leading medium word is a description of what is wanted, not its name:
    # "play some music by X" is a request for X. A *trailing* one is the same
    # ("play the Radiohead album"), so both ends are stripped.
    r"|^\s*(?:song|track|album|playlist|video|movie|film|music)\s+(?=\S)"
    r"|\b(?:please|for me|in here|in chat|out loud)\b"
    r"|\s+(?:song|track|album|video|movie|music)\s*$",
    re.IGNORECASE,
)

_VIDEO_HINT = re.compile(r"\b(?:video|movie|film|clip|watch)\b", re.IGNORECASE)
_AUDIO_HINT = re.compile(r"\b(?:song|track|album|music|listen|tune|playlist)\b", re.IGNORECASE)


@dataclass(slots=True)
class MediaResolution:
    """What playing this request would actually mean."""

    status: str  # "local" | "needs_network" | "offline" | "not_found" | "not_a_request"
    query: str = ""
    kind: str = ""
    item: MediaItem | None = None
    alternatives: tuple[MediaItem, ...] = ()
    searched: str = ""
    detail: str = ""

    @property
    def playable(self) -> bool:
        return self.status == "local" and self.item is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "query": self.query,
            "kind": self.kind,
            "item": self.item.to_dict() if self.item is not None else None,
            "alternatives": [alt.to_dict() for alt in self.alternatives],
            "searched": self.searched,
            "detail": self.detail,
        }


#: Words that point at something already under discussion rather than naming
#: anything. A request whose whole object is one of these is not answerable
#: from a media library.
_DEICTIC = frozenset(
    {"it", "this", "that", "these", "those", "them", "one", "the one", "something", "anything"}
)


def parse_play_request(message: str) -> tuple[str, str]:
    """Extract (what, kind) from a spoken or typed request, or ("", "").

    Returns an empty query for anything that is not a request to play — the
    common case, and the one where getting it wrong means starting music
    because somebody mentioned a band.
    """
    text = str(message or "").strip()
    if not text:
        return "", ""
    match = _PLAY_REQUEST.search(text)
    if not match:
        return "", ""

    kind = ""
    if _VIDEO_HINT.search(text):
        kind = "video"
    elif _AUDIO_HINT.search(text):
        kind = "audio"

    what = match.group("what").strip()
    # A title does not run past the end of a sentence.
    #
    # LIVE 2026-08-19: "2048 is open in Chrome. Play it — keep going until you
    # get a 128 tile. Tell me what you are doing as you go" was parsed as a
    # seventy-eight character query spanning two sentences, and a video in
    # Downloads started playing in the chat. Anything after the full stop is
    # the next thing being said, not more of the name.
    what = re.split(r"(?<=[.!?])\s+|\s+[—–]\s+|\n", what)[0]
    # Strip trailing clauses: "play Kind of Blue and turn the lights down".
    what = re.split(r"\b(?:and then|and also|,? then\b|and turn|and set)\b", what)[0]
    previous = None
    while previous != what:
        previous = what
        what = _STOPWORDS.sub("", what).strip()
    what = what.strip(" .!?\"'")
    # A pronoun is not a title.
    #
    # "Play it" is a request whose object is whatever was being discussed —
    # a game, a page, a video someone linked. Searching a media library for
    # "it" cannot be right, and answering it with a file is how a video
    # nobody asked for ends up on screen.
    if what.lower() in _DEICTIC:
        return "", ""
    return what, kind


def resolve_play_request(message: str) -> MediaResolution:
    """Decide what to do about a play request, and record why.

    Never raises for an ordinary miss: a request that cannot be satisfied is a
    normal outcome with facts attached, not an exception. The only failure
    reported as a degradation is the library itself misbehaving.
    """
    query, kind = parse_play_request(message)
    if not query:
        return MediaResolution(status="not_a_request")

    library = get_media_library()
    try:
        scan = library.index()
        matches = library.search(query, kind=kind, limit=6)
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
        record_degradation(
            "media.playback",
            exc,
            action="could not search the local media library for this request",
        )
        record_capability_failure(
            "media_playback",
            intent=f"play {query!r} from the local library",
            cause="failed",
            detail=f"the media index errored: {type(exc).__name__}",
        )
        return MediaResolution(status="not_found", query=query, kind=kind, detail=str(exc))

    searched = scan.narrative()

    if matches:
        best = matches[0]
        logger.info("play %r -> %s (%s)", query, best.title, best.kind)
        if not best.broadly_playable:
            # Findable but possibly not decodable in this browser. Say so up
            # front rather than showing a player that silently does nothing.
            record_capability_failure(
                "media_playback",
                intent=f"play {best.title!r} in the chat",
                cause="unavailable",
                detail=(
                    f"{best.extension} plays only in some browsers; the file is here "
                    "and the player may refuse it"
                ),
                still_possible=("open it in the system player instead",),
            )
        return MediaResolution(
            status="local",
            query=query,
            kind=best.kind,
            item=best,
            alternatives=tuple(matches[1:4]),
            searched=searched,
        )

    # Nothing local. Whether that is the end of it depends on the network.
    online = False
    try:
        from core.runtime.connectivity import get_connectivity_status

        online = bool(get_connectivity_status().online)
    except (RuntimeError, AttributeError, TypeError, ValueError, ImportError, OSError) as exc:
        record_degradation(
            "media.playback",
            exc,
            action="treated connectivity as unknown while resolving a play request",
            severity="debug",
        )

    if online:
        return MediaResolution(
            status="needs_network",
            query=query,
            kind=kind,
            searched=searched,
            detail="no local match; the network is up so it can be looked up",
        )

    # The honest offline case. Facts, not phrasing — including what she *can*
    # still do, so she does not over-generalise a bounded failure into "I
    # can't play anything".
    still_possible: list[str] = []
    if scan.items:
        sample = ", ".join(item.title for item in scan.items[:3])
        still_possible.append(
            f"{len(scan.items)} files are playable locally right now (e.g. {sample})"
        )
    record_offline_failure(
        "media_playback",
        intent=f"find and play {query!r}",
        still_possible=tuple(still_possible),
    )
    record_capability_failure(
        "media_playback",
        intent=f"find {query!r} in the local library",
        cause="empty_result",
        detail=f"searched {searched}; nothing matched",
        still_possible=tuple(still_possible),
    )
    return MediaResolution(
        status="offline",
        query=query,
        kind=kind,
        searched=searched,
        detail="no local match and no network path",
    )


__all__ = [
    "MediaResolution",
    "parse_play_request",
    "resolve_play_request",
]
