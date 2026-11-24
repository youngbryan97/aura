"""Put playing a place on the same record as the other things she does."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.GamesOnHerRecord")


def began_play(page: Any) -> None:
    # She is playing it from here until she keeps what she learned of it, and
    # every turn read in between is a turn of the playing.
    try:
        from core.conation.what_she_wants import get_what_she_wants

        where = str(getattr(page, "url", "") or "").strip()
        if where:
            get_what_she_wants().started(f"game:{where}", "play", label=where, where=where)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the game was not marked as under way: %s", exc)


def ended_play(page: Any) -> None:
    # And the game is a thing she went to, on the one record her talk and her
    # idle hours are on, so what playing it was worth is hers everywhere.
    try:
        from core.conation.what_she_wants import get_what_she_wants

        where = str(getattr(page, "url", "") or "").strip()
        if where:
            record = get_what_she_wants()
            record.went(f"game:{where}", "play", label=where, where=where)
            record.stopped("play", f"game:{where}")
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the game did not reach her record of what she has met: %s", exc)


def _this_game(page: Any) -> str:
    """The name her memory keeps this game under: its page, and for a file, that file as it now is.

    A file changed is another game. LIVE 2026-10-06 a Pong she had just
    mended was played from what she had kept of it broken ("the white bars
    cost me"), because both were kept under one address.
    """
    url = str(getattr(page, "url", "") or "")
    if url.startswith("file://"):
        import hashlib
        from pathlib import Path
        from urllib.parse import unquote, urlparse

        try:
            held = Path(unquote(urlparse(url).path)).read_bytes()
            return f"played as it happens at {url} ({hashlib.sha256(held).hexdigest()[:12]})"
        # not a failure: a game not on disk is named without its hash
        except OSError:
            pass
    return f"played as it happens at {url}"


def _what_she_kept_of(page: Any) -> dict[str, Any]:
    from core.agency.what_she_keeps_of_a_game import INDEXED_TABLES, kept_from
    from core.runtime.what_she_learned import recall

    began_play(page)
    held = recall(_this_game(page), indexed_tables=INDEXED_TABLES)
    if held:
        logger.info("she has played this game before: starting from what she kept of it")
    return kept_from(held)


def _keep_what_she_learned(page: Any, keep: dict[str, Any]) -> None:
    from core.agency.what_she_keeps_of_a_game import to_keep
    from core.cognition.when_more_thought_changes_the_answer import LOOKS_AHEAD, settled
    from core.runtime.what_she_learned import remember

    remember(_this_game(page), to_keep(keep))
    # How often looking a level further changed what she would do, in her looks ahead while she played: the curve
    # her looks stop by is fitted again on them, and kept.
    settled(LOOKS_AHEAD)

    ended_play(page)

