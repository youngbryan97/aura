"""Where a request is carried out: in a program on this machine, or on a page the conversation was already working.

Both questions are read by meaning, with the inference model her reply check reads with
(core/epistemics/what_her_record_refutes.py), and never by which words a request holds.

LIVE 23 November 2026, "Open En Croissant and play a game of chess against Stockfish" went to gmail.com, then to
lichess.org, then to linkedin.com. A request that names no page was taken to go on with the latest earlier message that
held anything shaped like a domain, when both held a word from one list ("game" in a résumé, "play" in the request):
the résumé pasted into the conversation that day held an address at gmail.com and a link at linkedin.com. And the
program was looked for only as the machine spells it, "en-croissant", which "En Croissant" is not.

- A program: each installed program whose name appears in the request, in any spelling of it (case, hyphens,
  underscores, dots or spaces between its words), is a candidate, and the reader is asked whether the request names an
  application by that name. "a game of chess" holds the word and names no program; "Open Chess" names one. Measured 23
  November 2026 on twelve requests: twelve read right.
- A page the conversation was working: the latest earlier message that names a place is a candidate, and the reader is
  asked whether what the new request asks for is done there. Measured on eleven pairs: eleven read right, among them the
  résumé with every request after it ("linkedin.com", "gmail.com", "github.com": not there), and "Now Salad Fingers.
  Pick whichever episode you want." after Allen Pan's channel on YouTube (there).

Without the reader a program is named only as written as a name (its spelling, or each word capitalized), and no page is
taken to be continued: a request is never sent to a page on a guess.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger("Aura.WhereARequestIsDone")

__all__ = ["NAMES_A_PROGRAM", "DONE_THERE", "continues_there", "installed_program_names", "program_it_names"]

#: What the reader is asked of a request and a program's name as the request writes it.
NAMES_A_PROGRAM = 'The request names an application called "{span}".'
#: What the reader is asked of an earlier request, a new one, and the site the earlier one named.
DONE_THERE = "What the second request asks for is done at {host}."

_SEPARATORS = re.compile(r"[\s\-_.]+")


def installed_program_names() -> list[str]:
    """The programs on this machine, by the names the machine gives them."""
    try:
        from core.container import ServiceContainer

        # Only a registry already running: reading where a request is done never starts a service.
        registry = ServiceContainer.peek("app_registry", default=None)
        if registry is not None and hasattr(registry, "all_apps"):
            names = [str(app.name) for app in registry.all_apps() if getattr(app, "name", "")]
            if names:
                return names
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug("the app registry could not be read: %s", exc)
    found: list[str] = []
    for folder in (Path("/Applications"), Path.home() / "Applications"):
        try:
            found += [entry.stem for entry in folder.iterdir() if entry.suffix == ".app"]
        except OSError:
            continue
    return found


def _spellings_of(name: str) -> re.Pattern[str] | None:
    words = [word for word in _SEPARATORS.split(str(name or "").strip()) if word]
    if not words or len("".join(words)) < 2:
        return None
    return re.compile(r"(?<![\w])" + r"[\s\-_.]*".join(re.escape(word) for word in words) + r"(?![\w])", re.IGNORECASE)


def _written_as_its_name(span: str, name: str) -> bool:
    return span == name or all(word[:1].isupper() for word in _SEPARATORS.split(span) if word)


def program_it_names(text: str, names: Iterable[str] | None = None) -> str:
    """The installed program ``text`` names, by the machine's name for it; "" when it names none."""
    said = str(text or "")
    if not said.strip():
        return ""
    candidates: list[tuple[str, str]] = []
    for name in sorted(set(installed_program_names() if names is None else names), key=len, reverse=True):
        spelled = _spellings_of(name)
        found = spelled.search(said) if spelled is not None else None
        if found:
            candidates.append((name, found.group(0)))
    if not candidates:
        return ""
    from core.epistemics.what_her_record_refutes import _reader

    read = _reader()
    for name, span in candidates:
        if read is None:
            if _written_as_its_name(span, name):
                return name
        elif read(said, NAMES_A_PROGRAM.format(span=span)) == "entailment":
            return name
    return ""


def _host(place: str) -> str:
    host = urlsplit(place if "://" in place else f"https://{place}").hostname or ""
    return host[4:] if host.startswith("www.") else host


def continues_there(earlier: str, request: str, place: str) -> bool:
    """Whether what ``request`` asks for is done at ``place``, which ``earlier`` named. False without the reader."""
    host = _host(str(place or ""))
    if not host or not str(request or "").strip():
        return False
    from core.epistemics.what_her_record_refutes import _reader

    read = _reader()
    if read is None:
        return False
    premise = f"First request: {' '.join(str(earlier or '').split())} Second request: {' '.join(str(request).split())}"
    there = read(premise, DONE_THERE.format(host=host)) == "entailment"
    logger.info("a request %s on with the conversation at %s: %r", "goes" if there else "does not go", host,
                str(request)[:120])
    return there
