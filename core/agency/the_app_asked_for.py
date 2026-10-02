"""Which app a request asks for, when it asks for one.

The fallback planner opens an app when a goal names one. It used to decide
that with three substring tests: any goal with "notes" in it opened Notes,
"terminal" opened Terminal and "browser" a browser — "take notes on the test",
"footnotes", "denotes", "the browser tab I left open". Bryan, 1 Oct 2026: "when
we run computer skills, for some reason my notes app always opens? and
sometimes a browser".

An app is asked for when its name follows a word that asks for opening it or
working in it, and it is an app on this machine.
"""
from __future__ import annotations

import re

__all__ = ["the_app_asked_for"]

#: "open the Notes app", "launch Terminal app".
_NAMED_AS_AN_APP = (
    r"\bopen(?:\s+up)?\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 ._-]{1,48}?)\s+(?:app|application)\b",
    r"\blaunch\s+(?:the\s+)?([A-Za-z][A-Za-z0-9 ._-]{1,48}?)\s+(?:app|application)?\b",
)

#: A word asking for an app to be opened or worked in, and what follows it.
_ASKED_FOR = re.compile(
    r"\b(?:open|launch|start|bring\s+up|switch\s+to|in|into|using)\s+(?:up\s+)?"
    r"(?:the\s+|my\s+|a\s+|an\s+)?([A-Za-z][A-Za-z0-9 ._-]{0,48})",
    re.IGNORECASE,
)


def the_app_asked_for(goal: str) -> str:
    """The app ``goal`` asks to have opened or worked in, by its own name, or ""."""
    text = str(goal or "")
    for pattern in _NAMED_AS_AN_APP:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return " ".join(match.group(1).split()).strip(" .")
    for asked in _ASKED_FOR.finditer(text):
        words = asked.group(1).split()
        for count in range(min(3, len(words)), 0, -1):
            named = " ".join(words[:count]).strip(" .")
            if named.lower() in {"browser", "web browser"}:
                return "Browser"
            installed = an_installed_app_called(named)
            if installed:
                return installed
    return ""


def an_installed_app_called(named: str) -> str:
    """The installed app ``named`` names, by its own name, or ""."""
    from core.runtime.app_target_resolution import (
        _normal_name,
        installed_app_inventory,
    )

    wanted = _normal_name(named)
    if not wanted:
        return ""
    for app in installed_app_inventory():
        if _normal_name(app.name) == wanted:
            return app.name
    return ""
