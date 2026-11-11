"""Her page: the one owner of what is done to a browser page.

What she did on a page went through 37 modules calling Playwright directly, 153 calls the effect-ownership gate
counted as ungoverned (Q06, 10 November 2026). They come here, and three things hold for every one of them.

A script run in a page is one its module declared, as written in the module (``page_script``): its argument is a
string literal, or the call stands where the module or a class is defined and reads only the module's own text.
tests/test_her_page_owner.py reads every call and fails on any other. Anything the script needs from the moment goes
in as an argument, never into its text, so nothing she read on a page and nothing a model wrote can become a script
she runs. ``SCRIPT % values`` or an f-string gives a plain string, which ``read`` refuses.

An address she goes to is one of the kinds she browses: a web page, a file of her own, or a blank page. A
``javascript:`` address is a script by another name and is refused.

Keys, clicks, typing and filling are made on a page she opened, each named by the source that asked for it, and
counted (``counts``).

Every method returns what the page gives back, so an async page is awaited by its caller as before and a stand-in
for a page in a test is driven as before.
"""
from __future__ import annotations

import sys
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

__all__ = ["PageGateway", "PageRefused", "PageScript", "get_page_gateway", "page_script"]

_DECLARING = object()

#: What an address may be: a web page, a file of hers, or a blank page.
_SCHEMES = frozenset({"http", "https", "file", "about"})


class PageRefused(PermissionError):
    """A script or an address her page owner does not run."""


class PageScript(str):
    """A script for a page, declared at import by the module that runs it. It is a string, so a page takes it as it
    took the text, and any operation on it gives back a plain string, which ``read`` refuses."""

    owner: str

    def __new__(cls, source: str, *, owner: str, _made_by: object = None) -> PageScript:
        if _made_by is not _DECLARING:
            raise TypeError("a page script is declared with page_script(), at import")
        made = super().__new__(cls, source)
        made.owner = owner
        return made


def page_script(source: str) -> PageScript:
    """Declare a script for a page: a literal, or the module's own text where the module or a class is defined."""
    if type(source) is not str or not source.strip():
        raise PageRefused("a page script is the module's own text")
    owner = str(sys._getframe(1).f_globals.get("__name__", "?"))
    return PageScript(source, owner=owner, _made_by=_DECLARING)


class PageGateway:
    """The single canonical owner of browser page effects."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str]] = Counter()

    def counts(self) -> dict[str, int]:
        """How many of each effect each source has made, as ``"effect:source"``."""
        return {f"{effect}:{source}": n for (effect, source), n in sorted(self._counts.items())}

    def read(self, target: Any, script: PageScript, *args: Any, source: str = "") -> Any:
        """Run a declared script in a page, or on an element of one, and give back what it returns."""
        if not isinstance(script, PageScript):
            raise PageRefused(
                "a page runs only a script its module declared with page_script(); values go in as arguments")
        self._counts["read", source or script.owner] += 1
        return target.evaluate(script, *args)

    def go_to(self, page: Any, address: str, *, source: str, **options: Any) -> Any:
        """Go to an address of a kind she browses."""
        scheme = urlsplit(str(address)).scheme.lower()
        if scheme not in _SCHEMES:
            raise PageRefused(f"{source} asked her page to go to a {scheme or 'schemeless'}: address")
        self._counts["go_to", source] += 1
        return page.goto(address, **options)

    def new_page(self, opener: Any, *, source: str, **options: Any) -> Any:
        """Open a page in a browser or a context she started."""
        self._counts["new_page", source] += 1
        return opener.new_page(**options)

    def press(self, target: Any, *args: Any, source: str, **options: Any) -> Any:
        """Press a key on a page, an element, or a page's keyboard."""
        self._counts["press", source] += 1
        return target.press(*args, **options)

    def type_text(self, target: Any, *args: Any, source: str, **options: Any) -> Any:
        """Type into a page, an element, or a page's keyboard."""
        self._counts["type", source] += 1
        return target.type(*args, **options)

    def click(self, target: Any, *args: Any, source: str, **options: Any) -> Any:
        """Click a page's element or a locator."""
        self._counts["click", source] += 1
        return target.click(*args, **options)

    def fill(self, target: Any, *args: Any, source: str, **options: Any) -> Any:
        """Fill a field."""
        self._counts["fill", source] += 1
        return target.fill(*args, **options)


_GATEWAY = PageGateway()


def get_page_gateway() -> PageGateway:
    return _GATEWAY
