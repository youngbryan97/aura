"""Leaving a page that leads nowhere, the way a person does: an address, or the words looked up.

LIVE 2026-10-05 a museum's game page answered her browser with a block page.
She said the way on was to find the game elsewhere, and every action she could
choose was a control on the page in front of her. A decision may now "go":
to an address it names, or to the web's answers to the words it gives.
"""
from __future__ import annotations

import re
import urllib.parse

__all__ = ["where_to_go"]


def where_to_go(value: str) -> str:
    """An address as given, or else the web's answers to the words, as a person types either into the address bar."""
    said = " ".join(str(value or "").split())
    if re.match(r"^(https?://|www\.)\S+$", said, re.IGNORECASE):
        return said if said.lower().startswith("http") else f"https://{said}"
    if re.match(r"^[a-z0-9-]+(\.[a-z0-9-]+)+(/\S*)?$", said, re.IGNORECASE):
        return f"https://{said}"
    return f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(said)}"
