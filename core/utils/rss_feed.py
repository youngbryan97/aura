"""Small RSS/Atom fallback parser for live autonomy.

The world-watcher should keep working even when the optional ``feedparser``
package is absent from a local runtime.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


def parse_feed_url(url: str, *, timeout: float = 12.0) -> Any:
    """Parse a feed URL with feedparser when available, otherwise stdlib XML."""
    try:
        import feedparser  # type: ignore

        return feedparser.parse(url)
    except ImportError:
        req = Request(url, headers={"User-Agent": "AuraWorldWatcher/1.0"})
        with urlopen(req, timeout=timeout) as response:
            data = response.read(1_000_000)
        return parse_feed_bytes(data)


def parse_feed_bytes(data: bytes) -> Any:
    root = ET.fromstring(data)
    channel = root.find("channel")
    if channel is not None:
        feed_title = _text(channel.find("title"))
        entries = []
        for item in channel.findall("item"):
            title = _text(item.find("title"))
            link = _text(item.find("link"))
            summary = _text(item.find("description"))
            guid = _text(item.find("guid")) or link or title
            if title:
                entries.append(SimpleNamespace(id=guid, title=title, link=link, summary=summary))
        return SimpleNamespace(entries=entries, feed={"title": feed_title})

    atom_ns = "{http://www.w3.org/2005/Atom}"
    feed_title = _text(root.find(f"{atom_ns}title"))
    entries = []
    for entry in root.findall(f"{atom_ns}entry"):
        title = _text(entry.find(f"{atom_ns}title"))
        summary = _text(entry.find(f"{atom_ns}summary")) or _text(entry.find(f"{atom_ns}content"))
        link_node = entry.find(f"{atom_ns}link")
        link = link_node.attrib.get("href", "") if link_node is not None else ""
        entry_id = _text(entry.find(f"{atom_ns}id")) or link or title
        if title:
            entries.append(SimpleNamespace(id=entry_id, title=title, link=link, summary=summary))
    return SimpleNamespace(entries=entries, feed={"title": feed_title})


def _text(node: ET.Element | None) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""
