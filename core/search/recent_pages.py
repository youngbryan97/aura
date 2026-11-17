"""A search held to the pages of the last few days, for a turn about the world as it is now.

LIVE 17 November 2026: "What do you make of the current Israel/Palestine situation and America's role in it? ... what
do you think of America's role in the conflict with Iran?" was searched as her whole question, with no time filter,
and came back with explainers from earlier in the year; Bryan wanted her informed of the situation as it is now. The
search engines offer a filter by age in steps (a day, a week, a month), and a search held to one is cached apart from
one that is not.
"""
from __future__ import annotations

__all__ = ["ddg_within", "ddgs_timelimit", "recency_key"]


def ddg_within(days: int) -> str:
    """DuckDuckGo's time filter for pages from the last ``days``: the smallest of its steps that covers them."""
    return "" if days <= 0 else "&df=d" if days <= 1 else "&df=w" if days <= 7 else "&df=m"


def ddgs_timelimit(days: int) -> dict[str, str]:
    """The same filter as the DDGS client takes it."""
    return {"timelimit": ddg_within(days)[4:]} if days > 0 else {}


def recency_key(query: str, days: int) -> str:
    """The cache key of a search held to recent pages, apart from the same words searched without the filter."""
    return f"{query} |within {days}d" if days > 0 else query
