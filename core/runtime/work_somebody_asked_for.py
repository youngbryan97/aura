"""Whether work somebody asked her for is under way: what she does for herself waits for it.

A build takes her model and her machine for minutes. LIVE 24 November 2026 a game she had been asked for was being written
while she practised another game between turns in the same process, and the writing fell from 26 tokens a second to 8.
"""
from __future__ import annotations

__all__ = ["begun", "ended", "under_way"]

_UNDER_WAY: set[str] = set()


def begun(work_id: str) -> None:
    _UNDER_WAY.add(str(work_id))


def ended(work_id: str) -> None:
    _UNDER_WAY.discard(str(work_id))


def under_way() -> bool:
    return bool(_UNDER_WAY)
