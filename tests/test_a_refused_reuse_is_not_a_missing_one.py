"""A cache that refuses to trim is a refusal, not an absence.

The prompt cache's miss line reports how far the prompt matched before it
diverged. Reading "matched 5202 (54.7%)" a person concludes the trie found
nothing usable. What can actually have happened is that it found the entry
holding those 5,202 tokens and mlx_lm declined to trim it, so the prefix was
re-read in full with nothing anywhere saying why.

LIVE, 2026-09-07: every turn of a five-step tool loop, each matching more of
the last than the one before — 17%, 35%, 47%, 55% — and each prefilling
everything. The loop exhausted its 178.8-second turn budget and the person got
an unfinished answer. Time to first token on the last turn was 83 seconds.
"""

from __future__ import annotations

import logging

import pytest

from core.brain.llm.prompt_cache import PromptCacheLRU


class _Cache(list):
    """Stands in for an mlx_lm KV cache."""


def _lru() -> PromptCacheLRU:
    return PromptCacheLRU(max_size=8)


def test_a_prefix_that_cannot_be_trimmed_says_so(caplog) -> None:
    lru = _lru()
    stored = [1, 2, 3, 4, 5, 6]
    lru.insert_cache("model", stored, _Cache([object()]))

    asked = [1, 2, 3, 9, 9, 9]
    with caplog.at_level(logging.INFO):
        cache, remainder = lru.fetch_nearest_cache(
            "model",
            asked,
            can_trim_prompt_cache=lambda _c: False,
            trim_prompt_cache=lambda _c, _n: None,
        )
    assert cache is None
    assert remainder == asked
    said = " ".join(record.getMessage() for record in caplog.records)
    assert "refuses to trim" in said, said


def test_a_trimmable_prefix_is_reused(caplog) -> None:
    lru = _lru()
    stored = [1, 2, 3, 4, 5, 6]
    held = _Cache([object()])
    lru.insert_cache("model", stored, held)

    trimmed: list[int] = []
    asked = [1, 2, 3, 9, 9, 9]
    with caplog.at_level(logging.INFO):
        cache, remainder = lru.fetch_nearest_cache(
            "model",
            asked,
            can_trim_prompt_cache=lambda _c: True,
            trim_prompt_cache=lambda _c, n: trimmed.append(n),
        )
    assert cache is not None
    assert remainder == [9, 9, 9]
    assert trimmed == [len(stored) - 3]
    assert "trimmed hit" in " ".join(r.getMessage() for r in caplog.records)


def test_a_real_absence_does_not_claim_a_refusal(caplog) -> None:
    """Nothing stored is a different fact and must read as one."""

    lru = _lru()
    with caplog.at_level(logging.INFO):
        cache, remainder = lru.fetch_nearest_cache(
            "model",
            [1, 2, 3],
            can_trim_prompt_cache=lambda _c: False,
            trim_prompt_cache=lambda _c, _n: None,
        )
    assert cache is None
    assert remainder == [1, 2, 3]
    assert "refuses to trim" not in " ".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("shared", [1, 3, 5])
def test_the_matched_length_is_still_reported(caplog, shared: int) -> None:
    lru = _lru()
    lru.insert_cache("model", [1, 2, 3, 4, 5, 6], _Cache([object()]))
    asked = [1, 2, 3, 4, 5, 6][:shared] + [7] * 4
    with caplog.at_level(logging.INFO):
        lru.fetch_nearest_cache(
            "model",
            asked,
            can_trim_prompt_cache=lambda _c: False,
            trim_prompt_cache=lambda _c, _n: None,
        )
    said = " ".join(record.getMessage() for record in caplog.records)
    assert f"matched {shared} " in said, said
