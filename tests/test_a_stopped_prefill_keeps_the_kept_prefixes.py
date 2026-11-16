"""A prefill stopped partway keeps the prefixes already kept under its key.

LIVE 15 November 2026, a typed turn of 49.4 s: her tool loop's third pass was stopped at 1,984 of 2,284 prefill
tokens, the cancel handler cleared every entry under the key, the 7,424-token prefix kept 3 s before went with it,
and the reply's pass prefilled 2,564 tokens from nothing (7.9 s). The stopped pass's cache was its own: every way a
generation borrows one takes the entry out of the trie or hands over a copy. Only an entry that still holds that
very object is retired.
"""

from __future__ import annotations

import copy

import pytest

from core.brain.llm.prompt_cache import PromptCacheLRU

pytestmark = pytest.mark.unit

KEY = (1, "user_surface")


class _Cache(list):
    """A stand-in KV cache: the tokens it holds."""


def _never_trims(_cache) -> bool:
    return False


def _trim(_cache, _count) -> None:
    raise AssertionError("an untrimmable cache is never trimmed")


def test_a_borrowed_cache_is_out_of_the_trie_and_retiring_it_keeps_the_rest() -> None:
    lru = PromptCacheLRU(max_size=8)
    prefix = list(range(10))
    lru.insert_cache(KEY, prefix, _Cache(prefix))
    turn = prefix + [10, 11, 12]
    lru.insert_cache(KEY, turn, _Cache(turn))

    borrowed, remaining = lru.fetch_nearest_cache(
        KEY, turn + [13, 14], can_trim_prompt_cache=_never_trims, trim_prompt_cache=_trim
    )
    assert borrowed == turn
    assert remaining == [13, 14]
    borrowed.extend([13])  # the prefill ran partway before it was stopped

    assert lru.retire_entries_holding(KEY, borrowed) == 0
    again, _ = lru.fetch_nearest_cache(
        KEY, prefix + [99], can_trim_prompt_cache=_never_trims, trim_prompt_cache=_trim
    )
    assert again == prefix


def test_an_entry_that_still_holds_the_stopped_cache_is_the_only_one_retired() -> None:
    lru = PromptCacheLRU(max_size=8)
    kept = _Cache(range(10))
    lru.insert_cache(KEY, list(range(10)), kept)
    aliased = _Cache(range(20))
    lru.insert_cache(KEY, list(range(20)), aliased)
    lru.insert_cache((2, "user_surface"), list(range(20)), aliased)

    assert lru.retire_entries_holding(KEY, aliased) == 1
    assert lru._holds(KEY, list(range(10)))
    assert not lru._holds(KEY, list(range(20)))
    assert lru._holds((2, "user_surface"), list(range(20)))


def test_a_stopped_job_with_no_cache_retires_nothing() -> None:
    lru = PromptCacheLRU(max_size=8)
    lru.insert_cache(KEY, [1, 2, 3], _Cache([1, 2, 3]))
    snapshot = copy.deepcopy(lru._cache)

    assert lru.retire_entries_holding(KEY, None) == 0
    assert lru._cache == snapshot


def test_the_cancel_handler_retires_by_object_and_not_by_key() -> None:
    """The handler's call, read where it is made, since the worker loop cannot run without a model."""
    from pathlib import Path

    worker = (Path(__file__).resolve().parents[1] / "core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    handler = worker.split("except _PrefillCancelledError as cancelled:", 1)[1].split("ipc_writer.put(", 1)[0]
    assert "retire_entries_holding(model_key, locals().get(\"cache\"))" in handler
    assert "clear_model_key" not in handler
