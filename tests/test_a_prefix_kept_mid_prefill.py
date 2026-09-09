"""The trie had the tokens and could not give them back.

On this model only a strict PREFIX can be reused. `can_trim_prompt_cache` is
`all(c.is_trimmable())`, the resident checkpoint is `Qwen3_5`, and
`ArraysCache.is_trimmable` is a bare `return False` — so the exact-hit and
trimmed-hit paths are dead, permanently.

The entry the previous turn stored is always LONGER than the match, because it
holds that turn's volatile block and the reply generated after it. LIVE,
2026-09-08, consecutive turns of one conversation:

    matched 571 (67.5%) ... the entry holding them refuses to trim (ArraysCache)
    matched 598 (92.7%) ... the entry holding them refuses to trim (ArraysCache)

Ninety-three per cent of the prompt reusable, and re-read in full.

The prefill happening right now is the one moment the KV for a shorter prefix
exists. Taking a copy of it on the way past turns the next turn's miss into a
prefix hit, which needs no trim.
"""

from __future__ import annotations

import copy

from core.brain.llm.prompt_cache import PromptCacheLRU

_KEY = ("a-model", "user_surface")


def _never_trims(_cache) -> bool:
    return False


def _refuse_to_trim(_cache, _n) -> None:
    raise AssertionError("this model cannot trim; nothing may call this")


def _fetch(lru: PromptCacheLRU, tokens: list[int]):
    return lru.fetch_nearest_cache(
        _KEY,
        tokens,
        can_trim_prompt_cache=_never_trims,
        trim_prompt_cache=_refuse_to_trim,
    )


def test_the_untrimmable_entry_is_a_miss_and_the_divergence_is_recorded():
    lru = PromptCacheLRU(max_size=8)
    lru.insert_cache(_KEY, [1, 2, 3, 4, 5], ["KV-for-five"])
    cache, remaining = _fetch(lru, [1, 2, 3, 9, 9])
    assert cache is None
    assert remaining == [1, 2, 3, 9, 9]
    assert lru.where_it_last_diverged(_KEY) == 3


def test_a_snapshot_at_the_divergence_makes_the_next_turn_a_prefix_hit():
    lru = PromptCacheLRU(max_size=8)
    lru.insert_cache(_KEY, [1, 2, 3, 4, 5], ["KV-for-five"])
    _fetch(lru, [1, 2, 3, 9, 9])
    assert lru.snapshot_prefix(_KEY, [1, 2, 3], ["KV-for-three"], deep_copy=copy.deepcopy)
    cache, remaining = _fetch(lru, [1, 2, 3, 9, 9])
    assert cache == ["KV-for-three"]
    assert remaining == [9, 9], "only the divergent tail is left to prefill"


def test_the_snapshot_is_a_copy_because_generation_keeps_writing():
    """Storing the live object stores whatever it has become by the time
    anybody reads it, which is the whole prompt."""
    lru = PromptCacheLRU(max_size=8)
    live = ["KV-so-far"]
    lru.snapshot_prefix(_KEY, [1, 2, 3], live, deep_copy=copy.deepcopy)
    live.append("KV-for-the-rest")
    cache, _ = _fetch(lru, [1, 2, 3, 7])
    assert cache == ["KV-so-far"]


def test_nothing_is_kept_when_there_is_nothing_worth_keeping():
    from core.brain.llm.mlx_worker import _a_prefix_worth_keeping

    lru = PromptCacheLRU(max_size=8)
    tokens = list(range(500))
    # Nothing searched yet: no measured divergence, so no snapshot.
    assert _a_prefix_worth_keeping(lru, _KEY, tokens, 0, ["KV"], 64) == (0, None)
    # No cache object.
    assert _a_prefix_worth_keeping(lru, _KEY, tokens, 0, None, 64) == (0, None)
    # No LRU at all.
    assert _a_prefix_worth_keeping(None, _KEY, tokens, 0, ["KV"], 64) == (0, None)


def test_the_point_is_a_chunk_boundary_short_of_the_divergence():
    from core.brain.llm.mlx_worker import _a_prefix_worth_keeping

    lru = PromptCacheLRU(max_size=8)
    lru.insert_cache(_KEY, list(range(500)), ["KV"])
    _fetch(lru, list(range(300)) + [9999] * 50)
    assert lru.where_it_last_diverged(_KEY) == 300
    at, keep = _a_prefix_worth_keeping(lru, _KEY, list(range(400)), 0, ["KV"], 64)
    # 300 rounded down to a 64-token boundary: the callback only reports there.
    assert at == 256
    assert keep is not None


def test_a_divergence_inside_the_first_chunk_is_not_worth_a_copy():
    from core.brain.llm.mlx_worker import _a_prefix_worth_keeping

    lru = PromptCacheLRU(max_size=8)
    lru.insert_cache(_KEY, list(range(500)), ["KV"])
    _fetch(lru, [0, 1, 2] + [9999] * 50)
    assert lru.where_it_last_diverged(_KEY) == 3
    assert _a_prefix_worth_keeping(lru, _KEY, list(range(400)), 0, ["KV"], 64) == (0, None)


def test_the_callback_takes_the_snapshot_once_and_only_past_the_point():
    from core.brain.llm.mlx_worker import _build_prefill_progress_callback

    class _Watchdog:
        def activity(self):
            return None

    class _Writer:
        def __init__(self):
            self.sent = []

        def put(self, item):
            self.sent.append(item)

    kept: list[int] = []
    report = _build_prefill_progress_callback(
        _Watchdog(),
        _Writer(),
        request_id="r",
        action="generate",
        snapshot_at=128,
        keep_prefix=kept.append,
    )
    report(64, 400)
    assert kept == [], "not yet past the point"
    report(128, 400)
    report(192, 400)
    report(256, 400)
    assert kept == [128], "once, at the first boundary past the point"


def test_progress_still_reports_when_nothing_is_being_kept():
    from core.brain.llm.mlx_worker import _build_prefill_progress_callback

    class _Watchdog:
        def __init__(self):
            self.beats = 0

        def activity(self):
            self.beats += 1

    class _Writer:
        def __init__(self):
            self.sent = []

        def put(self, item):
            self.sent.append(item)

    watchdog, writer = _Watchdog(), _Writer()
    report = _build_prefill_progress_callback(
        watchdog, writer, request_id="r", action="generate"
    )
    report(64, 400)
    assert watchdog.beats == 1
    assert writer.sent[0]["prompt_tokens_processed"] == 64
