"""A prefix that has served a hit is evicted after every entry that never has (a segmented LRU).

LIVE 7 November 2026, picking a video from a results page: the third of four browser calls reused a 3968-token prefix
of her system prompt, its own 8218-token prompt and reply went in after it, the byte budget evicted by age alone, and
the fourth call read all 7826 tokens again (26.8s).
"""

from core.brain.llm.prompt_cache import PromptCacheLRU

KEY = ("cortex", "default")


def _never_trims(_cache):
    return False


def _no_trim(_cache, _count):
    raise AssertionError("an untrimmable cache is never trimmed")


def _fetch(lru, tokens):
    return lru.fetch_nearest_cache(KEY, tokens, can_trim_prompt_cache=_never_trims, trim_prompt_cache=_no_trim)


def test_the_shared_prefix_survives_the_budget_the_finished_prompt_does_not():
    system = list(range(40))
    lru = PromptCacheLRU(max_size=12, max_total_tokens=110)
    lru.insert_cache(KEY, system, ["snapshot"])
    second = system + [100, 101, 102]
    cache, remaining = _fetch(lru, second)
    assert cache == ["snapshot"] and remaining == [100, 101, 102]
    lru.insert_cache(KEY, second + [103], ["second's prompt and reply"])
    third = system + list(range(200, 260))
    cache, remaining = _fetch(lru, third)
    assert remaining == list(range(200, 260))
    # The third call's prompt and reply, over budget beside the others: an entry nobody reused goes first.
    lru.insert_cache(KEY, third + [261], ["third's prompt and reply"])
    assert lru._holds(KEY, system)
    fourth = system + list(range(300, 320))
    cache, remaining = _fetch(lru, fourth)
    assert cache == ["snapshot"] and remaining == list(range(300, 320))


def test_with_nothing_reused_the_oldest_goes_first():
    lru = PromptCacheLRU(max_size=2)
    lru.insert_cache(KEY, [1, 2], ["a"])
    lru.insert_cache(KEY, [3, 4], ["b"])
    lru.insert_cache(KEY, [5, 6], ["c"])
    lru.insert_cache(KEY, [7, 8], ["d"])
    assert not lru._holds(KEY, [1, 2])
    assert lru._holds(KEY, [7, 8])


def test_a_reused_entry_is_kept_when_the_lane_is_full():
    lru = PromptCacheLRU(max_size=12)
    budget = lru._lane_budget("default")
    lru.insert_cache(KEY, [1, 2, 3], ["prefix"])
    _fetch(lru, [1, 2, 3, 9])
    for n in range(budget):
        lru.insert_cache(KEY, [50 + n, 60 + n], [f"finished {n}"])
    assert lru._holds(KEY, [1, 2, 3])
    assert not lru._holds(KEY, [50, 60])


def test_clearing_forgets_what_was_reused():
    lru = PromptCacheLRU(max_size=12)
    lru.insert_cache(KEY, [1, 2, 3], ["prefix"])
    _fetch(lru, [1, 2, 3, 9])
    assert lru._reused
    lru.clear()
    assert not lru._reused
