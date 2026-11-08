"""Entries that once served a hit keep only their share of the cache, and a shared prefix outlives its extensions.

LIVE 8 November 2026, dress rehearsal 8, picking an episode: 24,480 tokens of prefixes reused in the steps before held
the budget, so each new prompt was the only entry the budget could evict, and it was evicted on the call that stored it
("budget drained 9242 tokens (33722 -> 24480)"). The 7,936-token prefix of her system prompt kept mid-prefill went the
same way, and four page calls in a row each read all of their 8,972 to 11,952 tokens again: 115 s of prefill.
"""

from core.brain.llm.prompt_cache import PromptCacheLRU

PAGES = ("cortex", "default")
TALK = ("cortex", "user_surface")


def _never_trims(_cache):
    return False


def _no_trim(_cache, _count):
    raise AssertionError("an untrimmable cache is never trimmed")


def _fetch(lru, key, tokens):
    return lru.fetch_nearest_cache(key, tokens, can_trim_prompt_cache=_never_trims, trim_prompt_cache=_no_trim)


def test_old_reused_entries_past_their_share_go_before_the_new_one():
    lru = PromptCacheLRU(max_size=12, max_total_tokens=100)
    first, second = list(range(30)), list(range(100, 130))
    for prefix in (first, second):
        lru.insert_cache(PAGES, prefix, [f"prefix {prefix[0]}"])
        _fetch(lru, PAGES, prefix + [999])
    assert lru._reused_hold_too_much() is True
    new = list(range(500, 550))
    lru.insert_cache(PAGES, new, ["the new prompt"])
    assert lru._holds(PAGES, new)
    assert not lru._holds(PAGES, first) and lru._holds(PAGES, second)


def test_reused_entries_within_their_share_are_still_kept():
    lru = PromptCacheLRU(max_size=12, max_total_tokens=100)
    shared = list(range(30))
    lru.insert_cache(PAGES, shared, ["the shared prefix"])
    _fetch(lru, PAGES, shared + [999])
    lru.insert_cache(PAGES, list(range(200, 240)), ["one finished prompt"])
    lru.insert_cache(PAGES, list(range(300, 340)), ["another"])
    assert lru._holds(PAGES, shared)
    assert not lru._holds(PAGES, list(range(200, 240)))


def test_a_finished_prompt_goes_before_the_prefix_it_extends():
    lru = PromptCacheLRU(max_size=12, max_total_tokens=90)
    system = list(range(40))
    lru.insert_cache(PAGES, system, ["kept mid-prefill"])
    lru.insert_cache(PAGES, system + list(range(1000, 1030)), ["its prompt and reply"])
    assert lru._holds(PAGES, system)
    cache, remaining = _fetch(lru, PAGES, system + list(range(2000, 2020)))
    assert cache == ["kept mid-prefill"] and remaining == list(range(2000, 2020))


def test_the_conversation_keeps_the_turn_its_next_turn_extends():
    lru = PromptCacheLRU(max_size=12, max_total_tokens=90)
    system = list(range(40))
    turn = system + list(range(1000, 1030))
    lru.insert_cache(TALK, system, ["kept mid-prefill"])
    lru.insert_cache(TALK, turn, ["the last turn"])
    assert lru._holds(TALK, turn) and not lru._holds(TALK, system)
