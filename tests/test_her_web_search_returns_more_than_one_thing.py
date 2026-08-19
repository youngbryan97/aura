"""Every web search she ran returned one result: the encyclopedia entry.

LIVE 2026-08-19. Asked to find a 2048 game and play it, she searched,
considered exactly one destination, opened the Wikipedia article, and landed
somewhere with nothing to play. It looked like a judgement error and was not:
there was nothing else on the list to choose.

Organic results on DuckDuckGo Lite are redirects through the engine —
``//duckduckgo.com/l/?uddg=https%3A%2F%2Fplay2048.co%2F`` — and the reader
skipped anything containing "duckduckgo", so it discarded every result and
kept the one encyclopedia link that happened to be direct. Twice, titled with
its own hostname.

This is not about navigation. Every research question she has ever asked went
through here.
"""
from __future__ import annotations

from core.capabilities.browser_controller import _search_results_in, _unwrapped

PAGE = """
<a rel="nofollow" href="https://en.wikipedia.org/wiki/2048_(video_game)">2048 (video game)</a>
<a href="https://en.wikipedia.org/wiki/2048_(video_game)">More at <q>Wikipedia</q></a>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fplay2048.co%2F&amp;rut=abc">2048 &bull; Play the Free Online Game</a>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.2048.org%2F&amp;rut=def">2048 Game</a>
<a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js&amp;rut=x">Play 2048 Block Blast - ad</a>
<a href="https://duckduckgo.com/duckduckgo-help-pages/company/ads-by-microsoft/">more info</a>
<a href="//duckduckgo.com/lite/?q=2048+game&amp;s=20">Next Page</a>
"""


def test_a_result_link_is_unwrapped_to_where_it_goes():
    assert _unwrapped("//duckduckgo.com/l/?uddg=https%3A%2F%2Fplay2048.co%2F&rut=abc") == "https://play2048.co/"
    assert _unwrapped("https://en.wikipedia.org/wiki/2048") == "https://en.wikipedia.org/wiki/2048"
    assert _unwrapped("") == ""
    assert _unwrapped("/relative/only") == ""


def test_the_search_returns_the_results_and_not_just_the_direct_link():
    found = _search_results_in(PAGE, 8)
    urls = [row["url"] for row in found]
    assert "https://play2048.co/" in urls
    assert "https://www.2048.org/" in urls
    assert len(found) >= 3, "this is the defect: one result stood in for the whole page"


def test_every_result_carries_the_words_that_describe_it():
    """A decision between hostnames is not a decision."""
    found = _search_results_in(PAGE, 8)
    playable = next(row for row in found if row["url"] == "https://play2048.co/")
    assert playable["title"] == "2048 • Play the Free Online Game"
    assert all(row["title"] and row["title"] != row["url"] for row in found)


def test_the_engines_own_pages_are_not_results():
    urls = [row["url"] for row in _search_results_in(PAGE, 8)]
    assert not any("duckduckgo.com" in url for url in urls)
    assert not any("y.js" in url for url in urls)


def test_navigation_furniture_is_not_a_result():
    titles = [row["title"] for row in _search_results_in(PAGE, 8)]
    assert not any(title.lower().startswith(("more at", "more info", "next page")) for title in titles)


def test_the_same_destination_is_not_offered_twice():
    urls = [row["url"] for row in _search_results_in(PAGE, 8)]
    assert len(urls) == len(set(urls))


def test_the_count_is_respected():
    assert len(_search_results_in(PAGE, 2)) == 2


def test_an_empty_page_returns_nothing_rather_than_raising():
    assert _search_results_in("", 5) == []
    assert _search_results_in("<html><body>no links</body></html>", 5) == []


BING_PAGE = """
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=1&amp;u=a1aHR0cHM6Ly9wbGF5MjA0OC5jby8" >2048 &bull; Play the Free Online Game</a></h2></li>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=2&amp;u=a1aHR0cHM6Ly93d3cuMjA0OC5vcmcv">2048 Game</a></h2></li>
<li><h2><a href="https://www.bing.com/search?q=more">Related searches</a></h2></li>
"""


def test_a_throttled_provider_is_not_an_empty_internet():
    """LIVE: after a run of lookups the first source began answering 202 with
    an interstitial, and her entire ability to look anything up went with it.

    A challenge reads as "nothing matched" to a reader that only counts links.
    """
    from core.capabilities.browser_controller import _looks_like_a_challenge

    assert _looks_like_a_challenge(202, "<html>please wait</html>")
    assert _looks_like_a_challenge(429, "slow down")
    assert _looks_like_a_challenge(200, "<html>tiny</html>")
    assert not _looks_like_a_challenge(200, "<html>" + "result " * 500 + "</html>")


def test_there_is_more_than_one_place_to_look():
    """A single provider is a single point of failure for every question."""
    from core.capabilities.browser_controller import SEARCH_SOURCES

    assert len(SEARCH_SOURCES) >= 2
    hosts = {name for name, _template, _reader in SEARCH_SOURCES}
    assert len(hosts) >= 2


def test_a_wrapped_destination_is_unwrapped_to_where_it_goes():
    from core.capabilities.browser_controller import _bing_destination

    assert _bing_destination(
        "https://www.bing.com/ck/a?!&amp;&amp;p=1&amp;u=a1aHR0cHM6Ly9wbGF5MjA0OC5jby8"
    ) == "https://play2048.co/"
    assert _bing_destination("https://play2048.co/") == "https://play2048.co/"


def test_the_second_source_reads_results_and_titles_too():
    from core.capabilities.browser_controller import _bing_results_in

    found = _bing_results_in(BING_PAGE, 5)
    urls = [row["url"] for row in found]
    assert "https://play2048.co/" in urls
    assert "https://www.2048.org/" in urls
    assert not any("bing.com" in url for url in urls), "the engine's own pages are not results"
    playable = next(row for row in found if row["url"] == "https://play2048.co/")
    assert playable["title"] == "2048 • Play the Free Online Game"
