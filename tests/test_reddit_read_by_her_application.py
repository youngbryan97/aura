"""F13: she reads Reddit by an application registered to her account, not by a browser a CAPTCHA stops."""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit

LISTING = {"data": {"children": [
    {"kind": "t3", "data": {"title": "Rules of the sub", "stickied": True, "permalink": "/r/x/comments/0/rules/"}},
    {"kind": "t3", "data": {"title": "Is free will compatible with physics?", "author": "kant_fan", "score": 412,
                            "num_comments": 88, "permalink": "/r/philosophy/comments/abc/is_free_will/", "selftext": "Asking."}},
]}}
THREAD = [
    {"data": {"children": [{"kind": "t3", "data": {"title": "Is free will compatible with physics?", "author": "kant_fan",
                                                   "score": 412, "num_comments": 2, "permalink": "/r/philosophy/comments/abc/x/",
                                                   "selftext": "Compatibilists say yes. Why?"}}]}},
    {"data": {"children": [
        {"kind": "t1", "data": {"author": "hume", "score": 30, "body": "Because freedom is acting on your own reasons.",
                                "replies": {"data": {"children": [{"kind": "t1", "data": {"author": "spinoza", "score": 4,
                                                                                         "body": "Reasons are causes too."}}]}}}},
        {"kind": "more", "data": {"count": 9}},
    ]}},
]


def test_a_listing_and_a_thread_read_as_a_person_reads_them():
    from core.skills.reddit_reading import posts_in, thread_in

    posts = posts_in(LISTING, 5)
    assert [p["title"] for p in posts] == ["Is free will compatible with physics?"]   # the pinned rules left out
    assert posts[0]["url"] == "https://www.reddit.com/r/philosophy/comments/abc/is_free_will/" and posts[0]["comments"] == "88"
    thread = thread_in(THREAD)
    assert thread.startswith("Is free will compatible with physics?") and "Compatibilists say yes." in thread
    assert "- u/hume (30 points): Because freedom" in thread and "  - u/spinoza (4 points): Reasons are causes too." in thread


def test_without_her_applications_credentials_her_browser_reads_as_before(monkeypatch):
    from core.skills import reddit_reading
    from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

    monkeypatch.setattr(reddit_reading, "_credentials", lambda: None)
    assert asyncio.run(RedditAdapterSkill._by_her_application(RedditInput(mode="browse", subreddit="philosophy"))) is None


def test_with_them_a_subreddit_and_a_thread_are_read_by_the_api_with_its_token(monkeypatch):
    from core.skills import reddit_reading
    from core.skills.reddit_adapter import RedditAdapterSkill, RedditInput

    asked = []

    async def request(method, url, *, headers, data=None):
        asked.append((method, url, headers.get("Authorization", "")[:6]))
        if url == reddit_reading.TOKEN_URL:
            assert data == "grant_type=client_credentials"
            return {"access_token": "t0k", "expires_in": 3600}
        return THREAD if "/comments/" in url else LISTING

    monkeypatch.setattr(reddit_reading, "_credentials", lambda: ("id", "secret"))
    monkeypatch.setattr(reddit_reading, "_request", request)
    monkeypatch.setattr(reddit_reading, "_token", {"value": "", "until": 0.0})
    browsed = asyncio.run(RedditAdapterSkill._by_her_application(RedditInput(mode="browse", subreddit="philosophy", limit=3)))
    assert browsed["ok"] and browsed["read_by"] == "her application" and browsed["count"] == 1
    read = asyncio.run(RedditAdapterSkill._by_her_application(
        RedditInput(mode="read_post", url="https://www.reddit.com/r/philosophy/comments/abc/x/")))
    assert read["ok"] and "Reasons are causes too." in read["content"]
    assert [a[0] for a in asked] == ["POST", "GET", "GET"]   # one token, kept for the second read
    assert asked[1][1].startswith("https://oauth.reddit.com/r/philosophy/hot?") and asked[1][2] == "bearer"
