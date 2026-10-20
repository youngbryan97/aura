"""Reading Reddit through its API, as an application registered to her account: no browser, no sign-in, no CAPTCHA.

LIVE to 20 October 2026: her browser found a subreddit's posts on 20 of 36 tries and was stopped by a CAPTCHA each time
it checked her sign-in; Reddit refuses scripted reads of its pages ("You've been blocked by network security"), and
old Reddit now asks for an account. A script application registered on her account (reddit.com/prefs/apps) reads what
anyone can read, by OAuth's application-only grant, within Reddit's API rules (a user agent naming the application, at
most 100 requests a minute). Its identifier and secret are in her Keychain (``AURA_REDDIT_CLIENT_ID``,
``AURA_REDDIT_CLIENT_SECRET``, under ``AuraAutonomyEngine``), entered by Bryan; without them nothing here runs.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.parse
from typing import Any

logger = logging.getLogger("Skills.Reddit")

__all__ = ["available", "posts_in", "read_subreddit", "read_thread", "thread_in"]

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
USER_AGENT = "macos:aura-reading:1.0 (her own reader)"
_token: dict[str, Any] = {"value": "", "until": 0.0}


def _credentials() -> tuple[str, str] | None:
    from core.security.zenith_secrets import get_credential

    client, secret = get_credential("reddit", "client_id"), get_credential("reddit", "client_secret")
    return (client, secret) if client and secret else None


def available() -> bool:
    """Whether her application's credentials are in her Keychain."""
    return _credentials() is not None


async def _request(method: str, url: str, *, headers: dict[str, str], data: str | None = None) -> Any:
    from core.runtime.network_gateway import get_network_gateway

    answer = await get_network_gateway().request_async(method, url, headers={"User-Agent": USER_AGENT, **headers},
                                                       data=data, timeout=20.0, source="skills.reddit_reading",
                                                       read_only=method == "GET")
    status = int(answer.get("status_code", 0) or 0)
    if not 200 <= status < 300:
        raise RuntimeError(f"reddit answered {status} for {urllib.parse.urlsplit(url).path}")
    content = answer.get("content", b"")
    return json.loads(content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content))


async def _bearer() -> str:
    if _token["value"] and time.time() < _token["until"]:
        return str(_token["value"])
    credentials = _credentials()
    if credentials is None:
        raise RuntimeError("her Reddit application's credentials are not in her Keychain")
    basic = base64.b64encode(f"{credentials[0]}:{credentials[1]}".encode()).decode()
    granted = await _request("POST", TOKEN_URL, headers={"Authorization": f"Basic {basic}",
                                                         "Content-Type": "application/x-www-form-urlencoded"},
                             data="grant_type=client_credentials")
    token = granted.get("access_token") if isinstance(granted, dict) else None
    if not token:
        raise RuntimeError(f"reddit granted no token: {str(granted)[:120]}")
    _token.update(value=str(token), until=time.time() + float(granted.get("expires_in", 3600)) - 60)
    return str(_token["value"])


def posts_in(listing: Any, limit: int) -> list[dict[str, str]]:
    """The posts of a listing, in its order, less those pinned to its top."""
    children = ((listing.get("data") or {}).get("children") or []) if isinstance(listing, dict) else []
    posts = []
    for child in children:
        post = (child.get("data") or {}) if isinstance(child, dict) else {}
        if not isinstance(child, dict) or child.get("kind") != "t3" or post.get("stickied"):
            continue
        posts.append({"title": str(post.get("title") or ""), "author": str(post.get("author") or ""),
                      "score": str(post.get("score") or 0), "comments": str(post.get("num_comments") or 0),
                      "url": "https://www.reddit.com" + str(post.get("permalink") or ""),
                      "text": str(post.get("selftext") or ""), "links_to": str(post.get("url") or "")})
        if len(posts) >= limit:
            break
    return posts


def thread_in(thread: Any, *, at_most: int = 15000) -> str:
    """A thread as a person reads it: the post, then its comments in Reddit's order, each with its replies under it."""
    if not isinstance(thread, list) or len(thread) < 2:
        return ""
    posts = posts_in(thread[0], 1)
    if not posts:
        return ""
    post = posts[0]
    lines = [post["title"], f"posted by u/{post['author']}, {post['score']} points, {post['comments']} comments"]
    if post["text"].strip():
        lines += ["", post["text"]]
    elif post["links_to"] and "reddit.com" not in post["links_to"]:
        lines += ["", f"links to {post['links_to']}"]
    lines += ["", "Comments:"]

    def comments(listing: Any, depth: int) -> None:
        for child in ((listing.get("data") or {}).get("children") or []) if isinstance(listing, dict) else []:
            said = (child.get("data") or {}) if isinstance(child, dict) else {}
            if child.get("kind") != "t1" or not said.get("body"):
                continue
            lines.append(f"{'  ' * depth}- u/{said.get('author')} ({said.get('score', 0)} points): {said['body']}")
            if depth < 2 and isinstance(said.get("replies"), dict):
                comments(said["replies"], depth + 1)

    comments(thread[1], 0)
    return "\n".join(lines)[:at_most]


async def read_subreddit(subreddit: str, sort: str = "hot", limit: int = 10) -> list[dict[str, str]]:
    """A subreddit's posts by her application."""
    sort = sort if sort in {"hot", "new", "top", "rising"} else "hot"
    query = urllib.parse.urlencode({"limit": max(1, min(int(limit), 50)) + 2, "raw_json": 1})
    listing = await _request("GET", f"{API}/r/{urllib.parse.quote(subreddit or 'all')}/{sort}?{query}",
                             headers={"Authorization": f"bearer {await _bearer()}"})
    return posts_in(listing, limit)


async def read_thread(url: str, *, comments: int = 60) -> str:
    """A post and its comments by her application, from the post's address on Reddit."""
    path = urllib.parse.urlsplit(url).path.rstrip("/")
    if "/comments/" not in path:
        raise ValueError(f"not the address of a post: {url}")
    query = urllib.parse.urlencode({"limit": comments, "raw_json": 1})
    return thread_in(await _request("GET", f"{API}{path}?{query}", headers={"Authorization": f"bearer {await _bearer()}"}))
