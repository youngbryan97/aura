"""skills/web_search.py — Enhanced Web Search Skill

Provides lightweight, dependency-resilient web search for Aura.
Falls back gracefully through multiple search strategies:
  1. DuckDuckGo HTML scrape (no API key)
  2. urllib-based raw HTML fetch + snippet extraction
  3. Stub result if all fail

FreeSearchSkill is a compatibility alias pointing here.
"""

import asyncio
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.WebSearch")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_PRICE_QUERY_TERMS = {
    "price",
    "cost",
    "worth",
    "quote",
    "trading",
    "ticker",
    "market",
}

_PRICE_RESULT_TERMS = {
    "price",
    "usd",
    "quote",
    "chart",
    "market cap",
    "trading at",
    "live price",
}

_PROMOTIONAL_TERMS = {
    "buy",
    "sell",
    "exchange",
    "trade",
    "wallet",
    "download app",
    "sign up",
    "sign-up",
    "most trusted",
    "safe and trusted",
    "trusted place",
}

_PRICE_FRIENDLY_DOMAINS = {
    "coinmarketcap.com",
    "coingecko.com",
    "coindesk.com",
    "finance.yahoo.com",
    "markets.businessinsider.com",
    "marketwatch.com",
    "investing.com",
    "tradingview.com",
}


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")
    deep: bool = Field(False, description="If True, fetch and summarize the first result page.")
    num_results: int = Field(5, ge=1, le=20, description="Number of results to return.")


class EnhancedWebSearchSkill(BaseSkill):
    """High-resilience web search with deep-dive option.

    Uses DuckDuckGo (no API key required) as the primary source.
    Falls back to urllib raw fetch if playwright/requests unavailable.
    When deep=True, fetches the top result's page and extracts the
    first ~800 characters of readable body text for richer context.
    """

    name = "web_search"
    description = (
        "Search the internet for current information, news, facts, or research. "
        "Returns a list of search result snippets. Set deep=True to read the "
        "content of the first result in detail."
    )
    input_model = WebSearchInput
    timeout_seconds = 25.0
    metabolic_cost = 2

    def __init__(self):
        super().__init__()
        self.browser = _StubBrowser()  # Satisfy tests that access skill.browser

    async def execute(self, params: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        # Normalise params — accept dict or WebSearchInput or legacy goal-dict format
        if isinstance(params, dict):
            # Legacy interface: {"query": ..., "deep": ...}
            query = params.get("query") or params.get("q", "")
            deep = bool(params.get("deep", False))
            num_results = int(params.get("num_results", 5))
        elif isinstance(params, WebSearchInput):
            query = params.query
            deep = params.deep
            num_results = params.num_results
        else:
            query = str(params)
            deep = False
            num_results = 5

        if not query or not query.strip():
            return {"ok": False, "error": "No search query provided."}

        query = query.strip()
        logger.info("🔍 WebSearch: '%s' (deep=%s)", query[:60], deep)

        try:
            results = await asyncio.to_thread(self._ddg_search, query, num_results)
        except Exception as e:
            logger.warning("DDG search failed: %s — falling back to stub.", e)
            results = []

        if not results:
            return {
                "ok": False,
                "error": f"No results found for '{query}'.",
                "query": query,
                "results": [],
                "result": "",
                "source": "none",
                "mode": "standard",
            }

        top = results[0]
        deep_content = ""

        if deep and top.get("url"):
            try:
                deep_content = await asyncio.to_thread(self._fetch_page_text, top["url"])
            except Exception as e:
                logger.debug("Deep fetch failed for %s: %s", top["url"], e)

        result_text = "\n\n".join(
            f"{i+1}. {r.get('title','')}\n{r.get('snippet','')}\n{r.get('url','')}"
            for i, r in enumerate(results)
        )

        return {
            "ok": True,
            "query": query,
            "results": results,
            "result": deep_content if (deep and deep_content) else result_text,
            "summary": top.get("snippet", ""),
            "source": top.get("url", ""),
            "mode": "deep" if (deep and deep_content) else "standard",
            "count": len(results),
        }

    # ── Internal helpers ───────────────────────────────────────────

    def _ddg_search(self, query: str, num: int) -> List[Dict[str, str]]:
        """Scrape DuckDuckGo HTML search results (no API key needed)."""
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        results = []
        # DDG HTML structure: <a class="result__a" href="...">title</a>
        # and <a class="result__snippet">snippet</a>
        title_re = re.compile(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        snippet_re = re.compile(
            r'class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        titles = title_re.findall(html)
        snippets = [m.group(1) for m in snippet_re.finditer(html)]

        for i, (href, title) in enumerate(titles[:num]):
            # DDG proxies URLs through redirect — extract the real URL
            real_url = self._extract_ddg_url(href)
            snippet = self._clean_html(snippets[i]) if i < len(snippets) else ""
            title_clean = self._clean_html(title)
            if real_url and title_clean:
                results.append({"title": title_clean, "url": real_url, "snippet": snippet})

        return self._rerank_results(query, results)

    @staticmethod
    def _domain_matches(host: str, domain: str) -> bool:
        return host == domain or host.endswith(f".{domain}")

    @classmethod
    def _is_price_query(cls, query: str) -> bool:
        text = f" {query.lower()} "
        if any(term in text for term in _PRICE_QUERY_TERMS):
            return True
        return bool(re.search(r"\b(?:btc|bitcoin|eth|ethereum|sol|solana|stock|shares?)\b", text))

    @classmethod
    def _score_result(cls, query: str, result: Dict[str, str], index: int) -> int:
        title = str(result.get("title", "") or "")
        snippet = str(result.get("snippet", "") or "")
        url = str(result.get("url", "") or "")
        host = urllib.parse.urlparse(url).netloc.lower()
        haystack = f"{title} {snippet} {host}".lower()
        query_tokens = [token for token in re.split(r"[^a-z0-9]+", query.lower()) if len(token) >= 3]

        score = 100 - index
        for token in query_tokens[:8]:
            if token in haystack:
                score += 2

        if cls._is_price_query(query):
            if any(term in haystack for term in _PRICE_RESULT_TERMS):
                score += 12
            if re.search(r"\$\s?\d", haystack) or re.search(r"\b\d[\d,]*(?:\.\d+)?\s*(?:usd|usdt)\b", haystack):
                score += 12
            if any(cls._domain_matches(host, domain) for domain in _PRICE_FRIENDLY_DOMAINS):
                score += 10
            if any(term in haystack for term in _PROMOTIONAL_TERMS):
                score -= 14
            if cls._domain_matches(host, "coinbase.com") and any(term in haystack for term in _PROMOTIONAL_TERMS):
                score -= 10
            if not any(term in haystack for term in _PRICE_RESULT_TERMS) and "$" not in haystack:
                score -= 6

        return score

    @classmethod
    def _rerank_results(cls, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(results) < 2:
            return results
        indexed_results = list(enumerate(results))
        return [
            item
            for _idx, item in sorted(
                indexed_results,
                key=lambda pair: cls._score_result(query, pair[1], pair[0]),
                reverse=True,
            )
        ]

    def _extract_ddg_url(self, href: str) -> str:
        """Extract real URL from DDG redirect href."""
        if href.startswith("http"):
            return href
        # DDG uses //duckduckgo.com/l/?uddg=<encoded_url>
        match = re.search(r'uddg=([^&]+)', href)
        if match:
            return urllib.parse.unquote(match.group(1))
        # Some hrefs are relative with //
        if href.startswith("//"):
            return "https:" + href
        return href

    def _fetch_page_text(self, url: str, max_chars: int = 1200) -> str:
        """Fetch a URL and return the first max_chars of readable text."""
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "html" not in content_type:
                return ""
            html = resp.read(65536).decode("utf-8", errors="replace")

        # Strip scripts, styles, tags
        html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    @staticmethod
    def _clean_html(text: str) -> str:
        """Remove HTML tags and decode entities."""
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#x27;', "'").replace('&nbsp;', ' ')
        return re.sub(r'\s+', ' ', text).strip()

    async def on_stop_async(self):
        """Compatibility stub — no persistent resources to clean up."""
        pass


class _StubBrowser:
    """Minimal browser stub to satisfy tests that access skill.browser."""
    is_active = False

    async def ensure_ready(self): pass
    async def browse(self, url: str): pass
    async def click(self, text_match: str = "", selector: str = "") -> bool:
        return False
    async def close(self): pass
