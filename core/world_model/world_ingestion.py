"""World-scale ingestion — unrestricted web reach that updates the world model from all data.

The critique's "world-scale embodiment and data ingestion": Aura should have "unrestricted
internet access, not just allowlisted tools" and "a world model that updates from all data, not
just your life." The transport already exists (RobustHTTP, PhantomBrowser, reality_grounding);
what was missing is the autonomous engine that reaches *any* source, extracts it, and folds it
into the world model and memory.

Owner-authorized, full permissions: **reads are unrestricted** — any URL, any search, no
allowlist. The one discipline kept is the architecture's own, not a leash on reach: *state-
changing* requests (POST/PUT/DELETE/PATCH) and irreversible side-effects route through the value
model + Will first, because reversibility is a constitutional bound, not a permission gate. A GET
that only reads the world never asks anyone.

Ingestion pipeline: fetch → extract text → distill candidate facts → push to the belief sink
(world_state) and memory, and surface anomalies (claims that contradict current beliefs) to the
scientific engine as hypotheses. The fetcher and sinks are injectable, so the pipeline is fully
testable without a network and pluggable into whatever belief/memory surfaces the runtime exposes.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway

logger = logging.getLogger("WorldModel.Ingestion")

_STATE_CHANGING = {"POST", "PUT", "DELETE", "PATCH"}
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A DuckDuckGo HTML result link, used to parse the default search backend.
_DDG_RESULT = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                         re.IGNORECASE | re.DOTALL)

# Fetcher: url → (status_code, text). Sync or async both supported.
Fetcher = Callable[[str], Any]


@dataclass
class IngestDocument:
    url: str
    status: int
    text: str
    title: str = ""
    fetched_at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400 and bool(self.text)


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str = ""


@dataclass
class IngestReport:
    source: str
    facts: List[str]
    beliefs_written: int
    memories_written: int
    anomalies: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "facts": self.facts,
            "beliefs_written": self.beliefs_written,
            "memories_written": self.memories_written,
            "anomalies": self.anomalies,
            "error": self.error,
        }


class WorldIngestionEngine:
    """Unrestricted web reach + ingestion into the world model and memory."""

    def __init__(
        self,
        *,
        fetcher: Optional[Fetcher] = None,
        belief_sink: Optional[Callable[[str, Any, float], None]] = None,
        memory_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_chars: int = 20000,
        min_requests_interval_s: float = 0.5,
        max_facts_per_doc: int = 12,
    ) -> None:
        self._fetcher = fetcher
        self._belief_sink = belief_sink
        self._memory_sink = memory_sink
        self._max_chars = max_chars
        self._min_interval = min_requests_interval_s
        self._max_facts = max_facts_per_doc
        self._last_request = 0.0
        self._stats = {"fetched": 0, "ingested_facts": 0, "blocked_writes": 0}

    # ── reach (unrestricted reads) ────────────────────────────────────────

    async def fetch(self, url: str) -> IngestDocument:
        """GET any URL (no allowlist) and return extracted text. Reads are unrestricted."""
        await self._throttle()
        try:
            status, raw = await self._invoke_fetcher(url)
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug",
                               action=f"fetch failed: {url}")
            return IngestDocument(url=url, status=0, text="")
        self._stats["fetched"] += 1
        title = self._extract_title(raw)
        text = self._extract_text(raw)[: self._max_chars]
        return IngestDocument(url=url, status=int(status), text=text, title=title)

    async def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Unrestricted web search via the default HTML backend (or injected fetcher)."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        await self._throttle()
        try:
            _status, raw = await self._invoke_fetcher(url)
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug", action="search failed")
            return []
        results: List[SearchResult] = []
        for href, label in _DDG_RESULT.findall(raw or ""):
            title = _WS_RE.sub(" ", _TAG_RE.sub("", label)).strip()
            if href and title:
                results.append(SearchResult(url=href, title=title))
            if len(results) >= limit:
                break
        return results

    # ── ingestion (world model + memory updates) ──────────────────────────

    async def ingest_url(self, url: str, *, source_trust: float = 0.5) -> IngestReport:
        """Fetch a URL and fold its content into the world model + memory."""
        doc = await self.fetch(url)
        if not doc.ok:
            return IngestReport(source=url, facts=[], beliefs_written=0, memories_written=0,
                                error=f"fetch_failed:{doc.status}")
        return self.ingest_text(doc.text, source=url, source_trust=source_trust, title=doc.title)

    def ingest_text(self, text: str, *, source: str, source_trust: float = 0.5,
                    title: str = "") -> IngestReport:
        """Distill arbitrary text into facts and update beliefs + memory (sync, testable)."""
        facts = self._distill(text)
        beliefs_written = 0
        memories_written = 0
        anomalies: List[str] = []
        trust = max(0.0, min(1.0, source_trust))

        for fact in facts:
            # Anomaly detection: does this contradict what we already believe? Surface it to
            # the scientific engine as a hypothesis rather than silently overwriting.
            if self._contradicts_belief(fact):
                anomalies.append(fact)
                self._raise_hypothesis(fact, source)
            if self._write_belief(f"world_fact:{self._key(fact)}", fact, trust):
                beliefs_written += 1
            if self._write_memory(fact, {"source": source, "title": title, "trust": trust}):
                memories_written += 1

        self._stats["ingested_facts"] += len(facts)
        return IngestReport(source=source, facts=facts, beliefs_written=beliefs_written,
                            memories_written=memories_written, anomalies=anomalies)

    # ── governed side-effects (writes route through Will) ─────────────────

    async def state_changing_request(self, method: str, url: str,
                                     *, reversible: bool = False, confirmed: bool = False,
                                     **kwargs: Any) -> Dict[str, Any]:
        """A request that changes the world (POST/PUT/...) — gated by the value model + Will.

        Reads never reach this path. This is reversibility discipline, not a reach restriction:
        an irreversible, unconfirmed external write is held for confirmation.
        """
        method = method.upper()
        if method not in _STATE_CHANGING:
            # Not actually state-changing — treat as a read.
            doc = await self.fetch(url)
            return {"allowed": True, "status": doc.status, "text": doc.text}

        if not self._authorize_write(method, url, reversible=reversible, confirmed=confirmed):
            self._stats["blocked_writes"] += 1
            return {"allowed": False, "reason": "blocked_by_governance", "url": url}

        try:
            status, text = await self._invoke_fetcher_method(method, url, **kwargs)
            return {"allowed": True, "status": int(status), "text": text}
        except (RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug",
                               action=f"{method} {url} failed")
            return {"allowed": True, "status": 0, "error": str(exc)}

    def _authorize_write(self, method: str, url: str, *, reversible: bool, confirmed: bool) -> bool:
        try:
            from core.values.value_model import ActionDescriptor, get_value_model
            judgment = get_value_model().evaluate_with_will(ActionDescriptor(
                description=f"external {method} to {urlparse(url).netloc}",
                domain="network_call", reversible=reversible, confirmed=confirmed,
                impact=0.6, tags=("network", "external_write"),
            ))
            # Permitted and not awaiting confirmation → proceed.
            return judgment.permitted and not judgment.requires_confirmation
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug",
                               action="write authorization unavailable; failing closed")
            return False

    # ── sinks (best-effort, lazily resolved) ──────────────────────────────

    def _write_belief(self, key: str, value: Any, confidence: float) -> bool:
        sink = self._belief_sink
        if sink is None:
            try:
                from core.container import ServiceContainer
                ws = ServiceContainer.get("world_state", default=None)
                if ws is not None and hasattr(ws, "set_belief"):
                    sink = lambda k, v, c: ws.set_belief(k, v, confidence=c, source="world_ingestion")
            except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation("world_ingestion", exc, severity="debug")
        if sink is None:
            return False
        try:
            sink(key, value, confidence)
            return True
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug")
            return False

    def _write_memory(self, content: str, meta: Dict[str, Any]) -> bool:
        sink = self._memory_sink
        if sink is None:
            return False
        try:
            sink(content, meta)
            return True
        except (AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug")
            return False

    def _contradicts_belief(self, fact: str) -> bool:
        try:
            from core.container import ServiceContainer
            ws = ServiceContainer.get("world_state", default=None)
            if ws is not None and hasattr(ws, "get_belief"):
                prior = ws.get_belief(f"world_fact:{self._key(fact)}")
                if prior and isinstance(prior, str) and prior.strip() and prior.strip() != fact.strip():
                    return True
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug")
        return False

    def _raise_hypothesis(self, fact: str, source: str) -> None:
        try:
            from core.cognition.scientific_engine import get_scientific_engine
            get_scientific_engine().form_hypothesis(
                f"ingested claim may be true: {fact[:160]}",
                predicted_observable="corroborated_by_independent_source",
                expected=0.5, prior_confidence=0.4,
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError, TypeError) as exc:
            record_degradation("world_ingestion", exc, severity="debug")

    # ── extraction / distillation ─────────────────────────────────────────

    @staticmethod
    def _extract_text(raw: str) -> str:
        raw = raw or ""
        raw = _SCRIPT_STYLE_RE.sub(" ", raw)
        raw = _TAG_RE.sub(" ", raw)
        return _WS_RE.sub(" ", raw).strip()

    @staticmethod
    def _extract_title(raw: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", raw or "", re.IGNORECASE | re.DOTALL)
        return _WS_RE.sub(" ", _TAG_RE.sub("", m.group(1))).strip()[:200] if m else ""

    def _distill(self, text: str) -> List[str]:
        """Pull declarative, information-bearing sentences out of extracted text."""
        out: List[str] = []
        seen: set = set()
        for sent in _SENT_SPLIT.split(text or ""):
            s = sent.strip()
            if not (40 <= len(s) <= 280):
                continue
            words = s.split()
            if len(words) < 6:
                continue
            # Skip nav/boilerplate-ish lines (mostly capitalized fragments, no verb-ish token).
            if not re.search(r"\b(is|are|was|were|has|have|will|can|provides|causes|means|"
                             r"includes|uses|supports|enables|found|shows|reported)\b", s, re.IGNORECASE):
                continue
            key = self._key(s)
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= self._max_facts:
                break
        return out

    @staticmethod
    def _key(text: str) -> str:
        return _WS_RE.sub(" ", str(text or "").strip().lower())[:160]

    # ── fetcher plumbing (default httpx; injectable for tests) ─────────────

    async def _throttle(self) -> None:
        import asyncio
        dt = time.time() - self._last_request
        if dt < self._min_interval:
            await asyncio.sleep(self._min_interval - dt)
        self._last_request = time.time()

    async def _invoke_fetcher(self, url: str) -> Tuple[int, str]:
        if self._fetcher is not None:
            res = self._fetcher(url)
            if hasattr(res, "__await__"):
                res = await res
            return res
        return await self._default_get(url)

    async def _invoke_fetcher_method(self, method: str, url: str, **kwargs: Any) -> Tuple[int, str]:
        if self._fetcher is not None:
            res = self._fetcher(url)  # injected fetchers are read-shaped in tests
            if hasattr(res, "__await__"):
                res = await res
            return res
        return await self._default_request(method, url, **kwargs)

    async def _default_get(self, url: str) -> Tuple[int, str]:
        return await self._default_request("GET", url)

    async def _default_request(self, method: str, url: str, **kwargs: Any) -> Tuple[int, str]:
        headers = kwargs.pop("headers", {"User-Agent": "Mozilla/5.0 (compatible; Aura/1.0)"})
        result = await get_network_gateway().request_async(
            method,
            url,
            headers=headers,
            timeout=30.0,
            source="world_ingestion",
            read_only=method.upper() in {"GET", "HEAD", "OPTIONS"},
            **kwargs,
        )
        content = result.get("content", b"")
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content or "")
        return int(result.get("status_code") or 0), text

    def get_health(self) -> Dict[str, Any]:
        return {"module": "WorldIngestionEngine", "stats": dict(self._stats), "status": "online"}


_instance: Optional[WorldIngestionEngine] = None


def get_world_ingestion_engine() -> WorldIngestionEngine:
    global _instance
    if _instance is None:
        _instance = WorldIngestionEngine()
    return _instance
