"""core/knowledge/bottling.py

Knowledge Bottling  (lineage: Brainiac — DC)
===========================================
Brainiac bottles whole civilizations — collecting and compressing knowledge for
later retrieval. This compresses a topic/corpus into a structured "bottle"
(summary + key facts + retrieval keys), persists it, and retrieves it by keyword
later. Model-driven compression when a brain is available, deterministic
extraction otherwise. It lives in knowledge/ beside ingestion.py and retrieval.py,
which it complements: ingestion brings raw knowledge in; bottling compresses a
focused topic into a durable, indexed capsule.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.engine_support import (
    coerce_text,
    data_root,
    record_engine_degradation,
    resolve_brain,
)

logger = logging.getLogger("Aura.KnowledgeBottling")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("knowledge_bottling", exc, action=action, severity=severity)


@dataclass
class KnowledgeBottle:
    topic: str
    slug: str
    summary: str
    key_facts: list[str]
    keys: list[str]            # retrieval keys
    source_chars: int
    created_at: float = field(default_factory=lambda: time.time())


class KnowledgeBottlingEngine:
    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._bottles_made = 0
        try:
            self._store: Path | None = data_root("knowledge_bottles")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            _degrade(exc, action="disabled knowledge-bottle persistence after store path setup failed")
            self._store = None
        logger.info("🫙 KnowledgeBottlingEngine initialized (Brainiac lineage)")

    @staticmethod
    def _slugify(topic: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
        return (slug or "bottle")[:60]

    @staticmethod
    def _extract_keys(text: str, limit: int = 12) -> list[str]:
        stop = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to",
            "of", "in", "on", "for", "with", "that", "this", "it", "as", "by", "be",
        }
        words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:limit]]

    @staticmethod
    def _heuristic_summary(text: str, max_sentences: int = 3) -> tuple[str, list[str]]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        summary = " ".join(sentences[:max_sentences])
        key_facts = sentences[:5]
        return summary[:1000], key_facts

    async def bottle(self, topic: str, content: str) -> KnowledgeBottle:
        self._bottles_made += 1
        summary, key_facts = self._heuristic_summary(content)
        keys = self._extract_keys(topic + " " + content)

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think") and content:
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                out = coerce_text(await asyncio.wait_for(
                    brain.think(
                        f"Compress the following about '{topic}' into a 2-sentence summary:\n{content[:1500]}",
                        mode=ThinkingMode.FAST, origin="brainiac", is_background=True,
                    ),
                    timeout=25.0,
                ))
                if out:
                    summary = out.strip()[:1000]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="stored heuristic knowledge bottle after model compression failed")

        bottle = KnowledgeBottle(
            topic=topic[:200],
            slug=self._slugify(topic),
            summary=summary,
            key_facts=key_facts,
            keys=keys,
            source_chars=len(content),
        )
        if self._store is not None:
            try:
                await async_atomic_write_text(self._store / f"{bottle.slug}.json", json.dumps(asdict(bottle), indent=2))
            except (OSError, TypeError, ValueError) as exc:
                _degrade(exc, action="returned in-memory knowledge bottle after persistence failed")
        return bottle

    def retrieve(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        if self._store is None:
            return []
        qkeys = set(self._extract_keys(query, limit=20))
        scored: list[tuple[int, dict]] = []
        try:
            for path in self._store.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
                overlap = len(qkeys.intersection(set(data.get("keys", []))))
                if query.lower() in data.get("topic", "").lower():
                    overlap += 5
                if overlap:
                    scored.append((overlap, data))
        except (OSError, RuntimeError) as exc:
            _degrade(exc, action="returned empty knowledge-bottle retrieval after store scan failed")
            return []
        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def get_status(self) -> dict[str, Any]:
        count = 0
        if self._store is not None:
            try:
                count = sum(1 for _ in self._store.glob("*.json"))
            except OSError:
                count = self._bottles_made
        return {"bottles_made_session": self._bottles_made, "bottles_on_disk": count, "healthy": True}


_INSTANCE: KnowledgeBottlingEngine | None = None


def get_knowledge_bottling(orchestrator: Any = None) -> KnowledgeBottlingEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = KnowledgeBottlingEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_knowledge_bottling(orchestrator: Any = None) -> KnowledgeBottlingEngine:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.BRAINIAC, default=None) or get_knowledge_bottling(orchestrator)
    register_runtime_service(ServiceNames.BRAINIAC, inst, required=False, owner="core/knowledge/bottling.py", registered_by="register_knowledge_bottling")
    register_runtime_service("brainiac", inst, required=False, owner="core/knowledge/bottling.py", registered_by="register_knowledge_bottling")
    return inst


__all__ = [
    "KnowledgeBottle",
    "KnowledgeBottlingEngine",
    "get_knowledge_bottling",
    "register_knowledge_bottling",
]
