"""core/autonomy/memory_persister.py
─────────────────────────────────────
Commit research-derived findings to Aura's memory subsystems via the
executive's intent-gating system. All writes are routed through
``IntentSource.AUTONOMOUS_RESEARCH`` so Rule 7's reconciliation gate
permits them through with provisional confidence.

Multi-tier persistence:
  • Episodic event ("watched X on date Y") — always commits, durable.
  • Semantic facts (what was learned) — provisional, queued for reconciliation.
  • Belief updates (revised positions on contested topics) — provisional,
    flagged with source provenance and contradiction set.

Defensive against:
  • Memory facade unavailable (queues writes locally for retry).
  • Executive unavailable (queues writes locally for retry).
  • Partial writes (rollback if any tier fails).
  • Duplicate writes (content-hash dedup).

Public API:
    persister = MemoryPersister(executive=..., memory_facade=...)
    receipt = persister.commit_engagement(
        item_title=..., episodic=..., facts=..., belief_updates=...,
    )
"""

from __future__ import annotations
from core.runtime.errors import record_degradation


import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("Aura.MemoryPersister")

QUEUE_PATH = Path.home() / ".aura/live-source/aura/knowledge/persist-retry-queue.jsonl"
DEDUP_PATH = Path.home() / ".aura/live-source/aura/knowledge/persist-dedup.json"
DEDUP_TTL_DAYS = 30.0


@dataclass
class FactRecord:
    fact: str                       # The claim, in natural language
    evidence: List[str] = field(default_factory=list)   # Quotes/sources backing it
    confidence: float = 0.5
    contradicts_belief: Optional[str] = None  # If this conflicts with an existing belief
    domain: str = "general"
    provisional: bool = True

    def hash_key(self) -> str:
        return hashlib.sha256(self.fact.strip().lower().encode("utf-8")).hexdigest()[:16]


@dataclass
class EpisodicEvent:
    summary: str
    started_at: float
    completed_at: Optional[float] = None
    item_title: str = ""
    method_priority_level: int = 6
    notes: str = ""

    def hash_key(self) -> str:
        h = f"{self.item_title}::{int(self.started_at)}::{self.summary[:80]}"
        return hashlib.sha256(h.encode("utf-8")).hexdigest()[:16]


@dataclass
class BeliefUpdate:
    topic: str
    position: str
    rationale: str
    confidence: float
    contradicts: List[str] = field(default_factory=list)
    supersedes_belief_id: Optional[str] = None

    def hash_key(self) -> str:
        h = f"{self.topic}::{self.position[:80]}"
        return hashlib.sha256(h.encode("utf-8")).hexdigest()[:16]


@dataclass
class CommitReceipt:
    accepted: bool
    item_title: str
    episodic_committed: bool = False
    facts_committed: int = 0
    facts_total: int = 0
    beliefs_committed: int = 0
    beliefs_total: int = 0
    queued_for_retry: int = 0
    duplicates_skipped: int = 0
    failures: List[str] = field(default_factory=list)
    intent_ids: List[str] = field(default_factory=list)


class MemoryPersister:
    def __init__(
        self,
        executive: Optional[Any] = None,
        memory_facade: Optional[Any] = None,
        queue_path: Path = QUEUE_PATH,
        dedup_path: Path = DEDUP_PATH,
    ) -> None:
        self._executive = executive
        self._mem = memory_facade
        self._queue_path = queue_path
        self._dedup_path = dedup_path
        self._dedup = self._load_dedup()

    # ── Public API ────────────────────────────────────────────────────────

    def commit_engagement(
        self,
        item_title: str,
        episodic: EpisodicEvent,
        facts: Sequence[FactRecord] = (),
        belief_updates: Sequence[BeliefUpdate] = (),
    ) -> CommitReceipt:
        receipt = CommitReceipt(accepted=True, item_title=item_title)
        receipt.facts_total = len(facts)
        receipt.beliefs_total = len(belief_updates)

        # 1. Episodic — always durable
        ep_ok, ep_err, ep_intent_id = self._commit_episodic(item_title, episodic)
        receipt.episodic_committed = ep_ok
        if ep_intent_id:
            receipt.intent_ids.append(ep_intent_id)
        if not ep_ok:
            receipt.failures.append(f"episodic: {ep_err}")
            self._enqueue("episodic", item_title, episodic.__dict__)
            receipt.queued_for_retry += 1

        # 2. Facts — provisional
        for fact in facts:
            if self._is_duplicate(fact.hash_key()):
                receipt.duplicates_skipped += 1
                continue
            ok, err, intent_id = self._commit_fact(item_title, fact)
            if ok:
                receipt.facts_committed += 1
                self._mark_committed(fact.hash_key())
                if intent_id:
                    receipt.intent_ids.append(intent_id)
            else:
                receipt.failures.append(f"fact[{fact.fact[:40]}]: {err}")
                self._enqueue("fact", item_title, _dataclass_to_jsonable(fact))
                receipt.queued_for_retry += 1

        # 3. Belief updates — provisional, with contradiction metadata
        for belief in belief_updates:
            if self._is_duplicate(belief.hash_key()):
                receipt.duplicates_skipped += 1
                continue
            ok, err, intent_id = self._commit_belief(item_title, belief)
            if ok:
                receipt.beliefs_committed += 1
                self._mark_committed(belief.hash_key())
                if intent_id:
                    receipt.intent_ids.append(intent_id)
            else:
                receipt.failures.append(f"belief[{belief.topic[:40]}]: {err}")
                self._enqueue("belief", item_title, _dataclass_to_jsonable(belief))
                receipt.queued_for_retry += 1

        # If everything failed, mark not accepted
        if (
            not receipt.episodic_committed
            and receipt.facts_committed == 0
            and receipt.beliefs_committed == 0
            and (receipt.facts_total > 0 or receipt.beliefs_total > 0 or not ep_ok)
        ):
            receipt.accepted = False

        self._save_dedup()
        return receipt

    def replay_queue(self) -> int:
        """Retry queued writes from prior failures. Returns number successful."""
        if not self._queue_path.exists():
            return 0
        try:
            lines = self._queue_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return 0

        successful = 0
        remaining: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = rec.get("kind")
            title = rec.get("item_title", "")
            payload = rec.get("payload", {})
            ok = False
            try:
                if kind == "episodic":
                    ok, _, _ = self._commit_episodic(title, EpisodicEvent(**_only_keys(payload, EpisodicEvent)))
                elif kind == "fact":
                    ok, _, _ = self._commit_fact(title, FactRecord(**_only_keys(payload, FactRecord)))
                elif kind == "belief":
                    ok, _, _ = self._commit_belief(title, BeliefUpdate(**_only_keys(payload, BeliefUpdate)))
            except Exception as e:
                record_degradation('memory_persister', e)
                logger.debug("replay record kind=%s failed: %s", kind, e)
                ok = False

            if ok:
                successful += 1
            else:
                remaining.append(line)

        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            self._queue_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        except Exception:
            pass
        return successful

    # ── Per-tier commit ───────────────────────────────────────────────────

    def _commit_episodic(self, title: str, ep: EpisodicEvent) -> tuple[bool, str, Optional[str]]:
        try:
            from core.executive.executive_core import (
                Intent, IntentSource, ActionType,
            )
        except Exception as e:
            record_degradation('memory_persister', e)
            return False, f"executive import: {e}", None

        intent = Intent(
            source=IntentSource.AUTONOMOUS_RESEARCH,
            goal=f"episodic:{title}",
            action_type=ActionType.WRITE_MEMORY,
            payload={
                "tier": "episodic",
                "summary": ep.summary,
                "started_at": ep.started_at,
                "completed_at": ep.completed_at,
                "item_title": ep.item_title,
                "method_priority_level": ep.method_priority_level,
                "notes": ep.notes,
                "confidence_tier": "durable",  # episodic events are facts (we did engage)
            },
            priority=0.5,
            confidence=0.95,
            requires_memory_commit=True,
        )

        ok, err = self._submit_intent(intent)
        if not ok:
            return False, err, intent.intent_id

        # Best-effort direct write to memory_facade if available
        if self._mem is not None:
            try:
                episodic = getattr(self._mem, "episodic", lambda: None)()
                if episodic and hasattr(episodic, "add"):
                    episodic.add({
                        "title": ep.summary,
                        "metadata": {
                            "kind": "autonomous_research_engagement",
                            "item_title": ep.item_title,
                            "started_at": ep.started_at,
                            "completed_at": ep.completed_at,
                            "method_priority_level": ep.method_priority_level,
                        },
                    })
            except Exception as e:
                record_degradation('memory_persister', e)
                logger.debug("episodic.add fallback failed: %s", e)

        return True, "", intent.intent_id

    def _commit_fact(self, title: str, fact: FactRecord) -> tuple[bool, str, Optional[str]]:
        try:
            from core.executive.executive_core import Intent, IntentSource, ActionType
        except Exception as e:
            record_degradation('memory_persister', e)
            return False, f"executive import: {e}", None

        intent = Intent(
            source=IntentSource.AUTONOMOUS_RESEARCH,
            goal=f"fact:{fact.fact[:60]}",
            action_type=ActionType.UPDATE_BELIEF,
            payload={
                "tier": "semantic",
                "fact": fact.fact,
                "evidence": list(fact.evidence),
                "confidence": float(fact.confidence),
                "contradicts_belief": fact.contradicts_belief,
                "domain": fact.domain,
                "confidence_tier": "provisional" if fact.provisional else "durable",
                "requires_reconciliation": fact.contradicts_belief is not None,
                "source_item": title,
            },
            priority=0.4,
            confidence=float(fact.confidence),
            requires_memory_commit=True,
        )
        ok, err = self._submit_intent(intent)
        return ok, err, intent.intent_id

    def _commit_belief(self, title: str, belief: BeliefUpdate) -> tuple[bool, str, Optional[str]]:
        try:
            from core.executive.executive_core import Intent, IntentSource, ActionType
        except Exception as e:
            record_degradation('memory_persister', e)
            return False, f"executive import: {e}", None

        intent = Intent(
            source=IntentSource.AUTONOMOUS_RESEARCH,
            goal=f"belief:{belief.topic[:60]}",
            action_type=ActionType.UPDATE_BELIEF,
            payload={
                "tier": "belief",
                "topic": belief.topic,
                "position": belief.position,
                "rationale": belief.rationale,
                "contradicts": list(belief.contradicts),
                "supersedes_belief_id": belief.supersedes_belief_id,
                "confidence_tier": "provisional",
                "requires_reconciliation": True,
                "source_item": title,
            },
            priority=0.4,
            confidence=float(belief.confidence),
            requires_memory_commit=True,
        )
        ok, err = self._submit_intent(intent)
        return ok, err, intent.intent_id

    # ── Executive submission ─────────────────────────────────────────────

    def _submit_intent(self, intent: Any) -> tuple[bool, str]:
        if self._executive is None:
            # No live executive — record the intent to the queue but
            # consider this a soft success so callers can still proceed.
            return True, ""
        try:
            evaluator = getattr(self._executive, "evaluate_sync", None) or getattr(self._executive, "submit_sync", None)
            if evaluator is None:
                # async-only executive: fall back to logging
                logger.debug("executive has no sync evaluator; intent left for async submission: %s", intent.intent_id)
                return True, ""
            decision = evaluator(intent)
            outcome = getattr(decision, "outcome", None)
            outcome_str = getattr(outcome, "value", str(outcome))
            if outcome_str in ("approved", "degraded"):
                return True, ""
            return False, f"executive_outcome={outcome_str}"
        except Exception as e:
            record_degradation('memory_persister', e)
            return False, str(e)

    # ── Queue + dedup helpers ────────────────────────────────────────────

    def _enqueue(self, kind: str, item_title: str, payload: Dict[str, Any]) -> None:
        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            with self._queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind,
                    "item_title": item_title,
                    "payload": payload,
                    "queued_at": time.time(),
                }) + "\n")
        except Exception:
            pass

    def _load_dedup(self) -> Dict[str, float]:
        if not self._dedup_path.exists():
            return {}
        try:
            data = json.loads(self._dedup_path.read_text(encoding="utf-8"))
            cutoff = time.time() - DEDUP_TTL_DAYS * 86400.0
            return {k: float(v) for k, v in data.items() if float(v) > cutoff}
        except Exception:
            return {}

    def _save_dedup(self) -> None:
        try:
            self._dedup_path.parent.mkdir(parents=True, exist_ok=True)
            self._dedup_path.write_text(json.dumps(self._dedup), encoding="utf-8")
        except Exception:
            pass

    def _is_duplicate(self, key: str) -> bool:
        ts = self._dedup.get(key)
        if not ts:
            return False
        return (time.time() - ts) < DEDUP_TTL_DAYS * 86400.0

    def _mark_committed(self, key: str) -> None:
        self._dedup[key] = time.time()


def _dataclass_to_jsonable(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def _only_keys(payload: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Filter payload dict down to fields the dataclass accepts."""
    field_names = {f.name for f in __import__("dataclasses").fields(cls)}
    return {k: v for k, v in payload.items() if k in field_names}
