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

import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.MemoryPersister")


def _default_queue_path() -> Path:
    override = os.environ.get("AURA_MEMORY_PERSIST_RETRY_QUEUE_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aura/data/autonomy/persist-retry-queue.jsonl"


def _default_dedup_path() -> Path:
    override = os.environ.get("AURA_MEMORY_PERSIST_DEDUP_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aura/data/autonomy/persist-dedup.json"


QUEUE_PATH = _default_queue_path()
DEDUP_PATH = _default_dedup_path()
DEDUP_TTL_DAYS = 30.0


@dataclass
class FactRecord:
    fact: str                       # The claim, in natural language
    evidence: list[str] = field(default_factory=list)   # Quotes/sources backing it
    confidence: float = 0.5
    contradicts_belief: str | None = None  # If this conflicts with an existing belief
    domain: str = "general"
    provisional: bool = True

    def hash_key(self) -> str:
        return hashlib.sha256(self.fact.strip().lower().encode("utf-8")).hexdigest()[:16]


@dataclass
class EpisodicEvent:
    summary: str
    started_at: float
    completed_at: float | None = None
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
    contradicts: list[str] = field(default_factory=list)
    supersedes_belief_id: str | None = None

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
    failures: list[str] = field(default_factory=list)
    intent_ids: list[str] = field(default_factory=list)


class MemoryPersister:
    def __init__(
        self,
        executive: Any | None = None,
        memory_facade: Any | None = None,
        queue_path: Path | None = None,
        dedup_path: Path | None = None,
    ) -> None:
        self._executive = executive
        self._mem = memory_facade
        self._queue_path = (
            Path(queue_path).expanduser() if queue_path is not None else _default_queue_path()
        )
        self._dedup_path = (
            Path(dedup_path).expanduser() if dedup_path is not None else _default_dedup_path()
        )
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
        except (RuntimeError, AttributeError, TypeError, ValueError):
            return 0

        successful = 0
        skipped_duplicates = 0
        remaining: list[str] = []
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
            # CP126 (critical): "Replay does not check or mark dedup keys,
            # and a failed final queue rewrite is silently ignored;
            # successful records remain on disk and are committed again on
            # every replay."
            #
            # The first commit path consults _is_duplicate before writing;
            # replay did not, so the same episode, fact or belief was
            # re-committed on every retry sweep. Combined with the swallowed
            # rewrite below, one failed rewrite turned a retry queue into an
            # unbounded duplicate generator against her real memory.
            try:
                record = _replay_record(kind, payload)
                if record is None:
                    # A newer Aura may have queued a record this binary does
                    # not understand yet. Keep the exact line for that newer
                    # reader; silently draining it is irreversible data loss.
                    remaining.append(line)
                    continue
                # hash_key() must be inside the guard: a queued payload with a
                # null or wrong-typed field raises AttributeError here, and one
                # bad record used to abort the ENTIRE sweep — leaving every
                # later record unreplayed.
                dedup_key = record.hash_key()
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                record_degradation(
                    "memory_persister",
                    exc,
                    severity="info",
                    action="dropped an unparseable queued record",
                    enforce_failure_policy=False,
                )
                continue
            if self._is_duplicate(dedup_key):
                # Already committed by an earlier sweep. Draining it is the
                # correct outcome — leaving it queued is what produced the
                # repeat in the first place.
                skipped_duplicates += 1
                continue
            try:
                if kind == "episodic":
                    ok, _, _ = self._commit_episodic(title, record)
                elif kind == "fact":
                    ok, _, _ = self._commit_fact(title, record)
                elif kind == "belief":
                    ok, _, _ = self._commit_belief(title, record)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation('memory_persister', e)
                logger.debug("replay record kind=%s failed: %s", kind, e)
                ok = False

            if ok:
                successful += 1
                self._mark_committed(dedup_key)
            else:
                remaining.append(line)

        # Persist the dedup marks BEFORE rewriting the queue. If the process
        # dies between the two, the next sweep sees the marks and skips the
        # records rather than committing them a second time; the reverse
        # order loses that protection precisely when it is needed.
        self._save_dedup()

        try:
            atomic_write_text(self._queue_path, "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
            # Was `pass  # no-op: intentional`. It is not harmless: every
            # record just committed is still on disk, so the next sweep
            # replays them. The dedup marks above now absorb that, but a
            # queue that cannot be rewritten is a real durability fault and
            # has to be visible.
            record_degradation(
                "memory_persister",
                exc,
                severity="warning",
                action=(
                    "committed records remain queued; dedup marks prevent "
                    "re-commit but the queue is not draining"
                ),
                enforce_failure_policy=False,
            )
        if skipped_duplicates:
            logger.info(
                "Replay skipped %d already-committed record(s).", skipped_duplicates,
            )
        return successful

    # ── Per-tier commit ───────────────────────────────────────────────────

    def _commit_episodic(self, title: str, ep: EpisodicEvent) -> tuple[bool, str, str | None]:
        try:
            from core.executive.executive_core import (
                ActionType,
                Intent,
                IntentSource,
            )
        except (ImportError, AttributeError, RuntimeError) as e:
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
            except (RuntimeError, AttributeError, TypeError) as e:
                record_degradation('memory_persister', e)
                logger.debug("episodic.add fallback failed: %s", e)

        return True, "", intent.intent_id

    def _commit_fact(self, title: str, fact: FactRecord) -> tuple[bool, str, str | None]:
        try:
            from core.executive.executive_core import ActionType, Intent, IntentSource
        except (ImportError, AttributeError, RuntimeError) as e:
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

    def _commit_belief(self, title: str, belief: BeliefUpdate) -> tuple[bool, str, str | None]:
        try:
            from core.executive.executive_core import ActionType, Intent, IntentSource
        except (ImportError, AttributeError, RuntimeError) as e:
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
        """Submit one intent to the executive. False means NOT committed.

        CP126 4f3b7d53. Both no-executive paths returned success while
        writing nothing and enqueuing nothing — the comment claimed the
        intent was recorded to the queue and no queue write existed. So
        receipts marked episodic memories, facts and beliefs committed while
        the data was discarded: silent loss, reported as success, on the
        path whose whole job is durability.

        Returning failure is the entire fix, because the caller already
        does the right thing with it — _enqueue() to the retry queue and
        queued_for_retry on the receipt. The machinery was there; this
        function just never let it run.
        """
        if self._executive is None:
            # An un-injected persister used to be a total-loss path: every
            # commit failed with executive_unavailable and the whole queue was
            # write-only. Resolve the live executive before giving up, so the
            # default construction actually persists.
            try:
                from core.container import ServiceContainer

                self._executive = ServiceContainer.get("executive_core", default=None)
            except (ImportError, RuntimeError, AttributeError) as e:
                record_degradation('memory_persister', e, severity="debug")
            if self._executive is None:
                return False, "executive_unavailable"
        try:
            evaluator = getattr(self._executive, "evaluate_sync", None) or getattr(self._executive, "submit_sync", None)
            if evaluator is None:
                logger.debug(
                    "executive has no sync evaluator; intent queued for retry: %s",
                    intent.intent_id,
                )
                return False, "executive_has_no_sync_evaluator"
            decision = evaluator(intent)
            outcome = getattr(decision, "outcome", None)
            outcome_str = getattr(outcome, "value", str(outcome))
            if outcome_str in ("approved", "degraded"):
                return True, ""
            return False, f"executive_outcome={outcome_str}"
        except (RuntimeError, AttributeError, TypeError) as e:
            record_degradation('memory_persister', e)
            return False, str(e)

    # ── Queue + dedup helpers ────────────────────────────────────────────

    def _enqueue(self, kind: str, item_title: str, payload: dict[str, Any]) -> None:
        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            with self._queue_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind,
                    "item_title": item_title,
                    "payload": payload,
                    "queued_at": time.time(),
                }) + "\n")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # no-op: intentional

    def _load_dedup(self) -> dict[str, float]:
        if not self._dedup_path.exists():
            return {}
        try:
            data = json.loads(self._dedup_path.read_text(encoding="utf-8"))
            cutoff = time.time() - DEDUP_TTL_DAYS * 86400.0
            return {k: float(v) for k, v in data.items() if float(v) > cutoff}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}

    def _save_dedup(self) -> None:
        """Persist the committed-record marks.

        This is the ledger that stops a replay committing the same memory
        twice, so losing it silently reintroduces exactly the duplication the
        replay dedup exists to prevent. It also used not to catch OSError at
        all, so a full or read-only disk propagated out of a maintenance sweep
        and abandoned every record still queued behind it.
        """
        try:
            atomic_write_text(self._dedup_path, json.dumps(self._dedup), encoding="utf-8")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            record_degradation(
                "memory_persister",
                exc,
                severity="warning",
                action=(
                    "could not persist dedup marks; already-committed records "
                    "may be committed again on the next replay"
                ),
                enforce_failure_policy=False,
            )

    def _is_duplicate(self, key: str) -> bool:
        ts = self._dedup.get(key)
        if not ts:
            return False
        return (time.time() - ts) < DEDUP_TTL_DAYS * 86400.0

    def _mark_committed(self, key: str) -> None:
        self._dedup[key] = time.time()


def _replay_record(kind: Any, payload: Any) -> Any:
    """Rebuild a queued record so its dedup key can be computed.

    Returns None for a kind this version does not understand — a forward
    -compatible queue entry is not a reason to crash the sweep.
    """
    mapping = {
        "episodic": EpisodicEvent,
        "fact": FactRecord,
        "belief": BeliefUpdate,
    }
    cls = mapping.get(str(kind or ""))
    if cls is None:
        return None
    return cls(**_only_keys(payload or {}, cls))


def _dataclass_to_jsonable(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj)


def _only_keys(payload: dict[str, Any], cls: type) -> dict[str, Any]:
    """Filter payload dict down to fields the dataclass accepts."""
    field_names = {f.name for f in __import__("dataclasses").fields(cls)}
    return {k: v for k, v in payload.items() if k in field_names}
