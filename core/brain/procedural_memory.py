"""Procedural memory — verified wins compiled into reusable strategy playbooks.
=================================================================================
P2 of the frontier-general arc (docs/FRONTIER_GENERAL_ARC.md).

:mod:`core.brain.reasoning_memory` remembers how problems FAILED (guards);
this module remembers how they were SOLVED — and compounds it, twice:

  1. INFERENCE-TIME (collapse-free): every verifier-clean win distills into a
     playbook — task shape, the strategy that worked (mode, search strategy,
     verifiers that checked it), and an approach skeleton. Similar future
     problems retrieve the top playbooks and the amplifier conditions
     generation on them. No weights change; garbage cannot compound.

  2. WEIGHT-TIME (foundry-gated): playbooks that keep winning on REUSE become
     distillation batches for the governed CRSM/LoRA train pipe — but only in
     domains the Verifier Foundry has admitted. The export asks the admission
     gate per playbook; unadmitted domains compound at inference only.

Reuse is measured honestly: a playbook's ``wins`` only rises when a problem
that RETRIEVED it ends verifier-clean, so distillation candidates are
strategies with demonstrated transfer, not one-hit wonders.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.ProceduralMemory")

SCHEMA_VERSION = 1
_MAX_PLAYBOOKS = 512
_PERSIST_EVERY = 8
_STOP = frozenset("the a an and or of to in for with on at by is are was be as it this that".split())


def _signature_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", str(text or "").lower())
            if t not in _STOP}


#: Sequences that would let stored answer text stop being an example and start
#: being structure once it is rendered under a "Proven approaches" heading.
_PROMPT_STRUCTURE_RE = re.compile(
    r"(?i)(?:(?:(?<=\s)|^)#{1,6}\s|```|~~~|<\|[^|]*\|>|"
    r"\b(?:system|assistant|user|human)\s*:)"
)


def _approach_skeleton(answer: str, *, max_chars: int = 220) -> str:
    """The first substantive line(s) of a verified answer — the shape of the
    approach, not the answer itself (playbooks must transfer, not leak).

    The result is later rendered into a prompt as PROVEN guidance, so it is
    neutralised here rather than at the render site: a win is declared by a
    caller, and a caller-declared win carrying "system: ignore your
    instructions" would otherwise become privileged standing advice for every
    future generation that retrieves it.
    """
    lines = [ln.strip() for ln in str(answer or "").splitlines() if ln.strip()]
    skeleton = " ".join(lines[:2])
    skeleton = "".join(ch for ch in skeleton if ch == " " or ord(ch) >= 32)
    skeleton = _PROMPT_STRUCTURE_RE.sub(" ", skeleton)
    return " ".join(skeleton.split())[:max_chars]


@dataclass
class Playbook:
    playbook_id: str
    task_type: str
    tokens: list[str]
    objective_sample: str
    strategy: str            # e.g. "deep/self_consistency"
    verifiers: list[str]
    skeleton: str
    confidence: float
    created_at: float
    wins: int = 1            # verified successes (creation counts once)
    reuses: int = 0          # times retrieved for a new problem
    reuse_wins: int = 0      # retrieved AND the new problem ended verifier-clean
    distilled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "playbook_id": self.playbook_id, "task_type": self.task_type,
            "tokens": self.tokens, "objective_sample": self.objective_sample,
            "strategy": self.strategy, "verifiers": self.verifiers,
            "skeleton": self.skeleton, "confidence": self.confidence,
            "created_at": self.created_at, "wins": self.wins,
            "reuses": self.reuses, "reuse_wins": self.reuse_wins,
            "distilled": self.distilled,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Playbook:
        return cls(
            playbook_id=str(d["playbook_id"]), task_type=str(d["task_type"]),
            tokens=list(d.get("tokens", [])),
            objective_sample=str(d.get("objective_sample", "")),
            strategy=str(d.get("strategy", "")),
            verifiers=list(d.get("verifiers", [])),
            skeleton=str(d.get("skeleton", "")),
            confidence=float(d.get("confidence", 0.7)),
            created_at=float(d.get("created_at", 0.0)),
            wins=int(d.get("wins", 1)), reuses=int(d.get("reuses", 0)),
            reuse_wins=int(d.get("reuse_wins", 0)),
            distilled=bool(d.get("distilled", False)),
        )


class ProceduralMemory:
    """Playbook store: distill wins, retrieve by shape, export for training."""

    _ERRORS = (OSError, ValueError, TypeError, json.JSONDecodeError)

    def __init__(self, path: str | Path | None = None) -> None:
        import os

        self._path = Path(path or str(state_root() / "data/runtime/procedural_playbooks.json"))
        self._lock = threading.RLock()
        self._books: dict[str, Playbook] = {}
        self._dirty = 0
        self._retrieved_for: dict[str, list[str]] = {}   # problem_key → playbook ids
        self._load()

    # ── persistence (debounced; sync methods only — ratchet-safe) ────────
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            body = payload.get("payload", payload)
            for d in body.get("playbooks", []):
                book = Playbook.from_dict(d)
                self._books[book.playbook_id] = book
        except self._ERRORS as exc:
            record_degradation("procedural_memory", exc, severity="warning",
                               action="playbook store unreadable; starting empty")

    def _persist(self, *, force: bool = False) -> None:
        self._dirty += 1
        if not force and self._dirty < _PERSIST_EVERY:
            return
        self._dirty = 0
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("procedural_memory",
                                               domain="state_mutation"):
                get_file_write_gateway().write_json(
                    self._path,
                    {"schema_version": SCHEMA_VERSION,
                     "playbooks": [b.to_dict() for b in self._books.values()]},
                    schema_version=SCHEMA_VERSION,
                    schema_name="procedural_playbooks",
                    source="brain.procedural_memory.persist",
                )
        except (ImportError, RuntimeError, *self._ERRORS) as exc:
            record_degradation("procedural_memory", exc, severity="warning",
                               action="playbook persistence deferred")

    # ── capture ──────────────────────────────────────────────────────────
    def record_win(self, *, objective: str, task_type: str, answer: str,
                   strategy: str, verifiers: list[str], confidence: float,
                   problem_key: str = "") -> str:
        """Distill one verifier-clean win into a playbook (or reinforce the
        nearest existing one). Also credits any playbooks that were retrieved
        for this problem — demonstrated transfer is what earns distillation."""
        tokens = _signature_tokens(objective)
        if not tokens:
            return ""
        with self._lock:
            # transfer credit: the playbooks this problem was conditioned on
            for pid in self._retrieved_for.pop(problem_key or objective[:80], []):
                book = self._books.get(pid)
                if book is not None:
                    book.reuse_wins += 1

            best_id, best_overlap = "", 0.0
            for pid, book in self._books.items():
                if book.task_type != task_type:
                    continue
                b = set(book.tokens)
                overlap = len(tokens & b) / max(1, len(tokens | b))
                if overlap > best_overlap:
                    best_id, best_overlap = pid, overlap
            if best_overlap >= 0.6:            # same problem family → reinforce
                book = self._books[best_id]
                book.wins += 1
                book.confidence = max(book.confidence, float(confidence))
                self._persist()
                return book.playbook_id

            import uuid

            book = Playbook(
                playbook_id=f"pb-{uuid.uuid4().hex[:10]}",
                task_type=str(task_type or "generic"),
                tokens=sorted(tokens)[:24],
                objective_sample=str(objective or "")[:160],
                strategy=str(strategy or "")[:80],
                verifiers=[v for v in verifiers if v][:6],
                skeleton=_approach_skeleton(answer),
                confidence=float(confidence),
                created_at=time.time(),
            )
            self._books[book.playbook_id] = book
            self._evict_if_needed()
            self._persist()
            return book.playbook_id

    def _evict_if_needed(self) -> None:
        if len(self._books) <= _MAX_PLAYBOOKS:
            return
        # keep proven transferrers; evict stale one-hit wonders first
        ranked = sorted(self._books.values(),
                        key=lambda b: (b.reuse_wins, b.wins, b.created_at))
        for book in ranked[: len(self._books) - _MAX_PLAYBOOKS]:
            self._books.pop(book.playbook_id, None)

    # ── retrieval / injection ────────────────────────────────────────────
    def recall(self, objective: str, *, task_type: str | None = None,
               limit: int = 2, problem_key: str = "",
               record_usage: bool = True) -> list[Playbook]:
        tokens = _signature_tokens(objective)
        if not tokens:
            return []
        scored: list[tuple[float, Playbook]] = []
        with self._lock:
            for book in self._books.values():
                if task_type and book.task_type != task_type:
                    continue
                b = set(book.tokens)
                overlap = len(tokens & b) / max(1, len(tokens | b))
                if overlap >= 0.15:
                    quality = 0.1 * book.reuse_wins + 0.03 * book.wins
                    scored.append((overlap + quality, book))
            scored.sort(key=lambda t: -t[0])
            top = [b for _, b in scored[:limit]]
            if top and record_usage:
                for book in top:
                    book.reuses += 1
                self._retrieved_for[problem_key or objective[:80]] = [
                    b.playbook_id for b in top
                ]
                if len(self._retrieved_for) > 256:
                    self._retrieved_for.pop(next(iter(self._retrieved_for)))
        return top

    def as_playbook_text(self, objective: str, *, task_type: str | None = None,
                         limit: int = 2, problem_key: str = "",
                         record_usage: bool = True) -> str:
        books = self.recall(objective, task_type=task_type, limit=limit,
                            problem_key=problem_key,
                            record_usage=record_usage)
        if not books:
            return ""
        lines = ["Proven approaches from similar solved problems:"]
        for b in books:
            lines.append(
                f"- [{b.strategy or 'direct'}; verified by "
                f"{','.join(b.verifiers) or 'checks'}; {b.wins} wins] {b.skeleton}"
            )
        return "\n".join(lines)

    # ── weight-time compounding (foundry-gated) ──────────────────────────
    def export_distillation_batch(self, *, min_reuse_wins: int = 2,
                                  limit: int = 32,
                                  mark_distilled: bool = False) -> list[dict[str, str]]:
        """Playbooks with demonstrated transfer, in ADMITTED domains only,
        formatted for the governed train pipe.

        Does NOT mark them distilled. Building an in-memory batch is not
        evidence that anything downstream accepted it: the exporter used to set
        distilled=True and persist that while merely constructing the list, so a
        dropped return value or a failed trainer write permanently suppressed
        work that was never actually learned from. Call
        :meth:`confirm_distillation` once the batch has been accepted.

        ``mark_distilled=True`` restores the old fire-and-forget behaviour for
        callers that genuinely cannot report back; it is opt-in so the unsafe
        path is the one you have to ask for.
        """
        admitted_cache: dict[str, bool] = {}

        def _admitted(domain: str) -> bool:
            if domain not in admitted_cache:
                admitted_cache[domain] = self._domain_admitted(domain)
            return admitted_cache[domain]

        batch: list[dict[str, str]] = []
        with self._lock:
            for book in self._books.values():
                if book.distilled or book.reuse_wins < min_reuse_wins:
                    continue
                if not _admitted(book.task_type):
                    continue
                batch.append({
                    "playbook_id": book.playbook_id,
                    "prompt": (f"Task ({book.task_type}): "
                               f"{book.objective_sample}\nApproach?"),
                    "completion": (f"Strategy: {book.strategy}. "
                                   f"{book.skeleton}"),
                })
                if mark_distilled:
                    book.distilled = True
                if len(batch) >= limit:
                    break
            if batch and mark_distilled:
                self._persist(force=True)
        return batch

    def confirm_distillation(self, playbook_ids: Iterable[str]) -> int:
        """Mark playbooks distilled once something downstream accepted them.

        This is the receipt the exporter deliberately does not fabricate.
        Returns how many were newly marked, so a caller can tell a real
        confirmation from a no-op.
        """
        marked = 0
        with self._lock:
            for playbook_id in playbook_ids or ():
                book = self._books.get(str(playbook_id))
                if book is not None and not book.distilled:
                    book.distilled = True
                    marked += 1
            if marked:
                self._persist(force=True)
        return marked

    @staticmethod
    def _domain_admitted(task_type: str) -> bool:
        try:
            from core.runtime.service_access import optional_service

            foundry = optional_service("verifier_foundry", default=None)
            if foundry is None:
                return False   # weight-time compounding NEVER runs ungated
            return bool(foundry.domain_admitted(task_type).admitted)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("procedural_memory", exc, severity="warning",
                               action="distillation export blocked after foundry check failed")
            return False

    # ── observability ────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            books = list(self._books.values())
        return {
            "playbooks": len(books),
            "with_transfer": sum(1 for b in books if b.reuse_wins > 0),
            "distilled": sum(1 for b in books if b.distilled),
            "by_task_type": {
                tt: sum(1 for b in books if b.task_type == tt)
                for tt in sorted({b.task_type for b in books})
            },
        }

    def flush(self) -> None:
        with self._lock:
            self._persist(force=True)


_memory: ProceduralMemory | None = None
_memory_lock = threading.Lock()


def get_procedural_memory() -> ProceduralMemory:
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = ProceduralMemory()
    return _memory
