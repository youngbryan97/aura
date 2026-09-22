"""core/learning/deliberate_practice.py — the Practice Director.

Aura's learning stack has proven muscle with no proprioception: the self-play
flywheel practices verifiable tasks (uniformly), the domain-specialist trainer
grows adapter experts (least-recently-trained first), and the compounding loop
trains fused generations — but NOTHING decides what deserves practice. Real
receipts show the waste: the live 32B's own sealed eval scored program_output
0/5 and string_transform 1/5 while four domains sat at 5/5, and the flywheel
kept drilling all eight equally.

The Practice Director is the missing self-direction organ — deliberate
practice in the literal sense: study your own failures, drill exactly those,
and prove the improvement. It:

* **Observes** every real outcome that carries a domain — flywheel practice
  attempts (fed live per burst), sealed heldout evaluations from compounding
  runs, specialist gate receipts — each observation pinned to the receipt
  file it came from. No receipt, no observation.
* **Ranks** a curriculum: per-domain failure mass under exponential time
  decay (half-life 7 days), with honesty rails — a mastered domain
  (accuracy ≥ 0.95 with enough evidence) has zero need however old its
  ancient failures; a never-observed domain gets a fixed exploration floor
  rather than a fabricated score.
* **Directs, causally**: the flywheel asks it for a focused practice battery
  (concentrated on the top needs, with an exploration remainder); the
  specialist scheduler asks it which eligible domain to train next. Both
  degrade to their old uniform behavior if the director is disabled
  (AURA_DELIBERATE_PRACTICE=0) or has no evidence.
* **Explains itself**: ``why()`` renders the current direction with receipts,
  surfaced through the learning self-report ("why are you practicing X?"
  gets failure counts and file paths, not vibes) and /api/system/learning.

Honesty boundary: the director only directs domains whose outcomes are
mechanically verifiable (the heldout-battery domains). Conversational
failures (quality-gate exhaustions, user corrections) are a different
evidence stream and are NOT silently folded in as if drills could fix them.

Governance: the ledger lives under the runtime data dir, written through
the file-write gateway inside a governed scope; all consumption honors the
declared kill switch. Observation intake is in-memory and O(1); persistence
happens in ``flush()`` (worker threads) or ``flush_async()`` — never a sync
write on the event loop.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.flags import FlagKind, declare

logger = logging.getLogger("Aura.DeliberatePractice")

SERVICE_NAME = "practice_director"

_ENABLED_FLAG = declare(
    "AURA_DELIBERATE_PRACTICE",
    kind=FlagKind.BOOL,
    default=True,
    description=(
        "Practice Director: failure-directed curriculum steering the self-play "
        "flywheel's practice mix and the domain-specialist scheduler's next-domain "
        "choice. Off = observation continues, consumers fall back to uniform/LRT."
    ),
    owner="core/learning/deliberate_practice.py",
)

_RECOVERABLE = (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError)

# Evidence decay: a failure last week matters half as much as one today.
_HALF_LIFE_DAYS = 7.0
# Below this decayed attempt mass a domain is "unobserved" — we do not
# pretend to know its accuracy from two data points.
_MIN_EVIDENCE_MASS = 3.0
# Mastered = provably strong; ancient failures must not haunt the ranking.
_MASTERY_ACCURACY = 0.95
# Unobserved domains get a fixed exploration need, not a fabricated score.
_EXPLORATION_NEED = 0.25
# Confidence saturates once this much decayed attempt mass exists.
_CONFIDENCE_MASS = 10.0
# Ledger bounds: compact to the newest half when the file crosses the cap.
_LEDGER_MAX_LINES = 5000
_SEEN_RECEIPTS_MAX = 500

_LEDGER_NAME = "practice_curriculum.jsonl"
_SEEN_NAME = "practice_curriculum_seen.json"


@dataclass(frozen=True)
class PracticeNeed:
    """One domain's ranked practice need, with the receipts behind it."""

    domain: str
    need: float  # 0..1 — rank key
    accuracy: float | None  # decayed observed accuracy; None = unobserved
    attempts: float  # decayed attempt mass behind the estimate
    evidence: tuple[str, ...]  # receipt identifiers, newest first
    reason: str  # one honest human sentence

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "need": round(self.need, 4),
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "attempts": round(self.attempts, 2),
            "evidence": list(self.evidence[:5]),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _Observation:
    at: float
    domain: str
    attempts: int
    correct: int
    source: str
    receipt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "domain": self.domain,
            "attempts": self.attempts,
            "correct": self.correct,
            "source": self.source,
            "receipt": self.receipt,
        }


class PracticeDirector:
    """Failure-directed curriculum over the verifiable practice domains."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        now: Callable[[], float] = lambda: time.time(),
    ) -> None:
        if data_dir is None:
            from core.config import get_config

            data_dir = Path(get_config().paths.data_dir) / "learning"
        self._dir = Path(data_dir)
        self._ledger_path = self._dir / _LEDGER_NAME
        self._seen_path = self._dir / _SEEN_NAME
        self._now = now
        self._lock = threading.Lock()
        self._observations: list[_Observation] = []
        self._pending: list[_Observation] = []
        self._seen_receipts: dict[str, float] = {}
        self._loaded = False

    # ── lifecycle ────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(_ENABLED_FLAG.value())

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        observations: list[_Observation] = []
        try:
            if self._ledger_path.exists():
                for line in self._ledger_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        observations.append(
                            _Observation(
                                at=float(row["at"]),
                                domain=str(row["domain"]),
                                attempts=int(row["attempts"]),
                                correct=int(row["correct"]),
                                source=str(row.get("source", "")),
                                receipt=str(row.get("receipt", "")),
                            )
                        )
                    except (ValueError, TypeError, KeyError):
                        continue  # one corrupt line must not void the ledger
        except OSError as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="started with an empty practice ledger after read failure",
                classification=FallbackClassification.SAFE_FALLBACK,
            )
        seen: dict[str, float] = {}
        try:
            if self._seen_path.exists():
                raw = json.loads(self._seen_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    seen = {str(k): float(v) for k, v in raw.items()}
        except _RECOVERABLE:
            seen = {}
        with self._lock:
            self._observations = observations
            self._seen_receipts = seen

    # ── intake ───────────────────────────────────────────────────────────

    def observe(
        self,
        *,
        domain: str,
        attempts: int,
        correct: int,
        source: str,
        receipt: str,
    ) -> None:
        """Record one domain outcome. In-memory and O(1); persist via flush."""
        domain = str(domain or "").strip()
        if not domain or attempts <= 0:
            return
        self._ensure_loaded()
        observation = _Observation(
            at=float(self._now()),
            domain=domain,
            attempts=int(attempts),
            correct=max(0, min(int(correct), int(attempts))),
            source=str(source),
            receipt=str(receipt),
        )
        with self._lock:
            self._observations.append(observation)
            self._pending.append(observation)

    def observe_burst(
        self,
        domain_results: dict[str, tuple[int, int]],
        *,
        source: str,
        receipt: str,
    ) -> None:
        """Batch intake for a practice burst: {domain: (attempts, correct)}."""
        for domain, (attempts, correct) in sorted(domain_results.items()):
            self.observe(
                domain=domain,
                attempts=attempts,
                correct=correct,
                source=source,
                receipt=receipt,
            )

    # ── harvest: sweep external receipts ─────────────────────────────────

    def harvest(self) -> int:
        """Sweep receipt files the learning stack already writes — sealed
        heldout evaluations from compounding runs and specialist gate
        receipts — into observations. Idempotent per receipt file. Blocking
        (file I/O): call from a worker thread."""
        self._ensure_loaded()
        new = 0
        new += self._harvest_eval_reports()
        new += self._harvest_specialist_receipts()
        if new:
            self.flush()
        return new

    def _harvest_eval_reports(self) -> int:
        runs_dir = self._dir / "compounding" / "runs"
        if not runs_dir.is_dir():
            return 0
        new = 0
        try:
            report_paths = sorted(runs_dir.glob("*/*_eval.json"))
        # not a failure: a directory that will not list holds no new reports.
        except OSError:
            return 0
        for path in report_paths[-50:]:
            receipt = str(path)
            with self._lock:
                if receipt in self._seen_receipts:
                    continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                per_domain = dict(
                    (report.get("result") or {}).get("per_domain") or {}
                )
                created_at = float(report.get("created_at") or path.stat().st_mtime)
            except _RECOVERABLE:
                with self._lock:
                    self._seen_receipts[receipt] = float(self._now())
                continue
            for domain, bucket in sorted(per_domain.items()):
                try:
                    total = int(bucket.get("total", 0))
                    correct = int(bucket.get("correct", 0))
                except (TypeError, ValueError, AttributeError):
                    continue
                if total <= 0:
                    continue
                observation = _Observation(
                    at=created_at,
                    domain=str(domain),
                    attempts=total,
                    correct=max(0, min(correct, total)),
                    source="heldout_eval",
                    receipt=receipt,
                )
                with self._lock:
                    self._observations.append(observation)
                    self._pending.append(observation)
                new += 1
            with self._lock:
                self._seen_receipts[receipt] = float(self._now())
        return new

    def _harvest_specialist_receipts(self) -> int:
        specialists_dir = self._dir / "specialists"
        if not specialists_dir.is_dir():
            return 0
        new = 0
        try:
            receipt_paths = sorted(specialists_dir.rglob("receipt*.json"))
        # not a failure: a directory that will not list holds no new receipts.
        except OSError:
            return 0
        for path in receipt_paths[-50:]:
            receipt = str(path)
            with self._lock:
                if receipt in self._seen_receipts:
                    continue
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                domain = str(report.get("domain", "") or "")
                accuracy = report.get("domain_candidate_accuracy")
                battery_size = int(report.get("battery_size", 0) or 0)
                created_at = float(report.get("created_at") or path.stat().st_mtime)
            except _RECOVERABLE:
                with self._lock:
                    self._seen_receipts[receipt] = float(self._now())
                continue
            if domain and accuracy is not None and battery_size > 0:
                correct = int(round(float(accuracy) * battery_size))
                observation = _Observation(
                    at=created_at,
                    domain=domain,
                    attempts=battery_size,
                    correct=max(0, min(correct, battery_size)),
                    source="specialist_gate",
                    receipt=receipt,
                )
                with self._lock:
                    self._observations.append(observation)
                    self._pending.append(observation)
                new += 1
            with self._lock:
                self._seen_receipts[receipt] = float(self._now())
        return new

    # ── the curriculum ───────────────────────────────────────────────────

    def curriculum(self, *, top: int = 8) -> list[PracticeNeed]:
        """Ranked practice needs across every known domain, receipts attached."""
        self._ensure_loaded()
        from core.learning.heldout_battery import BatterySpec, generate_battery

        known_domains = sorted(
            {task.domain for task in generate_battery(BatterySpec(seed=0, size=16))}
        )
        with self._lock:
            observations = list(self._observations)
        now = float(self._now())

        by_domain: dict[str, list[_Observation]] = {d: [] for d in known_domains}
        for observation in observations:
            by_domain.setdefault(observation.domain, []).append(observation)

        needs: list[PracticeNeed] = []
        for domain, domain_observations in sorted(by_domain.items()):
            mass = 0.0
            weighted_correct = 0.0
            for observation in domain_observations:
                age_days = max(0.0, (now - observation.at) / 86400.0)
                weight = math.pow(0.5, age_days / _HALF_LIFE_DAYS)
                mass += weight * observation.attempts
                weighted_correct += weight * observation.correct
            evidence = tuple(
                observation.receipt
                for observation in sorted(
                    domain_observations, key=lambda item: item.at, reverse=True
                )
                if observation.receipt
            )[:5]
            if mass < _MIN_EVIDENCE_MASS:
                needs.append(
                    PracticeNeed(
                        domain=domain,
                        need=_EXPLORATION_NEED,
                        accuracy=None,
                        attempts=round(mass, 2),
                        evidence=evidence,
                        reason=(
                            f"not enough recent evidence for {domain} "
                            f"(decayed mass {mass:.1f} < {_MIN_EVIDENCE_MASS:.0f}) — explore"
                        ),
                    )
                )
                continue
            accuracy = weighted_correct / mass
            if accuracy >= _MASTERY_ACCURACY:
                needs.append(
                    PracticeNeed(
                        domain=domain,
                        need=0.0,
                        accuracy=accuracy,
                        attempts=round(mass, 2),
                        evidence=evidence,
                        reason=f"{domain} is holding at {accuracy:.0%} — maintain, don't drill",
                    )
                )
                continue
            confidence = min(1.0, mass / _CONFIDENCE_MASS)
            need = (1.0 - accuracy) * confidence
            needs.append(
                PracticeNeed(
                    domain=domain,
                    need=need,
                    accuracy=accuracy,
                    attempts=round(mass, 2),
                    evidence=evidence,
                    reason=(
                        f"{domain} at {accuracy:.0%} over ~{mass:.0f} recent verified "
                        f"attempts — drill it"
                    ),
                )
            )
        needs.sort(key=lambda item: (-item.need, item.domain))
        return needs[: max(1, int(top))]

    # ── causal consumers ─────────────────────────────────────────────────

    def choose_focus_domain(self, eligible: list[str]) -> str | None:
        """The highest-need domain among ``eligible`` (specialist scheduler
        seam). None when disabled or no eligible domain has a real need —
        the caller keeps its least-recently-trained fallback."""
        if not self.enabled or not eligible:
            return None
        eligible_set = set(eligible)
        for need in self.curriculum(top=32):
            if need.domain in eligible_set and need.need > 0.0 and need.accuracy is not None:
                return need.domain
        return None

    def focused_battery(self, *, seed: int, size: int) -> list[Any]:
        """A practice battery concentrated on the top needs (flywheel seam):
        half the tasks from the top-need domain, a quarter from the second,
        the remainder uniform exploration. Deterministic in (seed, size,
        current curriculum). Falls back to the uniform battery when disabled
        or evidence-free."""
        from core.learning.heldout_battery import BatterySpec, generate_battery

        uniform = cast(list[Any], generate_battery(BatterySpec(seed=seed, size=size)))
        if not self.enabled or size < 4:
            return uniform
        ranked = [
            need
            for need in self.curriculum(top=4)
            if need.need > 0.0 and need.accuracy is not None
        ]
        if not ranked:
            return uniform

        pool = cast(
            list[Any], generate_battery(BatterySpec(seed=seed, size=size * 10))
        )
        quotas: list[tuple[str, int]] = []
        first_quota = size // 2
        quotas.append((ranked[0].domain, first_quota))
        second_quota = size // 4 if len(ranked) > 1 else 0
        if second_quota:
            quotas.append((ranked[1].domain, second_quota))

        chosen: list[Any] = []
        chosen_ids: set[str] = set()
        for domain, quota in quotas:
            for task in pool:
                if quota <= 0:
                    break
                if task.domain == domain and task.task_id not in chosen_ids:
                    chosen.append(task)
                    chosen_ids.add(task.task_id)
                    quota -= 1
        # Exploration remainder: keep the uniform stream's coverage.
        for task in uniform:
            if len(chosen) >= size:
                break
            if task.task_id not in chosen_ids:
                chosen.append(task)
                chosen_ids.add(task.task_id)
        # Top up from the pool if dedup starved the count (tiny batteries).
        for task in pool:
            if len(chosen) >= size:
                break
            if task.task_id not in chosen_ids:
                chosen.append(task)
                chosen_ids.add(task.task_id)
        return chosen[:size]

    # ── self-knowledge ───────────────────────────────────────────────────

    def why(self) -> str:
        """The current practice direction as honest sentences with receipts."""
        needs = self.curriculum(top=3)
        drilling = [n for n in needs if n.need > 0.0 and n.accuracy is not None]
        if not drilling:
            observed = [n for n in needs if n.accuracy is not None]
            if observed:
                return (
                    "No domain currently needs drilling — every observed domain "
                    "is holding at mastery. Practice stays exploratory."
                )
            return (
                "No verified practice evidence yet — practicing uniformly "
                "until real outcomes accumulate."
            )
        lines = []
        for need in drilling:
            line = need.reason
            if need.evidence:
                line += f" (receipt: {Path(need.evidence[0]).name})"
            lines.append(line)
        return "Practice is failure-directed: " + "; ".join(lines) + "."

    def get_status(self) -> dict[str, Any]:
        self._ensure_loaded()
        with self._lock:
            observation_count = len(self._observations)
            pending = len(self._pending)
        return {
            "schema": "aura.practice_director.v1",
            "enabled": self.enabled,
            "observations": observation_count,
            "pending_flush": pending,
            "ledger_path": str(self._ledger_path),
            "top_needs": [need.to_dict() for need in self.curriculum(top=4)],
            "direction": self.why(),
        }

    # ── persistence ──────────────────────────────────────────────────────

    def _serialize_pending(self) -> tuple[str, str, bool] | None:
        """Snapshot pending state → (ledger_append_or_full, seen_json,
        full_rewrite). Returns None when nothing needs writing."""
        with self._lock:
            if not self._pending and not self._seen_receipts:
                return None
            pending = list(self._pending)
            self._pending = []
            observations = list(self._observations)
            if len(self._seen_receipts) > _SEEN_RECEIPTS_MAX:
                newest = sorted(
                    self._seen_receipts.items(), key=lambda item: -item[1]
                )[: _SEEN_RECEIPTS_MAX // 2]
                self._seen_receipts = dict(newest)
            seen_json = json.dumps(self._seen_receipts, indent=1, sort_keys=True)
        if len(observations) > _LEDGER_MAX_LINES:
            keep = observations[-(_LEDGER_MAX_LINES // 2) :]
            with self._lock:
                self._observations = keep
            body = "\n".join(json.dumps(o.to_dict(), sort_keys=True) for o in keep)
            return (body + "\n" if body else "", seen_json, True)
        body = "\n".join(json.dumps(o.to_dict(), sort_keys=True) for o in pending)
        return (body + "\n" if body else "", seen_json, False)

    def flush(self) -> None:
        """Persist pending observations. Blocking; call off the event loop."""
        snapshot = self._serialize_pending()
        if snapshot is None:
            return
        body, seen_json, full_rewrite = snapshot
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.receipts import PracticeCurriculumStore

            with local_internal_governed_scope(
                "deliberate_practice.ledger", domain="memory_write"
            ):
                PracticeCurriculumStore(self._dir).persist(
                    ledger_body=body,
                    seen_json=seen_json,
                    full_rewrite=full_rewrite,
                )
        except _RECOVERABLE as exc:
            record_degradation(
                SERVICE_NAME,
                exc,
                action="continued with unpersisted practice observations",
                classification=FallbackClassification.SAFE_FALLBACK,
            )

    async def flush_async(self) -> None:
        import asyncio

        await asyncio.to_thread(self.flush)


# ── singleton ────────────────────────────────────────────────────────────

_director: PracticeDirector | None = None
_director_lock = threading.Lock()


def get_practice_director() -> PracticeDirector:
    global _director
    with _director_lock:
        if _director is None:
            _director = PracticeDirector()
        return _director


def set_practice_director_for_test(director: PracticeDirector | None) -> None:
    global _director
    with _director_lock:
        _director = director


__all__ = [
    "PracticeDirector",
    "PracticeNeed",
    "get_practice_director",
    "set_practice_director_for_test",
]
