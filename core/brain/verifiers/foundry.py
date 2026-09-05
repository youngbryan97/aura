"""Verifier Foundry — measured verifier reliability and the self-training gate.
================================================================================
P1 of the frontier-general arc (docs/FRONTIER_GENERAL_ARC.md).

The ceiling on a self-improving local mind is the boundary of verification:
she may only compound (train on) what she can check, or she amplifies her own
garbage. That boundary is movable — but only honestly if every verifier's
actual reliability is MEASURED against later ground truth, and admission to
the self-training loop is granted by evidence, not by assumption.

The foundry is that bookkeeper and gatekeeper:

  * every truth-engine verdict can be recorded (verifier, domain, hard pass,
    soft score) and later GRADED when reality arrives — an audit, a resolved
    prediction, an action outcome;
  * per (verifier, domain) reliability is tracked with pessimistic statistics
    (Wilson lower bound on accuracy; false-pass rate, the metric that poisons
    training data, bounded from above);
  * ``domain_admitted()`` answers the only question the training pipe may ask:
    has this domain's verification EARNED the right to mint training data?
  * ``weight_for()`` gives the registry reliability weights for soft-score
    folding (the hard gate is never softened — a provable failure is final);
  * classically deterministic domains (code, math, logic) are seed-admitted —
    their engines are correct by construction — but evidence still accumulates
    and can REVOKE an admission: the gate ratchets on measurement, not faith.

Ledger: event-sourced (events.jsonl fold) + AuditChain tamper evidence, the
same proven pattern as the Ulysses covenant — a mind must not be able to
quietly rewrite the record of how trustworthy its own checkers are.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.audit_chain import AuditChain
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.VerifierFoundry")

SCHEMA_VERSION = 1

_FOUNDRY_DIR_FLAG = declare(
    "AURA_FOUNDRY_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for the verifier-foundry reliability ledger",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MIN_GRADED_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MIN_GRADED", kind=FlagKind.INT, default=50,
    description="Graded verdicts required before a domain can earn admission",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MIN_WILSON_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MIN_WILSON", kind=FlagKind.FLOAT, default=0.85,
    description="Wilson lower bound on accuracy required for admission",
    owner="core.brain.verifiers.foundry",
)
_ADMIT_MAX_FALSE_PASS_FLAG = declare(
    "AURA_FOUNDRY_ADMIT_MAX_FALSE_PASS", kind=FlagKind.FLOAT, default=0.05,
    description="Upper bound (Wilson) on false-pass rate tolerated for admission",
    owner="core.brain.verifiers.foundry",
)

# Domains whose engines are deterministic checkers (execution, proof, unit
# arithmetic): admitted from birth, revocable by evidence.
SEED_ADMITTED_DOMAINS = frozenset({"code", "math", "logic"})

#: Ground-truth channels allowed to grade a verdict. CP126 cd3bd98e:
#: grade_verdict accepted a free-form ``source`` string with no allowlist,
#: grader identity, or independence check — so anything that could call the
#: method could move reliability and admission by asserting a boolean.
#: These are the channels that carry their own evidence trail; a caller
#: naming anything else is refused rather than silently trusted.
TRUSTED_GRADE_SOURCES = frozenset({
    "audit",                 # governed audit trail
    "prediction_resolution", # a prediction that later resolved
    "action_outcome",        # an executed action's measured result
    "human",                 # explicit human adjudication
    "task_verifier",         # an independent task-level verifier
    "frontier_battery",      # the frontier-gap battery's known-answer grading
})

_Z95 = 1.6449  # one-sided 95%


def wilson_lower_bound(successes: int, n: int, z: float = _Z95) -> float:
    """Pessimistic estimate of a success rate: the Wilson score interval's
    lower bound. With no evidence it is 0 — trust must be earned."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def wilson_upper_bound(successes: int, n: int, z: float = _Z95) -> float:
    """Pessimistic (i.e. large) estimate of a failure-mode rate."""
    if n <= 0:
        return 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4 * n)) / n)
    return min(1.0, (centre + margin) / denom)


@dataclass
class ReliabilityCell:
    """Running reliability of one (verifier, domain) pair."""

    verifier: str
    domain: str
    recorded: int = 0
    graded: int = 0
    correct: int = 0
    passes: int = 0          # verdicts that said pass, among graded
    false_passes: int = 0    # said pass, truth was fail — the poison metric
    false_fails: int = 0     # said fail, truth was pass — wasted compute
    brier_sum: float = 0.0

    def accuracy_lb(self) -> float:
        return wilson_lower_bound(self.correct, self.graded)

    def false_pass_ub(self) -> float:
        """Upper bound on P(truth=fail | verdict=pass) — bounded pessimistically
        over the verdicts that could have poisoned training data."""
        if self.passes <= 0:
            return 1.0
        return wilson_upper_bound(self.false_passes, self.passes)

    def brier(self) -> float:
        return self.brier_sum / self.graded if self.graded else 1.0

    def snapshot(self) -> ReliabilityCell:
        """A detached copy for callers.

        ``reliability()`` used to return the LIVE cell; any caller could then
        mutate graded/correct/passes/false_passes directly, bypassing the
        lock, the event log, and the audit chain — corrupting governance
        state with no receipt. Observers get a copy instead.
        """
        return ReliabilityCell(
            verifier=self.verifier,
            domain=self.domain,
            recorded=self.recorded,
            graded=self.graded,
            correct=self.correct,
            passes=self.passes,
            false_passes=self.false_passes,
            false_fails=self.false_fails,
            brier_sum=self.brier_sum,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verifier": self.verifier,
            "domain": self.domain,
            "recorded": self.recorded,
            "graded": self.graded,
            "correct": self.correct,
            "false_passes": self.false_passes,
            "false_fails": self.false_fails,
            "accuracy_lb": round(self.accuracy_lb(), 4),
            "false_pass_ub": round(self.false_pass_ub(), 4),
            "brier": round(self.brier(), 4),
        }


@dataclass(frozen=True)
class AdmissionDecision:
    domain: str
    admitted: bool
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


class VerifierFoundry:
    """Reliability ledger + admission gate. Thread-safe; durable off-loop."""

    def __init__(self, *, root: Path | None = None,
                 clock: Callable[[], float] = time.time):
        env_root = str(_FOUNDRY_DIR_FLAG.value() or "")
        self.root = Path(root) if root else (
            Path(env_root) if env_root
            else (state_root() / "data" / "verifier_foundry")
        )
        self._ensure_root()
        self.events_path = self.root / "events.jsonl"
        self._chain = AuditChain(self.root / "chain")
        self._clock = clock
        self._lock = threading.RLock()

        self._cells: dict[tuple[str, str], ReliabilityCell] = {}
        self._pending: dict[str, dict[str, Any]] = {}   # verdict_id → verdict
        self._pending_order: list[str] = []
        self._revoked_seeds: set[str] = set()
        self._restore_errors = 0
        # Events folded into memory whose durable write did not complete
        # (CP126 1cd91c2e / 9d1a2e8c). Non-empty ⇒ in-memory governance state
        # is not backed by the tamper-evident ledger.
        self._unpersisted_events: set[str] = set()
        # Verdicts already graded — a grade is a one-shot input (CP126 cd3bd98e).
        self._graded_verdicts: set[str] = set()

        # durable writes on a dedicated thread — same no-on-loop-fsync
        # discipline as the covenant ledger
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._pending_writes = 0
        self._pending_writes_lock = threading.Lock()
        self._writer = threading.Thread(target=self._writer_loop,
                                        name="verifier-foundry-ledger", daemon=True)
        self._writer_started = False
        self._writer_running = True
        self._closed = False
        self._restore()

    # ── storage plumbing ─────────────────────────────────────────────────
    def _ensure_root(self) -> None:
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("verifier_foundry",
                                               domain="state_mutation"):
                get_file_write_gateway().ensure_directory(
                    self.root, source="verifier_foundry")
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry root could not be created")

    def _append_event(self, kind: str, body: dict[str, Any]) -> None:
        # A closed foundry has no live writer thread. Folding and queueing an
        # event here would mutate reliability state that will never be made
        # durable — a silent divergence between memory and the ledger.
        if self._closed:
            raise RuntimeError("verifier_foundry_closed")
        body = dict(body)
        body["event"] = kind
        body["event_id"] = body.get("event_id") or f"vf-{uuid.uuid4().hex[:12]}"
        body.setdefault("timestamp", self._clock())
        self._fold(body)
        if not self._writer_started:
            self._writer_started = True
            self._writer.start()
        with self._pending_writes_lock:
            self._pending_writes += 1
        self._queue.put(body)

    def _writer_loop(self) -> None:
        while self._writer_running:
            item = self._queue.get()
            if item is None:
                self._writer_running = False
                continue
            try:
                self._persist_event(item)
            finally:
                with self._pending_writes_lock:
                    self._pending_writes -= 1

    def _persist_event(self, body: dict[str, Any]) -> None:
        line = json.dumps(body, sort_keys=True, ensure_ascii=False,
                          default=str) + "\n"
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("verifier_foundry",
                                               domain="state_mutation"):
                get_file_write_gateway().append_text(
                    self.events_path, line, source="verifier_foundry")
                self._chain.append(receipt_id=str(body["event_id"]),
                                   kind=f"foundry_{body['event']}",
                                   body=body,
                                   timestamp=float(body["timestamp"]))
            with self._lock:
                self._unpersisted_events.discard(str(body["event_id"]))
        except (OSError, RuntimeError, ImportError, ValueError) as exc:
            # CP126 1cd91c2e + 9d1a2e8c. The event was ALREADY folded into
            # live reliability state and callers already hold verdict ids and
            # admission decisions derived from it, but it is not in the ledger
            # (or its body landed without a matching chain record). Swallowing
            # that as a degradation left memory governing admission while the
            # durable, tamper-evident record disagreed.
            #
            # We cannot un-ring the bell for callers, so we do the honest
            # thing: mark the divergence and let admission FAIL CLOSED until
            # the ledger is reconciled.
            with self._lock:
                self._unpersisted_events.add(str(body["event_id"]))
            record_degradation("verifier_foundry", exc, severity="critical",
                               action=("foundry event held in memory only; "
                                       "admission fails closed until reconciled"))

    def flush_ledger(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._pending_writes_lock:
                if self._pending_writes <= 0:
                    return True
            time.sleep(0.005)
        with self._pending_writes_lock:
            return self._pending_writes <= 0

    async def flush_ledger_async(self, timeout: float = 5.0) -> bool:
        """Await the ledger drain without blocking the event loop.

        CP126 750942aa. flush_ledger polls with time.sleep, so an async
        caller stalls every other task on the loop for up to `timeout`
        seconds — the failure mode that once froze this runtime for twenty
        minutes on an on-loop fsync. Async callers must use this.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            with self._pending_writes_lock:
                if self._pending_writes <= 0:
                    return True
            # asyncio.sleep yields the loop; time.sleep holds it.
            await asyncio.sleep(0.005)
        with self._pending_writes_lock:
            return self._pending_writes <= 0

    async def verify_ledger_async(self) -> tuple[bool, list[dict[str, Any]]]:
        """Verify the ledger off the event loop.

        CP126 750942aa. verify_ledger reads and parses the ENTIRE event
        file synchronously, so its cost scales with the whole ledger; on the
        loop that is an unbounded stall. The drain is awaited here and the
        parse runs in a worker thread.
        """
        await self.flush_ledger_async()
        return await asyncio.to_thread(self._verify_ledger_locked)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.flush_ledger()
        if self._writer_started and self._writer.is_alive():
            self._queue.put(None)
            self._writer.join(timeout=2.0)
        self._chain.close()

    def _restore(self) -> None:
        if not self.events_path.exists():
            return
        try:
            raw = self.events_path.read_text(encoding="utf-8")
        except OSError as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry ledger unreadable; starting empty")
            return
        # AUDIT-CHAIN FIRST. Folding raw event bodies straight into live
        # reliability state let any edited, deleted, injected, reordered, or
        # replayed line poison governance — the chain verification was
        # optional and separate. We verify the chain against the on-disk
        # bodies and fold ONLY chain-confirmed events, in chain order.
        parsed: dict[str, dict[str, Any]] = {}
        line_order: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._restore_errors += 1
                record_degradation("verifier_foundry", exc, severity="warning",
                                   action="skipped corrupt foundry event line")
                continue
            if not isinstance(event, dict):
                self._restore_errors += 1
                continue
            eid = str(event.get("event_id", ""))
            if not eid:
                self._restore_errors += 1
                continue
            parsed[eid] = event
            line_order.append(eid)

        verified_ids = self._chain_verified_event_ids(parsed)
        if verified_ids is None:
            # The chain itself could not be read/verified: fold nothing and
            # surface it, rather than trusting unverifiable bodies.
            record_degradation(
                "verifier_foundry",
                RuntimeError("foundry audit chain unverifiable at restore"),
                severity="critical",
                action="started with empty reliability state; ledger not trusted",
            )
            return
        seen: set[str] = set()
        for eid in line_order:
            if eid not in verified_ids or eid in seen:
                # Unverified body OR a duplicate id: not admissible evidence.
                if eid not in verified_ids:
                    self._restore_errors += 1
                continue
            seen.add(eid)
            try:
                self._fold(parsed[eid])
            except (KeyError, TypeError, ValueError) as exc:
                self._restore_errors += 1
                record_degradation("verifier_foundry", exc, severity="warning",
                                   action="skipped malformed verified foundry event")

    def _chain_verified_event_ids(
        self, bodies: dict[str, dict[str, Any]]
    ) -> set[str] | None:
        """Return the event ids whose on-disk body matches the audit chain.

        None means the chain could not be verified at all (treat everything
        as untrusted). A verified-but-mismatched body is simply excluded.
        """
        # The body_loader is called once per record the CHAIN contains, so it
        # doubles as the authoritative set of chain-present ids. An event in
        # events.jsonl that the chain never asks about was injected after the
        # fact and is not in the chain — membership, not absence-of-problem,
        # is what makes a body admissible.
        chain_ids: set[str] = set()

        def _loader(rid: str, kind: str) -> dict[str, Any] | None:
            chain_ids.add(str(rid))
            return bodies.get(rid)

        try:
            ok, problems = self._chain.verify(body_loader=_loader)
        except (OSError, RuntimeError, ValueError) as exc:
            record_degradation("verifier_foundry", exc, severity="critical",
                               action="foundry audit-chain verification raised")
            return None
        bad_ids = {
            str(problem.get("receipt_id", ""))
            for problem in problems
            if str(problem.get("receipt_id", ""))
        }
        if not ok and not bad_ids:
            # Chain broken with no attributable receipt (e.g. a broken link):
            # trust nothing.
            return None
        # Verified = present in the chain AND not flagged as a problem.
        return {eid for eid in bodies if eid in chain_ids and eid not in bad_ids}

    _MAX_PENDING = 5000

    def _fold(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        if kind == "verdict":
            vid = str(event.get("verdict_id", ""))
            cell = self._cell(str(event.get("verifier", "?")),
                              str(event.get("domain", "?")))
            cell.recorded += 1
            if vid:
                self._pending[vid] = {
                    "verifier": cell.verifier,
                    "domain": cell.domain,
                    "hard_pass": bool(event.get("hard_pass")),
                    "score": float(event.get("score", 0.5)),
                }
                self._pending_order.append(vid)
                while len(self._pending_order) > self._MAX_PENDING:
                    old = self._pending_order.pop(0)
                    self._pending.pop(old, None)
        elif kind == "grade":
            vid = str(event.get("verdict_id", ""))
            verdict = self._pending.pop(vid, None)
            if verdict is None:
                return
            # Keep pending_order consistent with _pending: leaving the graded
            # id in the order list made pending_verdicts() return stale ids
            # indefinitely (until unrelated capacity eviction) and allowed a
            # duplicate grade to be processed differently after the first pop.
            try:
                self._pending_order.remove(vid)
            except ValueError:
                pass
            truth = bool(event.get("truth_pass"))
            cell = self._cell(verdict["verifier"], verdict["domain"])
            cell.graded += 1
            said_pass = verdict["hard_pass"]
            if said_pass:
                cell.passes += 1
            if said_pass == truth:
                cell.correct += 1
            elif said_pass and not truth:
                cell.false_passes += 1
            else:
                cell.false_fails += 1
            cell.brier_sum += (verdict["score"] - (1.0 if truth else 0.0)) ** 2
        elif kind == "revoke_seed":
            self._revoked_seeds.add(str(event.get("domain", "")))

    def _cell(self, verifier: str, domain: str) -> ReliabilityCell:
        key = (verifier, domain)
        cell = self._cells.get(key)
        if cell is None:
            cell = ReliabilityCell(verifier=verifier, domain=domain)
            self._cells[key] = cell
        return cell

    # ── recording and grading ────────────────────────────────────────────
    def record_verdict(self, *, verifier: str, domain: str, hard_pass: bool,
                       score: float, checked: bool, task_key: str = "",
                       meta: dict[str, Any] | None = None) -> str:
        """Record one engine verdict; returns the verdict_id used to grade it
        later. Unchecked verdicts (engine had nothing to verify) are not
        reliability evidence and are not recorded."""
        if not checked:
            return ""
        verdict_id = f"vd-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if self._closed:
                return ""
            self._append_event("verdict", {
                "verdict_id": verdict_id,
                "verifier": str(verifier or "?"),
                "domain": str(domain or "?"),
                "hard_pass": bool(hard_pass),
                "score": max(0.0, min(1.0, float(score))),
                "task_key": str(task_key or "")[:64],
                "meta": dict(meta or {}),
            })
        return verdict_id

    def grade_verdict(self, verdict_id: str, *, truth_pass: bool,
                      source: str) -> bool:
        """Reality arrived: grade a recorded verdict against ground truth.

        ``source`` must name a TRUSTED ground-truth channel
        (:data:`TRUSTED_GRADE_SOURCES`). CP126 cd3bd98e: this accepted any
        free-form string, so a grade — the input that moves reliability and
        therefore admission — could come from an unidentified caller with no
        evidence trail behind it. An unrecognized source is refused and
        receipted, not recorded as truth.

        A verdict is also graded ONCE: re-grading the same verdict from a
        second source would let a caller move reliability by repetition.
        """
        channel = str(source or "").strip().lower()[:64]
        if channel not in TRUSTED_GRADE_SOURCES:
            record_degradation(
                "verifier_foundry",
                PermissionError(f"untrusted_grade_source:{channel or 'unset'}"),
                severity="error",
                action="refused a ground-truth grade from an unrecognized channel",
            )
            return False
        with self._lock:
            if self._closed or verdict_id not in self._pending:
                return False
            if verdict_id in self._graded_verdicts:
                record_degradation(
                    "verifier_foundry",
                    ValueError(f"duplicate_grade:{verdict_id}"),
                    severity="warning",
                    action="refused a second grade for an already-graded verdict",
                )
                return False
            self._graded_verdicts.add(verdict_id)
            self._append_event("grade", {
                "verdict_id": verdict_id,
                "truth_pass": bool(truth_pass),
                "source": channel,
            })
        return True

    def pending_verdicts(self, *, domain: str | None = None) -> list[str]:
        with self._lock:
            if domain is None:
                return list(self._pending_order)
            return [v for v in self._pending_order
                    if self._pending.get(v, {}).get("domain") == domain]

    # ── reliability and folding weights ──────────────────────────────────
    def reliability(self, verifier: str, domain: str) -> ReliabilityCell:
        # A DETACHED snapshot: the live cell must never leave the lock, or a
        # caller can mutate durable reliability state with no audit receipt.
        with self._lock:
            return self._cell(verifier, domain).snapshot()

    _WEIGHT_FLOOR = 0.25

    # An unmeasured verifier is not a trusted one. Giving it full weight
    # (1.0) was an OPTIMISTIC prior that let an uncalibrated or brand-new
    # verifier materially move soft scores before it had earned any evidence.
    # A skeptical prior weights it below a proven-good verifier until it has
    # graded evidence of its own, without silencing it entirely.
    _UNMEASURED_WEIGHT = 0.5

    def weight_for(self, verifier: str, domain: str) -> float:
        """Soft-score folding weight. Unmeasured verifiers get a SKEPTICAL
        prior (below a proven verifier, above the bad-verifier floor);
        measured ones are weighted by their pessimistic accuracy, floored so
        a bad verifier is muted, never inverted. The HARD gate is never
        weighted."""
        with self._lock:
            cell = self._cells.get((verifier, domain))
            graded = cell.graded if cell is not None else 0
            acc_lb = cell.accuracy_lb() if cell is not None else 0.0
        if cell is None or graded < 10:
            return self._UNMEASURED_WEIGHT
        return max(self._WEIGHT_FLOOR, acc_lb)

    # ── the admission gate ───────────────────────────────────────────────
    # A verifier needs at least this many graded verdicts before its own
    # accuracy floor is held against it — below it, one weak cell is noise,
    # not evidence of a weak verifier.
    _MIN_CELL_GRADED_FOR_FLOOR = 10

    def _domain_evidence(self, domain: str) -> tuple[int, float, float]:
        """Aggregate graded evidence across every verifier in a domain,
        scored by the WEAKEST relevant false-pass bound (a chain of checkers
        is only as trustworthy as the leakiest one actually used).

        The accuracy returned is the MINIMUM of the pooled lower bound and
        the worst individually-measured verifier's lower bound. Pooling alone
        let a high-volume accurate verifier swamp a low-accuracy one — the
        domain read admitted, its weak member unnoticed — while the false-
        pass bound already used the worst cell. Accuracy now uses the worst
        cell too, so the two bounds are consistent.
        """
        with self._lock:
            cells = [c for (v, d), c in self._cells.items()
                     if d == domain and c.graded > 0]
        if not cells:
            return 0, 0.0, 1.0
        graded = sum(c.graded for c in cells)
        correct = sum(c.correct for c in cells)
        pooled_acc = wilson_lower_bound(correct, graded)
        worst_fp = max(c.false_pass_ub() for c in cells)
        measured = [c for c in cells if c.graded >= self._MIN_CELL_GRADED_FOR_FLOOR]
        worst_acc = min((c.accuracy_lb() for c in measured), default=pooled_acc)
        return graded, min(pooled_acc, worst_acc), worst_fp

    def _evaluate_admission(
        self, domain: str, *, allow_revoke: bool
    ) -> AdmissionDecision:
        """Evaluate admission. Only writes durable state when allow_revoke.

        ``status()`` and other observers call with allow_revoke=False so a
        nominal read can never append a revoke_seed event and mutate durable
        governance (or block on ledger I/O). The admission GATE calls with
        allow_revoke=True so the measurement ratchet still fires — but a
        pending revocation is still REPORTED as not-admitted either way, so
        an observer and the gate agree on the verdict.
        """
        with self._lock:
            diverged = len(self._unpersisted_events)
        if diverged:
            # In-memory reliability is ahead of (or divergent from) the
            # durable ledger: admission would be governed by state no audit
            # can reproduce.
            return AdmissionDecision(
                domain, False, "ledger_divergence",
                {"unpersisted_events": diverged})

        graded, acc_lb, fp_ub = self._domain_evidence(domain)

        if domain in SEED_ADMITTED_DOMAINS and domain not in self._revoked_seeds:
            # seeds are admitted by construction — but measured evidence can
            # revoke them (the ratchet works on measurement, not faith)
            revocable = graded >= int(_ADMIT_MIN_GRADED_FLAG.value()) and (
                fp_ub > float(_ADMIT_MAX_FALSE_PASS_FLAG.value()) * 2)
            if revocable:
                if allow_revoke:
                    with self._lock:
                        self._append_event("revoke_seed", {
                            "domain": domain,
                            "false_pass_ub": round(fp_ub, 4),
                            "graded": graded,
                        })
                    logger.warning("Foundry: seed admission REVOKED for %r "
                                   "(false-pass UB %.3f over %d graded)",
                                   domain, fp_ub, graded)
                return AdmissionDecision(domain, False,
                                         "seed_revoked_by_evidence",
                                         {"false_pass_ub": fp_ub, "graded": graded})
            return AdmissionDecision(domain, True, "seed_admitted",
                                     {"graded": graded})

        min_graded = int(_ADMIT_MIN_GRADED_FLAG.value())
        if graded < min_graded:
            return AdmissionDecision(
                domain, False, "insufficient_evidence",
                {"graded": graded, "required": min_graded})
        if acc_lb < float(_ADMIT_MIN_WILSON_FLAG.value()):
            return AdmissionDecision(
                domain, False, "accuracy_below_threshold",
                {"accuracy_lb": round(acc_lb, 4)})
        if fp_ub > float(_ADMIT_MAX_FALSE_PASS_FLAG.value()):
            return AdmissionDecision(
                domain, False, "false_pass_rate_too_high",
                {"false_pass_ub": round(fp_ub, 4)})
        return AdmissionDecision(domain, True, "earned_by_evidence",
                                 {"graded": graded,
                                  "accuracy_lb": round(acc_lb, 4),
                                  "false_pass_ub": round(fp_ub, 4)})

    def domain_admitted(self, domain: str) -> AdmissionDecision:
        """May verifier-clean wins in this domain become training data?

        This is the GATE, so it applies the measurement ratchet (a seed can
        be revoked by durable evidence). Observers must use
        :meth:`domain_admitted_readonly`.
        """
        domain = str(domain or "").strip().lower()
        return self._evaluate_admission(domain, allow_revoke=True)

    def domain_admitted_readonly(self, domain: str) -> AdmissionDecision:
        """Same verdict as the gate, but never mutates governance state."""
        domain = str(domain or "").strip().lower()
        return self._evaluate_admission(domain, allow_revoke=False)

    # ── observability ────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        with self._lock:
            cells = [c.to_dict() for c in self._cells.values()]
            pending = len(self._pending)
        domains = sorted({c["domain"] for c in cells}
                         | set(SEED_ADMITTED_DOMAINS))
        return {
            "schema_version": SCHEMA_VERSION,
            "cells": cells,
            "pending_verdicts": pending,
            "restore_errors": self._restore_errors,
            # READ-ONLY: a status poll must never mutate durable governance.
            "admissions": {
                d: self.domain_admitted_readonly(d).admitted for d in domains
            },
            "chain_head": self._chain.head_hash(),
            "chain_length": self._chain.length(),
            "root": str(self.root),
        }

    def is_alive(self) -> bool:
        if self._closed or not self.events_path.parent.is_dir():
            return False
        with self._lock:
            if self._unpersisted_events:
                # Memory and ledger disagree: the component is running but its
                # governance evidence is not durable.
                return False
        if not self._writer_started:
            # No writes yet: nothing has failed, so it is alive if it can
            # persist — the audit chain must itself be healthy, not merely
            # the directory present.
            chain_ok = getattr(self._chain, "is_healthy", None)
            return bool(chain_ok()) if callable(chain_ok) else True
        if self._writer.is_alive():
            return True
        # The writer thread was started and is now DEAD. A dead writer cannot
        # consume the queue, so pending or future events stall forever —
        # reporting "alive" because pending happens to be zero hid exactly
        # that failure. A started-then-dead writer is not alive.
        return False

    def verify_ledger(self) -> tuple[bool, list[dict[str, Any]]]:
        """Synchronous verification. Async callers must use
        verify_ledger_async — this reads and parses the whole ledger."""
        self.flush_ledger()
        return self._verify_ledger_locked()

    def _verify_ledger_locked(self) -> tuple[bool, list[dict[str, Any]]]:
        """The parse itself, with no flushing, so it can run in a thread."""
        bodies: dict[str, dict[str, Any]] = {}
        problems: list[dict[str, Any]] = []
        if self.events_path.exists():
            for idx, line in enumerate(
                    self.events_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    bodies[str(event.get("event_id", f"line-{idx}"))] = event
                except json.JSONDecodeError:
                    problems.append({"seq": -1, "kind": "event_body",
                                     "receipt_id": f"line-{idx}",
                                     "reason": "event body is not valid JSON"})
        ok, chain_problems = self._chain.verify(
            body_loader=lambda rid, kind: bodies.get(rid))
        problems.extend(chain_problems)
        return (ok and not problems, problems)


_foundry: VerifierFoundry | None = None
_foundry_lock = threading.Lock()


def get_verifier_foundry() -> VerifierFoundry:
    global _foundry
    if _foundry is None:
        with _foundry_lock:
            if _foundry is None:
                _foundry = VerifierFoundry()
    return _foundry


def boot_verifier_foundry() -> VerifierFoundry:
    """Build the foundry and publish it on the service spine.

    aura_main has imported this name since the foundry landed, and it was
    never defined — so every boot logged "Verifier Foundry boot failed:
    cannot import name 'boot_verifier_foundry'", degraded, and carried on
    without it. The failure was recorded honestly and still meant the
    foundry had never once been live. Mirrors boot_verifier_curriculum.
    """

    foundry = get_verifier_foundry()
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("verifier_foundry", foundry, required=False)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "verifier_foundry",
            exc,
            severity="warning",
            action="foundry built but not registered on the service container",
        )
    return foundry
