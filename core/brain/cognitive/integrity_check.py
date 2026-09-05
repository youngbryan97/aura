"""Integrity Guard — Semantic Drift Prevention

Audits the belief graph for:
  - Low-confidence beliefs ("Logic Ulcers") → quarantined
  - Contradictory belief pairs → flagged
  - Stale beliefs with no recent reinforcement → decayed

Logs audit results to integrity_audit.log and emits to thought stream.
"""
import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Kernel.IntegrityGuard")


@dataclass
class AuditReport:
    beliefs_scanned: int = 0
    quarantined: int = 0
    contradictions_found: int = 0
    decayed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    # CP126 (high): "Contradiction scanning silently ignores most large
    # graphs. Only the first 500 beliefs are considered and reporting stops
    # at 50 pairs, with no ordering guarantee, pagination, coverage flag,
    # omitted count, or continuation cursor."
    #
    # The caps are defensible — this runs on a live belief graph and an
    # unbounded O(n) sweep with logging is not free. What was not defensible
    # is that a 5,000-belief graph reported "contradictions=0" in exactly
    # the same words as a genuinely clean 400-belief one. A consistency
    # audit that silently examines a tenth of the evidence is worse than no
    # audit, because it produces a clean bill of health nobody questions.
    contradiction_scan_considered: int = 0
    contradiction_scan_total: int = 0
    contradiction_scan_truncated: bool = False
    contradiction_reporting_capped: bool = False

    @property
    def contradiction_coverage(self) -> float:
        """Fraction of eligible beliefs the contradiction scan looked at."""
        if self.contradiction_scan_total <= 0:
            return 0.0
        return self.contradiction_scan_considered / self.contradiction_scan_total

    @property
    def contradiction_result_is_complete(self) -> bool:
        """Whether "contradictions_found" is an answer or a lower bound."""
        return not (self.contradiction_scan_truncated or self.contradiction_reporting_capped)

    def __str__(self) -> str:
        qualifier = ""
        if not self.contradiction_result_is_complete:
            qualifier = (
                f" [partial scan: {self.contradiction_scan_considered}"
                f"/{self.contradiction_scan_total} beliefs"
                f"{', report capped' if self.contradiction_reporting_capped else ''}]"
            )
        return (
            f"IntegrityAudit: scanned={self.beliefs_scanned}, "
            f"quarantined={self.quarantined}, "
            f"contradictions={self.contradictions_found}{qualifier}, "
            f"decayed={self.decayed} ({self.duration_s:.1f}s)"
        )


class IntegrityGuard:
    """Audits beliefs for semantic drift and logical inconsistency.
    Quarantines low-confidence beliefs, detects contradictions,
    and decays stale beliefs.
    """

    def __init__(
        self,
        belief_graph: Any = None,
        confidence_threshold: float = 0.15,
        staleness_days: int = 30,
        audit_log_path: Path | None = None,
    ):
        self.belief_graph = belief_graph
        self.confidence_threshold = confidence_threshold
        self.staleness_days = staleness_days
        self.audit_log_path = audit_log_path or (Path.cwd() / "logs" / "integrity_audit.log")

    async def audit_beliefs(self) -> AuditReport:
        """Run a full integrity sweep on the belief graph."""
        report = AuditReport()
        t0 = time.monotonic()
        logger.info("🛡️  Integrity audit starting...")

        if not self.belief_graph:
            logger.warning("No belief graph available — skipping audit.")
            report.errors.append("no_belief_graph")
            report.duration_s = time.monotonic() - t0
            return report

        try:
            from core.adaptation.immune_system import get_immune_system
            immune_sys = get_immune_system()
        except ImportError:
            immune_sys = None

        try:
            beliefs = self._get_beliefs()
            report.beliefs_scanned = len(beliefs)
            for belief in beliefs:
                if belief.get("confidence", 1.0) < self.confidence_threshold:
                    await self._quarantine(belief, report)
            self._detect_contradictions(beliefs, report)
            
            # Use the dedicated decay method
            await self._decay_stale(beliefs, report)
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation('integrity_check', exc)
            msg = f"Integrity audit error: {exc}"
            logger.error(msg, exc_info=True)
            report.errors.append(msg)

        report.duration_s = time.monotonic() - t0
        self._write_audit_log(report)
        logger.info("🛡️  %s", report)

        try:
            from core.thought_stream import get_emitter
            get_emitter().emit(
                "Integrity Audit 🛡️", str(report),
                level="info" if not report.quarantined else "warning",
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('integrity_check', exc)
            logger.debug("Suppressed thought-stream emit: %s", exc)

        return report

    # ------------------------------------------------------------------
    def _get_beliefs(self) -> list[dict]:
        try:
            if hasattr(self.belief_graph, "get_all_beliefs"):
                return self.belief_graph.get_all_beliefs()
            if hasattr(self.belief_graph, "beliefs"):
                return list(self.belief_graph.beliefs.values())
            if hasattr(self.belief_graph, "conn"):
                c = self.belief_graph._get_conn().cursor()
                c.execute("SELECT * FROM beliefs")
                return [dict(row) for row in c.fetchall()]
        except (sqlite3.Error, OSError) as exc:
            record_degradation('integrity_check', exc)
            logger.warning("Failed to retrieve beliefs: %s", exc)
        return []

    async def _quarantine(self, belief: dict, report: AuditReport) -> None:
        belief_id = belief.get("id", "unknown")
        logger.info(
            "🚨 Quarantining belief %s (conf=%.3f): %s...",
            belief_id, belief.get("confidence", 0),
            str(belief.get("content", ""))[:80],
        )
        try:
            if hasattr(self.belief_graph, "update_belief"):
                self.belief_graph.update_belief(belief_id, status="quarantined", quarantined_at=time.time())
            elif hasattr(self.belief_graph, "conn"):
                def _do_quarantine():
                    c = self.belief_graph.conn.cursor()
                    c.execute("UPDATE beliefs SET status='quarantined' WHERE id=?", (belief_id,))
                    self.belief_graph.conn.commit()
                await asyncio.to_thread(_do_quarantine)
            report.quarantined += 1
        except (sqlite3.Error, OSError) as exc:
            record_degradation('integrity_check', exc)
            report.errors.append(f"quarantine {belief_id}: {exc}")

    def _detect_contradictions(self, beliefs: list[dict], report: AuditReport) -> None:
        """Detect contradictions with O(n) prefix matching (PERF-02)."""
        MAX_SCAN = 500
        MAX_CONTRADICTIONS = 50
        NEGATION_PREFIXES = ("not ", "never ", "cannot ", "doesn't ", "isn't ", "won't ")
        
        eligible = [b for b in beliefs if b.get("status") != "quarantined"]
        # Deterministic order, so two runs over the same graph examine the
        # same subset. Without this the truncation was not just partial but
        # arbitrarily partial — a contradiction could appear and disappear
        # between audits with no change to the beliefs.
        eligible.sort(key=lambda b: str(b.get("id", "")))
        considered = eligible[:MAX_SCAN]
        report.contradiction_scan_total = len(eligible)
        report.contradiction_scan_considered = len(considered)
        report.contradiction_scan_truncated = len(eligible) > len(considered)
        if report.contradiction_scan_truncated:
            logger.warning(
                "Contradiction scan examined %d of %d eligible beliefs; "
                "contradictions=%d is a LOWER BOUND, not a clean bill of health.",
                len(considered),
                len(eligible),
                report.contradictions_found,
            )
        contents = [
            (b.get("id", "?"), str(b.get("content", "")).lower().strip())
            for b in considered
        ]
        
        # Build prefix-indexed lookup for O(n) instead of O(n^2)
        negated: dict[str, str] = {}  # stripped_text -> original_id
        for bid, text in contents:
            for prefix in NEGATION_PREFIXES:
                if text.startswith(prefix):
                    negated[text[len(prefix):]] = bid
        
        seen: set = set()
        for bid, text in contents:
            if text in negated and negated[text] != bid:
                pair = frozenset({bid, negated[text]})
                if pair not in seen:
                    seen.add(pair)
                    report.contradictions_found += 1
                    logger.warning("⚠️  Contradiction: [%s] vs [%s]", bid, negated[text])
                if report.contradictions_found >= MAX_CONTRADICTIONS:
                    report.contradiction_reporting_capped = True
                    logger.warning(
                        "Contradiction reporting capped at %d pairs; more may exist.",
                        MAX_CONTRADICTIONS,
                    )
                    break

    async def _decay_stale(self, beliefs: list[dict], report: AuditReport) -> None:
        """Decay all stale beliefs."""
        cutoff = time.time() - (self.staleness_days * 86400)
        for belief in beliefs:
            await self._decay_stale_single(belief, report, cutoff)

    async def _decay_stale_single(self, belief: dict, report: AuditReport, cutoff: float | None = None) -> None:
        """Decay a single stale belief."""
        if cutoff is None:
            cutoff = time.time() - (self.staleness_days * 86400)
            
        if belief.get("status") == "quarantined":
            return

        # [Phase 17.4] Protected Enclaves check.
        #
        # CP126 da6520fc. An unavailable immune system skipped protection and
        # let the belief decay — the enclave exists to prevent exactly that —
        # and operational or schema failures from get_immune_system /
        # is_protected were not caught here at all, so one malformed belief
        # could terminate the whole sweep.
        #
        # Both are now handled the same way: if protection cannot be
        # established, this belief is left alone. Skipping one decay is
        # harmless; decaying a protected belief is not, and neither is
        # aborting the sweep for every belief after it.
        try:
            from core.adaptation.immune_system import get_immune_system
            immune_sys = get_immune_system()
            if immune_sys:
                is_protected = immune_sys.is_protected(belief.get("metadata", {}) or belief)
                if is_protected:
                    return
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "integrity_guard",
                exc,
                severity="warning",
                action="left a belief undecayed because enclave protection could not be checked",
            )
            return

        # CP126 fb224bd0. A belief with neither last_reinforced nor
        # created_at fell back to 0, which is 1970 — so unknown provenance
        # was decayed as though it were confirmed ancient. Those are
        # different claims, and only one of them is evidence.
        last_reinforced = belief.get("last_reinforced", belief.get("created_at", None))
        if last_reinforced is None:
            report.skipped_unknown_age = getattr(report, "skipped_unknown_age", 0) + 1
            return
        try:
            last_reinforced = float(last_reinforced)
        except (TypeError, ValueError):
            report.skipped_unknown_age = getattr(report, "skipped_unknown_age", 0) + 1
            return
        if last_reinforced <= 0.0:
            # An explicit 0 is the same claim as an absent field: nobody
            # recorded when this belief was last touched. 1970 is not
            # evidence of staleness.
            report.skipped_unknown_age = getattr(report, "skipped_unknown_age", 0) + 1
            return
        if last_reinforced < cutoff:
            belief_id = belief.get("id", "unknown")
            old_conf = belief.get("confidence", 1.0)
            new_conf = max(0.0, old_conf - 0.05)
            if new_conf != old_conf:
                try:
                    if hasattr(self.belief_graph, "update_belief"):
                        self.belief_graph.update_belief(belief_id, confidence=new_conf)
                    elif hasattr(self.belief_graph, "conn"):
                        def _do_decay():
                            c = self.belief_graph.conn.cursor()
                            c.execute("UPDATE beliefs SET confidence=? WHERE id=?", (new_conf, belief_id))
                            self.belief_graph.conn.commit()
                        await asyncio.to_thread(_do_decay)
                    report.decayed += 1
                except (sqlite3.Error, OSError) as exc:
                    record_degradation('integrity_check', exc)
                    report.errors.append(f"decay {belief_id}: {exc}")

    def _write_audit_log(self, report: AuditReport) -> None:
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            # Internal maintenance write: unscoped, the live runtime refuses
            # it as a governance violation and the belief-integrity audit
            # trail is silently lost — an audit log that vanishes under
            # governance is worse than none (observed live 2026-07-18).
            with local_internal_governed_scope(
                "brain.cognitive.integrity_check.audit_log",
                domain="file_write",
            ):
                get_file_write_gateway().append_text(
                    self.audit_log_path,
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {report}\n",
                    source="brain.cognitive.integrity_check.audit_log",
                )
        except OSError as exc:
            record_degradation('integrity_check', exc)
            logger.warning("Failed to write audit log: %s", exc)
