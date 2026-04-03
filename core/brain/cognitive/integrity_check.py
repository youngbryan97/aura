"""Integrity Guard — Semantic Drift Prevention

Audits the belief graph for:
  - Low-confidence beliefs ("Logic Ulcers") → quarantined
  - Contradictory belief pairs → flagged
  - Stale beliefs with no recent reinforcement → decayed

Logs audit results to integrity_audit.log and emits to thought stream.
"""
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Kernel.IntegrityGuard")


@dataclass
class AuditReport:
    beliefs_scanned: int = 0
    quarantined: int = 0
    contradictions_found: int = 0
    decayed: int = 0
    errors: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def __str__(self) -> str:
        return (
            f"IntegrityAudit: scanned={self.beliefs_scanned}, "
            f"quarantined={self.quarantined}, "
            f"contradictions={self.contradictions_found}, "
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
        audit_log_path: Optional[Path] = None,
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
        except Exception as exc:
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
        except Exception as exc:
            logger.debug("Suppressed thought-stream emit: %s", exc)

        return report

    # ------------------------------------------------------------------
    def _get_beliefs(self) -> List[Dict]:
        try:
            if hasattr(self.belief_graph, "get_all_beliefs"):
                return self.belief_graph.get_all_beliefs()
            if hasattr(self.belief_graph, "beliefs"):
                return list(self.belief_graph.beliefs.values())
            if hasattr(self.belief_graph, "conn"):
                c = self.belief_graph._get_conn().cursor()
                c.execute("SELECT * FROM beliefs")
                return [dict(row) for row in c.fetchall()]
        except Exception as exc:
            logger.warning("Failed to retrieve beliefs: %s", exc)
        return []

    async def _quarantine(self, belief: Dict, report: AuditReport) -> None:
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
        except Exception as exc:
            report.errors.append(f"quarantine {belief_id}: {exc}")

    def _detect_contradictions(self, beliefs: List[Dict], report: AuditReport) -> None:
        """Detect contradictions with O(n) prefix matching (PERF-02)."""
        MAX_SCAN = 500
        MAX_CONTRADICTIONS = 50
        NEGATION_PREFIXES = ("not ", "never ", "cannot ", "doesn't ", "isn't ", "won't ")
        
        contents = [
            (b.get("id", "?"), str(b.get("content", "")).lower().strip())
            for b in beliefs[:MAX_SCAN] if b.get("status") != "quarantined"
        ]
        
        # Build prefix-indexed lookup for O(n) instead of O(n^2)
        negated: Dict[str, str] = {}  # stripped_text -> original_id
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
                    break

    async def _decay_stale(self, beliefs: List[Dict], report: AuditReport) -> None:
        """Decay all stale beliefs."""
        cutoff = time.time() - (self.staleness_days * 86400)
        for belief in beliefs:
            await self._decay_stale_single(belief, report, cutoff)

    async def _decay_stale_single(self, belief: Dict, report: AuditReport, cutoff: Optional[float] = None) -> None:
        """Decay a single stale belief."""
        if cutoff is None:
            cutoff = time.time() - (self.staleness_days * 86400)
            
        if belief.get("status") == "quarantined":
            return

        # [Phase 17.4] Protected Enclaves check
        try:
            from core.adaptation.immune_system import get_immune_system
            immune_sys = get_immune_system()
            if immune_sys:
                is_protected = immune_sys.is_protected(belief.get("metadata", {}) or belief)
                if is_protected:
                    return
        except ImportError:
            logger.debug("IntegrityGuard: Immune system not installed, skipping enclave check.")
            
        last_reinforced = belief.get("last_reinforced", belief.get("created_at", 0))
        if not last_reinforced or last_reinforced < cutoff:
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
                except Exception as exc:
                    report.errors.append(f"decay {belief_id}: {exc}")

    def _write_audit_log(self, report: AuditReport) -> None:
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.audit_log_path, "a") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {report}\n")
        except Exception as exc:
            logger.warning("Failed to write audit log: %s", exc)