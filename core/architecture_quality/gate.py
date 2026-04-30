"""Architecture-quality gate.

Snapshots a baseline before a self-modification session, evaluates
post-change, and produces a binary pass/fail decision under TOML rules.
Called from `core.self_modification.safe_modification` right after the
staging file is promoted, so a regressing patch is rolled back.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .scorer import (
    DEFAULT_EXCLUDES, DEFAULT_ROOTS, QualityScore, score_codebase,
)

try:
    import tomllib as _toml
except Exception:  # pragma: no cover
    import tomli as _toml  # type: ignore

logger = logging.getLogger("ArchitectureQuality.Gate")

DEFAULT_RULES: Dict[str, object] = {
    "max_score_drop": 200,
    "max_new_cycles": 0,
    "max_new_god_files": 0,
    "min_overall_score": 0,
    "metric_floors": {},
}


@dataclass
class QualityReport:
    """Difference between two QualityScores plus a rule verdict."""
    baseline: QualityScore
    current: QualityScore
    delta_score: int
    new_cycles: int
    new_god_files: List[str]
    metric_deltas: Dict[str, float]
    rules: Dict[str, object]
    passed: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "current": self.current.to_dict(),
            "delta_score": self.delta_score,
            "new_cycles": self.new_cycles,
            "new_god_files": self.new_god_files,
            "metric_deltas": self.metric_deltas,
            "rules": self.rules,
            "passed": self.passed,
            "reason": self.reason,
        }


def _load_rules(rules_path: Optional[Path]) -> Dict[str, object]:
    rules: Dict[str, object] = dict(DEFAULT_RULES)
    if rules_path is None:
        return rules
    try:
        with open(rules_path, "rb") as f:
            data = _toml.load(f)
    except FileNotFoundError:
        logger.warning("Rules file not found: %s (using defaults)", rules_path)
        return rules
    except Exception as e:
        logger.error("Failed to parse rules %s: %s (using defaults)", rules_path, e)
        return rules
    section = data.get("gate", data)
    if isinstance(section, dict):
        for k, v in section.items():
            rules[k] = v
    floors = data.get("metric_floors")
    if isinstance(floors, dict):
        rules["metric_floors"] = floors
    return rules


class ArchitectureQualityGate:
    """Stateful pre/post quality gate for self-modification sessions."""

    def __init__(self, root: Path, *,
                 rules_path: Optional[Path] = None,
                 roots=DEFAULT_ROOTS, exclude=DEFAULT_EXCLUDES) -> None:
        self.root = Path(root).resolve()
        if rules_path is None:
            default = Path(__file__).parent / "rules.toml"
            rules_path = default if default.exists() else None
        self.rules_path = Path(rules_path) if rules_path else None
        self.roots = tuple(roots)
        self.exclude = tuple(exclude)
        self.rules = _load_rules(self.rules_path)
        self._baseline: Optional[QualityScore] = None

    def baseline(self) -> QualityScore:
        """Snapshot the current quality of the tree."""
        score = score_codebase(self.root, roots=self.roots, exclude=self.exclude)
        self._baseline = score
        logger.info(
            "📐 baseline overall=%d modules=%d edges=%d cycles=%d",
            score.overall_score, score.module_count, score.edge_count, score.cycles,
        )
        return score

    def evaluate(self, *, since: Optional[QualityScore] = None) -> QualityReport:
        """Score now and diff against *since* (or the stored baseline)."""
        base = since or self._baseline
        if base is None:
            base = self.baseline()
        current = score_codebase(self.root, roots=self.roots, exclude=self.exclude)
        delta = current.overall_score - base.overall_score
        new_cycles = max(0, current.cycles - base.cycles)
        base_gods = set(base.god_files)
        new_gods = [m for m in current.god_files if m not in base_gods]
        metric_deltas = {
            k: round(current.metrics.get(k, 0.0) - base.metrics.get(k, 0.0), 6)
            for k in current.metrics
        }
        report = QualityReport(
            baseline=base, current=current, delta_score=delta,
            new_cycles=new_cycles, new_god_files=new_gods,
            metric_deltas=metric_deltas, rules=dict(self.rules),
        )
        passed, reason = self._apply_rules(report)
        report.passed = passed
        report.reason = reason
        return report

    def _apply_rules(self, report: QualityReport) -> Tuple[bool, str]:
        rules = self.rules
        max_drop = int(rules.get("max_score_drop", 200))  # type: ignore[arg-type]
        max_new_cycles = int(rules.get("max_new_cycles", 0))  # type: ignore[arg-type]
        max_new_gods = int(rules.get("max_new_god_files", 0))  # type: ignore[arg-type]
        min_overall = int(rules.get("min_overall_score", 0))  # type: ignore[arg-type]

        if -report.delta_score > max_drop:
            return False, f"score regressed by {-report.delta_score} (limit {max_drop})"
        if report.new_cycles > max_new_cycles:
            return False, f"introduced {report.new_cycles} new cycle(s) (limit {max_new_cycles})"
        if len(report.new_god_files) > max_new_gods:
            return False, (f"introduced {len(report.new_god_files)} new god file(s): "
                           f"{report.new_god_files[:3]}")
        if report.current.overall_score < min_overall:
            return False, (f"overall_score {report.current.overall_score} "
                           f"below floor {min_overall}")
        floors = rules.get("metric_floors") or {}
        if isinstance(floors, dict):
            for k, floor in floors.items():
                if report.current.metrics.get(k, 0.0) < float(floor):
                    return False, (f"metric {k}={report.current.metrics.get(k, 0.0):.3f} "
                                   f"below floor {floor}")
        return True, "architecture quality preserved"

    def gate(self, report: QualityReport) -> Tuple[bool, str]:
        """Return (passed, reason) for an evaluated report."""
        return report.passed, report.reason


# ---------- Module-level helpers ----------

_INSTALLED_GATE: Optional[ArchitectureQualityGate] = None


def install_gate(gate: ArchitectureQualityGate) -> None:
    """Register a process-wide gate that safe_modification will consult."""
    global _INSTALLED_GATE
    _INSTALLED_GATE = gate


def get_installed_gate() -> Optional[ArchitectureQualityGate]:
    return _INSTALLED_GATE


def baseline_session(root: Path, **kw) -> Tuple[ArchitectureQualityGate, QualityScore]:
    """Build a gate, snapshot baseline, install it, return both."""
    gate = ArchitectureQualityGate(root, **kw)
    score = gate.baseline()
    install_gate(gate)
    return gate, score


def evaluate_session(*, gate: Optional[ArchitectureQualityGate] = None) -> QualityReport:
    """Evaluate using the supplied gate or the installed one."""
    g = gate or _INSTALLED_GATE
    if g is None:
        raise RuntimeError("No ArchitectureQualityGate installed")
    return g.evaluate()
