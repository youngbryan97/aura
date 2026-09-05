"""core/learning/eval_before_promotion.py
Evaluates candidate model adapters against benchmark suites before production promotion.
"""
import json
import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Learning.EvalBeforePromotion")

_ADAPTER_EVAL_ERRORS = (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError)


class AdapterEvaluator:
    """Runs verification benchmarks on candidate adapters."""

    def evaluate_candidate(self, adapter_path: str) -> dict[str, Any]:
        """Read real evaluation evidence and decide whether promotion is allowed."""
        logger.info("Evaluating candidate adapter: %s", adapter_path)
        adapter = Path(adapter_path).expanduser()
        if not adapter.exists():
            return {
                "status": "blocked",
                "passed_safety": False,
                "accuracy_score": 0.0,
                "can_promote": False,
                "reason": "adapter_path_missing",
            }

        evidence_paths = []
        if adapter.is_dir():
            evidence_paths.append(adapter / "evaluation_report.json")
        evidence_paths.append(adapter.with_suffix(adapter.suffix + ".evaluation.json" if adapter.suffix else ".evaluation.json"))

        report_path = next((path for path in evidence_paths if path.exists()), None)
        if report_path is None:
            return {
                "status": "blocked",
                "passed_safety": False,
                "accuracy_score": 0.0,
                "can_promote": False,
                "reason": "evaluation_report_missing",
            }

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            passed_safety = bool(report.get("passed_safety", False))
            accuracy_score = float(report.get("accuracy_score", 0.0))
            regression_passed = bool(report.get("regression_passed", False))
            hidden_eval_passed = bool(report.get("hidden_eval_passed", False))
            can_promote = (
                passed_safety
                and regression_passed
                and hidden_eval_passed
                and accuracy_score >= float(report.get("promotion_threshold", 0.75))
            )
            return {
                "status": "evaluated",
                "passed_safety": passed_safety,
                "regression_passed": regression_passed,
                "hidden_eval_passed": hidden_eval_passed,
                "accuracy_score": accuracy_score,
                "can_promote": can_promote,
                "evidence_path": str(report_path),
            }
        except _ADAPTER_EVAL_ERRORS as exc:
            record_degradation("learning.adapter_evaluator", exc)
            return {
                "status": "failed",
                "passed_safety": False,
                "accuracy_score": 0.0,
                "can_promote": False,
                "reason": str(exc),
            }
