from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReadinessLevel(StrEnum):
    BOOTSTRAP = "bootstrap"
    MIXED = "mixed"
    VALIDATED = "validated"
    PRODUCTION = "production"


@dataclass
class ReadinessGate:
    vectors_dir: str | Path = "training/vectors"
    behavioral_results_path: str | Path | None = None

    def evaluate(
        self,
        registry_status: dict[str, Any],
        *,
        model_path: str | Path | None = None,
    ) -> dict[str, Any]:
        vectors_dir = Path(self.vectors_dir)
        behavioral_path = Path(self.behavioral_results_path) if self.behavioral_results_path else None
        validator_report = None
        if vectors_dir.exists():
            try:
                from training.caa_32b_validation import CAAModelValidator

                validator_report = CAAModelValidator(
                    vectors_dir=vectors_dir,
                    model_path=model_path,
                ).run(
                    behavioral_results=behavioral_path
                )
            except (
                ImportError,
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                validator_report = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        coverage = float(registry_status.get("coverage_ratio", 0.0) or 0.0)
        exact = int(registry_status.get("exact_match_count", 0) or 0)
        loaded = int(registry_status.get("loaded_total", 0) or 0)
        exact_ratio = float(exact / loaded) if loaded else 0.0
        extracted_ratio = float((registry_status.get("extracted_count", 0) or 0) / loaded) if loaded else 0.0
        validator_passed = bool(validator_report and validator_report.get("passed"))
        exact_model_bound = bool(
            validator_report
            and validator_report.get("model_descriptor_sha256")
            and validator_report.get("exact_model_detected")
        )
        if loaded == 0 or extracted_ratio < 0.5:
            level = ReadinessLevel.BOOTSTRAP
            detail = "insufficient extracted vectors"
        elif coverage < 1.0 or exact_ratio < 1.0:
            level = ReadinessLevel.MIXED
            detail = "nearest-layer or partial coverage still active"
        elif validator_passed and exact_model_bound:
            level = ReadinessLevel.PRODUCTION
            detail = "exact model-bound vectors and behavioral validation passed"
        else:
            level = ReadinessLevel.VALIDATED
            detail = "extracted vectors present; exact model-bound production evidence incomplete"
        return {
            "level": level.value,
            "detail": detail,
            "coverage_ratio": coverage,
            "exact_match_ratio": exact_ratio,
            "extracted_ratio": extracted_ratio,
            "exact_model_bound": exact_model_bound,
            "validator": validator_report,
        }
