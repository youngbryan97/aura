from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .alpha_controller import AlphaController
from .mode_collapse_detector import CollapseSignal, ModeCollapseDetector
from .quality_metrics import compute_vector_quality
from .readiness_gate import ReadinessGate
from .vector_registry import VectorRegistry


class ProductionCAA:
    """Coordinates vector provenance, readiness, collapse detection, and alpha."""

    def __init__(
        self,
        *,
        base_alpha: float = 5.0,
        vectors_dir: str | Path = "training/vectors",
        behavioral_results_path: str | Path | None = None,
    ) -> None:
        self.registry = VectorRegistry()
        self.readiness_gate = ReadinessGate(
            vectors_dir=vectors_dir,
            behavioral_results_path=behavioral_results_path,
        )
        self.alpha_controller = AlphaController(base_alpha=base_alpha)
        self.collapse_detector = ModeCollapseDetector()
        self.vector_quality_by_layer: Dict[int, Dict[str, Any]] = {}
        self.readiness: Dict[str, Any] = {
            "level": "bootstrap",
            "detail": "uninitialized",
            "coverage_ratio": 0.0,
            "exact_match_ratio": 0.0,
            "extracted_ratio": 0.0,
            "validator": None,
        }
        self.last_collapse: Dict[str, Any] = self.collapse_detector.status()["last_signal"]

    def ingest_registry(
        self,
        registry: VectorRegistry,
        *,
        expected_layers: list[int],
        expected_keys: list[str],
        model_path: str = "",
    ) -> Dict[str, Any]:
        self.registry = registry
        self.vector_quality_by_layer = {
            layer: compute_vector_quality(vectors, layer_idx=layer).to_dict()
            for layer, vectors in registry.layers().items()
        }
        registry_status = registry.status(expected_layers=expected_layers, expected_keys=expected_keys)
        self.readiness = self.readiness_gate.evaluate(registry_status, model_path=model_path)
        self.alpha_controller.update(
            readiness_level=self.readiness["level"],
            exact_match_ratio=float(self.readiness.get("exact_match_ratio", 0.0) or 0.0),
            extracted_ratio=float(self.readiness.get("extracted_ratio", 0.0) or 0.0),
        )
        return self.status()

    def observe_generation(self, text: str) -> Dict[str, Any]:
        signal: CollapseSignal = self.collapse_detector.observe(text)
        self.last_collapse = signal.to_dict()
        self.alpha_controller.update(
            readiness_level=self.readiness["level"],
            exact_match_ratio=float(self.readiness.get("exact_match_ratio", 0.0) or 0.0),
            extracted_ratio=float(self.readiness.get("extracted_ratio", 0.0) or 0.0),
            collapse_signal=signal,
        )
        return {"collapse": self.last_collapse, "alpha_state": self.alpha_controller.state.to_dict()}

    def status(self) -> Dict[str, Any]:
        return {
            "readiness": self.readiness,
            "alpha_state": self.alpha_controller.state.to_dict(),
            "collapse": self.collapse_detector.status(),
            "registry": self.registry.status(),
            "vector_quality_by_layer": self.vector_quality_by_layer,
        }
