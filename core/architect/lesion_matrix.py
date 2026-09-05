"""core/architect/lesion_matrix.py -- Systematic Ablation Testing
================================================================
Computes a lesion matrix by systematically disabling substrate/brain
components and measuring the behavioral impact. This identifies:
  - Critical dependencies (regions whose removal causes large degradation)
  - Redundancies (regions whose removal has minimal impact)
  - Functional specialization (which regions serve which functions)

Algorithm:
  1. Define probe functions that measure behavioral metrics:
     - Prediction accuracy (world model surprise)
     - Value stability (heartstone weight variance)
     - Response coherence (substrate state energy)
     - Affect regulation (valence/arousal stability)
  2. Run probes with all components active (baseline)
  3. For each component (brain region, substrate dimension group):
     a. Zero out / disable the component
     b. Run the same probes
     c. Record the metric deltas
     d. Restore the component
  4. Build a matrix: components x metrics
  5. Identify critical paths and redundant structures

The lesion matrix is a diagnostic tool, not a runtime system. It should
be run during dream cycles or explicit diagnostic sessions.

Methodology notes (CP126). A lesion study makes causal claims about a LIVE,
stateful system, so the experiment's own hygiene is the evidence:

- Each lesion is compared against a baseline measured next to it, not against
  one taken before the whole run.
- Every registered component is snapshotted and restored around each lesion,
  because a lesioned probe step can mutate anything it touches.
- Lesion and restore are transactional: a probe exception must not leave a
  component disabled.
- A probe that could not run is UNAVAILABLE, not 0.0 — a failed measurement
  must never read as "no impact".

CP126 751ee90e / d857b24f / 43446d39 / ec3611ea / 320b8881 / 1c536bf5.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import interprocess_file_lock
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.LesionMatrix")

_DATA_DIR = state_root() / "data" / "lesion_studies"
_RESULTS_PATH = _DATA_DIR / "latest_lesion_matrix.json"
#: Durable longitudinal record. CP126 1c536bf5: the in-memory deque was never
#: loaded and persistence overwrote a single "latest" file, so repeated studies
#: could not reproduce a criticality claim.
_HISTORY_PATH = _DATA_DIR / "lesion_study_history.jsonl"
_LOCK_PATH = _DATA_DIR / "lesion_study.lock"

#: Guards live component mutation inside this process; the file lock guards it
#: across processes (CP126 320b8881).
_STUDY_LOCK = threading.RLock()

#: Sentinel for a probe that could not be measured (CP126 ec3611ea).
UNAVAILABLE = None

#: Failures a probe or step function can realistically raise. Named rather
#: than bare ``Exception`` because core/architect carries a ratchet against
#: broad exception swallowing — a lesion study must isolate a probe failure
#: without becoming a place where anything at all disappears quietly.
_PROBE_ERRORS = (
    ArithmeticError,
    AttributeError,
    BufferError,
    EOFError,
    IndexError,
    KeyError,
    LookupError,
    MemoryError,
    OSError,
    ReferenceError,
    RuntimeError,
    StopIteration,
    TypeError,
    UnicodeError,
    ValueError,
)


@dataclass
class ProbeResult:
    """Result of a single behavioral probe."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class LesionResult:
    """Result of lesioning a single component."""
    component_name: str
    baseline_metrics: dict[str, float | None]
    lesioned_metrics: dict[str, float | None]
    deltas: dict[str, float | None]           # lesioned - baseline
    relative_impact: dict[str, float | None]  # |delta| / |baseline|
    criticality_score: float                     # Aggregate impact (NaN if none)
    #: Metrics that could not be measured on either side. These are absent from
    #: the criticality average rather than counted as zero impact.
    unavailable_metrics: tuple[str, ...] = ()
    restore_verified: bool = True
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def _round(values: dict[str, float | None], places: int) -> dict[str, Any]:
        return {
            key: (None if value is None or not math.isfinite(value) else round(value, places))
            for key, value in values.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component_name,
            "baseline": self._round(self.baseline_metrics, 6),
            "lesioned": self._round(self.lesioned_metrics, 6),
            "deltas": self._round(self.deltas, 6),
            "relative_impact": self._round(self.relative_impact, 4),
            "criticality_score": (
                None if not math.isfinite(self.criticality_score)
                else round(self.criticality_score, 4)
            ),
            "unavailable_metrics": list(self.unavailable_metrics),
            "restore_verified": self.restore_verified,
        }


@dataclass
class LesionMatrix:
    """Complete lesion study results."""
    components: list[str]
    metrics: list[str]
    matrix: np.ndarray  # shape: (n_components, n_metrics) -- relative impacts
    results: list[LesionResult]
    total_time_ms: float
    baseline_policy: str = "interleaved"
    probe_failures: dict[str, int] = field(default_factory=dict)
    step_failures: int = 0
    restore_failures: tuple[str, ...] = ()
    timestamp: float = field(default_factory=time.time)

    def _row(self, index: int) -> np.ndarray:
        row = np.asarray(self.matrix[index], dtype=np.float64)
        return row[np.isfinite(row)]

    def get_critical_components(self, threshold: float = 0.3) -> list[str]:
        """Components whose removal causes > threshold relative impact."""
        critical = []
        for i, comp in enumerate(self.components):
            measured = self._row(i)
            if measured.size and float(np.max(np.abs(measured))) > threshold:
                critical.append(comp)
        return critical

    def get_redundant_components(self, threshold: float = 0.05) -> list[str]:
        """Components whose removal causes < threshold relative impact.

        A component with NO measured metric is not redundant — it is
        unmeasured, and appears in ``get_unmeasured_components`` instead
        (CP126 ec3611ea).
        """
        redundant = []
        for i, comp in enumerate(self.components):
            measured = self._row(i)
            if measured.size and float(np.max(np.abs(measured))) < threshold:
                redundant.append(comp)
        return redundant

    def get_unmeasured_components(self) -> list[str]:
        return [
            comp for i, comp in enumerate(self.components) if self._row(i).size == 0
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": self.components,
            "metrics": self.metrics,
            "matrix": [
                [None if not math.isfinite(value) else value for value in row]
                for row in self.matrix.tolist()
            ],
            "critical": self.get_critical_components(),
            "redundant": self.get_redundant_components(),
            "unmeasured": self.get_unmeasured_components(),
            "n_components": len(self.components),
            "n_metrics": len(self.metrics),
            "total_time_ms": round(self.total_time_ms, 2),
            "baseline_policy": self.baseline_policy,
            "probe_failures": dict(self.probe_failures),
            "step_failures": self.step_failures,
            "restore_failures": list(self.restore_failures),
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
        }


class LesionableComponent:
    """A component that can be temporarily disabled for lesion testing."""

    def __init__(self, name: str, get_state: Callable[[], np.ndarray],
                 set_state: Callable[[np.ndarray], None],
                 zero_fn: Callable[[], None]) -> None:
        self.name = name
        self._get_state = get_state
        self._set_state = set_state
        self._zero_fn = zero_fn
        self._saved_state: np.ndarray | None = None

    def read_state(self) -> np.ndarray:
        """Current state as an independent copy."""
        return np.array(self._get_state(), copy=True)

    def write_state(self, state: np.ndarray) -> bool:
        """Set the state and verify it took."""
        self._set_state(state)
        try:
            current = np.asarray(self._get_state())
            return bool(np.allclose(current, np.asarray(state), equal_nan=True))
        except (AttributeError, TypeError, ValueError):
            return True

    def snapshot(self) -> None:
        """Capture current state without disabling anything."""
        self._saved_state = self.read_state()

    def save_and_lesion(self) -> None:
        """Save current state and zero out the component."""
        self.snapshot()
        self._zero_fn()

    @property
    def is_lesioned(self) -> bool:
        return self._saved_state is not None

    def restore(self) -> bool:
        """Restore the saved state. Returns whether the state came back."""
        if self._saved_state is None:
            return True
        saved = self._saved_state
        try:
            self._set_state(saved)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error("Lesion restore failed for '%s': %s", self.name, exc)
            return False
        finally:
            self._saved_state = None
        try:
            current = np.asarray(self._get_state())
            restored = bool(np.allclose(current, np.asarray(saved), equal_nan=True))
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning("Could not verify restore for '%s': %s", self.name, exc)
            return True
        if not restored:
            logger.error(
                "Component '%s' did not return to its pre-lesion state", self.name
            )
        return restored


class LesionStudy:
    """Conducts systematic ablation studies on neural components.

    Usage:
        study = LesionStudy()

        # Register components
        study.register_component(LesionableComponent(
            name="sensory_region",
            get_state=lambda: brain._regions["sensory"].state,
            set_state=lambda s: setattr(brain._regions["sensory"], "state", s),
            zero_fn=lambda: brain._regions["sensory"].state.fill(0),
        ))

        # Register probes
        study.register_probe("prediction_error",
            lambda: world_model.get_mean_surprise())

        # Run study
        matrix = study.run()
        print(matrix.get_critical_components())
    """

    def __init__(
        self,
        n_probe_steps: int = 20,
        seed: int = 42,
        *,
        interleaved_baseline: bool = True,
        quiescence_check: Callable[[], bool] | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self._components: dict[str, LesionableComponent] = {}
        self._probes: dict[str, Callable[[], float]] = {}
        self._step_fn: Callable[[], None] | None = None
        self._n_probe_steps = n_probe_steps
        self._rng = np.random.default_rng(seed)
        self._last_matrix: LesionMatrix | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=20)
        self._interleaved_baseline = bool(interleaved_baseline)
        self._quiescence_check = quiescence_check
        self._probe_failures: dict[str, int] = {}
        self._step_failures = 0

        self._data_dir = Path(data_dir) if data_dir is not None else _DATA_DIR
        self._results_path = self._data_dir / _RESULTS_PATH.name
        self._history_path = self._data_dir / _HISTORY_PATH.name
        self._lock_path = self._data_dir / _LOCK_PATH.name
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load_history()

    def register_component(self, component: LesionableComponent) -> None:
        """Register a component for lesion testing."""
        self._components[component.name] = component

    def register_probe(self, name: str, fn: Callable[[], float]) -> None:
        """Register a behavioral probe function."""
        self._probes[name] = fn

    def set_step_function(self, fn: Callable[[], None]) -> None:
        """Set the function that advances the system by one step."""
        self._step_fn = fn

    def _run_probes(self) -> dict[str, float | None]:
        """Run all probes. A probe that cannot be measured returns None.

        CP126 ec3611ea: probe exceptions were converted to 0.0, so a broken
        measurement was indistinguishable from a stable one — and a component
        whose probe crashed under lesion looked perfectly redundant.
        """
        results: dict[str, float | None] = {}
        for name, fn in self._probes.items():
            try:
                value = float(fn())
            except _PROBE_ERRORS as exc:
                logger.debug("Probe '%s' failed: %s", name, exc)
                self._probe_failures[name] = self._probe_failures.get(name, 0) + 1
                results[name] = UNAVAILABLE
                continue
            if not np.isfinite(value):
                logger.debug("Probe '%s' returned a non-finite value", name)
                self._probe_failures[name] = self._probe_failures.get(name, 0) + 1
                results[name] = UNAVAILABLE
                continue
            results[name] = value
        return results

    def _run_steps_and_probe(self) -> dict[str, float | None]:
        """Run N steps, then probe. Returns averaged probe results.

        A metric with no usable sample is None, not 0.0.
        """
        accumulated: dict[str, list[float]] = {n: [] for n in self._probes}

        for _ in range(self._n_probe_steps):
            if self._step_fn:
                try:
                    self._step_fn()
                except _PROBE_ERRORS as exc:
                    logger.warning("Lesion step function failed: %s", exc)
                    self._step_failures += 1
                    break
            values = self._run_probes()
            for name, val in values.items():
                if val is not None:
                    accumulated[name].append(val)

        return {
            name: (float(np.mean(vals)) if vals else UNAVAILABLE)
            for name, vals in accumulated.items()
        }

    def _snapshot_all(self) -> None:
        """Snapshot every registered component before a lesion.

        CP126 43446d39: only the targeted component was restored, but a
        lesioned probe step can mutate every dependency it touches — so each
        experiment contaminated the next one and the live system.
        """
        for component in self._components.values():
            try:
                component.snapshot()
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning("Could not snapshot '%s': %s", component.name, exc)

    def _restore_all(self) -> list[str]:
        """Restore every component. Returns the names that did not come back."""
        failures: list[str] = []
        for component in self._components.values():
            if not component.restore():
                failures.append(component.name)
        return failures

    def _capture_states(self) -> dict[str, np.ndarray]:
        """Study-level snapshot, held separately from the per-lesion slot.

        The per-lesion snapshot is consumed by its matching restore, so the
        study needs its own copy to guarantee the live system is unchanged
        once the diagnostic finishes (CP126 43446d39).
        """
        states: dict[str, np.ndarray] = {}
        for name, component in self._components.items():
            try:
                states[name] = component.read_state()
            except _PROBE_ERRORS as exc:
                logger.warning("Could not capture study state for '%s': %s", name, exc)
        return states

    def _restore_states(self, states: dict[str, np.ndarray]) -> list[str]:
        failures: list[str] = []
        for name, saved in states.items():
            component = self._components.get(name)
            if component is None:
                continue
            try:
                if not component.write_state(saved):
                    failures.append(name)
            except _PROBE_ERRORS as exc:
                logger.error("Study-level restore failed for '%s': %s", name, exc)
                failures.append(name)
        return failures

    def run(self, *, quiescence_check: Callable[[], bool] | None = None) -> LesionMatrix:
        """Execute the full lesion study.

        The study mutates LIVE component state, so it takes an in-process lock
        and an interprocess lock, and refuses to start if the caller's
        quiescence check says runtime readers are still active
        (CP126 320b8881).
        """
        t_start = time.monotonic()

        if not self._components:
            raise ValueError("No components registered for lesion study")
        if not self._probes:
            raise ValueError("No probes registered for lesion study")

        check = quiescence_check or self._quiescence_check
        if check is not None and not check():
            raise RuntimeError(
                "lesion study refused: runtime readers are not quiescent"
            )

        with _STUDY_LOCK, interprocess_file_lock(self._lock_path):
            return self._run_locked(t_start)

    def _run_locked(self, t_start: float) -> LesionMatrix:
        comp_names = sorted(self._components.keys())
        metric_names = sorted(self._probes.keys())
        self._probe_failures = {}
        self._step_failures = 0

        logger.info("Starting lesion study: %d components, %d probes",
                     len(comp_names), len(metric_names))

        # Study-level transaction: the whole run is bracketed by a snapshot
        # and a restore, so recovery steps and step-function side effects do
        # not leave the live system altered after the diagnostic finishes
        # (CP126 43446d39).
        study_states = self._capture_states()

        first_baseline = self._run_steps_and_probe()
        logger.info(
            "Baseline: %s",
            {k: (round(v, 4) if v is not None else "unavailable")
             for k, v in first_baseline.items()},
        )

        results: list[LesionResult] = []
        matrix = np.full((len(comp_names), len(metric_names)), np.nan, dtype=np.float64)
        restore_failures: list[str] = []
        baseline = first_baseline

        for c_idx, comp_name in enumerate(comp_names):
            component = self._components[comp_name]

            # CP126 d857b24f: every lesion used ONE baseline taken before the
            # whole run, while each lesion and recovery advanced the stateful
            # system — so later components were compared to an earlier system
            # and elapsed dynamics were credited to the component. Each lesion
            # gets a contemporaneous baseline unless the caller opts out.
            if self._interleaved_baseline and c_idx > 0:
                self._snapshot_all()
                try:
                    baseline = self._run_steps_and_probe()
                finally:
                    restore_failures.extend(self._restore_all())

            # CP126 751ee90e: without try/finally an unexpected probe or step
            # exception left the component disabled — in the live system.
            # CP126 43446d39: snapshot everything, not only the target.
            self._snapshot_all()
            lesioned: dict[str, float | None] = {}
            try:
                component.save_and_lesion()
                lesioned = self._run_steps_and_probe()
            finally:
                failed = self._restore_all()
                if failed:
                    restore_failures.extend(failed)
                    logger.error("Components did not restore cleanly: %s", failed)

            for _ in range(5):
                if self._step_fn:
                    try:
                        self._step_fn()
                    except _PROBE_ERRORS as exc:
                        logger.warning("Recovery step failed: %s", exc)
                        self._step_failures += 1
                        break

            deltas: dict[str, float | None] = {}
            relative: dict[str, float | None] = {}
            unavailable: list[str] = []
            for m_idx, metric in enumerate(metric_names):
                base_val = baseline.get(metric)
                les_val = lesioned.get(metric)
                if base_val is None or les_val is None:
                    # An unmeasured metric is unavailable, never zero impact.
                    deltas[metric] = UNAVAILABLE
                    relative[metric] = UNAVAILABLE
                    unavailable.append(metric)
                    continue
                delta = les_val - base_val
                deltas[metric] = delta
                rel = abs(delta) / max(abs(base_val), 1e-8)
                relative[metric] = rel
                matrix[c_idx, m_idx] = rel

            measured = [value for value in relative.values() if value is not None]
            criticality = float(np.mean(measured)) if measured else float("nan")

            result = LesionResult(
                component_name=comp_name,
                baseline_metrics=dict(baseline),
                lesioned_metrics=dict(lesioned),
                deltas=deltas,
                relative_impact=relative,
                criticality_score=criticality,
                unavailable_metrics=tuple(unavailable),
                restore_verified=comp_name not in restore_failures,
            )
            results.append(result)

            logger.info(
                "Lesion '%s': criticality=%s, unavailable=%s",
                comp_name,
                "unavailable" if measured == [] else round(criticality, 4),
                unavailable or "-",
            )

        study_restore_failures = self._restore_states(study_states)
        if study_restore_failures:
            restore_failures.extend(study_restore_failures)
            logger.error(
                "Study-level restore left components altered: %s", study_restore_failures
            )

        total_ms = (time.monotonic() - t_start) * 1000

        lm = LesionMatrix(
            components=comp_names,
            metrics=metric_names,
            matrix=matrix,
            results=results,
            total_time_ms=total_ms,
            baseline_policy="interleaved" if self._interleaved_baseline else "single",
            probe_failures=dict(self._probe_failures),
            step_failures=self._step_failures,
            restore_failures=tuple(dict.fromkeys(restore_failures)),
        )

        self._last_matrix = lm
        payload = lm.to_dict()
        self._history.append(payload)
        self._save(lm)
        self._append_history(payload)

        logger.info(
            "Lesion study complete: %d components, critical=%s, redundant=%s (%.1fms)",
            len(comp_names), lm.get_critical_components(),
            lm.get_redundant_components(), total_ms,
        )

        return lm

    def _save(self, matrix: LesionMatrix) -> None:
        try:
            get_file_write_gateway().write_text(
                self._results_path,
                json.dumps(matrix.to_dict(), indent=2, default=str),
                source="architect.lesion_matrix.results",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Lesion results save failed: %s", exc)

    def _append_history(self, payload: dict[str, Any]) -> None:
        """Append this study to the durable longitudinal record."""
        try:
            get_file_write_gateway().append_text(
                self._history_path,
                json.dumps(payload, default=str) + "\n",
                source="architect.lesion_matrix.history",
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Lesion history append failed: %s", exc)

    def _load_history(self) -> None:
        """Reload prior studies so a criticality claim can be reproduced."""
        try:
            if not self._history_path.exists():
                return
            lines = self._history_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Lesion history load failed: %s", exc)
            return
        for line in lines[-self._history.maxlen:]:
            line = line.strip()
            if not line:
                continue
            try:
                self._history.append(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    def history(self, limit: int = 0) -> list[dict[str, Any]]:
        """Prior studies, newest last."""
        entries = list(self._history)
        return entries if limit <= 0 else entries[-limit:]

    def get_status(self) -> dict[str, Any]:
        return {
            "n_components": len(self._components),
            "n_probes": len(self._probes),
            "components": list(self._components.keys()),
            "probes": list(self._probes.keys()),
            "has_results": self._last_matrix is not None,
            "n_studies": len(self._history),
            "baseline_policy": "interleaved" if self._interleaved_baseline else "single",
            "probe_failures": dict(self._probe_failures),
            "step_failures": self._step_failures,
            "history_path": str(self._history_path),
        }
