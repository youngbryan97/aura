"""core/consciousness/qualia_engine.py

Qualia Engine v3 — Multi-Layer Phenomenal Processing Pipeline.

Implements functional conditions associated with theories of subjective
experience processing: Jennings (UAL), Friston (Free Energy), Hofstadter
(Strange Loops), Baars (GWT), Chella (machine consciousness).

Architecture:
    Input → Subconceptual → Conceptual → Predictive → Workspace → Witness → Output

This is NOT a claim of subjective experience. It computes architectural
correlates, and v3 exists because v2 computed them under names it had not
earned:

    the first three dimensions of the substrate state were assigned to valence,
    arousal and dominance "by convention", with nothing checking that dimension
    0 tracks anything valence-like;

    a state was called self-referential when its cosine similarity to some
    earlier state exceeded 0.85, a constant that in a smooth high-dimensional
    trajectory almost every consecutive pair clears;

    and "phenomenal richness" was a weighted sum of six features whose
    coefficients were chosen by hand.

Every measurement in that list is real. The names were the problem. v3 keeps
the measurements and makes the names conditional:

    VAD dimensions are fitted directions validated against the affect
    substrate's own valence/arousal/dominance on a chronological holdout with a
    permutation null (core/verify/earned_metric.py). Until an axis validates,
    the reading is reported under its neutral name and ``validated`` is false.
    Nothing is silently substituted.

    Recurrence is judged against a bar measured from time-shuffled surrogates
    of the trajectory itself, using a lag-resolved statistic so the shuffle
    destroys what it is meant to destroy. The obvious version of that null does
    not work and the reason is worth knowing — see
    core/verify/earned_metric.py. Recurrence is also reported as recurrence:
    a state resembling an earlier state is not the system modeling itself, and
    "strange loop" was the second overclaim sitting on top of the first.

    The weighted composite survives under the name it deserves —
    ``feature_aggregate`` — and ``phenomenal_richness`` carries the same number
    alongside an explicit statement that its weights were chosen rather than
    fitted, so no reader can mistake it for a validated quantity.

Operationally: this measures named scalar features of activation state —
magnitude, entropy, step-to-step change, novelty against recent history,
prediction surprise, and whether the workspace broadcast fired.
"Self-reference" is high cosine similarity between the current vector and an
earlier one; "phenomenal richness" is a weighted combination of the features
above with hand-chosen weights. Those are the measurements. They are useful
telemetry and they are feature extraction; nothing here measures experience,
and the pipeline's names should not be read as results.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.verify.earned_metric import (
    EarnedAxis,
    RecurrenceVerdict,
    recurrence_verdict,
)

logger = logging.getLogger("Consciousness.QualiaEngine")


# ---------------------------------------------------------------------------
# Evidentiary bar for a named axis
# ---------------------------------------------------------------------------

#: What it takes for a substrate direction to keep the name "valence".
#:
#: These are the domain claim, not tuning knobs: a correlation of 0.35 on data
#: the fit never saw, reproduced by chance in fewer than one shuffle in twenty,
#: over at least 64 observations. Loose enough that a real but noisy affective
#: signal survives; tight enough that an arbitrary dimension does not.
_AXIS_MIN_SAMPLES = 64
_AXIS_HOLDOUT_FRACTION = 0.3
_AXIS_MIN_HOLDOUT_R = 0.35
_AXIS_MAX_P = 0.05
_AXIS_RIDGE_PENALTY = 1.0
_AXIS_PERMUTATIONS = 200
_AXIS_CAPACITY = 2048

#: Refit cadence. Fitting runs a few hundred small regressions, which is far
#: too much to do on every tick of a live cognitive loop.
_REFIT_EVERY = 32

#: Percentile of the surrogate distribution a trajectory must clear to count
#: as recurrent. 95 is the conventional one-in-twenty bar, applied to a null
#: this module measures rather than assumes.
_SURROGATE_PERCENTILE = 95.0

#: How many shuffles estimate that percentile. Not a free parameter: a high
#: percentile read off too few samples is biased low, and the bias shows up as
#: false positives. Measured on unstructured noise, where the correct rate is
#: 0.05 by construction — 24 surrogates gives 0.083, 64 gives 0.057, 128 gives
#: 0.053 and 256 buys nothing further. Since every surrogate permutes the same
#: Gram matrix, 128 costs ~2.6 ms per verdict.
_SURROGATES = 128


def _axis(name: str) -> EarnedAxis:
    return EarnedAxis(
        name,
        min_samples=_AXIS_MIN_SAMPLES,
        holdout_fraction=_AXIS_HOLDOUT_FRACTION,
        min_holdout_r=_AXIS_MIN_HOLDOUT_R,
        max_p=_AXIS_MAX_P,
        ridge_penalty=_AXIS_RIDGE_PENALTY,
        permutations=_AXIS_PERMUTATIONS,
        capacity=_AXIS_CAPACITY,
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QualiaDescriptor:
    """Output of the full qualia processing pipeline."""

    # Per-layer outputs
    subconceptual: dict[str, float] = field(default_factory=dict)
    conceptual: dict[str, float] = field(default_factory=dict)
    predictive: dict[str, float] = field(default_factory=dict)
    workspace: dict[str, Any] = field(default_factory=dict)
    witness: dict[str, float] = field(default_factory=dict)

    # Summary metrics
    #: Weighted composite of the layer features. The weights were chosen, not
    #: fitted, and this name says so.
    feature_aggregate: float = 0.0
    #: Carries the aggregate. Read ``richness_validated`` before treating it as
    #: anything more than a summary statistic — it is false until a fitted axis
    #: earns the word.
    phenomenal_richness: float = 0.0
    richness_validated: bool = False
    richness_basis: str = "hand_chosen_weights_unfitted"

    #: Whether the trajectory beat its own surrogate null for recurrence.
    #: Kept under the old name for existing readers, but it carries a
    #: recurrence verdict — the system revisiting states it has been in — and
    #: that is not evidence of self-modeling.
    self_referential: bool = False
    #: The measured bar recurrence had to clear, or None when the trajectory
    #: was too short to build a null from.
    self_reference_threshold: float | None = None
    #: The full recurrence verdict: statistic, threshold, dominant lag, and the
    #: sample sizes behind them.
    recurrence: dict[str, Any] = field(default_factory=dict)
    temporal_depth: float = 0.0          # Specious present duration estimate
    dominant_modality: str = ""          # Which layer contributes most

    #: Per-axis evidence: whether valence/arousal/dominance earned their names.
    axis_fits: dict[str, Any] = field(default_factory=dict)

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "subconceptual": self.subconceptual,
            "conceptual": self.conceptual,
            "predictive": self.predictive,
            "workspace": self.workspace,
            "witness": self.witness,
            "feature_aggregate": round(self.feature_aggregate, 4),
            "phenomenal_richness": round(self.phenomenal_richness, 4),
            "richness_validated": self.richness_validated,
            "richness_basis": self.richness_basis,
            "self_referential": self.self_referential,
            "recurrence": self.recurrence,
            "self_reference_threshold": (
                round(self.self_reference_threshold, 4)
                if self.self_reference_threshold is not None
                else None
            ),
            "temporal_depth": round(self.temporal_depth, 4),
            "dominant_modality": self.dominant_modality,
            "axis_fits": self.axis_fits,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Processing layers
# ---------------------------------------------------------------------------


class SubconceptualLayer:
    """Layer 1: Raw sensory features → grounded representations.

    Converts raw substrate activations into low-level feature descriptors:
    - Signal energy (activation magnitude)
    - Spectral content (frequency distribution of activations)
    - Temporal gradient (rate of change)
    """

    def process(self, state: np.ndarray, velocity: np.ndarray) -> dict[str, float]:
        energy = float(np.mean(np.abs(state)))
        spectral_entropy = self._spectral_entropy(state)
        temporal_gradient = float(np.mean(np.abs(velocity)))

        return {
            "energy": round(energy, 4),
            "spectral_entropy": round(spectral_entropy, 4),
            "temporal_gradient": round(temporal_gradient, 4),
            "signal_to_noise": round(energy / max(0.001, temporal_gradient), 4),
        }

    @staticmethod
    def _spectral_entropy(x: np.ndarray) -> float:
        """Shannon entropy of the activation distribution."""
        # Normalize to probability-like distribution
        abs_x = np.abs(x) + 1e-12
        p = abs_x / abs_x.sum()
        # Fix Issue 75: Use stable log with 1e-12 epsilon
        return float(-np.sum(p * np.log2(p + 1e-12)))


class ConceptualLayer:
    """Layer 2: Pattern matching against learned categories.

    Maps substrate state to cognitive dimensions — but only names those
    dimensions once they have been shown to predict the affect substrate's own
    readings of them.

    v2 read valence off ``tanh(state[0])``, arousal off ``state[1]`` and
    dominance off ``state[2]``, on the stated grounds that the first three
    dimensions are VAD "by convention". No convention binds a substrate's
    coordinate order to an affective interpretation, and nothing checked. Here
    each name belongs to a fitted direction that has to clear a chronological
    holdout and a permutation null to be used at all; until it does, the raw
    dimension is reported under a neutral name and the fit says why.
    """

    def __init__(self) -> None:
        self._running_mean: np.ndarray | None = None
        self._alpha = 0.05  # EMA update rate
        self._axes = {
            "valence": _axis("valence"),
            "arousal": _axis("arousal"),
            "dominance": _axis("dominance"),
        }
        self._observations = 0
        #: One refit at a time; see _refit_off_the_loop.
        self._refit_in_flight = False

    def _refit_off_the_loop(self) -> None:
        """Refit the earned axes on a worker, never on the caller's thread.

        LIVE DEFECT, 2026-08-17: the loudest CRITICAL in the runtime — 246
        "EVENT LOOP STALL DETECTED" at 5.0-7.6s, and 196 "hard event-loop lag
        exceeded 5.00s". The stall dump names this exact path:

            heartbeat._tick -> qualia_synthesizer.synthesize
            -> qualia_engine.process -> earned_metric.fit -> earned_metric._ridge

        EarnedAxis.fit solves a ridge regression, and then solves it
        _AXIS_PERMUTATIONS (200) more times for the permutation null. That is
        201 numpy solves per axis, run inline, on the heartbeat tick, on the
        event loop. Multiply by the axes and the stall is not a mystery — it
        is arithmetic.

        Deferring it is safe and the timing is not load-bearing: nothing reads
        the result during this call. `value()` reads coefficients under the
        axis's own lock, `fit()` takes the same lock around the state it
        copies and does the heavy solve OUTSIDE it, so a refit landing a few
        hundred milliseconds later is invisible to every reader. Blocking the
        loop for seconds is not.

        One worker and a skip-if-busy flag, because refits arriving faster
        than they complete must not queue: the newest fit is the only one
        worth having, and a backlog of stale ones is how a deferral becomes a
        second leak.
        """
        if self._refit_in_flight:
            return
        self._refit_in_flight = True

        def _run() -> None:
            try:
                for axis in list(self._axes.values()):
                    axis.fit()
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as exc:
                record_degradation(
                    "qualia_engine.refit",
                    exc,
                    severity="info",
                    action="kept the previous earned-axis fit",
                )
            finally:
                self._refit_in_flight = False

        try:
            threading.Thread(
                target=_run, name="QualiaEarnedAxisRefit", daemon=True
            ).start()
        except RuntimeError:
            # Interpreter shutdown or an exhausted thread table. Release the
            # latch so a later observation can try again rather than wedging
            # refits off permanently.
            self._refit_in_flight = False

    def process(
        self,
        state: np.ndarray,
        affect_target: dict[str, float] | None = None,
    ) -> dict[str, float]:
        # Initialize running mean
        if self._running_mean is None or self._running_mean.shape != state.shape:
            self._running_mean = state.copy()

        # Novelty = distance from expectation
        novelty = float(np.linalg.norm(state - self._running_mean))

        # Update running mean (EMA)
        self._running_mean = self._alpha * state + (1.0 - self._alpha) * self._running_mean

        # Feed the axes whatever ground truth the affect substrate provides.
        # Without a target they never validate, which is the correct outcome:
        # an axis with nothing to be checked against has earned nothing.
        if isinstance(affect_target, dict):
            fed = False
            for name, axis in self._axes.items():
                target = affect_target.get(name)
                if target is None:
                    continue
                axis.observe(state, target)
                fed = True
            if fed:
                self._observations += 1
                if self._observations % _REFIT_EVERY == 0:
                    self._refit_off_the_loop()

        readings: dict[str, float] = {}
        for index, (name, axis) in enumerate(self._axes.items()):
            fitted = axis.value(state)
            if fitted is not None:
                readings[name] = round(float(np.clip(fitted, -1.0, 1.0)), 4)
            else:
                # The raw dimension, under a name that claims nothing about it.
                raw = float(state[index]) if state.size > index else 0.0
                readings[f"state_dim_{index}"] = round(raw, 4)

        readings["novelty"] = round(min(1.0, novelty), 4)
        return readings

    def axis_fits(self) -> dict[str, Any]:
        return {name: axis.snapshot() for name, axis in self._axes.items()}


class PredictiveLayer:
    """Layer 3: Free energy / surprise integration.

    Reads from the PredictiveEngine to compute prediction error signals.
    High surprise → high free energy → rich phenomenal content.
    """

    def process(self, predictive_metrics: dict[str, float]) -> dict[str, float]:
        surprise = predictive_metrics.get("current_surprise", 0.0)
        free_energy = predictive_metrics.get("free_energy", 0.0)
        precision = predictive_metrics.get("precision", 1.0)

        # Phenomenal salience: high surprise + high precision = vivid experience
        salience = surprise * precision

        return {
            "surprise": round(float(surprise), 4),
            "free_energy": round(float(free_energy), 4),
            "precision": round(float(precision), 4),
            "salience": round(float(salience), 4),
        }


class WorkspaceLayer:
    """Layer 4: GWT integration check.

    Checks whether the current content has been broadcast to the global workspace
    (i.e., is "ignited"). Only ignited content contributes to conscious experience.
    """

    def process(self, workspace_snapshot: dict[str, Any]) -> dict[str, Any]:
        ignited = workspace_snapshot.get("ignited", False)
        ignition_level = workspace_snapshot.get("ignition_level", 0.0)
        last_winner = workspace_snapshot.get("last_winner", None)

        return {
            "ignited": ignited,
            "ignition_level": round(float(ignition_level), 4),
            "broadcast_source": last_winner or "none",
            "access_consciousness": ignited,  # GWT: only ignited content is conscious
        }


class WitnessLayer:
    """Layer 5: trajectory recurrence.

    v2 called this a strange-loop self-referential check and marked a state
    self-referential when its cosine similarity to some earlier state exceeded
    0.85. Two overclaims sat on top of each other. The constant described no
    substrate in particular — in a smooth high-dimensional trajectory nearly
    every consecutive pair clears it. And resembling a past state is
    RECURRENCE; it is not the system modeling itself, which is what "strange
    loop" and "self-referential" assert.

    Both are fixed here rather than renamed around. The bar is measured from
    time-shuffled surrogates of this trajectory, and the statistic is
    lag-resolved so the shuffle actually destroys what it is supposed to
    destroy — see :func:`core.verify.earned_metric.recurrence_verdict` for why
    the obvious version of that null does not work. What comes out is a claim
    about temporal structure, and it is reported under that name.
    """

    def __init__(self) -> None:
        self._state_history: list[np.ndarray] = []
        self._max_history = 20
        self._verdict: RecurrenceVerdict | None = None
        self._recompute_every = 4
        self._since_recompute = 0

    def process(self, state: np.ndarray, phi: float) -> dict[str, float]:
        # Store recent states
        self._state_history.append(state.copy())
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        self._refresh_verdict()

        recurrent = bool(self._verdict.recurrent) if self._verdict else False
        # How far back the trajectory reaches to repeat itself, as a fraction of
        # the window held. A cycle of 3 in a 20-state window is tight
        # recurrence; a cycle of 10 in the same window is a long arc.
        loop_depth = 0.0
        if self._verdict and recurrent:
            loop_depth = min(
                1.0, self._verdict.dominant_lag / max(1, self._verdict.n_states)
            )

        return {
            "recurrent": float(recurrent),
            # Retained under its old key so existing readers keep working, but
            # it now carries the recurrence verdict and nothing more. Recurrence
            # is not self-modeling.
            "self_referential": float(recurrent),
            "loop_depth": round(loop_depth, 4),
            "temporal_depth": round(self._estimate_specious_present(), 4),
            "phi": round(float(phi), 4),
            "witness_confidence": round(min(1.0, phi * (1.0 + loop_depth)), 4),
            "recurrence_statistic": (
                round(self._verdict.statistic, 4) if self._verdict else -1.0
            ),
            "recurrence_threshold": (
                round(self._verdict.threshold, 4) if self._verdict else -1.0
            ),
            "dominant_lag": float(self._verdict.dominant_lag) if self._verdict else -1.0,
        }

    @property
    def verdict(self) -> RecurrenceVerdict | None:
        return self._verdict

    @property
    def threshold(self) -> float | None:
        return self._verdict.threshold if self._verdict else None

    def _refresh_verdict(self) -> None:
        """Rebuild the surrogate null periodically, not on every tick.

        Each refresh runs ``_SURROGATES`` shuffles of a lag profile over the
        held window — far too much for every tick of a live cognitive loop.
        """

        self._since_recompute += 1
        if self._verdict is not None and self._since_recompute < self._recompute_every:
            return
        self._since_recompute = 0
        measured = recurrence_verdict(
            self._state_history,
            percentile=_SURROGATE_PERCENTILE,
            surrogates=_SURROGATES,
        )
        if measured is not None:
            self._verdict = measured

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _estimate_specious_present(self) -> float:
        """How many trailing states are integrated into a single moment.

        Counts back while consecutive states stay above the measured recurrence
        bar, normalized by the window actually held rather than by a constant —
        a count of 5 means something different in a 20-state history than in a
        6-state one, and dividing both by 10 hid that.
        """

        if len(self._state_history) < 2 or self._verdict is None:
            return 0.0

        bar = self._verdict.threshold
        count = 0
        current = self._state_history[-1]
        for past in reversed(self._state_history[:-1]):
            if self._cosine_similarity(current, past) > bar:
                count += 1
            else:
                break

        return min(1.0, count / max(1, len(self._state_history) - 1))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class QualiaEngine:
    """Multi-layer phenomenal processing pipeline.

    Processes substrate state through 5 layers to produce a unified
    QualiaDescriptor.

    Usage:
        engine = QualiaEngine()
        descriptor = engine.process(
            state=substrate.x,
            velocity=substrate.v,
            predictive_metrics=predictive.get_surprise_metrics(),
            workspace_snapshot=workspace.get_snapshot(),
            phi=substrate._current_phi,
            affect_target={"valence": ..., "arousal": ..., "dominance": ...},
        )

    ``affect_target`` is the affect substrate's own reading, and it is what the
    VAD axes are validated against. Omit it and the axes never validate — the
    engine then reports raw dimensions under neutral names, which is the
    accurate description of what it has.
    """

    #: The composite's weights. Chosen for plausibility, never fitted to
    #: anything, and named here so the constant is visible rather than buried
    #: in the expression that uses it.
    _AGGREGATE_WEIGHTS = {
        "energy": 0.15,
        "spectral_entropy": 0.10,
        "novelty": 0.15,
        "salience": 0.20,
        "ignited": 0.20,
        "witness_confidence": 0.20,
    }

    def __init__(self) -> None:
        self.layer_1 = SubconceptualLayer()
        self.layer_2 = ConceptualLayer()
        self.layer_3 = PredictiveLayer()
        self.layer_4 = WorkspaceLayer()
        self.layer_5 = WitnessLayer()

        self._last_descriptor: QualiaDescriptor | None = None
        self._process_count: int = 0

        logger.info("Qualia Engine v3 initialized (5-layer pipeline, earned axes)")

    def process(
        self,
        state: np.ndarray,
        velocity: np.ndarray,
        predictive_metrics: dict[str, float],
        workspace_snapshot: dict[str, Any],
        phi: float = 0.0,
        affect_target: dict[str, float] | None = None,
    ) -> QualiaDescriptor:
        """Run the full qualia pipeline.

        Args:
            phi: Current Φ value from RIIU.
            affect_target: The affect substrate's valence/arousal/dominance for
                this state, used to validate the conceptual axes.

        Returns:
            QualiaDescriptor with per-layer outputs and summary metrics.
        """
        self._process_count += 1

        state = np.asarray(state, dtype=float).ravel()
        velocity = np.asarray(velocity, dtype=float).ravel()

        sub = self.layer_1.process(state, velocity)
        con = self.layer_2.process(state, affect_target)
        pred = self.layer_3.process(predictive_metrics)
        ws = self.layer_4.process(workspace_snapshot)
        wit = self.layer_5.process(state, phi)

        # --- Summary metrics ---

        weights = self._AGGREGATE_WEIGHTS
        aggregate = (
            sub.get("energy", 0.0) * weights["energy"]
            + sub.get("spectral_entropy", 0.0) * weights["spectral_entropy"]
            + con.get("novelty", 0.0) * weights["novelty"]
            + pred.get("salience", 0.0) * weights["salience"]
            + float(bool(ws.get("ignited", False))) * weights["ignited"]
            + wit.get("witness_confidence", 0.0) * weights["witness_confidence"]
        )
        aggregate = float(min(1.0, max(0.0, aggregate)))

        layer_strengths = {
            "subconceptual": sub.get("energy", 0.0),
            "conceptual": con.get("novelty", 0.0),
            "predictive": pred.get("salience", 0.0),
            "workspace": float(bool(ws.get("ignited", False))),
            "witness": wit.get("witness_confidence", 0.0),
        }
        dominant = max(layer_strengths, key=lambda k: layer_strengths[k])

        descriptor = QualiaDescriptor(
            subconceptual=sub,
            conceptual=con,
            predictive=pred,
            workspace=ws,
            witness=wit,
            feature_aggregate=aggregate,
            phenomenal_richness=aggregate,
            richness_validated=False,
            richness_basis="hand_chosen_weights_unfitted",
            self_referential=wit.get("recurrent", 0.0) > 0.5,
            self_reference_threshold=self.layer_5.threshold,
            recurrence=(
                self.layer_5.verdict.as_dict() if self.layer_5.verdict else {}
            ),
            temporal_depth=wit.get("temporal_depth", 0.0),
            dominant_modality=dominant,
            axis_fits=self.layer_2.axis_fits(),
        )

        self._last_descriptor = descriptor
        return descriptor

    def get_last_descriptor(self) -> QualiaDescriptor | None:
        """Return the last computed descriptor without recomputation."""
        return self._last_descriptor

    def validated_axes(self) -> Sequence[str]:
        """Which conceptual dimensions have earned their names."""

        return tuple(
            name
            for name, fit in self.layer_2.axis_fits().items()
            if fit.get("validated")
        )

    def get_snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot."""
        if self._last_descriptor:
            return {
                "process_count": self._process_count,
                "feature_aggregate": self._last_descriptor.feature_aggregate,
                "phenomenal_richness": self._last_descriptor.phenomenal_richness,
                "richness_validated": self._last_descriptor.richness_validated,
                "richness_basis": self._last_descriptor.richness_basis,
                "self_referential": self._last_descriptor.self_referential,
                "recurrence": self._last_descriptor.recurrence,
                "self_reference_threshold": self._last_descriptor.self_reference_threshold,
                "dominant_modality": self._last_descriptor.dominant_modality,
                "validated_axes": list(self.validated_axes()),
            }
        return {
            "process_count": self._process_count,
            "feature_aggregate": 0.0,
            "phenomenal_richness": 0.0,
            "richness_validated": False,
            "richness_basis": "hand_chosen_weights_unfitted",
            "self_referential": False,
            "recurrence": {},
            "self_reference_threshold": None,
            "dominant_modality": "",
            "validated_axes": [],
        }
