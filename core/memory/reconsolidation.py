"""Memory reconsolidation — recalled memories become labile and are rewritten.

A consolidated memory is *not* a static recording. Retrieving it can return the
trace to a transiently labile state during which the present context "seeps in":
the memory can be strengthened, weakened, or updated before it restabilises.
Repeated retrieval makes a memory feel more vivid and confident while it drifts
further from the original encoding — vividness is not accuracy.

Grounded in the reconsolidation literature:
  * Nader, Lee & Schiller (2017) "An Update on Memory Reconsolidation Updating" —
    reactivation makes a stored memory labile; during restabilisation it can be
    updated with new information.
  * Speer et al. (2021, Nat Commun) — reinterpreting a memory in a new (positive)
    context adaptively *updates* it, but only after a reminder **and a delay**, not
    after a one-hour delay. → the labile window has a refractory period.
  * Sinclair et al. (2019) — prediction error (mismatch between what is recalled
    and the present context) drives how much a memory is updated.
  * Zhang et al. (2018); Paul & Asthana (2025) — *boundary conditions*: strong and
    strongly-emotional memories resist destabilisation and update less.
  * Richter, Cooper, Bays & Simons (2016) — vividness/confidence and accuracy are
    dissociable; a memory can be vivid yet wrong.

This module is deliberately pure (no DB, no I/O, no global state) so the dynamics
are unit-testable. :class:`EpisodicMemory` reads the current phenomenal/affective
context, calls :meth:`ReconsolidationEngine.reconsolidate`, and persists the
returned deltas. See ``core/memory/episodic_memory.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ReconsolidationOutcome:
    """Result of presenting a memory with the present context on recall.

    All numeric fields are the *new* values the caller should persist. ``fired``
    is True when the labile window was open and the trace was actually touched.
    ``drifted`` is True when the present context differed enough to rewrite the
    emotional/phenomenal content (a reconsolidation event, not just rehearsal).
    """

    fired: bool
    drifted: bool
    emotional_valence: float
    qualia_snapshot: dict[str, Any]
    importance: float
    decay_rate: float
    fidelity: float
    reconsolidation_count: int
    last_reconsolidated: float
    prediction_error: float = 0.0
    note: str = ""


@dataclass
class ReconsolidationEngine:
    """Computes how a recalled memory should change as the present seeps in.

    Two things can happen when a memory is brought to mind:

    1. **Rehearsal strengthening** — "telling everyone the story etches it deeper."
       Every effective recall nudges importance up a little and slows decay.
    2. **Reconsolidation drift** — if the trace is old enough to have consolidated
       and the present context differs from how it was encoded, the emotional tone
       and phenomenal snapshot blend toward *now*, and fidelity to the original
       drops. Strong / emotional memories resist this (boundary conditions).

    Both are bounded per event and gated by a refractory ``labile_cooldown`` so a
    burst of recalls in one turn cannot runaway-amplify a single memory.
    """

    # A trace re-enters the labile state at most once per this window. Rapid
    # re-recall within the window is a no-op (Speer 2021: updating needs a delay).
    labile_cooldown: float = 3600.0
    # Traces younger than this are still in synaptic consolidation; recall
    # rehearses (strengthens) them but does not yet rewrite their content.
    min_consolidation_age: float = 1800.0
    # Per-event ceilings so change is gradual and accumulates over many recalls.
    max_valence_drift: float = 0.18
    max_fidelity_loss: float = 0.12
    recall_strengthen: float = 0.03
    decay_stabilisation: float = 0.95  # decay_rate *= this on effective recall

    def reconsolidate(
        self,
        *,
        now: float,
        timestamp: float,
        emotional_valence: float,
        original_valence: float | None,
        importance: float,
        decay_rate: float,
        fidelity: float,
        reconsolidation_count: int,
        last_reconsolidated: float,
        current_strength: float,
        qualia_snapshot: dict[str, Any] | None,
        current_qualia: dict[str, Any] | None,
        lability: float = 1.0,
    ) -> ReconsolidationOutcome:
        """``lability`` is the neuromodulatory plasticity gain (ACh/dopamine raise
        it, cortisol impairs it). It scales how much the present can rewrite the
        trace — the model's "chemicals that make the neurons able to change."
        """
        qualia_snapshot = dict(qualia_snapshot or {})
        lability = max(0.2, min(2.5, lability))

        # Refractory window: a memory just touched stays stable for a while.
        if now - last_reconsolidated < self.labile_cooldown:
            return ReconsolidationOutcome(
                fired=False,
                drifted=False,
                emotional_valence=emotional_valence,
                qualia_snapshot=qualia_snapshot,
                importance=importance,
                decay_rate=decay_rate,
                fidelity=fidelity,
                reconsolidation_count=reconsolidation_count,
                last_reconsolidated=last_reconsolidated,
                note="refractory",
            )

        # ---- 1. Rehearsal strengthening (always, on an effective recall) -----
        # Strengthening tapers off as the memory is already strong.
        headroom = max(0.0, 1.0 - importance)
        new_importance = _clamp(importance + self.recall_strengthen * headroom)
        new_decay_rate = max(0.003, decay_rate * self.decay_stabilisation)

        # ---- 2. Reconsolidation drift (only if consolidated + context present)
        age = now - timestamp
        drifted = False
        prediction_error = 0.0
        new_valence = emotional_valence
        new_fidelity = fidelity
        note = "rehearsed"

        if age >= self.min_consolidation_age and current_qualia:
            cur_valence = float(current_qualia.get("valence", 0.0) or 0.0)
            cur_qnorm = float(current_qualia.get("q_norm", 0.0) or 0.0)
            old_qnorm = float(qualia_snapshot.get("q_norm", cur_qnorm) or 0.0)
            dim_mismatch = (
                1.0
                if qualia_snapshot.get("dominant_dim")
                and current_qualia.get("dominant_dim")
                and qualia_snapshot.get("dominant_dim") != current_qualia.get("dominant_dim")
                else 0.0
            )
            # Prediction error: how different is "now" from the stored trace?
            # Valence spans [-1, 1] (range 2), so normalise its delta by 2.
            prediction_error = _clamp(
                0.5 * abs(cur_valence - emotional_valence) / 2.0
                + 0.3 * abs(cur_qnorm - old_qnorm)
                + 0.2 * dim_mismatch
            )

            # Boundary conditions: strong, vivid, emotionally intense traces resist.
            emo_anchor = abs(original_valence if original_valence is not None else emotional_valence)
            emotional_resistance = min(0.85, emo_anchor)
            plasticity = _clamp((1.0 - current_strength) * (1.0 - emotional_resistance))

            # Blend fraction for this single event, scaled by how surprising "now"
            # is and by the current neuromodulatory plasticity gain.
            blend = plasticity * (0.4 + 0.6 * prediction_error) * lability

            if blend > 1e-3:
                # Emotional tone drifts toward the present mood (context seeps in),
                # capped so no single recall can swing it wildly.
                raw_delta = (cur_valence - emotional_valence) * blend
                delta = max(-self.max_valence_drift, min(self.max_valence_drift, raw_delta))
                new_valence = max(-1.0, min(1.0, emotional_valence + delta))

                # Phenomenal snapshot edges toward the present state too.
                qualia_snapshot = self._blend_qualia(qualia_snapshot, current_qualia, blend)

                # Each rewrite costs fidelity to the original, scaled by surprise.
                fidelity_loss = min(self.max_fidelity_loss, self.max_fidelity_loss * blend * (0.5 + prediction_error))
                new_fidelity = _clamp(fidelity - fidelity_loss)

                drifted = abs(delta) > 1e-4 or fidelity_loss > 1e-4
                note = "reconsolidated" if drifted else "rehearsed"

        return ReconsolidationOutcome(
            fired=True,
            drifted=drifted,
            emotional_valence=new_valence,
            qualia_snapshot=qualia_snapshot,
            importance=new_importance,
            decay_rate=new_decay_rate,
            fidelity=new_fidelity,
            reconsolidation_count=reconsolidation_count + (1 if drifted else 0),
            last_reconsolidated=now,
            prediction_error=prediction_error,
            note=note,
        )

    def reconsolidate_in_context(
        self,
        *,
        now: float,
        emotional_valence: float,
        qualia_snapshot: dict[str, Any] | None,
        importance: float,
        fidelity: float,
        reconsolidation_count: int,
        target_valence: float,
        intensity: float = 0.5,
        safe_context: dict[str, Any] | None = None,
    ) -> ReconsolidationOutcome:
        """Deliberate, therapeutic reconsolidation (the "therapy" of the model).

        Revisiting a hurtful memory inside a *safe* context — with helpful
        introspection — lets the present safety seep into the trace, so the
        memory restabilises a little less aversive. Speer et al. (2021) showed
        that finding positive meaning in a negative memory adaptively updates it.

        Unlike spontaneous recall this bypasses the refractory window (it is an
        intentional, repeated revisiting) but is still bounded per call.
        """
        qualia_snapshot = dict(qualia_snapshot or {})
        intensity = _clamp(intensity)
        # Move valence toward the chosen target; the size of the step scales with
        # intensity but is capped so reframing is gradual, never a hard overwrite.
        raw_delta = (target_valence - emotional_valence) * intensity
        cap = self.max_valence_drift * 1.5
        delta = max(-cap, min(cap, raw_delta))
        new_valence = max(-1.0, min(1.0, emotional_valence + delta))

        if safe_context:
            qualia_snapshot = self._blend_qualia(qualia_snapshot, safe_context, intensity * 0.5)

        # A reframed memory is still a *rewritten* memory: fidelity to the raw
        # original drops, even as the experience improves.
        new_fidelity = _clamp(fidelity - self.max_fidelity_loss * intensity)

        return ReconsolidationOutcome(
            fired=True,
            drifted=abs(delta) > 1e-4,
            emotional_valence=new_valence,
            qualia_snapshot=qualia_snapshot,
            importance=importance,  # therapeutic revisiting doesn't inflate salience
            decay_rate=max(0.003, 0.01),
            fidelity=new_fidelity,
            reconsolidation_count=reconsolidation_count + 1,
            last_reconsolidated=now,
            prediction_error=abs(delta),
            note="therapeutic",
        )

    @staticmethod
    def _blend_qualia(
        stored: dict[str, Any], current: dict[str, Any], blend: float
    ) -> dict[str, Any]:
        """Edge numeric phenomenal fields toward the present; adopt present
        categorical fields probabilistically via the blend weight."""
        out = dict(stored)
        for key in ("q_norm", "pri", "valence", "trend"):
            if key in current:
                old = float(stored.get(key, current[key]) or 0.0)
                cur = float(current.get(key, old) or 0.0)
                out[key] = round(old + (cur - old) * blend, 4)
        # The dominant dimension flips to "now" once enough context has bled in.
        if blend >= 0.5 and current.get("dominant_dim"):
            out["dominant_dim"] = current["dominant_dim"]
        out["reconsolidated"] = True
        return out


# Module-level default engine for callers that don't need custom thresholds.
default_engine = ReconsolidationEngine()
