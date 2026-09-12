"""Which memories come back, and in what order.

Recency alone gives her yesterday over the thing she needs; similarity alone
gives her the same three episodes forever. The competitive rank is the one that
decides, with the presentation history damping what has just been shown and the
qualia boost lifting what was felt strongly. Every score here is a number the
recall receipt can carry, so a strange recall can be read back rather than
guessed at.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only; that module imports this one
    from .episodic_memory import Episode

import math
import time

from core.cognition.actr_activation import base_level_activation
from core.memory.engram_association import (
    get_engram_association_field,
    is_engram_association_enabled,
)
from core.memory.engram_plasticity import (
    get_engram_plasticity_field,
    is_engram_plasticity_enabled,
)
from core.memory.hippocampus import HippocampalIndex
from core.memory.recall_observations import record_ranking
from core.runtime.errors import record_degradation
from core.utils.exceptions import capture_and_log


class _RanksWhatToRecall:
    """Lifted whole from EpisodicMemory; see episodic_memory.py."""

    @staticmethod
    def _presentation_history(ep: "Episode", now: float) -> list[float]:
        """Reconstruct when this trace was used, from the fields the store keeps.

        The store records an encoding time, an access count and the most recent
        access, but not the full list of accesses. The two obvious readings of
        that are both wrong at one end — treating every access as if it
        happened at encoding ignores rehearsal, treating them all as if they
        happened at ``last_accessed`` ignores age — so the intermediate
        accesses are spread evenly between the two timestamps that *are*
        recorded. Both endpoints are then exact and only the interior is
        modelled.

        Only ``timestamp`` is required. Rehearsal data is an optional strength
        signal, and a trace that carries none is a trace used once — which is a
        meaningful answer, not a missing one. Degraded and duck-typed records
        reach this path in practice, and refusing to rank them would turn a
        partial memory into no memory.
        """
        history = [ep.timestamp]
        extra = max(0, int(getattr(ep, "access_count", 0) or 0))
        if extra:
            recorded = float(getattr(ep, "last_accessed", 0.0) or 0.0)
            last = recorded if recorded > ep.timestamp else now
            if extra == 1:
                history.append(last)
            else:
                step = (last - ep.timestamp) / extra
                history.extend(ep.timestamp + step * (i + 1) for i in range(extra))
        return history

    @staticmethod
    def _recency_score(ep: "Episode", now: float | None = None) -> float:
        """Base-level activation of this trace: ``B = ln(Σ t_j^-d)``.

        This was ``min(1.0, max(0.0, ep.timestamp - 1774000000) / 2000000)``,
        which is not a recency score. It measured position against a hardcoded
        wall-clock epoch, so it was a step function: 0.0 before 2026-03-20, a
        23-day ramp, then a flat 1.0 for everything after 2026-04-12. Evaluated
        on 2026-08-12 an episode from one minute ago and one from thirty days
        ago both scored exactly 1.000000, so this term added a constant 0.4 to
        every candidate in ``_static_rank`` and the ranking was importance-only.
        Every day that passed pushed the usable window further into the past.

        Activation depends only on *elapsed* time, so it cannot saturate and
        has no epoch to go stale, and it reads frequency as well as recency — a
        trace recalled often is stronger than one merely stored recently, which
        the previous scorer had no way to express even inside the window where
        it still had a gradient.

        The value is unbounded below (log activation, ``-inf`` for a trace with
        no recorded use). :meth:`_static_rank` normalises across the candidate
        set rather than squashing here, so no threshold or noise constant has
        to be invented to make the number comparable to ``importance``.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .episodic_memory import (
            EpisodicMemory,
        )

        moment = time.time() if now is None else now
        return base_level_activation(
            EpisodicMemory._presentation_history(ep, moment), moment
        )

    def _static_rank(
        self, episodes: list["Episode"], now: float | None = None
    ) -> list["Episode"]:
        """Rank by importance blended with activation, normalised over the batch.

        The 0.6/0.4 split is unchanged. What changed is that the second term
        now varies: activation is min-max normalised across exactly the
        candidates being ranked, so the most active trace contributes the full
        0.4 and the least contributes none. Normalising against the batch
        instead of an absolute scale keeps the blend meaningful without
        introducing a retrieval threshold — the memory system declares a
        retention policy in items, not in seconds, so there is no principled
        wall-clock anchor available to convert activation into [0, 1].

        ``now`` is injectable so the ordering can be tested at a chosen wall
        clock. That matters more than it sounds: the defect this replaced was
        precisely a dependence on *when* the code ran, and a test that cannot
        move the clock cannot detect one.
        """
        if len(episodes) < 2:
            return list(episodes)

        moment = time.time() if now is None else now
        activations = [self._recency_score(ep, moment) for ep in episodes]
        finite = [a for a in activations if math.isfinite(a)]
        low = min(finite) if finite else 0.0
        high = max(finite) if finite else 0.0
        span = high - low

        def normalised(value: float) -> float:
            if not math.isfinite(value):
                return 0.0  # never used: below every recorded trace
            return (value - low) / span if span > 0.0 else 1.0

        scored = [
            ((ep.importance * 0.6) + (normalised(a) * 0.4), index, ep)
            for index, (ep, a) in enumerate(zip(episodes, activations, strict=True))
        ]
        # index breaks ties deterministically; Episode is not orderable.
        scored.sort(key=lambda item: (-item[0], item[1]))

        return [ep for _, _, ep in scored]

    def _observe_ranked_recall(
        self,
        episodes: list["Episode"],
        *,
        returned_count: int,
        now: float | None = None,
    ) -> None:
        """Record ACT-R activation against the final live retrieval decision.

        Observation belongs after every ranking stage. Emitting it inside the
        static fallback mislabeled competitive-ranker decisions and could not
        know the caller's actual limit.
        """
        if len(episodes) < 2:
            return
        moment = time.time() if now is None else now
        record_ranking(
            (self._recency_score(episode, moment) for episode in episodes),
            returned_count=returned_count,
        )

    def _competitive_rank(self, episodes: list["Episode"], query: str) -> list["Episode"]:
        """Re-rank candidates through the engram plasticity competition field.

        Salience for each engram blends query-cue relevance (so the trace that
        actually matches drives hardest), current strength, and importance. The
        field's substrate context (arousal/valence) modulates the activation
        threshold and temperature. On any failure this degrades cleanly to the
        static importance+recency ranking.
        """
        from .episodic_memory import (
            logger,
        )

        if len(episodes) < 2 or not is_engram_plasticity_enabled():
            return self._static_rank(episodes)
        try:
            query_cues = set(HippocampalIndex.extract_cues(query or ""))
            assoc_field = (
                get_engram_association_field()
                if is_engram_association_enabled() else None
            )
            salience: list[float] = []
            for ep in episodes:
                ep_cues = set(HippocampalIndex.extract_cues(ep.full_description))
                overlap = (
                    len(query_cues & ep_cues) / max(1, len(query_cues))
                    if query_cues else 0.0
                )
                relevance = max(0.05, min(1.0, overlap))
                strength = ep.current_strength()
                drive = relevance * (0.6 + 0.4 * strength) * (0.7 + 0.3 * ep.importance)
                # Learned-association boost: engrams this query has become wired to
                # through prior co-recall (voltage-STDP) get surfaced even when
                # their surface cues don't overlap — associative pattern completion.
                if assoc_field is not None and query_cues:
                    boost = assoc_field.association_boost(list(query_cues), list(ep_cues))
                    drive *= (1.0 + min(0.5, boost))
                salience.append(float(drive))

            qualia = self._current_qualia() or {}
            # Substrate coupling: qualia intensity (q_norm) is the arousal proxy —
            # the membrane-potential context that gates how readily engrams stay
            # above threshold — and ual valence warms the escape-rate temperature.
            arousal = float(qualia.get("arousal", qualia.get("q_norm", 0.5)))
            valence = float(qualia.get("valence", qualia.get("emotional_valence", 0.0)))

            field = get_engram_plasticity_field()
            result = field.compete(salience, arousal=arousal, valence=valence)
            # Stash per-engram competitive weights so _register_recall can apply
            # bounded LTP consolidation to the winners (recall → strengthening).
            self._last_competition_weights = {
                episodes[i].episode_id: float(result.weights[i])
                for i in range(len(episodes))
                if 0 <= i < len(result.weights)
            }
            if result.governance_breach:
                logger.info(
                    "🧠 [EngramPlasticity] recall homeostatic pressure high "
                    "(%.2f) — one attractor dominating; competition damping it.",
                    result.pressure,
                )
            ranked = [episodes[i] for i in result.order if 0 <= i < len(episodes)]
            # Append any indices the competition dropped (gated-out) at the tail,
            # preserving them as low-priority rather than losing them entirely.
            seen = {id(ep) for ep in ranked}
            ranked.extend(ep for ep in episodes if id(ep) not in seen)
            return ranked or self._static_rank(episodes)
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            record_degradation("episodic_memory", exc)
            logger.debug("Competitive rank fell back to static: %s", exc)
            return self._static_rank(episodes)

    def _apply_qualia_boost(self, episodes: list[Episode]) -> list[Episode]:
        """Re-rank episodes by qualia congruence with current phenomenal state."""

        try:
            from core.container import ServiceContainer
            qualia = ServiceContainer.get("qualia_synthesizer", default=None)
            if not qualia or qualia.q_norm < 0.1:
                return episodes  # No qualia data — skip boosting

            current = qualia.get_qualia_for_memory()
            current_norm = current.get("q_norm", 0.0)
            current_dim = current.get("dominant_dim", "")

            def congruence_score(ep: Episode) -> float:
                qs = ep.qualia_snapshot
                if not qs:
                    return ep.importance
                # Similarity: norm proximity + dimension match
                norm_sim = 1.0 - min(1.0, abs(qs.get("q_norm", 0) - current_norm))
                dim_bonus = 0.2 if qs.get("dominant_dim") == current_dim else 0.0
                return ep.importance + (norm_sim * 0.3) + dim_bonus

            episodes.sort(key=congruence_score, reverse=True)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('episodic_memory', e)
            capture_and_log(e, {'module': __name__})
        return episodes
