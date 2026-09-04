"""core/interiority/service.py — the runtime service, and where it lands.

This is the object the rest of the runtime holds. It runs the appraisal
engine, the forty-three faculties, and arbitration, and then it does the
part that makes any of it matter: it pushes the arbitrated effects into
subsystems that were already reading those numbers, and it answers
questions those subsystems already asked.

Push, in :meth:`apply`:

* affect deltas reach the one canonical affect engine, so valence,
  arousal and engagement move where the response generator and the
  reasoning-depth budget already read them;
* somatic markers reach the somatic marker gate as an option bias,
  before deliberation rather than after;
* goal deltas reach the drive budgets;
* attention biases reach the curiosity engine's queue.

Pull, which is the half that makes this an organ rather than a
publisher:

* :meth:`appraise` replaces the affect engine's keyword fallback;
* :meth:`attune` replaces the resonance module's word lists;
* :meth:`retention_held` answers the memory-edit ethics check;
* :meth:`permitted` filters an action set before anything scores it;
* :meth:`turn_budget` supplies depth, deadline and the ceiling on how
  irreversible an action this turn may take.

Every push is best-effort and records a degradation rather than raising:
an interior that can take the runtime down is worse than one that is
occasionally quiet. Every pull is deterministic and side-effect free, so
a consumer can call it on a hot path.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from core.interiority.appraisal import AppraisalEngine, AppraisalFrame
from core.interiority.arbitration import Arbitrated, arbitrate, permitted as _permitted
from core.interiority.core_affect import core_affect
from core.interiority.cleft import get_cleft
from core.interiority.effects import BudgetDelta
from core.interiority.evidence import Reading, measured
from core.interiority.event import EventKind, InteriorEvent
from core.interiority.faculty import Activation, FacultyContext, registry
from core.interiority.homes import HOMES
from core.interiority.interoception import get_interoception
from core.interiority.ledger import RelationalLedger
from core.interiority.other_minds import OtherEstimate, get_other_minds_model
from core.interiority.receptors import get_receptor_bank
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Interiority")

SERVICE_NAME = "interiority"


class InteriorityService:
    """Runs the faculties and lands their effects on the runtime."""

    def __init__(self, *, ledger: RelationalLedger | None = None) -> None:
        self._lock = threading.RLock()
        self.ledger = ledger or RelationalLedger()
        self.appraisal = AppraisalEngine(self.ledger)
        self.other_minds = get_other_minds_model()
        self.interoception = get_interoception()
        self._last: Arbitrated | None = None
        self._last_frame: AppraisalFrame | None = None
        self._last_activations: tuple[Activation, ...] = ()
        self._ticks = 0
        self._applied = 0
        #: Faculties switched off, for ablation measurement.
        self._disabled: set[str] = set()
        registry_size = len(registry())
        if registry_size == 0:
            from core.interiority.faculties import load_all

            load_all()

    # ── ablation ──────────────────────────────────────────────────────
    def disable(self, *faculty_ids: str) -> None:
        """Switch faculties off, so their contribution can be measured by
        difference. This is the mechanism the proof harness uses; a claim
        that a faculty matters is a measured delta or it is a hope."""
        with self._lock:
            self._disabled.update(faculty_ids)

    def enable(self, *faculty_ids: str) -> None:
        with self._lock:
            for faculty_id in faculty_ids:
                self._disabled.discard(faculty_id)

    def enable_all(self) -> None:
        with self._lock:
            self._disabled.clear()

    @property
    def disabled(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._disabled))

    # ── the tick ──────────────────────────────────────────────────────
    def tick(
        self,
        event: InteriorEvent,
        *,
        other: OtherEstimate | None = None,
        species: str = "human",
        interior: Mapping[str, Any] | None = None,
        dt: float | None = None,
    ) -> Arbitrated:
        """Appraise, run every enabled faculty, arbitrate, and record."""
        if other is None and event.subject is not None:
            other = self.other_minds.estimate(event, species=species)

        frame = self.appraisal.appraise(event, other)
        readings = dict(self.interoception.read())
        if interior:
            readings.update(interior)

        ctx = FacultyContext(
            frame=frame,
            ledger=self.ledger,
            other=other,
            interior=readings,
            now=event.at,
        )

        with self._lock:
            disabled = set(self._disabled)

        activations: list[Activation] = []
        for faculty in registry().all():
            if faculty.id in disabled:
                continue
            activations.append(faculty.evaluate(ctx))

        state = arbitrate(activations, dt=dt)
        # Core affect first, faculties on top. Without the general term an
        # event no faculty is about produces nothing, and a blocked
        # commitment with nobody to be angry at reads as neutral.
        base = core_affect(frame)
        state = replace(state, affect=base + state.affect)
        self._absorb_ledger_writes(state)

        with self._lock:
            self._ticks += 1
            self._last = state
            self._last_frame = frame
            self._last_activations = tuple(activations)
        return state

    def _absorb_ledger_writes(self, state: Arbitrated) -> None:
        for write in state.ledger:
            method = getattr(self.ledger, write.op, None)
            if not callable(method):
                record_degradation(
                    "interiority.service",
                    ValueError(f"unknown ledger op {write.op!r}"),
                    action="ledger write dropped",
                )
                continue
            try:
                method(**dict(write.args))
            except (TypeError, ValueError) as exc:
                record_degradation(
                    "interiority.service", exc, action=f"ledger op {write.op} rejected"
                )

    # ── push ──────────────────────────────────────────────────────────
    def apply(self, state: Arbitrated | None = None) -> dict[str, Any]:
        """Land the arbitrated effects on the subsystems that read them."""
        target = state or self.last()
        if target is None:
            return {"applied": False, "reason": "nothing has been appraised yet"}

        landed: dict[str, Any] = {}
        landed["affect"] = self._push_affect(target)
        landed["somatic"] = self._push_somatic(target)
        landed["drives"] = self._push_drives(target)
        landed["curiosity"] = self._push_curiosity(target)
        with self._lock:
            self._applied += 1
        return {"applied": True, "landed": landed}

    def _push_affect(self, state: Arbitrated) -> dict[str, Any]:
        if state.affect.empty:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("affect_engine", default=None)
            if engine is None:
                return {"moved": False, "reason": "no affect engine registered"}
            markers = getattr(engine, "markers", None)
            if markers is None or not hasattr(markers, "somatic_update"):
                return {"moved": False, "reason": "affect engine exposes no update path"}
            markers.somatic_update(
                valence=state.affect.valence,
                arousal=state.affect.arousal,
                engagement=state.affect.engagement,
            )
            self.interoception.note_affect(state.affect.valence)
            return {"moved": True, "delta": state.affect.to_dict()}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="affect delta not applied")
            return {"moved": False, "error": type(exc).__name__}

    def _push_somatic(self, state: Arbitrated) -> dict[str, Any]:
        if not state.somatic:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("somatic_marker_gate", default=None)
            if gate is None or not hasattr(gate, "set_interior_bias"):
                # The gate has no interior-bias channel in this build; the
                # markers stay available through last() and permitted().
                return {"moved": False, "reason": "gate has no interior bias channel"}
            gate.set_interior_bias(
                {m.option: m.bias for m in state.somatic},
                source="interiority",
            )
            return {"moved": True, "options": len(state.somatic)}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="somatic bias not applied")
            return {"moved": False, "error": type(exc).__name__}

    def _push_drives(self, state: Arbitrated) -> dict[str, Any]:
        if not state.goals:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            drives = ServiceContainer.get("drive_system", default=None) or (
                ServiceContainer.get("drive_engine", default=None)
            )
            if drives is None or not hasattr(drives, "satisfy"):
                return {"moved": False, "reason": "no drive system registered"}
            moved = 0
            for goal in state.goals:
                budget = _DRIVE_FOR_GOAL.get(goal.goal.split(":")[0])
                if budget is None:
                    continue
                if goal.delta >= 0:
                    drives.satisfy(budget, goal.delta)
                else:
                    drives.punish(budget, -goal.delta)
                moved += 1
            return {"moved": moved > 0, "budgets_touched": moved}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="drive delta not applied")
            return {"moved": False, "error": type(exc).__name__}

    def _push_curiosity(self, state: Arbitrated) -> dict[str, Any]:
        wanted = [a for a in state.attention if a.target.startswith("source:") and a.weight > 0]
        if not wanted:
            return {"moved": False}
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("curiosity_engine", default=None)
            if engine is None or not hasattr(engine, "add_curiosity"):
                return {"moved": False, "reason": "no curiosity engine registered"}
            for bias in wanted[:3]:
                engine.add_curiosity(
                    bias.target.split(":", 1)[1], bias.reason, priority=abs(bias.weight)
                )
            return {"moved": True, "topics": len(wanted[:3])}
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("interiority.service", exc, action="curiosity bias not applied")
            return {"moved": False, "error": type(exc).__name__}

    # ── pull ──────────────────────────────────────────────────────────
    def appraise(self, trigger: str, context: Mapping[str, Any] | None = None) -> dict[str, float]:
        """Valence, arousal and engagement for the affect engine.

        Replaces a keyword scorer. The numbers come from the appraisal
        frame and the faculties, so an event with nothing at stake reads
        neutral however emotive the words in it are, and an event that
        touches a commitment reads even if it is phrased flatly. That is
        the whole difference, and it is measurable on a fixed pair of
        inputs.
        """
        payload = dict(context or {})
        kind = _KIND_FOR_SOURCE.get(str(payload.get("source", "")), EventKind.WORLD)
        observations: dict[str, Reading] = {}
        intensity = payload.get("intensity")
        if isinstance(intensity, (int, float)):
            observations["instrument"] = measured(
                max(0.0, min(1.0, float(intensity))), source="affect_engine:intensity"
            )
        event = InteriorEvent(
            kind=kind,
            summary=str(trigger)[:200],
            subject=payload.get("subject"),
            object=payload.get("object") or str(trigger)[:64],
            observations=observations,
            confidence=float(payload.get("confidence", 1.0) or 1.0),
            source=str(payload.get("source", "affect_engine")),
        )
        state = self.tick(event)
        return {
            "v": max(-1.0, min(1.0, state.affect.valence)),
            "a": max(0.0, min(1.0, abs(state.affect.arousal))),
            "e": max(0.0, min(1.0, abs(state.affect.engagement))),
        }

    def attune(
        self, message: str, *, subject: str | None = None, species: str = "human",
        observations: Mapping[str, Reading] | None = None,
    ) -> OtherEstimate:
        """A read on another agent. Replaces the resonance word lists."""
        event = InteriorEvent(
            kind=EventKind.SOCIAL,
            summary=str(message)[:200],
            subject=subject or "unknown",
            observations=dict(observations or {}),
            source="attune",
        )
        return self.other_minds.estimate(event, species=species)

    def retention_held(self, memory_key: str) -> tuple[bool, str]:
        """Whether a memory is held against deletion, and by which faculty."""
        state = self.last()
        if state is None:
            return (False, "")
        now = time.time()
        for claim in state.retention:
            if memory_key == claim.memory_key or memory_key.endswith(claim.memory_key):
                return (True, f"{claim.held_by}: {claim.reason}")
        return (False, "")

    def permitted(self, candidates: Iterable[str]) -> tuple[tuple[str, ...], dict[str, str]]:
        state = self.last()
        if state is None:
            return (tuple(candidates), {})
        return _permitted(candidates, state)

    def turn_budget(self) -> BudgetDelta:
        state = self.last()
        return state.budget if state is not None else BudgetDelta()

    # ── reporting ─────────────────────────────────────────────────────
    def last(self) -> Arbitrated | None:
        with self._lock:
            return self._last

    def last_frame(self) -> AppraisalFrame | None:
        with self._lock:
            return self._last_frame

    def last_activations(self) -> tuple[Activation, ...]:
        with self._lock:
            return self._last_activations

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._last
            ticks = self._ticks
            applied = self._applied
            disabled = sorted(self._disabled)
        return {
            "faculties": len(registry()),
            "homes_declared": len(HOMES),
            "ticks": ticks,
            "applied": applied,
            "disabled": disabled,
            "ledger": self.ledger.counts(),
            "receptors": get_receptor_bank().gains(),
            "cleft": get_cleft().snapshot(),
            "other_minds": self.other_minds.status(),
            "interoception": self.interoception.status(),
            "state": state.to_dict() if state else None,
        }


#: Which drive budget a goal prefix satisfies. Only prefixes that map to a
#: budget the drive engine actually has are listed; an unmapped goal moves
#: no budget rather than inventing one.
_DRIVE_FOR_GOAL: Mapping[str, str] = {
    "welfare": "social",
    "be_present_for": "social",
    "repair": "social",
    "secure": "social",
    "prepare": "purpose",
    "resume": "purpose",
    "repeat_the_policy_that_produced_this": "purpose",
    "externalise_unencoded_structure": "curiosity",
    "recruit_others_to_hold_this": "purpose",
    "meet_standard": "purpose",
    "active_care_for_those_at_risk": "social",
    "seek_resolvable_external_structure": "energy",
}

_KIND_FOR_SOURCE: Mapping[str, EventKind] = {
    "user": EventKind.SOCIAL,
    "conversation": EventKind.SOCIAL,
    "tool": EventKind.OWN_ACTION,
    "skill": EventKind.OWN_ACTION,
    "memory": EventKind.EPISTEMIC,
    "research": EventKind.EPISTEMIC,
    "health": EventKind.INTEROCEPTIVE,
    "soma": EventKind.INTEROCEPTIVE,
    "goal": EventKind.GOAL,
    "loss": EventKind.LOSS,
}


_SERVICE: InteriorityService | None = None
_SERVICE_LOCK = threading.Lock()


def get_interiority() -> InteriorityService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = InteriorityService()
        return _SERVICE


def register_interiority(container: Any = None) -> InteriorityService:
    """Register the service so the rest of the runtime can reach it."""
    service = get_interiority()
    try:
        from core.container import ServiceContainer

        target = container or ServiceContainer
        register = getattr(target, "register", None)
        if callable(register):
            register(SERVICE_NAME, service)
    except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
        record_degradation(
            "interiority.service", exc, action="service not registered in container"
        )
    return service


__all__ = [
    "SERVICE_NAME",
    "InteriorityService",
    "get_interiority",
    "register_interiority",
]
