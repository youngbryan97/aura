"""core/brain/lane_admission.py — declarative memory admission for model lanes.

The dominant historical reliability ceiling on this host is model-serving
memory over-commitment: lanes spawn imperatively, each spot-checking
*instantaneous* free RAM, so concurrent warmups + trainers + a resident 32B
can jointly commit more than the host holds. The result is the recorded
stall → force-kill → cold-reload doom loop.

This module is the Kubernetes-style fix (roadmap K3): every model lane
*declares* a memory request up front, a single controller admits only within
an explicit host budget, and QoS classes decide who yields when the budget
is contended:

  GUARANTEED  — the primary cortex. Never named as an eviction candidate.
  BURSTABLE   — brainstem / reflex / solver. Evictable for a GUARANTEED
                candidate, in largest-footprint-first order.
  BEST_EFFORT — trainers, compounding, auxiliary lanes. Evicted first.

The controller is deliberately pure arithmetic + bookkeeping: callers pass
the observed active lanes and the candidate's projected footprint (the
existing, battle-tested projection in mlx_client). Decisions come in three
shapes:

  * fits            → admitted, no conditions.
  * fits-if-yield   → admitted, with an ``evict_first`` advisory naming the
                      lower-QoS lanes that must yield. The existing swap
                      machinery (and, next, the K1 reconciler) executes it.
  * envelope breach → REFUSED with a named reason. This is the case that
                      today ends in an OOM SIGKILL with empty stderr —
                      e.g. the 72B solver over a committed host. Refusing
                      here is aerospace envelope protection (A4), not a
                      regression: the spawn was never going to survive.

Every decision is recorded in a bounded in-memory ring for the health
surface and the incident narrator. ``AURA_LANE_ADMISSION`` selects
``enforce`` (default) or ``advise``.

``advise`` is NOT a kill switch, whatever this docstring used to say
(CP126 87a971fa). It only flips the ``enforced`` field on the decision;
``admitted`` still reads False on an envelope breach. The one production
caller — ``core/runtime/model_lane_control.py`` — deliberately ignores
``enforced`` entirely, because a durable reservation is a safety boundary
and must not become a reservation just because advisory mode was set. So
``advise`` is a DIAGNOSTIC ANNOTATION on refusals, not a way to disable
them, and describing it as a kill switch told an operator they had an
escape hatch that does not exist.

What this module does NOT own, and must not be read as owning: atomic
reservation, leases, committed-state reconciliation, eviction commands,
acknowledgements or release (CP126 f075c288 / f1728553 / eb09e5c0). It is
arithmetic over a snapshot the caller supplies. ``model_lane_control``
wraps it in the durable transaction that provides those, and calling this
module directly gives you a calculation, not a guarantee.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import ResourceObserver, get_resource_observer

logger = logging.getLogger("Aura.LaneAdmission")

_DECISION_RING_SIZE = 64

_BUDGET_GB_FLAG = declare(
    "AURA_LANE_BUDGET_GB",
    kind=FlagKind.FLOAT,
    default=0.0,
    description="Absolute lane memory budget in GB; 0 = derive from fraction",
    owner="core.brain.lane_admission",
)
_BUDGET_FRACTION_FLAG = declare(
    "AURA_LANE_BUDGET_FRACTION",
    kind=FlagKind.FLOAT,
    default=0.72,
    description="Fraction of host RAM all model lanes may jointly commit",
    owner="core.brain.lane_admission",
)
_EVICTION_SHIELD_FLAG = declare(
    "AURA_LANE_EVICTION_SHIELD_S",
    kind=FlagKind.FLOAT,
    default=180.0,
    description="Seconds after a user-facing turn during which a lane is shielded from eviction",
    owner="core.brain.lane_admission",
)
_ADMISSION_MODE_FLAG = declare(
    "AURA_LANE_ADMISSION",
    kind=FlagKind.STRING,
    default="enforce",
    description="Lane admission mode: enforce (refusals bind) or advise (log-only kill switch)",
    owner="core.brain.lane_admission",
)


class QoSClass(StrEnum):
    GUARANTEED = "guaranteed"
    BURSTABLE = "burstable"
    BEST_EFFORT = "best_effort"


_QOS_RANK = {QoSClass.BEST_EFFORT: 0, QoSClass.BURSTABLE: 1, QoSClass.GUARANTEED: 2}


def _finite_gb(value: Any) -> float | None:
    """A non-negative, finite GB quantity, or None when it is not one.

    CP126 0021d73a. `max(0.0, float("nan"))` is nan and every comparison
    against nan is False, so a malformed quantity did not refuse — it slid
    past the fit check into the eviction path. Infinity admits nothing but
    produces an eviction advisory naming every lane on the host.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if number < 0.0:
        return None
    return number


def classify_lane(
    model_path: str, *, purpose: str = "serve", qos: QoSClass | None = None
) -> tuple[str, QoSClass]:
    """Map a model path (and purpose) to a (lane, QoS) declaration.

    Token matching mirrors the projection heuristics in mlx_client so the
    two layers never disagree about which lane a path belongs to.

    CP126 4c7e8933: the tokens are a HEURISTIC over a caller-supplied
    string. A legitimately renamed 32B does not match, falls to
    BEST_EFFORT, and becomes first in line for eviction while it is
    serving. An explicit ``qos`` overrides the guess; the lane name is
    still derived from the path because that is only a label.
    """
    if qos is not None:
        lane, _inferred = classify_lane(model_path, purpose=purpose)
        return lane, qos
    if purpose in {"train", "compound", "fuse", "benchmark", "evaluate", "eval"}:
        return "trainer", QoSClass.BEST_EFFORT
    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        return "solver", QoSClass.BURSTABLE
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        return "cortex", QoSClass.GUARANTEED
    if any(token in lowered for token in ("14b", "7b", "brainstem")):
        return "brainstem", QoSClass.BURSTABLE
    if any(token in lowered for token in ("1.5b", "1p5b", "0.5b", "reflex")):
        return "reflex", QoSClass.BURSTABLE
    return "auxiliary", QoSClass.BEST_EFFORT


@dataclass(frozen=True)
class ActiveLane:
    """One observed live model lane (a running worker holding memory)."""

    lane: str
    qos: QoSClass
    footprint_gb: float
    model_path: str = ""
    # Age since this lane last completed a user-facing generation; recently
    # user-facing lanes are shielded from eviction advisories (the 180s
    # cortex-protection precedent from the live thrash findings).
    last_user_facing_age_s: float | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    lane: str
    qos: QoSClass
    request_gb: float
    committed_gb: float
    budget_gb: float
    evict_first: tuple[str, ...] = ()
    enforced: bool = True
    observation_source: str = "unavailable"
    observation_scenario_id: str = ""
    resource_observation_available: bool = False
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "lane": self.lane,
            "qos": str(self.qos),
            "request_gb": round(self.request_gb, 2),
            "committed_gb": round(self.committed_gb, 2),
            "budget_gb": round(self.budget_gb, 2),
            "evict_first": list(self.evict_first),
            "enforced": self.enforced,
            "observation_source": self.observation_source,
            "observation_scenario_id": self.observation_scenario_id,
            "resource_observation_available": self.resource_observation_available,
            "at": self.at,
        }


def _host_total_gb(observer: ResourceObserver | None = None) -> tuple[float, bool]:
    """Observed host RAM in GB, and whether it was actually observed.

    CP126 ec83a2ed / 360473eb: this used to return 64.0 when observation
    failed. On the developer host that is the right number, which is
    exactly why it survived — everywhere else it invented a host large
    enough to admit ~46GB of model lanes, recorded "observation
    unavailable", and then bound the decision as a normal fit. An
    envelope check against a guessed envelope is not a check.

    The guess is gone. Callers get 0.0 and the False flag, and admit()
    refuses rather than sizing a budget it cannot justify.
    """
    memory = (observer or get_resource_observer()).memory()
    if memory.available and memory.total_bytes > 0:
        return float(memory.total_bytes) / float(1024**3), True
    return 0.0, False


def _budget_for_total_gb(host_total_gb: float) -> float:
    absolute = float(_BUDGET_GB_FLAG.value())
    if absolute > 0.0:
        return absolute
    try:
        from core.runtime.desktop_boot_safety import (
            compute_process_rss_limit,
            desktop_resource_guard_enabled,
        )

        if desktop_resource_guard_enabled():
            return compute_process_rss_limit(int(host_total_gb * 1024**3)) / float(1024**3)
    except (ImportError, RuntimeError, TypeError, ValueError):
        pass
    fraction = max(0.30, min(0.95, float(_BUDGET_FRACTION_FLAG.value())))
    return host_total_gb * fraction


def lane_budget_gb(*, observer: ResourceObserver | None = None) -> float:
    """The total memory envelope all model lanes may jointly commit.

    Default: 72% of host RAM. On the 64 GB production host that is ~46 GB —
    room for the resident 4-bit 32B (~20), brainstem (~5), reflex (~2) and a
    trainer burst, while the OS, the orchestrator process, and page cache
    keep the remainder. Override with AURA_LANE_BUDGET_GB (absolute) or
    AURA_LANE_BUDGET_FRACTION.
    """
    host_total_gb, available = _host_total_gb(observer)
    if not available:
        # No observation, no budget. 0.0 is the honest answer and makes
        # every fit comparison fail, which is the fail-closed direction.
        # An absolute operator override still stands: that is a declared
        # quantity, not an inferred one.
        absolute = float(_BUDGET_GB_FLAG.value())
        return absolute if absolute > 0.0 else 0.0
    return _budget_for_total_gb(host_total_gb)


def _eviction_shield_s() -> float:
    return float(_EVICTION_SHIELD_FLAG.value())


def enforcement_mode() -> str:
    mode = str(_ADMISSION_MODE_FLAG.value()).strip().lower()
    if mode in {"off", "0", "false", "advise", "advisory"}:
        return "advise"
    return "enforce"


class LaneAdmissionController:
    """Pure budget/QoS arithmetic with a bounded decision ring.

    Thread-safe; safe to call from worker-spawn executor threads.
    """

    def __init__(self, *, observer: ResourceObserver | None = None) -> None:
        self._observer = observer
        self._decisions: deque[AdmissionDecision] = deque(maxlen=_DECISION_RING_SIZE)
        self._lock = threading.Lock()

    # ── decision engine ────────────────────────────────────────────

    def admit(
        self,
        *,
        model_path: str,
        request_gb: float,
        active: Iterable[ActiveLane],
        purpose: str = "serve",
        allow_disruptive_eviction: bool = False,
        qos: QoSClass | None = None,
    ) -> AdmissionDecision:
        """Decide whether one lane may commit its declared memory.

        ``qos`` lets a caller that KNOWS declare it. Without it the class is
        inferred from tokens in the model path (CP126 4c7e8933), which is a
        heuristic: a renamed 32B falls to BEST_EFFORT and becomes evictable
        while serving. Inference stays the default because every current
        caller relies on it, but a declaration now beats a guess.
        """
        lane, qos = classify_lane(model_path, purpose=purpose, qos=qos)
        # CP126 0021d73a: NaN, infinity and negatives reached the arithmetic.
        # `max(0.0, nan)` is nan, and `nan + committed <= budget` is False —
        # so a NaN request did not refuse loudly, it fell through to the
        # eviction path and could emit an evict_first advisory for a request
        # that means nothing. A malformed request is refused as malformed.
        requested_value = request_gb
        finite_request_gb = _finite_gb(requested_value)
        if finite_request_gb is None:
            return self._record(
                AdmissionDecision(
                    admitted=False,
                    reason=f"malformed_request_gb:{requested_value!r}",
                    lane=lane,
                    qos=qos,
                    request_gb=0.0,
                    committed_gb=0.0,
                    budget_gb=0.0,
                )
            )
        request_gb = finite_request_gb
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        host_total, observation_available = _host_total_gb(observer)
        budget = _budget_for_total_gb(host_total) if observation_available else 0.0
        absolute_override = float(_BUDGET_GB_FLAG.value())
        if not observation_available and absolute_override > 0.0:
            # A declared absolute budget is a quantity the operator supplied.
            # Honouring it is not a guess.
            budget = absolute_override
            observation_available = True

        if not observation_available:
            # CP126 ec83a2ed: the envelope cannot be established. Refusing is
            # the only honest answer — admitting here is what the old 64GB
            # fallback did, and it recorded the observation as unavailable
            # while binding the decision as a normal fit.
            #
            # GUARANTEED is the one exception and it is NOT an admission of
            # fit: the primary cortex must be able to come up on a host whose
            # memory we cannot read, or an observation fault becomes a total
            # outage. It is admitted with the envelope explicitly unverified,
            # so no reader can mistake it for a checked decision.
            return self._record(
                AdmissionDecision(
                    admitted=qos is QoSClass.GUARANTEED,
                    reason=(
                        "admitted_without_envelope:memory_unobservable"
                        if qos is QoSClass.GUARANTEED
                        else "memory_unobservable:cannot_establish_lane_budget"
                    ),
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=0.0,
                    budget_gb=0.0,
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=False,
                )
            )
        # A lane replacing itself (worker recycle) must not double-count:
        # callers exclude the candidate's own lane from `active`.
        # A malformed footprint in the ACTIVE set is worse than in the
        # request: silently dropping it (the old `> 0.0` filter, which is
        # False for NaN) under-counts what is committed and admits over the
        # envelope. Count what we can read; refuse if we cannot read it all.
        lanes: list[ActiveLane] = []
        unreadable: list[str] = []
        for active_lane in active:
            footprint = _finite_gb(active_lane.footprint_gb)
            if footprint is None:
                unreadable.append(active_lane.model_path or active_lane.lane)
            elif footprint > 0.0:
                lanes.append(active_lane)
        if unreadable:
            return self._record(
                AdmissionDecision(
                    admitted=False,
                    reason=f"unreadable_active_footprint:{','.join(unreadable[:3])}",
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=0.0,
                    budget_gb=budget,
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=observation_available,
                )
            )
        committed = sum(active_lane.footprint_gb for active_lane in lanes)

        if committed + request_gb <= budget:
            decision = self._record(
                AdmissionDecision(
                    admitted=True,
                    reason="fits",
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=committed,
                    budget_gb=budget,
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=observation_available,
                )
            )
            return decision

        # Over budget: can lower-QoS lanes yield enough room? The recently-
        # user-facing shield protects warm lanes from background churn, but
        # a GUARANTEED candidate (the primary cortex) outranks it — the
        # cortex must always be able to come up.
        shield_s = _eviction_shield_s()
        # CP126 8de2c849 asked whether allow_disruptive_eviction should be
        # able to pierce the QoS floor and name a GUARANTEED cortex lane.
        # It should — an explicit operator handoff (a 72B solver taking the
        # host, a cortex-for-cortex model swap) has to be able to replace the
        # resident lane, and the existing arithmetic tests encode exactly
        # that. Restricting it here broke the legitimate case.
        #
        # The finding is still real, but it is a TRUST-BOUNDARY question, not
        # an arithmetic one: the flag is the caller ASSERTING authority, and
        # this module cannot tell an authorised handoff from an untrusted
        # trainer because everything it is given is caller-supplied
        # (CP126 7b277475 says the same thing about the whole input set).
        #
        # The check therefore lives where the state actually is:
        # core/runtime/model_lane_control.py refuses to fence an owner that
        # is not `preemptible`, one already fenced by someone else, and a
        # GUARANTEED owner that is currently SERVING. That is verified
        # against committed state this module never sees. Duplicating a
        # weaker version of it here would be a second, disagreeing authority.
        evictable = [
            active_lane
            for active_lane in lanes
            if (
                allow_disruptive_eviction
                or _QOS_RANK[active_lane.qos] < _QOS_RANK[qos]
            )
            and (
                allow_disruptive_eviction
                or qos is QoSClass.GUARANTEED
                or active_lane.last_user_facing_age_s is None
                or active_lane.last_user_facing_age_s >= shield_s
            )
        ]
        # Best-effort first, then largest footprint — free the most with the
        # least collateral.
        evictable.sort(
            key=lambda active_lane: (
                _QOS_RANK[active_lane.qos],
                -active_lane.footprint_gb,
            )
        )

        freed = 0.0
        chosen: list[ActiveLane] = []
        for candidate in evictable:
            if committed - freed + request_gb <= budget:
                break
            chosen.append(candidate)
            freed += candidate.footprint_gb

        if committed - freed + request_gb <= budget:
            decision = self._record(
                AdmissionDecision(
                    admitted=True,
                    reason="fits_after_yield",
                    lane=lane,
                    qos=qos,
                    request_gb=request_gb,
                    committed_gb=committed,
                    budget_gb=budget,
                    evict_first=tuple(
                        c.model_path or c.lane for c in chosen
                    ),
                    observation_source=provenance.source.value,
                    observation_scenario_id=provenance.scenario_id,
                    resource_observation_available=observation_available,
                )
            )
            return decision

        # Envelope breach: even with every eviction we own, this spawn
        # exceeds the host budget. Refuse with the arithmetic in the reason.
        enforced = enforcement_mode() == "enforce"
        decision = self._record(
            AdmissionDecision(
                admitted=False,
                reason=(
                    f"lane_budget_exceeded:{lane} request {request_gb:.1f}GB "
                    f"+ committed {committed - freed:.1f}GB (after max yield) "
                    f"> budget {budget:.1f}GB"
                ),
                lane=lane,
                qos=qos,
                request_gb=request_gb,
                committed_gb=committed,
                budget_gb=budget,
                evict_first=tuple(c.model_path or c.lane for c in chosen),
                enforced=enforced,
                observation_source=provenance.source.value,
                observation_scenario_id=provenance.scenario_id,
                resource_observation_available=observation_available,
            )
        )
        return decision

    # ── observability ──────────────────────────────────────────────

    def _record(self, decision: AdmissionDecision) -> AdmissionDecision:
        with self._lock:
            self._decisions.append(decision)
        if decision.admitted and decision.reason == "fits":
            logger.debug("Lane admission: %s", decision.to_dict())
        elif decision.admitted:
            logger.info("Lane admission (yield advised): %s", decision.to_dict())
        else:
            logger.warning("Lane admission REFUSED: %s", decision.to_dict())
        return decision

    def snapshot(self) -> dict[str, Any]:
        observer = self._observer or get_resource_observer()
        provenance = observer.provenance
        _total, observation_available = _host_total_gb(observer)
        budget = lane_budget_gb(observer=observer)
        with self._lock:
            recent = [d.to_dict() for d in list(self._decisions)[-10:]]
            unverified = sum(
                1
                for d in self._decisions
                if not d.resource_observation_available
            )
        return {
            "alive": True,
            "ready": self.is_ready(),
            "budget_gb": round(budget, 2),
            "mode": enforcement_mode(),
            # CP126 e405a76a: the surface reported healthy while the
            # controller could not observe memory. Say what is true instead.
            "resource_observation_available": observation_available,
            "envelope_established": budget > 0.0,
            "decisions_without_envelope": unverified,
            "observation_source": provenance.source.value,
            "observation_scenario_id": provenance.scenario_id,
            "recent_decisions": recent,
        }

    def is_alive(self) -> bool:
        """The controller object exists and can be called.

        Distinct from readiness on purpose: the arithmetic is always
        available, and that is not the same as being able to bind an
        envelope.
        """
        return True

    def is_ready(self) -> bool:
        """Whether admission decisions can actually be BOUND right now.

        CP126 e405a76a: this returned True unconditionally, so a controller
        that could not read host memory — and was therefore refusing every
        non-GUARANTEED lane — reported ready. That is the absence of a check
        published as a passed check, and it is the reason the fail-open 64GB
        default went unnoticed for so long: nothing on the health surface
        could ever have said otherwise.
        """
        return lane_budget_gb(observer=self._observer) > 0.0

    def get_status(self) -> dict[str, Any]:
        return self.snapshot()


_CONTROLLER: LaneAdmissionController | None = None
_CONTROLLER_LOCK = threading.Lock()


def get_lane_admission_controller() -> LaneAdmissionController:
    global _CONTROLLER
    if _CONTROLLER is None:
        with _CONTROLLER_LOCK:
            if _CONTROLLER is None:
                _CONTROLLER = LaneAdmissionController()
    return _CONTROLLER


def reset_lane_admission_controller_for_test() -> None:
    global _CONTROLLER
    with _CONTROLLER_LOCK:
        _CONTROLLER = None
