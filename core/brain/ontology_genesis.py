"""core/brain/ontology_genesis.py — Aura 3.0: Experimental Containment Mode
=======================================================================
Implements the "Hibernation Mode" protocol for high-compute autonomous
scientific discovery.

Refactored for ZENITH Protocol efficiency:
  - Background discovery loop is DISABLED by default.
  - Must be explicitly triggered by user or deep-research soul drive.
  - Mandatory resource_anxiety abort if system load exceeds thresholds.

CP126 c0a3c26e found that the advertised "autonomous formation of cognitive
laws" was a loop that logged, slept sixty seconds, and reached a comment saying
the real logic went there. It produced no candidate, experiment, verifier
result or artifact, and never wrote to its own discovery log — while logging
"Discovery loop active" and reporting ``active: true``. The module was made to
say so: ``DISCOVERY_IMPLEMENTED = False``, and a note naming the four things a
real step owes.

The step exists now, in :mod:`core.brain.ontology_discovery`. Each cycle reads
the runtime's own degradation ring, induces a conjunctive predicate over
observable features on a training split, measures it on a held-out split the
search never saw, tests it against a permutation null, prunes conjuncts that do
not pay for themselves, requires it to hold on a third split later in time, and
only then writes it — into the shared heuristic pool that
``curiosity_explorer``, ``dreamer_v2`` and ``dream_skill`` already read, so a
discovery reaches a consumer rather than a file nobody opens.

Most cycles find nothing, and that is the design. A discovery loop that cannot
come back empty is not measuring anything: on twenty runs of pure noise the
induction returns no law twenty times. ``last_refusal`` carries the reason.

The admission checks around it — volition, authority, resource pressure,
supervision — were already real and are unchanged.

CP126 c0a3c26e / 72c94940 / d52e6a29 / 96cb9483 / 478804d4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.brain.ontology_discovery import (
    DiscoveredLaw,
    DiscoveryOutcome,
    Observation,
    OntologyDiscovery,
)
from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_unit
from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.OntologyGenesis")

#: Whether a real discovery step exists. Flipping this to True without writing
#: one re-creates CP126 c0a3c26e. It is True because
#: ``core.brain.ontology_discovery`` produces a candidate, runs an experiment
#: on data the search never saw, obtains a verifier result against a
#: permutation null, and writes a discovered result to a pool with readers.
DISCOVERY_IMPLEMENTED = True

#: Gap between discovery cycles. The ring it reads holds 500 records and turns
#: over slowly; sampling it faster re-measures the same episodes and spends
#: compute on a question whose answer cannot have changed.
DISCOVERY_INTERVAL_S = 300.0

#: Degradation records pulled per cycle. The tracker keeps 500, which is the
#: ceiling; asking for it means a cycle sees everything still in memory.
OBSERVATION_LIMIT = 500

#: Window for the burst features. Long enough that a retry storm is one burst
#: rather than several, short enough that an idle hour is not.
BURST_WINDOW_S = 60.0

#: Discoveries kept in memory. This engine runs for the life of the process.
MAX_DISCOVERY_LOG = 50

#: Single source of truth for the abort boundary. CP126 96cb9483: status
#: advertised 0.2 while start_discovery used 0.5 and the loop used 0.3 or 0.6,
#: so an operator could not infer the real boundary from the service status.
ANXIETY_THRESHOLD_ADMISSION = 0.5
ANXIETY_THRESHOLD_RUNNING = 0.3
ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION = 0.6
HIGH_VOLITION_LEVEL = 3

#: Resource pressure attributed when telemetry is unavailable. CP126 d52e6a29:
#: this was 0.0 — the SAFEST possible value — so a missing homeostasis service
#: admitted high-compute work instead of deferring it for missing telemetry.
UNKNOWN_ANXIETY = 1.0

_GENESIS_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
)


def degradation_observations(
    records: list[dict[str, Any]] | None = None,
) -> list[Observation]:
    """Turn the runtime's degradation ring into episodes to learn from.

    The outcome is whether a record was serious — error or critical. Severity
    is therefore NOT a feature: a rule that reads the answer off its own input
    is the leak that makes an induction look brilliant and predict nothing, and
    excluding it is what makes the held-out lift mean anything.

    Everything else the record carries is available, plus three features about
    its neighbourhood in time. Those are where the interesting laws live: a
    single degradation says little, and "the fourth one in a minute, from a
    third distinct subsystem" says a great deal.
    """
    if records is None:
        try:
            from core.runtime.errors import recent_degradations

            records = recent_degradations(limit=OBSERVATION_LIMIT)
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Degradation ring unavailable: %s", exc)
            return []

    ordered = sorted(records or [], key=lambda r: float(r.get("at", 0.0) or 0.0))
    observations: list[Observation] = []
    for index, record in enumerate(ordered):
        at = float(record.get("at", 0.0) or 0.0)
        window_start = at - BURST_WINDOW_S
        window = [
            other
            for other in ordered[:index]
            if float(other.get("at", 0.0) or 0.0) >= window_start
        ]
        previous_at = float(ordered[index - 1].get("at", 0.0) or 0.0) if index else at
        severity = str(record.get("severity", "") or "").lower()
        observations.append(
            Observation(
                features={
                    "subsystem": str(record.get("subsystem", "") or "unknown"),
                    "error_type": str(record.get("error_type", "") or "unknown"),
                    "seconds_since_previous": round(max(0.0, at - previous_at), 3),
                    "burst_count": float(len(window)),
                    "distinct_subsystems": float(
                        len({str(o.get("subsystem", "")) for o in window})
                    ),
                    "repeat_of_previous": bool(
                        index
                        and record.get("subsystem") == ordered[index - 1].get("subsystem")
                    ),
                },
                outcome=severity in {"error", "critical"},
                at=at,
            )
        )
    return observations


class OntologyGenesisEngine:
    """
    Manages autonomous discovery of new cognitive laws and heuristics.

    ZENITH Protocol:
      - Default state is inert.
      - Throttled by resource anxiety.

    The discovery step itself is not implemented; see the module docstring.
    """

    def __init__(self, discovery: OntologyDiscovery | None = None) -> None:
        self._active = False
        self._genesis_task: asyncio.Task | None = None
        self._discovery_log: list[dict[str, Any]] = []
        self._last_refusal = ""
        self._started_at = 0.0
        self._discovery = discovery or OntologyDiscovery(
            outcome_name="a serious degradation"
        )
        self._cycles = 0
        self._integrated = 0
        self._last_cycle_at = 0.0
        self._last_outcome: DiscoveryOutcome | None = None

    # -- resource telemetry ----------------------------------------------
    def resource_anxiety(self) -> tuple[float, bool]:
        """Current resource pressure and whether it was actually measured."""
        try:
            homeostasis = get_runtime_service("homeostasis", default=None)
            if homeostasis is not None and hasattr(homeostasis, "anxiety"):
                raw = homeostasis.anxiety
                if callable(raw):
                    raw = raw()
                scalar = validated_unit(raw, name="anxiety", cautious_high=True)
                if not scalar.fault:
                    return float(scalar), True
                logger.debug("Homeostasis anxiety unusable: %s", scalar.fault)
        except _GENESIS_ERRORS as exc:
            logger.debug("Homeostasis probe failed: %s", exc)
        # Unknown pressure is treated as maximum: high-compute work is deferred
        # for missing telemetry rather than admitted on a convenient default.
        return UNKNOWN_ANXIETY, False

    def _get_resource_anxiety(self) -> float:
        """Backwards-compatible scalar accessor."""
        return self.resource_anxiety()[0]

    # -- admission --------------------------------------------------------
    async def start_discovery(
        self, mode: str = "manual", *, capability_token: str = ""
    ) -> bool:
        """Trigger a discovery cycle.

        Returns False and records the reason whenever admission is refused.
        """
        self._last_refusal = ""

        # CP126 c0a3c26e: there is nothing to start. Refusing here is the
        # honest answer; the alternative is an idle timer that reports itself
        # as active research.
        if not DISCOVERY_IMPLEMENTED:
            self._last_refusal = "not_implemented"
            logger.info(
                "OntologyGenesis: no discovery step is implemented; refusing to "
                "report an active discovery loop."
            )
            return False

        volition = self._volition_level()

        # CP126 72c94940: `mode` is a caller-supplied string, and passing
        # "deep_research" skipped the volition check entirely — no principal,
        # standing authority, signed plan, scope or compute lease. The mode
        # alone no longer grants anything; it must be attested.
        authorized_mode = mode == "deep_research" and self._mode_authorized(
            mode, capability_token
        )
        if not authorized_mode and volition < 1:
            self._last_refusal = "insufficient_volition"
            logger.info(
                "OntologyGenesis: refused — volition %d < 1 and deep_research was "
                "not attested.", volition,
            )
            return False

        anxiety, measured = self.resource_anxiety()
        if not measured:
            self._last_refusal = "resource_telemetry_unavailable"
            logger.warning(
                "OntologyGenesis: refused — resource telemetry unavailable, so "
                "high-compute work is deferred rather than admitted."
            )
            return False
        if anxiety >= ANXIETY_THRESHOLD_ADMISSION:
            self._last_refusal = f"resource_pressure={anxiety:.2f}"
            logger.warning(
                "OntologyGenesis: Abort. Resource pressure too high (%.2f >= %.2f).",
                anxiety, ANXIETY_THRESHOLD_ADMISSION,
            )
            return False

        if self._active:
            # CP126 478804d4: a task that died left _active True forever, so
            # this reported an already-active service that was not running.
            if self._genesis_task is not None and self._genesis_task.done():
                logger.warning(
                    "OntologyGenesis: previous discovery task is finished; clearing "
                    "the stale active flag before restarting."
                )
                self._active = False
            else:
                return True

        self._active = True
        self._started_at = time.time()
        self._genesis_task = get_task_tracker().create_task(
            self._discovery_loop(volition), name="ontology_genesis.discovery"
        )
        self._genesis_task.add_done_callback(self._on_task_done)
        logger.info(
            "OntologyGenesis: Hibernation ended (Volition=%d). Discovery loop active.",
            volition,
        )
        return True

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Clear the active flag however the task ended (CP126 478804d4)."""
        self._active = False
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            record_degradation(
                "ontology_genesis",
                exc,
                action="cleared the active flag after the discovery task failed",
                severity="error",
            )
            logger.error("OntologyGenesis discovery task failed: %s", exc)

    @staticmethod
    def _volition_level() -> int:
        kernel = get_runtime_service("aura_kernel", default=None)
        try:
            return int(getattr(kernel, "volition_level", 0) or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _mode_authorized(mode: str, capability_token: str) -> bool:
        """Whether a deep_research request carries real authority."""
        token = str(capability_token or "").strip()
        if not token:
            return False
        try:
            from core.agency.capability_token import get_token_store

            get_token_store().validate(
                token, domain="self_modification", action="deep_research"
            )
        except (PermissionError, *_GENESIS_ERRORS) as exc:
            logger.info("OntologyGenesis: deep_research token rejected (%s)", exc)
            return False
        return True

    # -- lifecycle --------------------------------------------------------
    async def stop_discovery(self) -> None:
        self._active = False
        task, self._genesis_task = self._genesis_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("OntologyGenesis: Returning to hibernation.")

    def running_threshold(self, volition: int) -> float:
        return (
            ANXIETY_THRESHOLD_RUNNING_HIGH_VOLITION
            if volition >= HIGH_VOLITION_LEVEL
            else ANXIETY_THRESHOLD_RUNNING
        )

    async def run_discovery_cycle(
        self, observations: list[Observation] | None = None
    ) -> DiscoveryOutcome:
        """One candidate → experiment → verifier → integration pass.

        Separated from the loop so it can be driven directly by a test or an
        operator. The loop's job is admission and pacing; the evidence work is
        here, and it returns the refusal when there is one rather than logging
        and moving on.
        """
        episodes = observations if observations is not None else degradation_observations()
        # The induction is CPU-bound over up to 500 episodes with a
        # 999-permutation null. On the loop it would hold the event loop for
        # long enough to matter, and this runs beside a live desktop runtime.
        outcome = await asyncio.to_thread(self._discovery.discover, episodes)
        self._last_cycle_at = time.time()
        self._cycles += 1
        self._last_outcome = outcome

        if not outcome.found or outcome.discovered is None:
            self._last_refusal = outcome.refusal or "no law survived validation"
            logger.debug("OntologyGenesis: no law this cycle — %s", self._last_refusal)
            return outcome

        self._last_refusal = ""
        discovered = outcome.discovered
        self._discovery_log.append(discovered.to_dict())
        # The log is bounded: this runs for the life of the process and an
        # unbounded list of discoveries is a leak with a scientific name.
        del self._discovery_log[:-MAX_DISCOVERY_LOG]
        self._integrate(discovered)
        return outcome

    def _integrate(self, discovered: DiscoveredLaw) -> bool:
        """Put the law where something reads it.

        The shared heuristic pool, not a private file. CP126's finding across
        this codebase is that a writer with no reader is indistinguishable from
        no writer at all, and a discovery nobody consults is exactly that.
        """
        rule = (
            f"{discovered.law.describe()} "
            f"(held-out lift {discovered.evidence.heldout_lift:.2f}, "
            f"p={discovered.evidence.p_value:.3f}, "
            f"transfer lift {discovered.evidence.transfer_lift:.2f})"
        )
        try:
            from core.adaptation.heuristic_synthesizer import get_heuristic_synthesizer

            accepted = bool(
                get_heuristic_synthesizer().ingest_external_heuristic(
                    rule, domain="runtime_ontology", source="ontology_genesis"
                )
            )
        except (*_GENESIS_ERRORS, OSError) as exc:
            record_degradation(
                "ontology_genesis",
                exc,
                action="discovered law was not integrated into the heuristic pool",
                severity="warning",
            )
            return False
        self._integrated += int(accepted)
        if accepted:
            logger.info("🔬 OntologyGenesis integrated a discovered law: %s", rule)
        return accepted

    async def _discovery_loop(self, volition: int = 0) -> None:
        """Admission, pacing, and the abort boundary around the discovery step."""
        while self._active:
            # CP126 478804d4: volition was captured once at start, so a
            # revocation mid-run never tightened the boundary.
            current_volition = self._volition_level()
            threshold = self.running_threshold(current_volition)
            anxiety, measured = self.resource_anxiety()
            if not measured or anxiety >= threshold:
                logger.warning(
                    "OntologyGenesis: Emergency hibernation (anxiety=%.2f "
                    "threshold=%.2f measured=%s).", anxiety, threshold, measured,
                )
                self._active = False
                break

            try:
                await self.run_discovery_cycle()
            except _GENESIS_ERRORS as exc:
                record_degradation(
                    "ontology_genesis",
                    exc,
                    action="discovery cycle failed; the loop continues to the next",
                    severity="warning",
                )
            await asyncio.sleep(DISCOVERY_INTERVAL_S)

    # -- status -----------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        anxiety, measured = self.resource_anxiety()
        volition = self._volition_level()
        return {
            "active": self._active and DISCOVERY_IMPLEMENTED,
            # CP126 c0a3c26e: the honest headline.
            "implemented": DISCOVERY_IMPLEMENTED,
            "discoveries": len(self._discovery_log),
            # Cycles run and laws integrated are separate numbers on purpose.
            # Most cycles find nothing, so reporting only "discoveries" would
            # make an engine that is working look like one that never ran —
            # and an engine that never ran look identical to a working one.
            "cycles": self._cycles,
            "integrated": self._integrated,
            "last_cycle_at": self._last_cycle_at,
            "last_law": self._discovery_log[-1] if self._discovery_log else None,
            # CP126 96cb9483: the thresholds actually in force, all of them.
            "anxiety_threshold_admission": ANXIETY_THRESHOLD_ADMISSION,
            "anxiety_threshold_running": self.running_threshold(volition),
            "current_anxiety": anxiety,
            "anxiety_measured": measured,
            "volition_level": volition,
            "last_refusal": self._last_refusal,
            "task_done": self._genesis_task.done() if self._genesis_task else None,
            "uptime_s": round(time.time() - self._started_at, 3) if self._started_at else 0.0,
        }


def register_ontology_genesis(orchestrator: Any | None = None) -> OntologyGenesisEngine:
    """Legacy registration helper expected by boot code."""
    engine = OntologyGenesisEngine()
    register_runtime_service("ontology_genesis", engine)
    if orchestrator is not None:
        try:
            orchestrator.ontology_genesis = engine
        except _GENESIS_ERRORS as _exc:
            record_degradation('ontology_genesis', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
    return engine
