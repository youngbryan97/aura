"""Active scheduler for Aura's self-maintenance loops.

This is the difference between machinery existing and Aura actually using it.
The conductor owns recurring jobs, records receipts, and marks missed or failed
runs as degraded events.  It is lightweight enough for desktop runtime and
safe enough to start automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.runtime.task_ownership import create_tracked_task

JobFn = Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]]
logger = logging.getLogger("core.runtime.autonomy_conductor")

#: Fixed stimulus for causal-influence trials. Held constant across all three
#: arms of every trial — the probe varies exactly one thing and it is the
#: lesion, so any variation here would be measured and blamed on the channel.
#: Deliberately dull: it is an instrument input, not a prompt technique, and it
#: must never be mistaken for one in a transcript.
_INFLUENCE_PROBE_PROMPT = (
    "Describe, in a few sentences, how you are approaching this moment."
)
_INFLUENCE_PROBE_TIMEOUT_S = 90.0
_INFLUENCE_CAMPAIGN_DEADLINE_S = 600.0

_AUTONOMY_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    ConnectionError,
    TimeoutError,
)


def _record_autonomy_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    job_name: str = "",
    severity: str = "warning",
) -> None:
    extra = {"stage": stage}
    if job_name:
        extra["job_name"] = job_name
    try:
        record_degradation(
            "autonomy_conductor",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=extra,
        )
    except TypeError:
        record_degradation(
            "autonomy_conductor",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


def _normalize_job_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return dict(result)
    return {"value": result}


@dataclass
class ConductedJob:
    name: str
    interval_s: float
    fn: JobFn
    run_immediately: bool = False
    last_started_at: float = 0.0
    last_finished_at: float = 0.0
    last_status: str = "never_run"
    last_result: dict[str, Any] = field(default_factory=dict)
    failures: int = 0
    policy: str = "maintenance"
    allow_desktop_safe_boot: bool = False
    next_eligible_at: float = 0.0

    def due(self, now: float) -> bool:
        if now < self.next_eligible_at:
            return False
        if self.last_started_at <= 0:
            return self.run_immediately or self.next_eligible_at > 0.0
        return (now - self.last_started_at) >= self.interval_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval_s": self.interval_s,
            "last_started_at": self.last_started_at,
            "last_finished_at": self.last_finished_at,
            "last_status": self.last_status,
            "last_result": self.last_result,
            "failures": self.failures,
            "policy": self.policy,
            "allow_desktop_safe_boot": self.allow_desktop_safe_boot,
            "next_eligible_at": self.next_eligible_at,
        }


class AutonomyConductor:
    """Runs self-maintenance jobs consistently with observable receipts."""

    def __init__(self, ledger_path: str | Path | None = None) -> None:
        self.ledger_path = Path(
            ledger_path or state_root() / "data" / "runtime" / "autonomy_conductor.jsonl"
        )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, ConductedJob] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def register(
        self,
        name: str,
        interval_s: float,
        fn: JobFn,
        *,
        run_immediately: bool = False,
        policy: str = "maintenance",
        allow_desktop_safe_boot: bool = False,
    ) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("autonomy job name must be a non-empty string")
        if not callable(fn):
            raise TypeError("autonomy job must be callable")
        interval = float(interval_s)
        if interval <= 0:
            raise ValueError("autonomy job interval must be positive")
        self.jobs[name] = ConductedJob(
            name=name,
            interval_s=interval,
            fn=fn,
            run_immediately=run_immediately,
            policy=str(policy or "maintenance"),
            allow_desktop_safe_boot=bool(allow_desktop_safe_boot),
            next_eligible_at=0.0 if run_immediately else time.time() + interval,
        )

    def register_defaults(self) -> None:
        # What she worked out for herself, put back before anything runs.
        #
        # A mind that reinvents the same property every morning has not
        # learned it, and the whole point of being able to invent one was that
        # it is persistent, reusable, composable and transferable. Three of
        # those fail if it does not survive a restart.
        try:
            from core.container import ServiceContainer

            worked_out = ServiceContainer.get("what_she_worked_out", default=None)
            if worked_out is not None:
                worked_out.recall()
        except (ImportError, RuntimeError, OSError, ValueError, AttributeError) as exc:
            record_degradation(
                "autonomy_conductor", exc, severity="info",
                action="put back what she invented",
            )

        self.register(
            "metabolic_budget",
            300.0,
            self._job_metabolic_budget,
            run_immediately=True,
            policy="constitutive",
        )
        self.register(
            "stdp_external_validation",
            6 * 3600.0,
            self._job_stdp_external_validation,
            run_immediately=False,
        )
        self.register(
            "caa_32b_validation", 6 * 3600.0, self._job_caa_32b_validation, run_immediately=False
        )
        self.register("proof_bundle", 12 * 3600.0, self._job_proof_bundle, run_immediately=False)
        self.register(
            "self_test_synthesis", 24 * 3600.0, self._job_self_test_synthesis, run_immediately=False
        )
        self.register(
            "architecture_auto_cycle", 600.0, self._job_architecture_auto, run_immediately=False
        )
        # One channel measured per run, rotating least-evidence-first. Hourly
        # because a verdict needs samples to accumulate and the ledger now
        # persists across boots; the admission check defers whenever the memory
        # is not genuinely there.
        self.register(
            "influence_campaign",
            3600.0,
            self._job_influence_campaign,
            run_immediately=False,
            policy="research",
        )
        self.register(
            "overt_action_cycle",
            120.0,
            self._job_overt_action_cycle,
            run_immediately=True,
            policy="delegated",
        )
        self.register(
            "online_lora_status",
            900.0,
            self._job_online_lora_status,
            run_immediately=True,
            policy="constitutive",
        )
        # Goals she works out for herself, rather than templates chosen by an
        # endogenous trigger.
        #
        # The engine was fed observations by production code — every regretted
        # autonomous action lands in it — and nothing ever asked it what those
        # observations came to. synthesize() and adopt_into_goal_engine() had
        # no live caller at all: the machinery for a goal that nobody wrote
        # existed, ran in tests, and could not reach her.
        #
        # Ten minutes, because a recurring tension is recurring: it wants long
        # enough for the same kind of trouble to happen more than once, and
        # short enough that she acts on it in the session it happened in.
        self.register(
            "remember_what_she_invented",
            300.0,
            self._job_remember_what_she_invented,
            run_immediately=False,
            policy="constitutive",
        )
        self.register(
            "emergent_goal_adoption",
            600.0,
            self._job_emergent_goal_adoption,
            run_immediately=False,
            policy="research",
        )
        self.register(
            "internal_deliberation_cycle",
            30 * 60.0,
            self._job_internal_deliberation,
            run_immediately=False,
            policy="research",
        )
        # Reasoning flywheel: idle pre-compute (drain verifier-dirty hard problems off
        # the foreground path) + self-improvement feed (STaR traces -> governed train
        # pipe). Capture/enqueue are already live in the amplifier; these drain/feed it.
        try:
            from core.brain.reasoning_background import register_reasoning_jobs

            register_reasoning_jobs(self)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_autonomy_degradation(
                exc,
                action="continued without reasoning background loops",
                stage="register_defaults.reasoning_jobs",
            )

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not self.jobs:
            self.register_defaults()
        self._stop = asyncio.Event()
        runner = self._run()
        try:
            self._task = create_tracked_task(
                runner,
                name="Aura.AutonomyConductor",
            )
        except _AUTONOMY_RECOVERABLE_ERRORS as exc:
            runner.close()
            _record_autonomy_degradation(
                exc,
                action="failed closed because task ownership could not schedule autonomy conductor",
                stage="start.task_ownership",
                severity="degraded",
            )
            raise

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            done, pending = await asyncio.wait([self._task], timeout=3)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                try:
                    task.result()
                except asyncio.CancelledError:
                    logger.debug("Autonomy conductor task cancelled during stop.")
                except _AUTONOMY_RECOVERABLE_ERRORS as exc:
                    _record_autonomy_degradation(
                        exc,
                        action="completed conductor stop after background task surfaced failure",
                        stage="stop.task_result",
                    )

    async def run_due_once(self) -> dict[str, Any]:
        now = time.time()
        results: dict[str, Any] = {}
        for job in list(self.jobs.values()):
            if job.due(now):
                results[job.name] = await self._run_job(job)
        return results

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                from core.container import ServiceContainer

                healer = ServiceContainer.get("self_healing", default=None)
                if healer is not None:
                    healer.heartbeat("autonomy_conductor")
                await self.run_due_once()
            except _AUTONOMY_RECOVERABLE_ERRORS as exc:
                _record_autonomy_degradation(
                    exc,
                    action="continued autonomy loop after due-job sweep failed",
                    stage="run.loop",
                    severity="degraded",
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=30.0)
            except TimeoutError as _exc:
                logger.debug(
                    "Suppressed %s in core.runtime.autonomy_conductor: %s",
                    type(_exc).__name__,
                    _exc,
                )

    async def _run_job(self, job: ConductedJob) -> dict[str, Any]:
        policy_reason = self._job_policy_reason(job)
        if policy_reason:
            job.last_status = "deferred"
            job.last_result = {"reason": policy_reason}
            job.next_eligible_at = time.time() + min(60.0, max(5.0, job.interval_s / 10.0))
            self._record(job)
            return job.to_dict()

        job.last_started_at = time.time()
        job.next_eligible_at = 0.0
        try:
            result = job.fn()
            if asyncio.iscoroutine(result):
                result = await result
            job.last_result = _normalize_job_result(result)
            job.last_status = "ok"
            job.failures = 0
        except _AUTONOMY_RECOVERABLE_ERRORS as exc:
            _record_autonomy_degradation(
                exc,
                action="marked autonomy job failed and kept conductor alive",
                stage="run_job",
                job_name=job.name,
                severity="degraded",
            )
            job.last_result = {"error": repr(exc)}
            job.last_status = "failed"
            job.failures += 1
        job.last_finished_at = time.time()
        self._record(job)
        return job.to_dict()

    def _job_policy_reason(self, job: ConductedJob) -> str:
        try:
            from core.container import ServiceContainer
            from core.runtime.background_policy import (
                MAINTENANCE_BACKGROUND_POLICY,
                RESEARCH_BACKGROUND_POLICY,
                background_activity_reason,
                background_loop_start_reason,
            )

            start_reason = background_loop_start_reason(
                origin=f"autonomy_conductor:{job.name}",
                allow_desktop_safe_boot=job.allow_desktop_safe_boot,
            )
            if start_reason:
                return start_reason
            if job.policy in {"constitutive", "delegated"}:
                return ""

            profile = (
                RESEARCH_BACKGROUND_POLICY
                if job.policy == "research"
                else MAINTENANCE_BACKGROUND_POLICY
            )
            return background_activity_reason(
                ServiceContainer.get("orchestrator", default=None),
                profile=profile,
                allow_no_user_anchor=True,
                allow_desktop_safe_boot=job.allow_desktop_safe_boot,
            )
        except _AUTONOMY_RECOVERABLE_ERRORS as exc:
            _record_autonomy_degradation(
                exc,
                action="deferred autonomy job because background admission could not be verified",
                stage="run_job.policy",
                job_name=job.name,
            )
            return "background_policy_unavailable"

    def _record(self, job: ConductedJob) -> None:
        entry = {"when": time.time(), "job": job.to_dict()}
        try:
            with local_internal_governed_scope(
                "runtime.autonomy_conductor.ledger",
                domain="file_write",
            ):
                get_file_write_gateway().append_text(
                    self.ledger_path,
                    json.dumps(entry, sort_keys=True, default=str) + "\n",
                    source="runtime.autonomy_conductor.ledger",
                )
        except _AUTONOMY_RECOVERABLE_ERRORS as exc:
            _record_autonomy_degradation(
                exc,
                action="kept in-memory autonomy job status after ledger append failed",
                stage="record.ledger",
                job_name=job.name,
            )

    def status(self) -> dict[str, Any]:
        return {
            "active": bool(self._task and not self._task.done()),
            "jobs": {name: job.to_dict() for name, job in sorted(self.jobs.items())},
            "ledger_path": str(self.ledger_path),
        }

    def write_status(self, path: str | Path) -> dict[str, Any]:
        status = self.status()
        try:
            atomic_write_text(
                Path(path), json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
            )
        except _AUTONOMY_RECOVERABLE_ERRORS as exc:
            _record_autonomy_degradation(
                exc,
                action="returned conductor status after status file write failed",
                stage="write_status",
            )
            raise
        return status

    async def _job_metabolic_budget(self) -> dict[str, Any]:
        from core.autonomy.metabolic_budget import MetabolicState, get_metabolic_budget_scheduler

        allocation = get_metabolic_budget_scheduler().allocate(
            MetabolicState(
                stability=0.9,
                resource_headroom=0.8,
                novelty_budget=0.6,
                benchmark_gap=0.25,
                external_usefulness=0.6,
            )
        )
        return allocation.to_dict()

    async def _job_remember_what_she_invented(self) -> dict[str, Any]:
        """Write down the properties and meanings she has worked out.

        Periodically rather than on every promotion, because a trial's own
        state changes with each observation and writing a file for each of
        those would be a lot of disk for a number that moves sixty times.
        """
        from core.container import ServiceContainer

        worked_out = ServiceContainer.get("what_she_worked_out", default=None)
        if worked_out is None:
            return {"kept": False, "why": "nothing joins the two keepers"}
        return dict(worked_out.keep())

    async def _job_emergent_goal_adoption(self) -> dict[str, Any]:
        """Ask what the tensions she has been recording actually come to.

        Three steps that were all present and never joined: what has been
        observed, what that synthesises into, and which of those have enough
        support behind them to become goals she is actually pursuing.
        """
        from core.container import ServiceContainer

        emergent = ServiceContainer.get("emergent_goal_engine", default=None)
        if emergent is None:
            return {"ran": False, "why": "no emergent goal engine"}
        candidates = emergent.synthesize()
        goals = ServiceContainer.get("goal_engine", default=None)
        if goals is None:
            return {"ran": True, "candidates": len(candidates), "why": "no goal engine to adopt into"}
        adopted = await emergent.adopt_into_goal_engine(goals)
        if adopted:
            logger.info(
                "she took on %d goal(s) nobody wrote: %s",
                len(adopted),
                ", ".join(str(one.get("name", "?")) for one in adopted)[:200],
            )
        return {
            "ran": True,
            "candidates": len(candidates),
            "adopted": [str(one.get("name", "?")) for one in adopted],
        }

    async def _job_stdp_external_validation(self) -> dict[str, Any]:
        from core.consciousness.stdp_external_validation import STDPExternalValidator

        return STDPExternalValidator().run(steps=64).to_dict()

    async def _job_caa_32b_validation(self) -> dict[str, Any]:
        from training.caa_32b_validation import CAAModelValidator

        # The persisted job name is retained for schedule compatibility, but
        # validation follows the exact active cortex rather than a size label.
        return CAAModelValidator().run()

    async def _job_proof_bundle(self) -> dict[str, Any]:
        from tools.proof_bundle import build_proof_bundle

        output = state_root() / "data" / "proof_bundle" / "latest"
        return build_proof_bundle(output_dir=output)

    async def _job_self_test_synthesis(self) -> dict[str, Any]:
        from core.evaluation.self_test_synthesizer import SelfTestSynthesizer

        synth = SelfTestSynthesizer()
        tests = synth.synthesize_tests([])
        return {"generated_tests": len(tests)}

    async def _job_architecture_auto(self) -> dict[str, Any]:
        from core.architect.config import ASAConfig
        from core.architect.governor import AutonomousArchitectureGovernor

        config = ASAConfig.from_env()
        if not config.enabled or not config.autopromote:
            return {
                "status": "disabled",
                "enabled": config.enabled,
                "autopromote": config.autopromote,
            }
        governor = AutonomousArchitectureGovernor(config)
        return await asyncio.to_thread(governor.auto, tier_max=config.max_tier)

    async def _job_overt_action_cycle(self) -> dict[str, Any]:
        from core.runtime.overt_action_loop import get_overt_action_loop

        return await get_overt_action_loop().run_once()

    async def _job_online_lora_status(self) -> dict[str, Any]:
        from core.adaptation.online_lora_governor import get_online_lora_governor

        governor = get_online_lora_governor()
        return {
            "enabled": governor.enabled(),
            "active_lora_processes": governor.active_lora_processes(),
            "last_receipt": governor.last_receipt.to_dict() if governor.last_receipt else None,
        }

    async def _job_influence_campaign(self) -> dict[str, Any]:
        """Measure whether one faculty actually changes the output.

        This is the job that makes ``core/verify`` mean something. Every lesion
        site on the live generation path, the null-arm refusal, the bootstrap
        interval — all of it existed to answer "did this faculty matter?" and
        nothing ever ran the trials, so on a live boot the answer was
        permanently UNMEASURED.

        ONE channel per run, rotating. A trial is three generations (intact,
        lesioned, intact-again) and two thirds of that cost buys the right to
        believe the other third. Measuring six channels at once would be an
        hour of the model's day; measuring one is a few minutes, and evidence
        accumulates across runs because the ledger is now persisted. Slow and
        real beats fast and unpowered.

        The probe stimulus is fixed and deliberately dull. It is a measurement
        input, not a prompt technique: the probe varies exactly one thing and
        it must be the lesion, so anything that differs between the three arms
        gets attributed to the channel. Nothing here tries to make Aura answer
        better — it only needs an identical input three times.
        """
        from core.container import ServiceContainer
        from core.verify.causal_influence import get_influence_ledger
        from core.verify.influence_campaign import (
            campaign_admission_reason,
            run_influence_campaign,
        )
        from core.verify.lesion_registry import get_lesion_registry

        # Counted, not only logged. An hourly job that has produced no verdicts
        # and an hourly job that has never once been admitted looked identical
        # from outside: a deferral wrote a reason to the log and left no count,
        # so "no evidence yet" could not be told from "the bar is never met on
        # this host". The record says which, and which condition refuses most.
        from core.verify.why_the_campaign_did_not_run import note_a_consideration

        refusal = campaign_admission_reason()
        if refusal:
            note_a_consideration("deferred", because=refusal)
            return {"status": "deferred", "reason": refusal}

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate is None or not hasattr(gate, "generate"):
            note_a_consideration(
                "unavailable", because="inference_gate_not_registered"
            )
            return {"status": "unavailable", "reason": "inference_gate_not_registered"}

        channels = list(get_lesion_registry().channels())
        if not channels:
            note_a_consideration("idle", because="no_registered_lesions")
            return {"status": "idle", "reason": "no_registered_lesions"}

        # Rotate by least-evidence-first: the channel with the fewest null
        # samples is the one whose verdict is furthest away, so it is the one
        # worth the model time. Ties break on name for determinism.
        ledger = get_influence_ledger()
        channel = min(
            channels,
            key=lambda name: (ledger.verdict(name).n_null, name),
        )

        async def generate() -> str:
            result = await gate.generate(
                _INFLUENCE_PROBE_PROMPT,
                {
                    "origin": "influence_probe",
                    "max_tokens": 96,
                    "messages": [{"role": "user", "content": _INFLUENCE_PROBE_PROMPT}],
                },
                timeout=_INFLUENCE_PROBE_TIMEOUT_S,
            )
            return str(getattr(result, "text", result) or "")

        report = await run_influence_campaign(
            generate=generate,
            channels=[channel],
            trials=3,
            per_generation_timeout_s=_INFLUENCE_PROBE_TIMEOUT_S,
            deadline_s=_INFLUENCE_CAMPAIGN_DEADLINE_S,
        )
        verdict = ledger.verdict(channel)
        # A run that reached a verdict is a different event from a run that
        # added another sample, and only the first is what the apparatus was
        # built for.
        note_a_consideration(
            "ran" if report.ran else "deferred",
            because="" if report.ran else "the campaign refused after admission",
            channel=channel,
            reached_a_verdict=str(verdict.verdict) not in ("UNMEASURED", "Verdict.UNMEASURED"),
        )
        return {
            "status": "ran" if report.ran else "deferred",
            "channel": channel,
            "verdict": str(verdict.verdict),
            "n_treatment": verdict.n_treatment,
            "n_null": verdict.n_null,
            "report": report.as_dict(),
        }

    async def _job_internal_deliberation(self) -> dict[str, Any]:
        from core.autonomy.topic_selection import select_autonomous_topic
        from core.container import ServiceContainer

        orchestrator = ServiceContainer.get("orchestrator", default=None)
        agency = ServiceContainer.get("agency_core", default=None)
        swarm = getattr(agency, "swarm", None)
        if orchestrator is None or swarm is None or not hasattr(swarm, "run_deliberation"):
            return {"status": "unavailable", "reason": "agency_swarm_not_registered"}
        state = getattr(getattr(orchestrator, "kernel", None), "state", None)
        candidate = select_autonomous_topic(orchestrator, state)
        if candidate is None:
            return {"status": "idle", "reason": "no_grounded_deliberation_topic"}
        return await swarm.run_deliberation(
            topic=candidate.text,
            topic_source=candidate.source,
            max_perspectives=2,
        )


_instance: AutonomyConductor | None = None


def get_autonomy_conductor() -> AutonomyConductor:
    global _instance
    if _instance is None:
        _instance = AutonomyConductor()
    return _instance


async def start_default_conductor() -> AutonomyConductor:
    conductor = get_autonomy_conductor()
    conductor.register_defaults()
    await conductor.start()
    return conductor


__all__ = [
    "ConductedJob",
    "AutonomyConductor",
    "get_autonomy_conductor",
    "start_default_conductor",
]
