"""core/runtime/worker_liveness.py — is this worker wedged, or just busy?

Clean-room adoption of the request-slot discipline that mature inference
servers (llama.cpp's ``llama-server``, vLLM, TGI) settled on: **the unit of
failure is one request, not the loaded model.** A server that unloads its
weights because a caller stopped waiting has confused two unrelated events.

Aura's resident Cortex holds roughly 20GB of wired memory. Destroying that
worker costs a cold reload, and the reload itself has historically triggered a
cascade: a second worker stacking beside the first, memory doubling, and a
death cluster. So a kill is one of the most expensive actions the runtime can
take, and it was being decided by a state machine that never asked whether the
worker was actually doing anything.

The evidence to answer that question already existed. The worker publishes
``active_job``, ``job_age_s`` and ``loop_stalled`` in its heartbeat; the client
tracks ``_last_heartbeat`` and per-request first-token budgets. What was missing
was a single place where those signals decide whether killing is warranted —
so instead, ``_reset_stale_lane_state`` killed a possibly-healthy worker because
the LANE STATE had gone stale, which is a statement about bookkeeping, not
about the model.

This module is that single place. It is deliberately:

* **general** — it classifies any long-lived worker from generic evidence, not
  just the MLX lane, so browser workers, tool runners and future model lanes
  can share one vocabulary for "is it dead or is it thinking?";
* **pure** — no I/O, no imports above the runtime foundation, so it can be
  called from anywhere including shutdown paths;
* **conservative** — when evidence is missing the verdict is UNKNOWN and
  killing is NOT justified. An expensive irreversible action must not be taken
  on absent information.

The graded response:

  GENERATING  → never kill. Cancel the *request* if the caller left.
  IDLE        → never kill for staleness. Recycle gracefully if you must.
  STALLED     → cancel the request and escalate; the model is probably fine.
  WEDGED      → killing is justified: nothing has proven liveness in too long.
  DEAD        → already gone; reap it.
  UNKNOWN     → do not kill.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "LivenessPolicy",
    "LivenessVerdict",
    "WorkerEvidence",
    "WorkerLiveness",
    "classify_worker",
    "kill_is_justified",
]


class LivenessVerdict(StrEnum):
    """What the evidence actually supports saying about a worker."""

    #: Process is gone. Reap it.
    DEAD = "dead"
    #: Nothing has proven liveness within tolerance. A kill is justified.
    WEDGED = "wedged"
    #: Alive and holding a job, but that job has stopped making progress.
    #: The REQUEST is the problem; cancel it before touching the process.
    STALLED = "stalled"
    #: Alive and actively producing. Killing this destroys working state.
    GENERATING = "generating"
    #: Alive with no work in flight.
    IDLE = "idle"
    #: Not enough evidence to say. Never a licence to kill.
    UNKNOWN = "unknown"


#: Verdicts under which destroying the process is a defensible action.
_KILLABLE = frozenset({LivenessVerdict.DEAD, LivenessVerdict.WEDGED})


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    try:
        value = float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return max(low, min(high, value))


@dataclass(frozen=True)
class LivenessPolicy:
    """Thresholds for turning evidence into a verdict.

    Defaults are deliberately generous. These decide whether to destroy an
    expensive resident process, so the cost of waiting one more interval is far
    lower than the cost of a wrongful kill.
    """

    #: No heartbeat for longer than this and the worker has proven nothing.
    #: Well above the 2s heartbeat interval so ordinary scheduling jitter, a
    #: GC pause, or a burst of IPC backlog cannot look like death.
    heartbeat_timeout_s: float = 90.0
    #: An active job with no progress for longer than this is STALLED. Long
    #: enough to cover prompt evaluation on a large resident model, which
    #: legitimately produces no tokens for a while.
    job_stall_s: float = 120.0
    #: A stalled job that stays stalled this long escalates to WEDGED, because
    #: at that point the decode loop is not merely slow.
    job_wedged_s: float = 600.0

    @classmethod
    def from_env(cls) -> LivenessPolicy:
        return cls(
            heartbeat_timeout_s=_env_float(
                "AURA_WORKER_HEARTBEAT_TIMEOUT_S", 90.0, low=5.0, high=3600.0
            ),
            job_stall_s=_env_float(
                "AURA_WORKER_JOB_STALL_S", 120.0, low=5.0, high=3600.0
            ),
            job_wedged_s=_env_float(
                "AURA_WORKER_JOB_WEDGED_S", 600.0, low=10.0, high=7200.0
            ),
        )


@dataclass
class WorkerEvidence:
    """What is actually known about a worker at one moment.

    Every field is optional-shaped on purpose: callers assemble this from
    whatever signals they happen to have, and missing evidence must produce
    UNKNOWN rather than a confident wrong answer.
    """

    #: Whether the OS process exists. None when the caller could not check.
    process_alive: bool | None = None
    #: Seconds since the last heartbeat. None when no heartbeat has ever been
    #: seen (which is NOT the same as a stale one — a worker that has not yet
    #: reported is starting, not dying).
    last_heartbeat_age_s: float | None = None
    #: Whether the worker says it is holding a job.
    active_job: bool = False
    #: How long the current job has run, per the worker.
    job_age_s: float = 0.0
    #: Whether the worker's own watchdog reports its decode loop stalled.
    loop_stalled: bool = False
    #: Seconds since this worker last produced observable output (a token, a
    #: chunk, a partial result). The strongest available proof of progress.
    last_progress_age_s: float | None = None
    #: Free-form provenance for the receipt.
    source: str = ""
    observed_at: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_alive": self.process_alive,
            "last_heartbeat_age_s": _round(self.last_heartbeat_age_s),
            "active_job": self.active_job,
            "job_age_s": _round(self.job_age_s),
            "loop_stalled": self.loop_stalled,
            "last_progress_age_s": _round(self.last_progress_age_s),
            "source": self.source,
            "observed_at": round(self.observed_at, 3),
        }


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 3)


@dataclass
class WorkerLiveness:
    """A verdict, the reason for it, and whether it licenses a kill."""

    verdict: LivenessVerdict
    reason: str
    evidence: WorkerEvidence
    policy: LivenessPolicy

    @property
    def kill_justified(self) -> bool:
        return self.verdict in _KILLABLE

    @property
    def should_cancel_request(self) -> bool:
        """A stalled job is a REQUEST problem; cancel before escalating."""
        return self.verdict is LivenessVerdict.STALLED

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "kill_justified": self.kill_justified,
            "should_cancel_request": self.should_cancel_request,
            "evidence": self.evidence.to_dict(),
        }

    def __str__(self) -> str:
        return f"{self.verdict.value}({self.reason})"


def classify_worker(
    evidence: WorkerEvidence, policy: LivenessPolicy | None = None
) -> WorkerLiveness:
    """Turn worker evidence into a verdict.

    Order matters: cheap certainties first, then progress, then silence. The
    ordering encodes the priority — *proof of life outranks proof of
    bookkeeping staleness*, which is precisely the inversion that made a stale
    lane state destroy a working model.
    """
    policy = policy or LivenessPolicy.from_env()

    def verdict(v: LivenessVerdict, reason: str) -> WorkerLiveness:
        return WorkerLiveness(v, reason, evidence, policy)

    # 1. The process is gone. Nothing else matters.
    if evidence.process_alive is False:
        return verdict(LivenessVerdict.DEAD, "process_not_alive")

    # 2. Direct proof of output is the strongest signal there is. A worker
    #    producing tokens is not wedged, no matter what any state machine says.
    progress_age = evidence.last_progress_age_s
    if progress_age is not None and progress_age <= policy.job_stall_s:
        return verdict(
            LivenessVerdict.GENERATING,
            f"produced output {progress_age:.1f}s ago",
        )

    # 3. Silence. A worker that has never reported is starting, not dying —
    #    None and "stale" are different claims and must not collapse.
    heartbeat_age = evidence.last_heartbeat_age_s
    if heartbeat_age is None:
        if evidence.process_alive is None:
            return verdict(LivenessVerdict.UNKNOWN, "no evidence available")
        return verdict(
            LivenessVerdict.UNKNOWN,
            "process alive but no heartbeat observed yet",
        )
    if heartbeat_age > policy.heartbeat_timeout_s:
        return verdict(
            LivenessVerdict.WEDGED,
            f"no heartbeat for {heartbeat_age:.1f}s "
            f"(> {policy.heartbeat_timeout_s:.0f}s)",
        )

    # 4. The heartbeat is fresh, so the process is demonstrably running its own
    #    loop. From here the question is only about the JOB.
    if not evidence.active_job:
        return verdict(LivenessVerdict.IDLE, "heartbeat fresh, no active job")

    job_age = max(0.0, float(evidence.job_age_s or 0.0))
    if evidence.loop_stalled or job_age > policy.job_stall_s:
        if job_age > policy.job_wedged_s:
            return verdict(
                LivenessVerdict.WEDGED,
                f"job stalled for {job_age:.1f}s "
                f"(> {policy.job_wedged_s:.0f}s) despite a live heartbeat",
            )
        return verdict(
            LivenessVerdict.STALLED,
            f"job active {job_age:.1f}s with no progress; cancel the request",
        )

    return verdict(
        LivenessVerdict.GENERATING,
        f"heartbeat fresh, job running {job_age:.1f}s within budget",
    )


def kill_is_justified(
    evidence: WorkerEvidence, policy: LivenessPolicy | None = None
) -> bool:
    """Convenience predicate for call sites that only need the yes/no."""
    return classify_worker(evidence, policy).kill_justified
