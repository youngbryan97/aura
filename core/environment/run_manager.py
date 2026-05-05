"""Multi-run environment manager with death handling, restart, and metrics.

This is the outer loop that sits above EnvironmentKernel. It handles:
- Durable run records
- Death detection → postmortem → restart
- Aggregate metrics across runs
- Mode tracking (strict_real, simulated_canary, fixture_replay)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .postmortem import PostmortemGenerator, PostmortemReport

logger = logging.getLogger("Aura.RunManager")

RunMode = Literal["strict_real", "simulated_canary", "fixture_replay"]


@dataclass
class RunRecord:
    """Durable record of a single environment run."""
    run_id: str
    environment_id: str
    mode: RunMode
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    terminal_reason: str = ""  # "death", "success", "crash", "timeout", "contamination"
    total_steps: int = 0
    final_score: float = 0.0
    policy_version: str = ""
    source_commit: str = ""
    adapter_config: dict[str, Any] = field(default_factory=dict)
    boundary_config: dict[str, Any] = field(default_factory=dict)
    simulated: bool = False
    contaminated: bool = False
    postmortem: PostmortemReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateMetrics:
    """Metrics across multiple runs."""
    total_runs: int = 0
    deaths: int = 0
    crashes: int = 0
    successes: int = 0
    contaminated: int = 0
    timeouts: int = 0
    avg_steps_per_run: float = 0.0
    avg_score: float = 0.0
    best_score: float = 0.0
    run_ids: list[str] = field(default_factory=list)


class RunManager:
    """Manages multi-episode environment runs with lifecycle handling."""

    def __init__(self, mode: RunMode = "strict_real"):
        self.mode = mode
        self.records: list[RunRecord] = []
        self.current_record: RunRecord | None = None
        self.postmortem_gen = PostmortemGenerator()
        self.learned_affordances: dict[str, float] = {}  # persists across runs
        self.learned_procedures: list[dict[str, Any]] = []  # persists across runs

    def start_run(
        self,
        *,
        run_id: str,
        environment_id: str,
        policy_version: str = "",
        source_commit: str = "",
        adapter_config: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Create a durable run record before first action."""
        record = RunRecord(
            run_id=run_id,
            environment_id=environment_id,
            mode=self.mode,
            policy_version=policy_version,
            source_commit=source_commit,
            adapter_config=adapter_config or {},
            simulated=self.mode == "simulated_canary",
        )
        self.current_record = record
        self.records.append(record)
        logger.info(f"Run started: {run_id} mode={self.mode}")
        return record

    def record_step(self, frame) -> None:
        """Called after each kernel.step()."""
        if self.current_record:
            self.current_record.total_steps += 1

    def detect_death(self, observation_text: str) -> bool:
        """Check if the observation indicates terminal death."""
        death_markers = (
            "You die", "You have died", "DYWYPI",
            "Do you want your possessions identified",
            "You were killed", "You are dead", "Goodbye ",
        )
        return any(marker in observation_text for marker in death_markers)

    def end_run(
        self,
        *,
        terminal_reason: str,
        frames: list,
        final_score: float = 0.0,
    ) -> RunRecord:
        """End the current run, generate postmortem if applicable."""
        if not self.current_record:
            raise RuntimeError("No active run to end")

        record = self.current_record
        record.ended_at = time.time()
        record.terminal_reason = terminal_reason
        record.final_score = final_score

        # Generate postmortem for deaths and crashes
        if terminal_reason in ("death", "crash"):
            record.postmortem = self.postmortem_gen.generate(
                run_id=record.run_id,
                environment_id=record.environment_id,
                mode=record.mode,
                terminal_reason=terminal_reason,
                frames=frames,
                started_at=record.started_at,
            )

        self.current_record = None
        logger.info(f"Run ended: {record.run_id} reason={terminal_reason} steps={record.total_steps}")
        return record

    def mark_contaminated(self, reason: str) -> None:
        """Flag the current run as contaminated by boundary violation."""
        if self.current_record:
            self.current_record.contaminated = True
            self.current_record.metadata["contamination_reason"] = reason

    def restart(self) -> None:
        """Prepare for a new run. Clears ephemeral state, preserves learning."""
        # Ephemeral world state is cleared by the kernel getting a new adapter.start()
        # Learned affordances and procedures persist.
        self.current_record = None

    def get_metrics(self) -> AggregateMetrics:
        """Compute aggregate metrics across all runs."""
        metrics = AggregateMetrics()
        total_steps = 0
        total_score = 0.0

        for record in self.records:
            metrics.total_runs += 1
            metrics.run_ids.append(record.run_id)
            total_steps += record.total_steps
            total_score += record.final_score
            metrics.best_score = max(metrics.best_score, record.final_score)

            if record.contaminated:
                metrics.contaminated += 1
            elif record.terminal_reason == "death":
                metrics.deaths += 1
            elif record.terminal_reason == "crash":
                metrics.crashes += 1
            elif record.terminal_reason == "success":
                metrics.successes += 1
            elif record.terminal_reason == "timeout":
                metrics.timeouts += 1

        if metrics.total_runs > 0:
            metrics.avg_steps_per_run = total_steps / metrics.total_runs
            metrics.avg_score = total_score / metrics.total_runs

        return metrics


__all__ = ["RunRecord", "RunMode", "AggregateMetrics", "RunManager"]
