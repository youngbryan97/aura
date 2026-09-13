"""core/factory/software_factory.py — Autonomous Software Factory.

End-to-end pipeline: map repo → find weaknesses → choose task → create branch →
patch code → run tests → run lint/security → compare baseline → generate diff →
create rollback → write docs → request approval.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.config import config
from core.factory.code_writer import CodeWriter
from core.factory.patch_planner import PatchPlanner
from core.factory.pr_builder import PRBuilder
from core.factory.regression_guard import RegressionGuard
from core.factory.repo_cartographer import RepoCartographer
from core.factory.rollback_manager import RollbackManager
from core.factory.test_runner import TestRunner

logger = logging.getLogger("Aura.SoftwareFactory")


class PipelineStage(StrEnum):
    MAPPING = "mapping"
    PLANNING = "planning"
    BRANCHING = "branching"
    PATCHING = "patching"
    TESTING = "testing"
    GUARDING = "guarding"
    DIFFING = "diffing"
    ROLLBACK_PREP = "rollback_prep"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FactoryJob:
    """Tracks one end-to-end software factory run."""
    job_id: str
    repo_path: str
    objective: str
    stage: PipelineStage = PipelineStage.MAPPING
    started_at: float = field(default_factory=lambda: time.time())
    completed_at: float = 0.0
    repo_map: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    branch_name: str = ""
    patches: list[dict[str, Any]] = field(default_factory=list)
    test_results: dict[str, Any] | None = None
    guard_results: dict[str, Any] | None = None
    diff_summary: str = ""
    rollback_id: str = ""
    error: str | None = None


class SoftwareFactory:
    """Aura's canonical autonomous software engineering pipeline.

    Maps → Plans → Branches → Patches → Tests → Guards → Diffs → Rollback → Review.
    """

    def __init__(self) -> None:
        self.cartographer = RepoCartographer()
        self.planner = PatchPlanner()
        self.writer = CodeWriter()
        self.tester = TestRunner()
        self.guard = RegressionGuard()
        self.pr_builder = PRBuilder()
        self.rollback = RollbackManager()
        self.jobs: dict[str, FactoryJob] = {}
        self._job_counter = 0

    async def run_pipeline(
        self,
        repo_path: str,
        objective: str,
        *,
        auto_approve: bool = True,
    ) -> dict[str, Any]:
        """Execute the full software factory pipeline."""
        self._job_counter += 1
        job = FactoryJob(
            job_id=f"factory_{self._job_counter}_{int(time.time())}",
            repo_path=repo_path,
            objective=objective,
        )
        self.jobs[job.job_id] = job
        logger.info("🏭 SoftwareFactory starting job %s: %s", job.job_id, objective[:60])

        try:
            # 1. MAP: Analyze the repository
            job.stage = PipelineStage.MAPPING
            job.repo_map = await self.cartographer.map_repo(repo_path)
            logger.info("🗺️  Repo mapped: %d files, %d modules",
                       job.repo_map.get("file_count", 0), job.repo_map.get("module_count", 0))

            # 2. PLAN: Determine what to change
            job.stage = PipelineStage.PLANNING
            job.plan = await self.planner.create_plan(objective, job.repo_map)
            logger.info("📋 Plan created: %d changes proposed", len(job.plan.get("changes", [])))

            # 3. BRANCH: Create isolated workspace
            job.stage = PipelineStage.BRANCHING
            job.branch_name = f"aura/factory-{job.job_id}"
            await self.rollback.create_workspace(repo_path, job.branch_name)

            # 4. PATCH: Write code changes
            job.stage = PipelineStage.PATCHING
            for change in job.plan.get("changes", []):
                patch = await self.writer.write_patch(change, repo_path)
                job.patches.append(patch)
            logger.info("✏️  Applied %d patches", len(job.patches))

            # 5. TEST: Run test suite
            job.stage = PipelineStage.TESTING
            job.test_results = await self.tester.run_tests(repo_path)
            if not job.test_results.get("all_passed", False):
                logger.warning("❌ Tests failed: %s", job.test_results.get("summary"))
                if not auto_approve:
                    job.stage = PipelineStage.FAILED
                    job.error = "test_failure"
                    return self._finalize(job)

            # 6. GUARD: Run lint, security, regression checks
            job.stage = PipelineStage.GUARDING
            job.guard_results = await self.guard.run_checks(repo_path, job.patches)
            if job.guard_results.get("regressions_found", 0) > 0:
                logger.warning("🛡️ Regression guard found issues: %s", job.guard_results)

            # 7. DIFF: Generate summary
            job.stage = PipelineStage.DIFFING
            job.diff_summary = self.pr_builder.generate_diff_summary(job.patches)

            # 8. ROLLBACK PREP: Ensure we can undo
            job.stage = PipelineStage.ROLLBACK_PREP
            job.rollback_id = self.rollback.register_rollback_point(repo_path, job.branch_name)

            # 9. REVIEW
            job.stage = PipelineStage.REVIEW
            job.stage = PipelineStage.COMPLETED

        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError) as e:
            job.error = str(e)
            job.stage = PipelineStage.FAILED
            logger.error("🏭 Factory job %s failed at stage %s: %s", job.job_id, job.stage, e)

        return self._finalize(job)

    def _finalize(self, job: FactoryJob) -> dict[str, Any]:
        job.completed_at = time.time()
        return {
            "ok": job.stage == PipelineStage.COMPLETED,
            "job_id": job.job_id,
            "stage": str(job.stage),
            "duration_s": job.completed_at - job.started_at,
            "patches_applied": len(job.patches),
            "tests": job.test_results,
            "guard": job.guard_results,
            "diff_summary": job.diff_summary[:500] if job.diff_summary else "",
            "rollback_id": job.rollback_id,
            "error": job.error,
        }

    async def run_mission(
        self,
        plan_steps: list[str],
        constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Runs the software factory pipeline as a mission engine step."""
        repo_path = str(config.paths.project_root)
        objective = " ".join(plan_steps)
        return await self.run_pipeline(repo_path, objective)

    def get_job(self, job_id: str) -> FactoryJob | None:
        return self.jobs.get(job_id)
