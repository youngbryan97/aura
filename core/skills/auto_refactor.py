"""Governed skill adapter for Aura's incremental code-health metabolism."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.errors import record_degradation
from core.self_modification.incremental_code_health import IncrementalCodeHealthScanner
from core.self_modification.safe_modification_harness import SafeModificationHarness
from core.skills.base_skill import BaseSkill
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Skills.AutoRefactor")

_MAX_VALIDATION_FILES = 64
_MAX_VALIDATION_BYTES = 2 * 1024 * 1024


class AutoRefactorParams(BaseModel):
    path: str = Field(".", description="Repository-relative Python file or directory to scan.")
    run_tests: bool = Field(
        False,
        description="Run a bounded pytest target when path names a specific test file or subtree.",
    )
    max_files: int = Field(
        100,
        ge=1,
        le=2_000,
        description="Maximum Python files examined in this incremental batch.",
    )
    time_budget_s: float = Field(
        1.5,
        ge=0.25,
        le=10.0,
        description="Hard wall-clock budget for inventory and static analysis in this batch.",
    )


class AutoRefactorSkill(BaseSkill):
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "auto_refactor"
    description = (
        "Incrementally analyzes repository code health under explicit time and file budgets, "
        "with cached coverage receipts and deduplicated repair proposals."
    )
    input_model = AutoRefactorParams
    timeout_seconds = 12.0
    metabolic_cost = 2
    retry_safe = True

    def __init__(self, *, root_path: Path | None = None) -> None:
        super().__init__()
        configured_root = os.environ.get("AURA_PROJECT_ROOT", "").strip()
        resolved_root = (
            root_path.expanduser().resolve()
            if root_path is not None
            else (
                Path(configured_root).expanduser().resolve()
                if configured_root
                else Path(__file__).resolve().parents[2]
            )
        )
        self._scanner = IncrementalCodeHealthScanner(resolved_root)
        self._execution_lock = asyncio.Lock()
        self._proposal_last_published: dict[str, float] = {}
        self._validation_jobs: dict[str, dict[str, Any]] = {}

    @property
    def root_path(self) -> Path:
        return self._scanner.root_path

    async def execute(
        self,
        params: AutoRefactorParams,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = AutoRefactorParams(**params)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("auto_refactor", exc)
                return {"ok": False, "error": f"Invalid input: {exc}"}

        if self._execution_lock.locked():
            return {
                "ok": False,
                "status": "deferred",
                "reason": "scan_in_progress",
                "message": "A code-health batch is already active; its cached cycle will continue.",
            }

        async with self._execution_lock:
            try:
                report = await asyncio.to_thread(
                    self._scan_codebase,
                    params.path,
                    max_files=params.max_files,
                    time_budget_s=params.time_budget_s,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "auto_refactor",
                    exc,
                    severity="warning",
                    action="returned bounded code-health scan failure",
                    extra={"target_path": params.path},
                )
                return {"ok": False, "error": str(exc)}

            issues = list(report.pop("issues", []))
            top_issues = issues[:10]
            proposals_published = self._publish_proposals(top_issues)
            test_results = None
            if params.run_tests:
                test_results = await self._request_targeted_validation(
                    Path(report["target"]),
                )

            coverage = dict(report["coverage"])
            complete = bool(coverage.get("coverage_complete"))
            status = "completed" if complete else "partial"
            completion = "complete" if complete else "partial"
            return {
                "ok": True,
                "status": status,
                "issues_found": len(issues),
                "issues_known": len(issues),
                "top_issues": top_issues,
                "coverage": coverage,
                "scan_errors": report["scan_errors"],
                "proposals_published": proposals_published,
                "test_results": test_results,
                "message": (
                    f"Code-health batch {completion} for {report['display_target']}: "
                    f"{coverage['files_examined_this_batch']} file(s) examined, "
                    f"{len(issues)} cached issue(s) known."
                ),
            }

    def _scan_codebase(
        self,
        path_str: str,
        *,
        max_files: int = 100,
        time_budget_s: float = 1.5,
    ) -> dict[str, Any]:
        return self._scanner.scan(
            path_str,
            max_files=max_files,
            time_budget_s=time_budget_s,
        )

    async def _request_targeted_validation(
        self,
        target: Path,
    ) -> dict[str, Any]:
        display = self._scanner.display_path(target)
        parts = Path(display).parts
        target_is_file = await asyncio.to_thread(target.is_file)
        target_is_dir = await asyncio.to_thread(target.is_dir)
        targeted = target_is_file and target.name.startswith("test_")
        targeted = targeted or bool(
            target_is_dir and parts and parts[0] == "tests" and display != "tests"
        )
        if not targeted:
            return {
                "ok": False,
                "status": "deferred",
                "reason": "targeted_test_scope_required",
                "message": "Test execution requires a specific test file or tests subtree.",
            }

        try:
            validation_inputs = await asyncio.to_thread(
                self._collect_validation_inputs,
                target,
            )
        except (OSError, UnicodeError) as exc:
            return {"ok": False, "status": "failed", "error": str(exc)}
        if validation_inputs.get("status") == "deferred":
            return validation_inputs

        changed_files = list(validation_inputs["changed_files"])
        patch_content = dict(validation_inputs["patch_content"])
        digest = hashlib.sha256()
        for relative_path in changed_files:
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(patch_content[relative_path].encode("utf-8")).digest())
        job_id = hashlib.sha256(
            f"{display}\0{digest.hexdigest()}".encode()
        ).hexdigest()[:20]
        existing = self._validation_jobs.get(job_id)
        if existing is not None:
            return dict(existing)

        job = {
            "ok": False,
            "status": "queued",
            "job_id": job_id,
            "target": display,
            "submitted_at": time.time(),
            "owner": "safe_modification_harness",
            "source_file_count": len(changed_files),
        }
        self._validation_jobs[job_id] = job
        coroutine = self._run_validation_job(
            job_id,
            display,
            changed_files,
            patch_content,
        )
        try:
            get_task_tracker().create_task(
                coroutine,
                name=f"auto_refactor.validation.{job_id}",
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            coroutine.close()
            job.update({"status": "failed", "error": str(exc), "completed_at": time.time()})
            record_degradation(
                "auto_refactor",
                exc,
                severity="warning",
                action="returned canonical validation scheduling failure with static scan preserved",
                extra={"target_path": display},
            )
        self._prune_validation_jobs()
        return dict(job)

    def _collect_validation_inputs(self, target: Path) -> dict[str, Any]:
        if target.is_file():
            candidates = [target]
        else:
            candidates = []
            for current, directory_names, file_names in os.walk(target, followlinks=False):
                directory_names[:] = sorted(
                    name
                    for name in directory_names
                    if not name.startswith(".") and name != "__pycache__"
                )
                for name in sorted(file_names):
                    if not name.endswith(".py"):
                        continue
                    candidates.append(Path(current) / name)
                    if len(candidates) > _MAX_VALIDATION_FILES:
                        return {
                            "ok": False,
                            "status": "deferred",
                            "reason": "validation_scope_too_large",
                            "message": (
                                "Targeted validation is limited to "
                                f"{_MAX_VALIDATION_FILES} Python files per job."
                            ),
                        }

        if not candidates:
            return {
                "ok": False,
                "status": "deferred",
                "reason": "no_python_tests_in_scope",
                "message": "The requested test scope contains no Python source files.",
            }

        patch_content: dict[str, str] = {}
        total_bytes = 0
        for candidate in candidates:
            source_bytes = candidate.read_bytes()
            total_bytes += len(source_bytes)
            if total_bytes > _MAX_VALIDATION_BYTES:
                return {
                    "ok": False,
                    "status": "deferred",
                    "reason": "validation_scope_too_large",
                    "message": (
                        "Targeted validation source exceeds the "
                        f"{_MAX_VALIDATION_BYTES}-byte job limit."
                    ),
                }
            relative_path = self._scanner.display_path(candidate)
            patch_content[relative_path] = source_bytes.decode("utf-8")

        return {
            "changed_files": sorted(patch_content),
            "patch_content": patch_content,
        }

    async def _run_validation_job(
        self,
        job_id: str,
        display: str,
        changed_files: list[str],
        patch_content: dict[str, str],
    ) -> None:
        job = self._validation_jobs[job_id]
        job.update({"status": "running", "started_at": time.time()})
        try:
            result = await SafeModificationHarness(self.root_path).run(
                changed_files,
                patch_content=patch_content,
                extra_test_targets=[display],
                require_distributed_sandbox=False,
            )
            job.update(
                {
                    "ok": bool(result.passed),
                    "status": "passed" if result.passed else "failed",
                    "checks": dict(result.checks),
                    "errors": list(result.errors[:10]),
                    "duration_s": round(float(result.duration_s), 3),
                    "completed_at": time.time(),
                }
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            job.update(
                {
                    "ok": False,
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": time.time(),
                }
            )
            record_degradation(
                "auto_refactor",
                exc,
                severity="warning",
                action="preserved failed canonical validation job receipt",
                extra={"target_path": display, "job_id": job_id},
            )

    def _prune_validation_jobs(self) -> None:
        if len(self._validation_jobs) <= 64:
            return
        completed = [
            job_id
            for job_id, job in self._validation_jobs.items()
            if job.get("status") not in {"queued", "running"}
        ]
        for job_id in completed[: len(self._validation_jobs) - 64]:
            self._validation_jobs.pop(job_id, None)

    def _publish_proposals(self, issues: list[dict[str, Any]]) -> int:
        if not issues:
            return 0
        from core.event_bus import get_event_bus

        now = time.time()
        cooldown_s = 6 * 3600.0
        published = 0
        bus = get_event_bus()
        for issue_data in issues:
            payload = json.dumps(
                {
                    "rule_id": issue_data.get("rule_id"),
                    "file": issue_data.get("file"),
                    "line": issue_data.get("line"),
                    "message": issue_data.get("message"),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if now - self._proposal_last_published.get(key, 0.0) < cooldown_s:
                continue
            try:
                bus.publish_threadsafe(
                    "refactor_proposal",
                    {"source": "AutoRefactorSkill", "issue": issue_data},
                )
                self._proposal_last_published[key] = now
                published += 1
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("auto_refactor", exc)
                logger.error("Failed to publish refactor proposal: %s", exc)

        if len(self._proposal_last_published) > 2_000:
            stale = [
                key
                for key, published_at in self._proposal_last_published.items()
                if now - published_at >= cooldown_s
            ]
            for key in stale:
                self._proposal_last_published.pop(key, None)
        return published


__all__ = ["AutoRefactorParams", "AutoRefactorSkill"]
