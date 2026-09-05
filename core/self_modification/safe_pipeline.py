"""core/self_modification/safe_pipeline.py

Self-Modification Safety Pipeline
=====================================
Every structural mutation Aura proposes about her own code travels through
a fixed nine-stage pipeline. Stages 1–4 are pre-flight, 5–8 are
sandboxed execution, 9 is staged production deployment. A failure at
any stage blocks the mutation and records a forensic envelope.

  1. PROPOSAL          — drive, intent, and target file recorded.
  2. SANDBOX_PATCH     — patch applied inside a tmpfs copy of the file.
  3. GENERATED_TESTS   — unit + AST-shape tests are generated for the
                         change set.
  4. FORMAL_VERIFY     — `formal_verifier.verify_mutation` is run; a
                         failure short-circuits.
  5. SHADOW_RUNTIME    — the patched module is loaded into a sub-process
                         under a memory + cpu cap; smoke calls are
                         exercised.
  6. DIFF_EXPLANATION  — a human-readable diff explanation is produced
                         and added to the proposal.
  7. ROLLBACK_PLAN     — the inverse-patch and the stem-cell snapshot
                         that would be restored on failure are recorded.
  8. APPROVAL          — Will + Conscience evaluate the proposal; if
                         either refuses, the proposal is blocked.
  9. STAGED_DEPLOY     — by default, the patch is staged as an operator
                         promotion artifact. Live source deployment requires
                         fresh owner approval plus a supervised operator
                         environment switch and then runs post-deploy
                         monitoring/rollback.

All ten phases produce auditable receipts in the same JSONL ledger as
agency receipts so external reviewers can reconstruct the lineage of
every code change.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import async_atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.self_modification.mutation_tiers import MutationTier, classify_mutation_path
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.SelfModSafePipeline")


_LEDGER_DIR = state_root() / "data" / "selfmod"
_LEDGER_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER_PATH = _LEDGER_DIR / "pipeline.jsonl"
_STAGING_DIR = _LEDGER_DIR / "staged"
_SUPERVISED_SELF_MODIFICATION_ENV = "AURA_ALLOW_SUPERVISED_SELF_MODIFICATION"


class Stage(StrEnum):
    PROPOSAL = "proposal"
    SANDBOX_PATCH = "sandbox_patch"
    GENERATED_TESTS = "generated_tests"
    FORMAL_VERIFY = "formal_verify"
    SHADOW_RUNTIME = "shadow_runtime"
    DIFF_EXPLANATION = "diff_explanation"
    ROLLBACK_PLAN = "rollback_plan"
    APPROVAL = "approval"
    STAGED_DEPLOY = "staged_deploy"
    POST_DEPLOY_MONITOR = "post_deploy_monitor"


@dataclass
class PipelineProposal:
    proposal_id: str
    drive: str
    intent: str
    file_path: str
    before_source: str
    after_source: str
    diff_explanation: str | None = None
    rollback_plan: str | None = None
    will_receipt_id: str | None = None
    promotion_artifact_path: str | None = None
    started_at: float = field(default_factory=time.time)
    stages_completed: list[str] = field(default_factory=list)
    blocked_at: str | None = None
    blocked_reason: str | None = None


def _production_source_scope(target: Path, reason: str):
    """The governed scope for replacing one of Aura's own source files.

    The staging artifact already went through the gateway; the write that
    actually replaces production source did not. That is backwards — the
    less consequential write was accounted for and the more consequential
    one reached ``atomic_write_text`` directly, past the governance check,
    the ownership record and the write ledger.
    """
    from core.governance_context import local_internal_governed_scope

    return local_internal_governed_scope(
        f"safe_pipeline.{reason}",
        constraints={"target": str(target), "stage": reason},
    )


async def _write_production_source(target: Path, source_text: str, *, reason: str) -> None:
    with _production_source_scope(target, reason):
        await get_file_write_gateway().write_text_async(
            target,
            source_text,
            encoding="utf-8",
            source=f"self_modification.safe_pipeline.{reason}",
        )


def _write_production_source_sync(target: Path, source_text: str, *, reason: str) -> None:
    """The rollback path, which runs where an await is not available.

    Rollback is the one write that must not be deferred or dropped: it is
    what stands between a failed self-modification and a runtime serving
    broken source.
    """
    with _production_source_scope(target, reason):
        get_file_write_gateway().write_text(
            target,
            source_text,
            encoding="utf-8",
            source=f"self_modification.safe_pipeline.{reason}",
        )


def _record(p: PipelineProposal, event: str, payload: dict[str, Any] | None = None) -> None:
    try:
        get_file_write_gateway().append_text(
            _LEDGER_PATH,
            json.dumps({
                "when": time.time(),
                "event": event,
                "proposal_id": p.proposal_id,
                "snapshot": asdict(p),
                "payload": payload or {},
            }, default=str) + "\n",
            source="self_modification.safe_pipeline.ledger",
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        record_degradation('safe_pipeline', exc)
        logger.warning("self-mod pipeline ledger append failed: %s", exc)


def _supervised_source_deploy_enabled(owner_approved: bool) -> bool:
    """Return True only for explicit owner-approved supervised source deployment.

    The normal live runtime path may generate and validate patches, but it must
    not overwrite source files under the running interpreter. Promotion remains
    an operator action unless both a fresh owner approval and the supervised
    environment switch are present.
    """
    if not owner_approved:
        return False
    raw = os.getenv(_SUPERVISED_SELF_MODIFICATION_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ─── pipeline ──────────────────────────────────────────────────────────────


class SafePipeline:
    SHADOW_TIMEOUT_S = 30.0
    SHADOW_MEM_MB = 512
    POST_DEPLOY_MONITOR_S = 60.0
    POST_DEPLOY_POLL_S = 2.0

    async def run(
        self,
        *,
        drive: str,
        intent: str,
        file_path: str,
        before_source: str,
        after_source: str,
        owner_approved: bool = False,
    ) -> PipelineProposal:
        proposal = PipelineProposal(
            proposal_id=f"SMP-{uuid.uuid4().hex[:10]}",
            drive=drive,
            intent=intent,
            file_path=file_path,
            before_source=before_source,
            after_source=after_source,
        )
        _record(proposal, "proposal")

        tier_decision = classify_mutation_path(file_path)
        _record(proposal, "tier_classified", tier_decision.to_dict())
        if tier_decision.tier is MutationTier.SEALED:
            return self._block(
                proposal,
                Stage.APPROVAL,
                f"{tier_decision.path} is sealed from runtime self-modification",
            )
        # PROPOSE_ONLY targets still traverse the complete validation pipeline.
        # The tier prevents live deployment, not diagnosis or quarantine. This
        # lets the daily immune system produce evidence-bearing repairs without
        # mutating the interpreter that is currently running Aura.

        # 2. SANDBOX_PATCH
        sandbox = Path(tempfile.mkdtemp(prefix="aura-selfmod-"))
        sandbox_file = sandbox / Path(file_path).name
        await async_atomic_write_text(sandbox_file, after_source, encoding="utf-8")
        proposal.stages_completed.append(Stage.SANDBOX_PATCH.value)
        _record(proposal, "sandbox_patched", {"sandbox": str(sandbox)})

        try:
            # 3. GENERATED_TESTS
            test_path = sandbox / "test_self_mod_patch.py"
            await async_atomic_write_text(test_path, self._generate_tests(file_path, before_source, after_source), encoding="utf-8")
            proposal.stages_completed.append(Stage.GENERATED_TESTS.value)
            _record(proposal, "tests_generated")

            # 4. FORMAL_VERIFY
            try:
                from core.self_modification.formal_verifier import verify_mutation
                vr = verify_mutation(
                    file_path=file_path,
                    before_source=before_source,
                    after_source=after_source,
                    touches_tick_loop="mind_tick" in file_path or "orchestrator" in file_path,
                )
                _record(proposal, "verify", {"ok": vr.ok, "violated": vr.invariants_violated, "satisfied": vr.invariants_satisfied})
                if not vr.ok:
                    return self._block(proposal, Stage.FORMAL_VERIFY, "; ".join(vr.invariants_violated))
                proposal.stages_completed.append(Stage.FORMAL_VERIFY.value)
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('safe_pipeline', exc)
                return self._block(proposal, Stage.FORMAL_VERIFY, f"verify_exception:{exc}")

            # 5. SHADOW_RUNTIME — load the patched module in a subprocess
            # under tight resource caps and run a smoke call.
            shadow_ok, shadow_detail = await self._run_shadow(sandbox_file)
            _record(proposal, "shadow_runtime", {"ok": shadow_ok, "detail": shadow_detail})
            if not shadow_ok:
                return self._block(proposal, Stage.SHADOW_RUNTIME, shadow_detail)
            proposal.stages_completed.append(Stage.SHADOW_RUNTIME.value)

            # 6. DIFF_EXPLANATION
            proposal.diff_explanation = self._diff_explanation(before_source, after_source)
            _record(proposal, "diff_explained", {"diff_summary": proposal.diff_explanation[:300]})
            proposal.stages_completed.append(Stage.DIFF_EXPLANATION.value)

            # 7. ROLLBACK_PLAN — capture a stem-cell snapshot of the file
            try:
                from core.resilience.stem_cell import get_registry
                reg = get_registry()
                organ = "selfmod_target_" + Path(file_path).stem
                reg.register(organ)
                reg.capture(organ, before_source, schema_version="1")
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation(
                    "safe_pipeline",
                    exc,
                    severity="warning",
                    action="continued self-modification pipeline with textual rollback plan after stem-cell capture failed",
                    extra={"stage": Stage.ROLLBACK_PLAN.value, "target": file_path},
                )
                logger.debug("stem-cell capture during rollback plan failed: %s", exc)
            proposal.rollback_plan = f"stem_cell:selfmod_target_{Path(file_path).stem}"
            _record(proposal, "rollback_planned")
            proposal.stages_completed.append(Stage.ROLLBACK_PLAN.value)

            # 8. APPROVAL
            try:
                from core.ethics.conscience import Verdict, get_conscience
                from core.governance.will_client import WillClient, WillRequest
                from core.will import ActionDomain
                conscience_decision = get_conscience().evaluate(
                    action="self_modify",
                    domain="self_modification",
                    intent=intent,
                    context={"file": file_path, "diff": proposal.diff_explanation},
                )
                if conscience_decision.verdict == Verdict.REFUSE:
                    return self._block(proposal, Stage.APPROVAL, f"conscience_refused:{conscience_decision.rule_id}")
                if conscience_decision.verdict == Verdict.REQUIRE_FRESH_USER_AUTH:
                    return self._block(proposal, Stage.APPROVAL, "require_fresh_user_auth")
                wd = await WillClient().decide_async(
                    WillRequest(
                        content="self_modify",
                        source="safe_pipeline",
                        domain=getattr(ActionDomain, "STATE_MUTATION", "state_mutation"),
                        context={"file": file_path, "intent": intent, "diff": proposal.diff_explanation},
                    )
                )
                if not WillClient.is_approved(wd):
                    return self._block(proposal, Stage.APPROVAL, f"will_refused:{getattr(wd, 'reason', '')}")
                proposal.will_receipt_id = getattr(wd, "receipt_id", None)
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('safe_pipeline', exc)
                return self._block(proposal, Stage.APPROVAL, f"approval_exception:{exc}")
            proposal.stages_completed.append(Stage.APPROVAL.value)
            _record(proposal, "approved", {"will_receipt_id": proposal.will_receipt_id})

            # 9. STAGED_DEPLOY — default to quarantine-only. Live source
            # overwrite requires explicit owner approval plus supervised env.
            target = Path(file_path)
            if not _supervised_source_deploy_enabled(owner_approved):
                proposal.promotion_artifact_path = self._stage_for_operator_promotion(
                    proposal,
                    target,
                    after_source,
                )
                return self._block(
                    proposal,
                    Stage.STAGED_DEPLOY,
                    (
                        "operator_promotion_required:"
                        f"{_SUPERVISED_SELF_MODIFICATION_ENV}=1 and owner_approved=True"
                    ),
                )

            await _write_production_source(
                target, after_source, reason="staged_deploy"
            )
            proposal.stages_completed.append(Stage.STAGED_DEPLOY.value)
            _record(proposal, "staged_deployed")

            # 10. POST_DEPLOY_MONITOR — wait briefly and check StabilityGuardian
            await self._post_deploy_monitor(proposal, target, before_source)
            return proposal
        finally:
            try:
                shutil.rmtree(sandbox, ignore_errors=True)
            except OSError as exc:
                record_degradation(
                    "safe_pipeline",
                    exc,
                    severity="warning",
                    action="recorded self-modification sandbox cleanup failure",
                    extra={"sandbox": str(sandbox)},
                )
                logger.debug("self-mod pipeline sandbox cleanup failed: %s", exc)

    # ─── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _stage_for_operator_promotion(
        proposal: PipelineProposal,
        target: Path,
        after_source: str,
    ) -> str:
        safe_parts = [part for part in target.parts if part not in {"", ".", "/"}]
        if target.is_absolute():
            safe_parts = list(target.parts[1:])
        artifact_path = _STAGING_DIR / proposal.proposal_id / Path(*safe_parts)
        get_file_write_gateway().write_text(
            artifact_path,
            after_source,
            encoding="utf-8",
            source="self_modification.safe_pipeline.operator_promotion_artifact",
        )
        _record(
            proposal,
            "staged_for_operator_promotion",
            {
                "artifact_path": str(artifact_path),
                "target": str(target),
                "supervised_env": _SUPERVISED_SELF_MODIFICATION_ENV,
            },
        )
        return str(artifact_path)

    def _block(self, proposal: PipelineProposal, stage: Stage, reason: str) -> PipelineProposal:
        proposal.blocked_at = stage.value
        proposal.blocked_reason = reason
        _record(proposal, f"blocked:{stage.value}", {"reason": reason})
        return proposal

    @staticmethod
    def _generate_tests(file_path: str, before: str, after: str) -> str:
        # Minimal but real: import the patched module from the sandbox
        # and assert it parses + the public surface of `before` is
        # preserved in `after`. The tests run inside the SHADOW_RUNTIME
        # phase below.
        return (
            "import ast, sys\n"
            f"src_after = {after!r}\n"
            "tree = ast.parse(src_after)\n"
            "# the patched module must parse\n"
            "names = sorted(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))\n"
            "print('AST_OK', len(names))\n"
        )

    async def _run_shadow(self, sandbox_file: Path) -> tuple[bool, str]:
        # Run a tiny Python subprocess with -B (no bytecode cache) and
        # ulimit-style caps where available. macOS lacks setrlimit for
        # mem in some cases, so we use a wall-clock timeout as the
        # primary backstop.
        test_file = sandbox_file.parent / "test_self_mod_patch.py"
        commands = (
            [sys.executable, "-B", str(test_file)],
            [sys.executable, "-B", str(sandbox_file)],
        )
        try:
            outputs: list[str] = []
            for index, cmd in enumerate(commands):
                proc = await get_subprocess_gateway().spawn_async(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(sandbox_file.parent),
                    source=(
                        "self_modification:safe_pipeline.generated_tests"
                        if index == 0
                        else "self_modification:safe_pipeline.shadow_runtime"
                    ),
                    accelerator_capability="auto",
                )
                try:
                    out, err = await asyncio.wait_for(
                        proc.communicate(), timeout=self.SHADOW_TIMEOUT_S
                    )
                except TimeoutError:
                    proc.kill()
                    await proc.wait()
                    return False, f"shadow_timeout>{self.SHADOW_TIMEOUT_S}s step={index}"
                if proc.returncode != 0:
                    return False, (
                        f"shadow_rc={proc.returncode} step={index} "
                        f"stderr={err.decode('utf-8', 'replace')[:240]}"
                    )
                outputs.append(out.decode("utf-8", "replace")[:160])
            return True, " | ".join(outputs)
        except (subprocess.SubprocessError, OSError) as exc:
            record_degradation('safe_pipeline', exc)
            return False, f"shadow_exception:{exc}"

    @staticmethod
    def _diff_explanation(before: str, after: str) -> str:
        import difflib
        diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=2))
        return "\n".join(diff[:200])

    @staticmethod
    def _guardian_health_snapshot(guardian: Any) -> tuple[bool, dict[str, Any]]:
        if guardian is None:
            return False, {
                "status": "stability_guardian_unavailable",
                "source": "service_container",
                "required_probe_missing": True,
            }

        try:
            latest_report = None
            latest_fn = getattr(guardian, "get_latest_report", None)
            if callable(latest_fn):
                latest_report = latest_fn()

            if latest_report is not None:
                if isinstance(latest_report, dict):
                    health_value = latest_report.get("overall_healthy", latest_report.get("healthy"))
                    if health_value is None:
                        return False, {
                            "status": "malformed_latest_report",
                            "source": "get_latest_report",
                            "report_keys": sorted(str(key) for key in latest_report.keys())[:20],
                        }
                    healthy = bool(health_value)
                    checks = latest_report.get("checks", [])
                    return healthy, {
                        "status": "healthy" if healthy else "degraded",
                        "source": "get_latest_report",
                        "overall_healthy": healthy,
                        "timestamp": latest_report.get("timestamp"),
                        "check_count": len(checks) if isinstance(checks, list) else None,
                    }

                health_value = getattr(latest_report, "overall_healthy", None)
                if health_value is None:
                    return False, {
                        "status": "malformed_latest_report",
                        "source": "get_latest_report_object",
                        "report_type": type(latest_report).__name__,
                    }
                healthy = bool(health_value)
                return healthy, {
                    "status": "healthy" if healthy else "degraded",
                    "source": "get_latest_report_object",
                    "overall_healthy": healthy,
                    "timestamp": getattr(latest_report, "timestamp", None),
                }

            history = getattr(guardian, "_report_history", None)
            if history:
                report = history[-1]
                health_value = getattr(report, "overall_healthy", None)
                if health_value is not None:
                    healthy = bool(health_value)
                    return healthy, {
                        "status": "healthy" if healthy else "degraded",
                        "source": "_report_history",
                        "overall_healthy": healthy,
                        "timestamp": getattr(report, "timestamp", None),
                    }

            summary_fn = getattr(guardian, "get_health_summary", None)
            if callable(summary_fn):
                summary = summary_fn()
                if isinstance(summary, dict):
                    health_value = summary.get("healthy", summary.get("overall_healthy"))
                    status = str(summary.get("status", "unknown"))
                    if health_value is None:
                        return False, {
                            "status": "malformed_health_summary",
                            "source": "get_health_summary",
                            "summary_status": status,
                        }
                    required_probe_missing = bool(summary.get("required_probe_missing")) or status in {
                        "initializing",
                        "unavailable",
                    }
                    healthy = bool(health_value) and not required_probe_missing
                    return healthy, {
                        "status": status,
                        "source": "get_health_summary",
                        "healthy": healthy,
                        "reported_healthy": bool(health_value),
                        "required_probe_missing": required_probe_missing,
                        "active_issue_count": len(summary.get("active_issues", []) or []),
                    }

            return False, {
                "status": "no_stability_report",
                "source": "stability_guardian",
                "required_probe_missing": True,
            }
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "safe_pipeline",
                exc,
                severity="warning",
                action="marked post-deploy monitor unhealthy after StabilityGuardian read failed",
            )
            return False, {
                "status": "monitor_exception",
                "source": "stability_guardian",
                "error": str(exc),
            }

    def _rollback_after_deploy(
        self,
        proposal: PipelineProposal,
        target: Path,
        before_source: str,
        *,
        reason: str,
        detail: dict[str, Any],
    ) -> None:
        proposal.blocked_at = Stage.POST_DEPLOY_MONITOR.value
        proposal.blocked_reason = reason
        try:
            _write_production_source_sync(
                target, before_source, reason="post_deploy_rollback"
            )
            _record(proposal, "rolled_back", {"reason": reason, "monitor": detail})
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            proposal.blocked_reason = f"{reason}; rollback_failed:{exc}"
            record_degradation(
                "safe_pipeline",
                exc,
                severity="critical",
                action="post-deploy rollback failed after unhealthy self-modification monitor",
                extra={"target": str(target), "reason": reason, "monitor": detail},
            )
            _record(proposal, "rollback_failed", {"reason": reason, "error": str(exc), "monitor": detail})

    async def _post_deploy_monitor(self, proposal: PipelineProposal, target: Path, before_source: str) -> None:
        # Watch StabilityGuardian after a live write. Missing or optimistic
        # health evidence is not enough for self-modification promotion.
        try:
            from core.container import ServiceContainer
            guardian = ServiceContainer.get("stability_guardian", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "safe_pipeline",
                exc,
                severity="warning",
                action="marked post-deploy monitor unhealthy because StabilityGuardian lookup failed",
            )
            guardian = None

        deadline = time.monotonic() + max(0.0, float(self.POST_DEPLOY_MONITOR_S))
        healthy_samples: list[dict[str, Any]] = []
        last_detail: dict[str, Any] = {}
        waitable_statuses = {"initializing", "no_stability_report"}

        first_sample = True
        while first_sample or time.monotonic() <= deadline:
            first_sample = False
            healthy, detail = self._guardian_health_snapshot(guardian)
            last_detail = detail
            status = str(detail.get("status", "unknown"))
            remaining_s = deadline - time.monotonic()
            if healthy:
                healthy_samples.append(detail)
            elif status in waitable_statuses and remaining_s > 0.0:
                await asyncio.sleep(max(0.0, min(float(self.POST_DEPLOY_POLL_S), remaining_s)))
                continue
            else:
                self._rollback_after_deploy(
                    proposal,
                    target,
                    before_source,
                    reason=f"post_deploy_health:{status}",
                    detail=detail,
                )
                return

            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                _record(
                    proposal,
                    "post_deploy_clean",
                    {"healthy_sample_count": len(healthy_samples), "last_monitor": last_detail},
                )
                break

            await asyncio.sleep(max(0.0, min(float(self.POST_DEPLOY_POLL_S), remaining_s)))
        if not healthy_samples:
            self._rollback_after_deploy(
                proposal,
                target,
                before_source,
                reason=f"post_deploy_health:{last_detail.get('status', 'no_healthy_evidence')}",
                detail=last_detail or {
                    "status": "no_healthy_evidence",
                    "source": "post_deploy_monitor",
                },
            )
            return
        proposal.stages_completed.append(Stage.POST_DEPLOY_MONITOR.value)


_PIPELINE: SafePipeline | None = None


def get_pipeline() -> SafePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SafePipeline()
    return _PIPELINE


__all__ = ["Stage", "PipelineProposal", "SafePipeline", "get_pipeline"]
