"""Safety gate for ASA autonomous operations.

Enforces operational constraints that prevent runaway or unsafe autonomous
architecture changes.  Every freeze event writes a structured autopsy record
to ``{artifacts}/autopsies/`` so failed autonomy becomes future LoRA /
self-repair training data.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.architect.config import ASAConfig
from core.architect.models import (
    ArchitectureGraph,
    MutationTier,
    PromotionStatus,
    RefactorPlan,
)
from core.runtime.atomic_writer import atomic_write_text


# ---------------------------------------------------------------------------
# Freeze autopsy record — structured for future LoRA / self-repair traces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FreezeAutopsy:
    """Immutable record of why ASA froze itself."""

    autopsy_id: str
    timestamp: float
    trigger: str          # rollback | consecutive_failures | diff_budget | repeat_file | manual
    freeze_duration_s: float
    thaw_at: float
    context: dict[str, Any] = field(default_factory=dict)
    plan_snapshot: dict[str, Any] | None = None
    git_state: dict[str, Any] = field(default_factory=dict)
    recent_promotions: list[dict[str, Any]] = field(default_factory=list)
    lesson: str = ""       # human-readable one-liner for training prompt

    def to_training_example(self) -> dict[str, Any]:
        """Return a dict shaped for LoRA fine-tune ingestion."""
        return {
            "type": "asa_freeze_autopsy",
            "instruction": (
                f"ASA froze due to '{self.trigger}'. Analyse the context and "
                "explain what went wrong, what the correct next action is, "
                "and how to prevent recurrence."
            ),
            "input": json.dumps(
                {
                    "trigger": self.trigger,
                    "context": self.context,
                    "plan": self.plan_snapshot,
                    "git_state": self.git_state,
                    "recent_promotions": self.recent_promotions,
                },
                sort_keys=True,
                default=str,
            ),
            "output": self.lesson or f"Frozen for {self.freeze_duration_s}s due to {self.trigger}. Review context and resume cautiously.",
            "metadata": {
                "autopsy_id": self.autopsy_id,
                "timestamp": self.timestamp,
                "thaw_at": self.thaw_at,
            },
        }


# ---------------------------------------------------------------------------
# Freeze durations and budgets
# ---------------------------------------------------------------------------

FREEZE_DURATIONS: dict[str, float] = {
    "rollback":              7200.0,   # 2 hours
    "consecutive_failures":  3600.0,   # 1 hour
    "diff_budget_exceeded":  1800.0,   # 30 min
    "repeat_file_edit":      3600.0,   # 1 hour
}

MAX_PLAN_FILES = 5
MAX_PLAN_LOC = 200
REPEAT_FILE_WINDOW_S = 7200.0        # 2 hours
T3_OBSERVATION_WINDOW_S = 1800.0     # 30 min before next T3 allowed

# Surfaces that trigger indirect escalation for T3 helpers
ESCALATION_SURFACES = frozenset({
    "authority/governance",
    "llm/model_routing",
    "self_modification",
    "training/finetune",
    "boot/runtime/kernel",
    "identity/persona/heartstone",
})


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------

class ASASafetyGate:
    """Enforces operational safety constraints on autonomous architecture changes."""

    def __init__(self, config: ASAConfig) -> None:
        self.config = config
        self.state_path = config.artifacts / "safety_state.json"
        self.autopsy_dir = config.artifacts / "autopsies"
        self.autopsy_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    # -- state persistence ---------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return self._empty_state()
        return self._empty_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "frozen_until": None,
            "freeze_reason": "",
            "consecutive_failures": 0,
            "recent_promotions": [],
            "active_t3_observations": [],
        }

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.state_path,
            json.dumps(self.state, indent=2, sort_keys=True, default=str),
        )

    # -- freeze / thaw -------------------------------------------------------

    def is_frozen(self) -> tuple[bool, str]:
        until = self.state.get("frozen_until")
        if until is None:
            return False, ""
        if time.time() >= until:
            self.state["frozen_until"] = None
            self.state["freeze_reason"] = ""
            self._save()
            return False, ""
        remaining = int(until - time.time())
        reason = self.state.get("freeze_reason", "unknown")
        return True, f"{reason} ({remaining}s remaining)"

    def freeze(
        self,
        trigger: str,
        *,
        duration_s: float | None = None,
        context: dict[str, Any] | None = None,
        plan: RefactorPlan | None = None,
    ) -> FreezeAutopsy:
        dur = duration_s if duration_s is not None else FREEZE_DURATIONS.get(trigger, 3600.0)
        thaw_at = time.time() + dur
        self.state["frozen_until"] = thaw_at
        self.state["freeze_reason"] = trigger
        self._save()
        return self._write_autopsy(trigger, dur, thaw_at, context or {}, plan)

    # -- preflight (run before every auto cycle) -----------------------------

    def preflight(self, monitor_fn: Any | None = None) -> tuple[bool, str]:
        """Run all precondition checks.  Returns (ok, reason)."""
        # 1. Frozen?
        frozen, reason = self.is_frozen()
        if frozen:
            return False, f"frozen: {reason}"

        # 2. Run monitor on armed observations before starting new work
        if monitor_fn is not None:
            rollback_triggered = self._run_armed_monitors(monitor_fn)
            if rollback_triggered:
                return False, "monitor detected regression and triggered rollback"

        # 3. Clean git baseline
        ok, reason = self._check_git_clean()
        if not ok:
            return False, reason

        # 4. Active T3 observation limit
        ok, reason = self._check_t3_observation_limit()
        if not ok:
            return False, reason

        return True, "ok"

    # -- per-plan checks (after plan selection, before shadow) ---------------

    def check_plan(self, plan: RefactorPlan) -> tuple[bool, str]:
        """Validate a specific plan before committing to shadow run."""
        # Repeat-file check
        ok, reason = self._check_repeat_file(plan.changed_files)
        if not ok:
            self.freeze("repeat_file_edit", context={"files": list(plan.changed_files)}, plan=plan)
            return False, reason

        # Per-plan file budget
        if len(plan.changed_files) > MAX_PLAN_FILES:
            self.freeze(
                "diff_budget_exceeded",
                context={"file_count": len(plan.changed_files), "max": MAX_PLAN_FILES},
                plan=plan,
            )
            return False, f"plan touches {len(plan.changed_files)} files (max {MAX_PLAN_FILES})"

        return True, "ok"

    # -- indirect T3 escalation ----------------------------------------------

    def check_indirect_escalation(
        self,
        plan: RefactorPlan,
        graph: ArchitectureGraph,
    ) -> bool:
        """Return True if a T3 plan indirectly touches sensitive surfaces
        (training, model-routing, self-mod, governance) through helper files.

        When True, the caller should escalate to PROPOSAL_ONLY.
        """
        if plan.risk_tier < MutationTier.T3_BEHAVIORAL_IMPROVEMENT:
            return False

        changed = set(plan.changed_files)
        # Find reverse-dependents: files that IMPORT any changed file
        for edge in graph.edges:
            if edge.kind != "imports":
                continue
            if edge.target not in changed:
                continue
            # edge.source imports our changed file — check its surface
            source_surfaces = graph.semantic_surfaces.get(edge.source, ())
            for surface in source_surfaces:
                if (surface.value if hasattr(surface, "value") else str(surface)) in ESCALATION_SURFACES:
                    return True
            # Also check if source is protected/sealed
            if self.config.is_protected(edge.source) or self.config.is_sealed(edge.source):
                return True

        return False

    # -- outcome recording ---------------------------------------------------

    def record_promotion(
        self,
        run_id: str,
        plan: RefactorPlan,
    ) -> None:
        """Record a successful promotion: reset failures, track file, commit."""
        self.state["consecutive_failures"] = 0
        promo = {
            "run_id": run_id,
            "files": list(plan.changed_files),
            "timestamp": time.time(),
            "tier": plan.risk_tier.name,
            "objective": plan.objective[:200],
        }
        self.state.setdefault("recent_promotions", []).append(promo)
        self.state["recent_promotions"] = self.state["recent_promotions"][-50:]

        if plan.risk_tier >= MutationTier.T3_BEHAVIORAL_IMPROVEMENT:
            self.state.setdefault("active_t3_observations", []).append(
                {"run_id": run_id, "armed_at": time.time()}
            )
        self._save()
        self._git_commit(plan.changed_files, run_id, plan.objective)

    def record_failure(self, run_id: str, plan: RefactorPlan | None = None) -> None:
        self.state["consecutive_failures"] = self.state.get("consecutive_failures", 0) + 1
        if self.state["consecutive_failures"] >= 2:
            self.freeze(
                "consecutive_failures",
                context={"consecutive": self.state["consecutive_failures"], "last_run_id": run_id},
                plan=plan,
            )
            self.state["consecutive_failures"] = 0
        self._save()

    def record_rollback(self, run_id: str, plan: RefactorPlan | None = None) -> None:
        self.freeze(
            "rollback",
            context={"run_id": run_id},
            plan=plan,
        )
        self.state["consecutive_failures"] = 0
        self._save()

    # -- internal checks -----------------------------------------------------

    def _check_git_clean(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.config.repo_root,
                capture_output=True, text=True, timeout=10,
            )
            dirty = result.stdout.strip()
            if dirty:
                return False, f"git not clean: {dirty[:200]}"
        except (subprocess.TimeoutExpired, OSError) as exc:
            return False, f"git check failed: {exc}"
        return True, "ok"

    def _check_t3_observation_limit(self) -> tuple[bool, str]:
        now = time.time()
        active = [
            obs for obs in self.state.get("active_t3_observations", [])
            if now - obs.get("armed_at", 0) < T3_OBSERVATION_WINDOW_S
        ]
        self.state["active_t3_observations"] = active
        if active:
            return False, f"T3 observation active: {active[0].get('run_id', '?')} — waiting {int(T3_OBSERVATION_WINDOW_S - (now - active[0]['armed_at']))}s"
        return True, "ok"

    def _check_repeat_file(self, files: tuple[str, ...]) -> tuple[bool, str]:
        cutoff = time.time() - REPEAT_FILE_WINDOW_S
        recent_files: set[str] = set()
        for promo in self.state.get("recent_promotions", []):
            if promo.get("timestamp", 0) >= cutoff:
                recent_files.update(promo.get("files", []))
        overlap = set(files) & recent_files
        if overlap:
            return False, f"file edited twice in {int(REPEAT_FILE_WINDOW_S // 60)} min: {sorted(overlap)[:5]}"
        return True, "ok"

    def _run_armed_monitors(self, monitor_fn: Any) -> bool:
        """Run monitor.check_once() for each armed observation.
        Returns True if any rollback was triggered."""
        triggered = False
        for obs in list(self.state.get("active_t3_observations", [])):
            try:
                result = monitor_fn(obs["run_id"])
                if getattr(result, "rollback_triggered", False) or (
                    isinstance(result, dict) and result.get("rollback_triggered")
                ):
                    self.record_rollback(obs["run_id"])
                    triggered = True
            except (KeyError, TypeError, ValueError, OSError):
                continue  # monitor failure doesn't block; logged elsewhere
        return triggered

    # -- git commit after promotion ------------------------------------------

    def _git_commit(self, files: tuple[str, ...], run_id: str, objective: str) -> bool:
        try:
            for f in files:
                subprocess.run(
                    ["git", "add", str(self.config.repo_root / f)],
                    cwd=self.config.repo_root,
                    capture_output=True, text=True, timeout=10,
                )
            msg = f"asa(auto): {objective[:80]} [{run_id[:12]}]"
            result = subprocess.run(
                ["git", "commit", "-m", msg, "--allow-empty"],
                cwd=self.config.repo_root,
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # -- git state snapshot (for autopsy) ------------------------------------

    def _git_snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {}
        try:
            snap["head"] = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.repo_root,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            snap["status"] = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.config.repo_root,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()[:500]
            snap["diff_stat"] = subprocess.run(
                ["git", "diff", "--stat", "HEAD"],
                cwd=self.config.repo_root,
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()[:500]
        except (subprocess.TimeoutExpired, OSError, subprocess.CalledProcessError) as exc:
            snap["error"] = repr(exc)
        return snap

    # -- freeze autopsy ------------------------------------------------------

    def _write_autopsy(
        self,
        trigger: str,
        duration_s: float,
        thaw_at: float,
        context: dict[str, Any],
        plan: RefactorPlan | None,
    ) -> FreezeAutopsy:
        import hashlib

        now = time.time()
        autopsy_id = hashlib.sha256(f"{trigger}-{now}".encode()).hexdigest()[:16]

        plan_snapshot = None
        if plan is not None:
            from core.architect.refactor_planner import plan_to_dict
            plan_snapshot = plan_to_dict(plan)

        autopsy = FreezeAutopsy(
            autopsy_id=autopsy_id,
            timestamp=now,
            trigger=trigger,
            freeze_duration_s=duration_s,
            thaw_at=thaw_at,
            context=context,
            plan_snapshot=plan_snapshot,
            git_state=self._git_snapshot(),
            recent_promotions=list(self.state.get("recent_promotions", [])[-10:]),
            lesson=f"ASA self-froze: {trigger}. Context: {json.dumps(context, default=str)[:300]}",
        )

        # Persist autopsy as structured JSON
        autopsy_path = self.autopsy_dir / f"{autopsy_id}.json"
        atomic_write_text(
            autopsy_path,
            json.dumps(asdict(autopsy), indent=2, sort_keys=True, default=str),
        )

        # Also append a LoRA-shaped training example
        training_path = self.autopsy_dir / "training_examples.jsonl"
        with open(training_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(autopsy.to_training_example(), sort_keys=True, default=str) + "\n")

        return autopsy
