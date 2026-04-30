"""Post-promotion observations and automatic rollback."""
from __future__ import annotations

import ast
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.architect.behavior_fingerprint import BehaviorFingerprinter
from core.architect.config import ASAConfig
from core.architect.models import BehaviorFingerprint, PostPromotionObservation, PromotionStatus, RefactorPlan, RollbackPacket
from core.architect.rollback_manager import RollbackManager
from core.runtime.atomic_writer import atomic_write_text


class PostPromotionMonitor:
    """Track promoted runs and rollback on delayed regression."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.root = self.config.artifacts / "observations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.fingerprinter = BehaviorFingerprinter(self.config)
        self.rollback_manager = RollbackManager(self.config)

    def arm(self, plan: RefactorPlan, rollback_packet: RollbackPacket, baseline: BehaviorFingerprint | None = None) -> PostPromotionObservation:
        metrics = {
            "armed": True,
            "plan_id": plan.id,
            "tier": plan.risk_tier.name,
            "changed_files": list(plan.changed_files),
            "observation_window": self.config.observation_window,
            "baseline": asdict(baseline) if baseline is not None else "not_available",
        }
        observation = PostPromotionObservation(
            run_id=rollback_packet.run_id,
            timestamp=time.time(),
            status=PromotionStatus.MONITORING,
            metrics=metrics,
        )
        self._write(observation)
        return observation

    def check_once(self, run_id: str, *, force_regression: bool = False) -> PostPromotionObservation:
        packet = self.rollback_manager.load_packet(run_id)
        failures = self._compile_changed(packet.changed_files)
        regression = force_regression or bool(failures)
        rollback_triggered = False
        status = PromotionStatus.MONITORING
        reason = "lightweight checks passed"
        if regression:
            self.rollback_manager.restore(packet)
            rollback_triggered = True
            status = PromotionStatus.ROLLED_BACK
            reason = "delayed regression detected; rollback restored original files"
        observation = PostPromotionObservation(
            run_id=run_id,
            timestamp=time.time(),
            status=status,
            metrics={"compile_failures": failures},
            regression_detected=regression,
            rollback_triggered=rollback_triggered,
            reason=reason,
        )
        self._write(observation)
        return observation

    def latest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            records.append(payload)
        return records[-20:]

    def _compile_changed(self, changed_files: tuple[str, ...]) -> list[str]:
        failures: list[str] = []
        for rel in changed_files:
            path = self.config.repo_root / rel
            if not path.exists() or not rel.endswith(".py"):
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError, OSError) as exc:
                failures.append(f"{rel}:{exc}")
        return failures

    def _write(self, observation: PostPromotionObservation) -> None:
        path = self.root / f"{observation.run_id}-{int(observation.timestamp * 1000)}.json"
        atomic_write_text(path, json.dumps(asdict(observation), indent=2, sort_keys=True, default=str))
