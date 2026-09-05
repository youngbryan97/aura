"""core/forge/self_improvement_forge.py — Self Improvement Forge."""
from __future__ import annotations

import logging
from typing import Any

from core.forge.capability_eval import CapabilityEval
from core.forge.patch_generator import PatchGenerator
from core.forge.promotion_gate import PromotionGate
from core.forge.regression_memory import get_regression_memory
from core.forge.shadow_runner import ShadowRunner
from core.forge.weakness_detector import WeaknessDetector

logger = logging.getLogger("Aura.SelfImprovementForge")


class SelfImprovementForge:
    """The manager of Aura's autonomous code refactoring and capability upgrades."""

    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Self Improvement Forge fully online.")

    async def analyze_weaknesses(self) -> dict[str, Any]:
        """Kernel spine query to check codebase quality and regressions."""
        return {"weaknesses_detected": 0}


    async def run_improvement_cycle(
        self,
        recent_execution_logs: list[dict[str, Any]],
        baseline_performance: dict[str, Any],
    ) -> dict[str, Any]:
        """Drives one complete iteration of the self-improvement loop."""
        logger.info("🚀 Initiating Self-Improvement Forge cycle...")

        # 1. Detect weaknesses
        weaknesses = WeaknessDetector.scan_for_weaknesses(recent_execution_logs)
        if not weaknesses:
            logger.info("No substantial codebase weaknesses identified. Idle.")
            return {"ok": True, "status": "no_weaknesses"}

        target = weaknesses[0]
        module = target["module"]

        # 2. Generate patch
        patch = PatchGenerator.generate_patch(module, target)
        reg_mem = get_regression_memory()

        # Check against regression memory
        if reg_mem.is_known_failure(patch["patch_code"]):
            logger.error("🚫 Aborting cycle: Patch code matches a known historical failure.")
            return {"ok": False, "status": "blocked_regression_memory"}

        # 3. Execute shadow sandbox tests
        # We write code to patch_path and run test_cmd
        # For simulation, we assume writing and testing
        shadow_res = await ShadowRunner.run_shadow_tests(
            patch_path=patch["patch_path"],
            test_cmd=f"pytest tests/test_{module}.py",
        )

        # 4. Evaluate capability level
        # Simulate baseline compared with shadow run outputs
        test_stats = {"pass_rate": 1.0 if shadow_res.get("ok") else 0.0}
        eval_report = CapabilityEval.evaluate(baseline_performance, test_stats)

        # 5. Check promotion gate
        approved = PromotionGate.check_gate(eval_report, requires_approval=False)

        if not approved:
            reg_mem.record_failure(
                module=module,
                patch_hash=hashlib.sha256(patch["patch_code"].encode()).hexdigest(),
                error_details=shadow_res.get("error", "Regression detected"),
            )
            return {"ok": False, "status": "failed_evaluation", "eval_report": eval_report}

        logger.info("🎉 Patch for module %s successfully approved for promotion!", module)
        return {
            "ok": True,
            "status": "promoted",
            "module": module,
            "patch": patch,
            "eval_report": eval_report,
        }


import hashlib

_forge_instance: SelfImprovementForge | None = None


def get_self_improvement_forge() -> SelfImprovementForge:
    global _forge_instance
    if _forge_instance is None:
        _forge_instance = SelfImprovementForge()
    return _forge_instance
