"""core/forge — Self-Improvement Forge package."""
from __future__ import annotations

from core.forge.weakness_detector import WeaknessDetector
from core.forge.patch_generator import PatchGenerator
from core.forge.shadow_runner import ShadowRunner
from core.forge.capability_eval import CapabilityEval
from core.forge.promotion_gate import PromotionGate
from core.forge.regression_memory import RegressionMemory, get_regression_memory
from core.forge.self_improvement_forge import SelfImprovementForge, get_self_improvement_forge

__all__ = [
    "WeaknessDetector",
    "PatchGenerator",
    "ShadowRunner",
    "CapabilityEval",
    "PromotionGate",
    "RegressionMemory",
    "get_regression_memory",
    "SelfImprovementForge",
    "get_self_improvement_forge",
]
