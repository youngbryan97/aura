"""Risk-tier classifier for architecture mutations."""
from __future__ import annotations

from fnmatch import fnmatch

from core.architect.config import ASAConfig
from core.architect.models import MutationTier, SemanticSurface


class MutationClassifier:
    """Assign the maximum risk tier across paths and semantic surfaces."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()

    def classify(
        self,
        paths: list[str] | tuple[str, ...],
        *,
        surfaces: list[SemanticSurface] | tuple[SemanticSurface, ...] = (),
        symbols: list[str] | tuple[str, ...] = (),
        dynamic_dependencies: list[str] | tuple[str, ...] = (),
    ) -> MutationTier:
        tier = MutationTier.T0_SYNTAX_STYLE
        for path in paths:
            tier = max(tier, self.classify_path(path))
        for dep in dynamic_dependencies:
            tier = max(tier, self.classify_path(dep))
        for surface in surfaces:
            tier = max(tier, self.classify_surface(surface))
        for symbol in symbols:
            tier = max(tier, self.classify_symbol(symbol))
        return tier

    def classify_path(self, path: str) -> MutationTier:
        rel = _clean_rel(path.replace("\\", "/"))
        if self.config.is_sealed(rel):
            return MutationTier.T5_SEALED
        if self.config.is_protected(rel):
            return MutationTier.T4_GOVERNANCE_SENSITIVE
        if rel.startswith(("docs/", "tests/", "scratch/")):
            return MutationTier.T0_SYNTAX_STYLE
        if rel.startswith(("core/", "skills/", "scripts/", "tools/")):
            return MutationTier.T1_CLEANUP
        if rel.endswith(".py"):
            return MutationTier.T1_CLEANUP
        return MutationTier.T2_REFACTOR

    def classify_surface(self, surface: SemanticSurface) -> MutationTier:
        if surface in {
            SemanticSurface.AUTHORITY_GOVERNANCE,
            SemanticSurface.MEMORY_WRITE_READ,
            SemanticSurface.STATE_MUTATION,
            SemanticSurface.BOOT_RUNTIME_KERNEL,
            SemanticSurface.LLM_MODEL_ROUTING,
            SemanticSurface.IDENTITY_PERSONA,
            SemanticSurface.SELF_MODIFICATION,
        }:
            return MutationTier.T4_GOVERNANCE_SENSITIVE
        if surface is SemanticSurface.PROOF_TEST_EVALUATION:
            return MutationTier.T5_SEALED
        if surface is SemanticSurface.CONSCIOUSNESS_SUBSTRATE:
            return MutationTier.T3_BEHAVIORAL_IMPROVEMENT
        if surface in {SemanticSurface.CAPABILITY_TOOL_EXECUTION, SemanticSurface.UI_API, SemanticSurface.TRAINING_FINETUNE}:
            return MutationTier.T2_REFACTOR
        return MutationTier.T1_CLEANUP

    def classify_symbol(self, symbol: str) -> MutationTier:
        lowered = symbol.lower()
        sealed_tokens = (
            "proof",
            "rollback",
            "mutationclassifier",
            "architect",
            "constitution",
            "unifiedwill",
        )
        governance_tokens = (
            "authority",
            "capability",
            "memorywrite",
            "stategateway",
            "modelrouter",
            "identity",
            "heartstone",
        )
        if any(token in lowered for token in sealed_tokens):
            return MutationTier.T5_SEALED
        if any(token in lowered for token in governance_tokens):
            return MutationTier.T4_GOVERNANCE_SENSITIVE
        return MutationTier.T1_CLEANUP

    def touches_sealed(self, paths: tuple[str, ...] | list[str]) -> bool:
        return any(self.classify_path(path) is MutationTier.T5_SEALED for path in paths)

    def explain_patterns(self, path: str) -> tuple[str, ...]:
        rel = _clean_rel(path.replace("\\", "/"))
        matches = [pattern for pattern in self.config.sealed_paths if fnmatch(rel, pattern)]
        matches.extend(pattern for pattern in self.config.protected_paths if fnmatch(rel, pattern))
        return tuple(matches)


def _clean_rel(path: str) -> str:
    if path.startswith("./"):
        return path[2:]
    return path
