"""Semantic-surface classifier for Aura source files."""
from __future__ import annotations

from pathlib import Path

from core.architect.models import SemanticSurface


class SemanticClassifier:
    """Classify files into mutation-relevant architecture surfaces."""

    def classify_path(self, path: str, *, names: set[str] | None = None, effects: set[str] | None = None) -> tuple[SemanticSurface, ...]:
        rel = path.replace("\\", "/")
        tokens = " ".join(sorted((names or set()) | (effects or set()))).lower()
        haystack = f"{rel.lower()} {tokens}"
        surfaces: list[SemanticSurface] = []
        self._add_if(
            surfaces,
            SemanticSurface.AUTHORITY_GOVERNANCE,
            haystack,
            ("will", "authority", "constitution", "governance", "executive_authority", "conscience"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.CAPABILITY_TOOL_EXECUTION,
            haystack,
            ("capability", "tool", "skill", "executor", "omni_tool"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.MEMORY_WRITE_READ,
            haystack,
            ("memory", "episodic", "semantic", "vector", "rag", "scar", "memory_write"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.STATE_MUTATION,
            haystack,
            ("state", "vault", "state_repo", "state_gateway", "state_write", "aura_state"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.BOOT_RUNTIME_KERNEL,
            haystack,
            ("aura_main", "runtime", "kernel", "bootstrap", "boot", "service_manifest", "service_registration"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.CONSCIOUSNESS_SUBSTRATE,
            haystack,
            ("consciousness", "substrate", "phi", "global_workspace", "neurochemical", "qualia", "phenomen"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.LLM_MODEL_ROUTING,
            haystack,
            ("llm", "model", "mlx", "router", "inference", "token", "prompt"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.IDENTITY_PERSONA,
            haystack,
            ("identity", "persona", "heartstone", "canonical_self", "self_model", "base_persona"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.SELF_MODIFICATION,
            haystack,
            ("self_modification", "self_improvement", "mutation", "repair", "architect"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.PROOF_TEST_EVALUATION,
            haystack,
            ("test", "proof", "eval", "benchmark", "verifier", "validation", "receipt"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.UI_API,
            haystack,
            ("api", "server", "route", "ui", "interface", "websocket", "http"),
        )
        self._add_if(
            surfaces,
            SemanticSurface.TRAINING_FINETUNE,
            haystack,
            ("training", "finetune", "lora", "adapter", "dataset"),
        )
        if not surfaces:
            surfaces.append(SemanticSurface.UTILITY_PERIPHERAL)
        return tuple(dict.fromkeys(surfaces))

    @staticmethod
    def _add_if(surfaces: list[SemanticSurface], surface: SemanticSurface, haystack: str, needles: tuple[str, ...]) -> None:
        if any(needle in haystack for needle in needles):
            surfaces.append(surface)

    def module_name_for_path(self, path: str) -> str:
        rel = path.replace("\\", "/")
        return Path(rel).with_suffix("").as_posix().replace("/", ".")
