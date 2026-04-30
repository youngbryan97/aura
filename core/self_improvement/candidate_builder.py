"""core/self_improvement/candidate_builder.py — Code generation from spec.

Constructs a replacement implementation from a ModuleSpec. Uses an injectable
CodeGenerator callable so tests can use deterministic generators while
production routes through core/llm/.

This is the paper's "agent reimplementation" step.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Protocol

from core.self_improvement.interface_contract import (
    CandidateModule,
    ModuleSpec,
)
from core.self_improvement.blinded_workspace import BlindedWorkspace

logger = logging.getLogger("Aura.CandidateBuilder")


class CodeGenerator(Protocol):
    """Protocol for code generators — injectable for testing."""

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate source code from a prompt and context."""
        ...


class PromptBuilder:
    """Builds structured prompts from ModuleSpec for code generation."""

    def build(self, spec: ModuleSpec) -> str:
        """Build a code generation prompt from a ModuleSpec."""
        sections = []
        sections.append("# Module Reimplementation Task")
        sections.append("")
        sections.append(f"## Module: {spec.module_path}")
        sections.append("")

        if spec.module_docstring:
            sections.append("## Module Documentation")
            sections.append(spec.module_docstring)
            sections.append("")

        # Interface contract
        sections.append("## Required Interface")
        sections.append("")
        for func in spec.interface.functions:
            keyword = "async def" if func.is_async else "def"
            params = ", ".join(func.parameters)
            ret = f" -> {func.return_annotation}" if func.return_annotation else ""
            sections.append(f"- `{keyword} {func.name}({params}){ret}`")
            if func.docstring:
                sections.append(f"  - {func.docstring[:200]}")
        sections.append("")

        for cls in spec.interface.classes:
            sections.append(f"- `class {cls.name}`")
            if cls.docstring:
                sections.append(f"  - {cls.docstring[:200]}")
            for method in cls.methods:
                keyword = "async def" if method.is_async else "def"
                params = ", ".join(method.parameters)
                sections.append(f"  - `{keyword} {method.name}({params})`")
        sections.append("")

        # Dependencies
        if spec.dependencies:
            sections.append("## Allowed Dependencies")
            for dep in spec.dependencies:
                sections.append(f"- {dep}")
            sections.append("")

        # Imports
        if spec.interface.imports:
            sections.append("## Required Imports")
            for imp in spec.interface.imports:
                sections.append(f"- `{imp}`")
            sections.append("")

        # Invariants
        if spec.invariants:
            sections.append("## Behavioral Invariants")
            for inv in spec.invariants:
                sections.append(f"- **{inv.name}**: {inv.description}")
            sections.append("")

        # __all__
        if spec.interface.all_names:
            sections.append("## Exported Names (__all__)")
            names = ", ".join(f'"{n}"' for n in sorted(spec.interface.all_names))
            sections.append(f"```python\n__all__ = [{names}]\n```")
            sections.append("")

        sections.append("## Instructions")
        sections.append("Implement the complete module from scratch.")
        sections.append("You must preserve the exact public interface.")
        sections.append("Do not hardcode test outputs.")
        sections.append("Return ONLY valid Python source code.")

        return "\n".join(sections)


class StubGenerator:
    """Deterministic code generator for testing — returns the stub as-is."""

    def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        return context.get("stub_code", "# No implementation generated\npass\n")


class CandidateBuilder:
    """Generates candidate modules from specs using a pluggable code generator."""

    def __init__(self, generator: Optional[CodeGenerator] = None):
        self.generator = generator or StubGenerator()
        self.prompt_builder = PromptBuilder()

    async def build(
        self,
        spec: ModuleSpec,
        workspace: BlindedWorkspace,
        attempt: int = 1,
    ) -> CandidateModule:
        """Generate a candidate implementation from the spec.

        Args:
            spec: The behavioral specification to implement.
            workspace: The blinded workspace (for context, not for source).
            attempt: Current attempt number.

        Returns:
            CandidateModule with generated source code.
        """
        start = time.monotonic()

        prompt = self.prompt_builder.build(spec)
        context: Dict[str, Any] = {
            "module_path": spec.module_path,
            "module_name": spec.module_name,
            "attempt": attempt,
            "stub_code": workspace.stub_path.read_text(encoding="utf-8")
            if workspace.stub_path.exists()
            else "",
        }

        source_code = self.generator.generate(prompt, context)
        elapsed = time.monotonic() - start

        candidate = CandidateModule(
            source_code=source_code,
            module_path=spec.module_path,
            generation_metadata={
                "generator": type(self.generator).__name__,
                "prompt_length": len(prompt),
                "attempt": attempt,
            },
            generation_time_s=elapsed,
            attempt_number=attempt,
        )

        logger.info(
            "Generated candidate for %s (attempt %d, %.2fs, %d chars)",
            spec.module_path, attempt, elapsed, len(source_code),
        )
        return candidate


__all__ = ["CandidateBuilder", "CodeGenerator", "PromptBuilder", "StubGenerator"]
