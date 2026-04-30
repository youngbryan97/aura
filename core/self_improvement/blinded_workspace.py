"""core/self_improvement/blinded_workspace.py — Isolated reconstruction environment.

Creates a temp directory containing only the module's interface contract,
docstrings, spec, and test files. The original implementation is explicitly
blocked. All file-access paths are recorded for anti-cheating audit.

This is the paper's "blinding" step: the agent sees the spec but never
the original code or original results.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from core.self_improvement.interface_contract import (
    ClassSignature,
    FunctionSignature,
    ModuleSpec,
)

logger = logging.getLogger("Aura.BlindedWorkspace")


@dataclass
class BlindedWorkspace:
    """An isolated directory for clean-room reconstruction."""

    workspace_dir: Path
    spec: ModuleSpec
    forbidden_paths: Set[str] = field(default_factory=set)
    access_log: List[str] = field(default_factory=list)
    _created: bool = False

    @property
    def stub_path(self) -> Path:
        """Path to the interface stub file."""
        return self.workspace_dir / self.spec.module_path

    @property
    def candidate_path(self) -> Path:
        """Path where the candidate implementation should be written."""
        return self.workspace_dir / self.spec.module_path

    @property
    def test_dir(self) -> Path:
        return self.workspace_dir / "tests"

    def record_access(self, path: str) -> None:
        """Record a file access for audit purposes."""
        self.access_log.append(path)

    def is_forbidden(self, path: str) -> bool:
        """Check if a path is forbidden (original implementation)."""
        normalized = os.path.normpath(path)
        for fp in self.forbidden_paths:
            if normalized == os.path.normpath(fp) or normalized.endswith(fp):
                return True
        return False

    def cleanup(self) -> None:
        """Remove the workspace directory."""
        if self._created and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)
            self._created = False


class BlindedWorkspaceFactory:
    """Creates blinded workspaces for clean-room reconstruction."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or ".").resolve()

    def create(self, spec: ModuleSpec, original_module_path: str) -> BlindedWorkspace:
        """Create a blinded workspace from a ModuleSpec.

        The workspace contains:
        1. Interface stub (.py with signatures + docstrings, no implementations)
        2. Test files (copied from the project)
        3. Dependency stubs (minimal)
        4. The spec as a JSON-like reference file

        The workspace does NOT contain:
        - The original implementation
        - Any file that could leak the implementation
        """
        workspace_dir = Path(tempfile.mkdtemp(prefix="aura_blind_"))
        abs_original = str((self.project_root / original_module_path).resolve())

        workspace = BlindedWorkspace(
            workspace_dir=workspace_dir,
            spec=spec,
            forbidden_paths={abs_original, original_module_path},
            _created=True,
        )

        # 1. Write interface stub
        stub_code = self._generate_stub(spec)
        stub_path = workspace_dir / spec.module_path
        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub_path.write_text(stub_code, encoding="utf-8")

        # 2. Write __init__.py files for package structure
        self._write_init_files(workspace_dir, spec.module_path)

        # 3. Copy test files
        self._copy_tests(workspace, spec)

        # 4. Write spec reference
        spec_ref_path = workspace_dir / "SPEC.txt"
        spec_ref_path.write_text(spec.summary(), encoding="utf-8")

        logger.info("Created blinded workspace at %s for %s", workspace_dir, spec.module_path)
        return workspace

    def _generate_stub(self, spec: ModuleSpec) -> str:
        """Generate a Python stub with signatures and docstrings only."""
        lines: List[str] = []

        # Module docstring
        if spec.module_docstring:
            lines.append(f'"""{spec.module_docstring}"""')
            lines.append("")

        # Imports
        for imp in spec.interface.imports:
            lines.append(imp)
        if spec.interface.imports:
            lines.append("")
            lines.append("")

        # Functions
        for func in spec.interface.functions:
            lines.append(self._stub_function(func))
            lines.append("")

        # Classes
        for cls in spec.interface.classes:
            lines.append(self._stub_class(cls))
            lines.append("")

        # Constants (as type annotations)
        for name, type_str in spec.interface.constants.items():
            lines.append(f"{name}: {type_str}  # TODO: implement")
            lines.append("")

        # __all__
        if spec.interface.all_names:
            names_str = ", ".join(f'"{n}"' for n in sorted(spec.interface.all_names))
            lines.append(f"__all__ = [{names_str}]")
            lines.append("")

        return "\n".join(lines)

    def _stub_function(self, func: FunctionSignature, indent: str = "") -> str:
        """Generate a stub for a single function."""
        parts: List[str] = []
        for dec in func.decorators:
            parts.append(f"{indent}@{dec}")
        keyword = "async def" if func.is_async else "def"
        params = ", ".join(func.parameters)
        ret = f" -> {func.return_annotation}" if func.return_annotation else ""
        parts.append(f"{indent}{keyword} {func.name}({params}){ret}:")
        if func.docstring:
            parts.append(f'{indent}    """{func.docstring}"""')
        parts.append(f"{indent}    raise NotImplementedError(\"Clean-room reimplementation required\")")
        return "\n".join(parts)

    def _stub_class(self, cls: ClassSignature) -> str:
        """Generate a stub for a single class."""
        parts: List[str] = []
        for dec in cls.decorators:
            parts.append(f"@{dec}")
        bases = f"({', '.join(cls.bases)})" if cls.bases else ""
        parts.append(f"class {cls.name}{bases}:")
        if cls.docstring:
            parts.append(f'    """{cls.docstring}"""')
        if not cls.methods:
            parts.append("    pass")
        else:
            for method in cls.methods:
                parts.append("")
                parts.append(self._stub_function(method, indent="    "))
        return "\n".join(parts)

    def _write_init_files(self, workspace_dir: Path, module_path: str) -> None:
        """Create __init__.py files for the package structure."""
        parts = Path(module_path).parts[:-1]
        current = workspace_dir
        for part in parts:
            current = current / part
            current.mkdir(parents=True, exist_ok=True)
            init = current / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")

    def _copy_tests(self, workspace: BlindedWorkspace, spec: ModuleSpec) -> None:
        """Copy test files into the workspace."""
        test_dir = workspace.test_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "__init__.py").write_text("", encoding="utf-8")

        for tc in spec.test_cases:
            if tc.file_path:
                src = self.project_root / tc.file_path
                if src.exists():
                    dst = test_dir / Path(tc.file_path).name
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        logger.debug("Could not copy test %s: %s", tc.file_path, e)


__all__ = ["BlindedWorkspace", "BlindedWorkspaceFactory"]
