"""core/self_improvement/blinded_workspace.py — Isolated reconstruction environment.

Creates a temp directory containing only the module's interface contract,
docstrings, spec, and test files. The original implementation is explicitly
blocked. All file-access paths are recorded for anti-cheating audit.

This is the paper's "blinding" step: the agent sees the spec but never
the original code or original results.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway
from core.self_improvement.interface_contract import (
    ClassSignature,
    FunctionSignature,
    ModuleSpec,
)

logger = logging.getLogger("Aura.BlindedWorkspace")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class BlindedWorkspace:
    """An isolated directory for clean-room reconstruction."""

    workspace_dir: Path
    spec: ModuleSpec
    forbidden_paths: set[str] = field(default_factory=set)
    access_log: list[str] = field(default_factory=list)
    _created: bool = False

    @property
    def interface_path(self) -> Path:
        """Path to the interface stub file."""
        return self.workspace_dir / self.spec.module_path

    def __getattr__(self, name: str):
        if name == "stub_path":
            return self.interface_path
        raise AttributeError(name)

    @property
    def candidate_path(self) -> Path:
        """Path where the candidate implementation should be written."""
        return self.workspace_dir / self.spec.module_path

    @property
    def test_dir(self) -> Path:
        return self.workspace_dir / "tests"

    @property
    def audit_manifest_path(self) -> Path:
        return self.workspace_dir / "AUDIT_MANIFEST.json"

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

    def __init__(self, project_root: str | None = None):
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
        # One scope around the whole workspace, because building it is one
        # act of internal maintenance and every step of it writes.
        #
        # Without it the Will refused the first write and the Reimplementation
        # Lab failed before it had a stub to hand anybody — 103 refusals in
        # the live log, each one a self-improvement attempt that could never
        # start. The workspace is a temporary directory of her own, holding
        # her own stubs; nothing here is a person's file.
        with local_internal_governed_scope(
            "core.self_improvement.blinded_workspace",
            domain="file_write",
            constraints={"op": "create_blinded_workspace"},
        ):
            return self._create(spec, original_module_path)

    def _create(self, spec: ModuleSpec, original_module_path: str) -> BlindedWorkspace:
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
        get_file_write_gateway().write_text(
            stub_path,
            stub_code,
            encoding="utf-8",
            source="core.self_improvement.blinded_workspace.interface_stub",
        )

        # 2. Write __init__.py files for package structure
        self._write_init_files(workspace_dir, spec.module_path)

        # 3. Copy test files
        copied_tests = self._copy_tests(workspace, spec)

        # 4. Write spec reference
        spec_ref_path = workspace_dir / "SPEC.txt"
        get_file_write_gateway().write_text(
            spec_ref_path,
            spec.summary(),
            encoding="utf-8",
            source="core.self_improvement.blinded_workspace.spec_reference",
        )

        # 5. Write clean-room audit manifest without original implementation text.
        self._write_audit_manifest(workspace, spec, original_module_path, copied_tests)

        logger.info("Created blinded workspace at %s for %s", workspace_dir, spec.module_path)
        return workspace

    def _generate_stub(self, spec: ModuleSpec) -> str:
        """Generate a Python stub with signatures and docstrings only."""
        lines: list[str] = []

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
            lines.append(f"{name}: {type_str}  # pending implementation")
            lines.append("")

        # __all__
        if spec.interface.all_names:
            names_str = ", ".join(f'"{n}"' for n in sorted(spec.interface.all_names))
            lines.append(f"__all__ = [{names_str}]")
            lines.append("")

        return "\n".join(lines)

    def _stub_function(self, func: FunctionSignature, indent: str = "") -> str:
        """Generate a stub for a single function."""
        parts: list[str] = []
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
        parts: list[str] = []
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
                get_file_write_gateway().write_text(
                    init,
                    "",
                    encoding="utf-8",
                    source="core.self_improvement.blinded_workspace.package_init",
                )

    def _copy_tests(self, workspace: BlindedWorkspace, spec: ModuleSpec) -> list[dict[str, str | int]]:
        """Copy test files into the workspace."""
        test_dir = workspace.test_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        get_file_write_gateway().write_text(
            test_dir / "__init__.py",
            "",
            encoding="utf-8",
            source="core.self_improvement.blinded_workspace.tests_init",
        )

        copied_tests: list[dict[str, str | int]] = []
        for tc in spec.test_cases:
            if tc.file_path:
                src = self.project_root / tc.file_path
                if src.exists():
                    dst = test_dir / Path(tc.file_path).name
                    try:
                        payload = src.read_bytes()
                        get_file_write_gateway().write_bytes(
                            dst,
                            payload,
                            source="core.self_improvement.blinded_workspace.copied_test",
                        )
                        copied_tests.append(
                            {
                                "source_path_hash": _sha256_text(str(src.resolve())),
                                "source_name": Path(tc.file_path).name,
                                "destination": dst.relative_to(workspace.workspace_dir).as_posix(),
                                "sha256": _sha256_bytes(payload),
                                "size_bytes": len(payload),
                            }
                        )
                    except OSError as e:
                        logger.debug("Could not copy test %s: %s", tc.file_path, e)
        return copied_tests

    def _write_audit_manifest(
        self,
        workspace: BlindedWorkspace,
        spec: ModuleSpec,
        original_module_path: str,
        copied_tests: list[dict[str, str | int]],
    ) -> None:
        interface_rel = workspace.interface_path.relative_to(workspace.workspace_dir).as_posix()
        manifest = {
            "schema": "aura.blinded_workspace.audit_manifest.v1",
            "module_path": spec.module_path,
            "original_module_path_hash": _sha256_text(
                str((self.project_root / original_module_path).resolve())
            ),
            "forbidden_path_hashes": sorted(
                _sha256_text(path) for path in workspace.forbidden_paths
            ),
            "forbidden_path_count": len(workspace.forbidden_paths),
            "generated_interface": {
                "path": interface_rel,
                "sha256": _sha256_bytes(workspace.interface_path.read_bytes()),
            },
            "copied_tests": copied_tests,
            "claim_supported": "clean_room_workspace_inputs_are_hash_manifested",
            "claim_not_supported": [
                "candidate_correctness",
                "source_equivalence",
                "original_implementation_access",
            ],
        }
        get_file_write_gateway().write_text(
            workspace.audit_manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
            source="core.self_improvement.blinded_workspace.audit_manifest",
        )


__all__ = ["BlindedWorkspace", "BlindedWorkspaceFactory"]
