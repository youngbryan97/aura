"""Static/runtime checks for Will/Authority governance closure."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


CONSEQUENTIAL_PATTERNS = {
    "subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.patch",
    "requests.delete",
    "urllib.request.urlopen",
    "Path.write_text",
    "Path.write_bytes",
}

APPROVED_ADAPTER_HINTS = {
    "authority_gateway",
    "capability_engine.py",
    "environment/command.py",
    "environment/environment_kernel.py",
    "runtime/atomic_writer.py",
    "self_modification/safe_modification.py",
    "self_modification/mutation_safety.py",
    "reproducibility/proof_substrate.py",
}


@dataclass
class ClosureFinding:
    path: str
    line: int
    pattern: str
    severity: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClosureAuditReport:
    scanned_files: int
    findings: list[ClosureFinding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "scanned_files": self.scanned_files,
            "passed": self.passed,
            "findings": [f.to_dict() for f in self.findings],
        }


class GovernanceClosureAuditor:
    """AST-based sweep for consequential paths outside approved adapters."""

    def __init__(self, *, approved_hints: Iterable[str] = APPROVED_ADAPTER_HINTS):
        self.approved_hints = tuple(approved_hints)

    def audit_tree(self, root: str | Path) -> ClosureAuditReport:
        root_path = Path(root)
        report = ClosureAuditReport(scanned_files=0)
        for path in root_path.rglob("*.py"):
            if any(part in {".venv", "__pycache__", ".git"} for part in path.parts):
                continue
            report.scanned_files += 1
            report.findings.extend(self.audit_file(path, root=root_path))
        return report

    def audit_file(self, path: str | Path, *, root: str | Path | None = None) -> list[ClosureFinding]:
        file_path = Path(path)
        rel = str(file_path.relative_to(root)) if root and file_path.is_relative_to(Path(root)) else str(file_path)
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return []
        approved = any(hint in rel for hint in self.approved_hints)
        findings: list[ClosureFinding] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._call_name(node.func)
            if not name:
                continue
            matched = self._match(name)
            if not matched:
                continue
            severity = "warning" if approved else "error"
            findings.append(
                ClosureFinding(
                    path=rel,
                    line=getattr(node, "lineno", 0),
                    pattern=matched,
                    severity=severity,
                    reason=(
                        "approved adapter must maintain explicit AuthorityGateway receipt"
                        if approved
                        else "consequential path outside approved adapter"
                    ),
                )
            )
        return findings

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = GovernanceClosureAuditor._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    @staticmethod
    def _match(name: str) -> str:
        for pattern in CONSEQUENTIAL_PATTERNS:
            if name == pattern or name.endswith("." + pattern.split(".", 1)[-1]) and pattern.startswith("Path."):
                return pattern
        return ""
