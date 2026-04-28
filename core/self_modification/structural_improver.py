"""Autonomous structural code improvement.

This is the deterministic half of Aura's source-level self-improvement:
it scans the actual repository for concrete flaws, applies repairs for known
safe patterns, validates the changed files, and rolls back failed repairs.
Larger semantic changes are still routed through the sandboxed
SelfModificationEngine; this module handles the classes of defects that should
not need a model to fix.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.StructuralImprover")


@dataclass(frozen=True)
class StructuralIssue:
    file_path: str
    line: int
    kind: str
    message: str
    severity: float = 0.5
    repairable: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRepairResult:
    issue: StructuralIssue
    changed: bool
    success: bool
    message: str
    validation: Dict[str, Any] = field(default_factory=dict)


class StructuralImprover:
    """Find and repair deterministic source-level defects in Aura."""

    EXCLUDED_PARTS = {
        ".git",
        ".venv",
        "__pycache__",
        "site-packages",
        "node_modules",
        "models",
        "training/adapters",
        "training/fused-model",
    }

    def __init__(
        self,
        root: Path | str,
        *,
        ledger_path: Optional[Path] = None,
        validation_timeout_s: int = 60,
    ):
        self.root = Path(root).resolve()
        self.validation_timeout_s = max(10, int(validation_timeout_s))
        self.ledger_path = ledger_path or (self.root / "data" / "structural_improvements.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def scan(self, *, max_files: int = 2000) -> List[StructuralIssue]:
        issues: List[StructuralIssue] = []
        scanned = 0
        for path in self._iter_python_files():
            if scanned >= max_files:
                break
            scanned += 1
            try:
                issues.extend(self._scan_file(path))
            except Exception as exc:
                record_degradation("structural_improver", exc)
                logger.debug("Structural scan skipped %s: %s", path, exc)
        return sorted(issues, key=lambda i: (-i.severity, i.file_path, i.line))

    def find_and_fix(
        self,
        *,
        max_repairs: int = 3,
        kinds: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        allowed = set(kinds or [])
        issues = [
            issue for issue in self.scan()
            if issue.repairable and (not allowed or issue.kind in allowed)
        ]
        results: List[StructuralRepairResult] = []
        for issue in issues[:max(0, int(max_repairs))]:
            results.append(self.apply_known_repair(issue))

        summary = {
            "ok": all(result.success for result in results),
            "issues_found": len(issues),
            "repairs_attempted": len(results),
            "repairs_successful": sum(1 for result in results if result.success),
            "results": [asdict(result) for result in results],
        }
        self._append_ledger(summary)
        return summary

    def apply_known_repair(self, issue: StructuralIssue) -> StructuralRepairResult:
        path = (self.root / issue.file_path).resolve()
        if not self._is_within_root(path):
            return StructuralRepairResult(issue, False, False, "path outside repository")
        if not path.exists():
            return StructuralRepairResult(issue, False, False, "file missing")

        original = path.read_text(encoding="utf-8")
        repaired = original
        try:
            if issue.kind == "missing_import_os":
                repaired = self._ensure_import(original, "os")
            elif issue.kind == "unsafe_async_gateway_mkdir":
                repaired = self._repair_gateway_mkdir(original)
                repaired = self._ensure_from_import(repaired, "pathlib", "Path")
            elif issue.kind == "unsafe_async_gateway_delete_tree":
                repaired = self._repair_gateway_delete_tree(original)
                repaired = self._ensure_import(repaired, "shutil")
            elif issue.kind == "turn_taking_state_gateway_mutation":
                repaired = self._repair_turn_taking_state_mutations(original)
            else:
                return StructuralRepairResult(issue, False, False, f"no deterministic repair for {issue.kind}")

            if repaired == original:
                return StructuralRepairResult(issue, False, False, "repair pattern did not change file")

            atomic_write_text(path, repaired, encoding="utf-8")
            validation = self._validate_files([path])
            if not validation.get("ok", False):
                atomic_write_text(path, original, encoding="utf-8")
                return StructuralRepairResult(issue, True, False, "validation failed; rolled back", validation)

            return StructuralRepairResult(issue, True, True, "repair applied", validation)
        except Exception as exc:
            record_degradation("structural_improver", exc)
            try:
                atomic_write_text(path, original, encoding="utf-8")
            except Exception:
                pass
            return StructuralRepairResult(issue, repaired != original, False, f"{type(exc).__name__}: {exc}")

    def _iter_python_files(self) -> Iterable[Path]:
        for path in self.root.rglob("*.py"):
            rel_parts = path.relative_to(self.root).parts
            rel_string = "/".join(rel_parts)
            if any(part in self.EXCLUDED_PARTS for part in rel_parts):
                continue
            if any(excluded in rel_string for excluded in self.EXCLUDED_PARTS if "/" in excluded):
                continue
            yield path

    def _scan_file(self, path: Path) -> List[StructuralIssue]:
        rel = str(path.relative_to(self.root))
        text = path.read_text(encoding="utf-8")
        issues: List[StructuralIssue] = []

        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            issues.append(StructuralIssue(
                file_path=rel,
                line=exc.lineno or 1,
                kind="syntax_error",
                message=str(exc),
                severity=1.0,
                repairable=False,
            ))

        if "os.environ" in text and not self._has_import(text, "os"):
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, "os.environ"),
                kind="missing_import_os",
                message="uses os.environ without importing os",
                severity=0.9,
                repairable=True,
            ))

        if "get_storage_gateway().create_dir(" in text:
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, "get_storage_gateway().create_dir("),
                kind="unsafe_async_gateway_mkdir",
                message="runtime path creation still depends on generated async storage gateway helper",
                severity=0.8,
                repairable=True,
            ))

        if "get_storage_gateway().delete_tree(" in text:
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, "get_storage_gateway().delete_tree("),
                kind="unsafe_async_gateway_delete_tree",
                message="runtime deletion still depends on generated async storage gateway helper",
                severity=0.8,
                repairable=True,
            ))

        if rel == "core/social/turn_taking.py" and "get_state_gateway().mutate(StateMutationRequest(" in text:
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, "get_state_gateway().mutate(StateMutationRequest("),
                kind="turn_taking_state_gateway_mutation",
                message="turn-taking state mutators still call generated state gateway helpers",
                severity=0.8,
                repairable=True,
            ))

        if 'raise NotImplementedError("Aura Pass' in text:
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, 'raise NotImplementedError("Aura Pass'),
                kind="unimplemented_stub",
                message="audited pass marker still raises NotImplementedError",
                severity=0.7,
                repairable=False,
            ))

        return issues

    @staticmethod
    def _has_import(text: str, module: str) -> bool:
        return bool(re.search(rf"(?m)^\s*import\s+.*\b{re.escape(module)}\b", text)) or bool(
            re.search(rf"(?m)^\s*from\s+{re.escape(module)}\s+import\b", text)
        )

    @staticmethod
    def _first_line_containing(text: str, needle: str) -> int:
        for idx, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                return idx
        return 1

    def _ensure_import(self, text: str, module: str) -> str:
        if self._has_import(text, module):
            return text
        return self._insert_import_line(text, f"import {module}")

    def _ensure_from_import(self, text: str, module: str, symbol: str) -> str:
        pattern = rf"(?m)^from\s+{re.escape(module)}\s+import\s+.*\b{re.escape(symbol)}\b"
        if re.search(pattern, text):
            return text
        return self._insert_import_line(text, f"from {module} import {symbol}")

    @staticmethod
    def _insert_import_line(text: str, import_line: str) -> str:
        lines = text.splitlines()
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if insert_at < len(lines) and re.match(r"^[rubfRUBF]*[\"']", lines[insert_at].strip()):
            quote = lines[insert_at].strip()[:3]
            insert_at += 1
            while insert_at < len(lines):
                if quote in lines[insert_at]:
                    insert_at += 1
                    break
                insert_at += 1
        while insert_at < len(lines) and (
            lines[insert_at].startswith("from __future__ import")
            or lines[insert_at].strip() == ""
        ):
            insert_at += 1
        while insert_at < len(lines) and (
            lines[insert_at].startswith("import ")
            or lines[insert_at].startswith("from ")
        ):
            insert_at += 1
        lines.insert(insert_at, import_line)
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    @staticmethod
    def _repair_gateway_mkdir(text: str) -> str:
        pattern = re.compile(
            r"(?P<indent>^[ \t]*)get_task_tracker\(\)\.create_task\("
            r"get_storage_gateway\(\)\.create_dir\((?P<target>[^,\n]+),\s*cause=(?P<cause>[^)]*)\)\)\s*$",
            re.MULTILINE,
        )
        return pattern.sub(
            lambda m: f"{m.group('indent')}Path({m.group('target').strip()}).mkdir(parents=True, exist_ok=True)",
            text,
        )

    @staticmethod
    def _repair_gateway_delete_tree(text: str) -> str:
        pattern = re.compile(
            r"(?P<indent>^[ \t]*)get_task_tracker\(\)\.create_task\("
            r"get_storage_gateway\(\)\.delete_tree\((?P<target>[^,\n]+),\s*cause=(?P<cause>[^)]*)\)\)\s*$",
            re.MULTILINE,
        )
        return pattern.sub(
            lambda m: f"{m.group('indent')}shutil.rmtree({m.group('target').strip()}, ignore_errors=True)",
            text,
        )

    @staticmethod
    def _repair_turn_taking_state_mutations(text: str) -> str:
        replacements = {
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='mode', new_value=mode, cause='TurnTakingEngine.set_mode')))": "self.state.mode = mode",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='user_speaking', new_value=True, cause='TurnTakingEngine.user_started_speaking')))": "self.state.user_speaking = True",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_user_speech_at', new_value=self._clock(), cause='TurnTakingEngine.user_started_speaking')))": "self.state.last_user_speech_at = self._clock()",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='user_speaking', new_value=False, cause='TurnTakingEngine.user_stopped_speaking')))": "self.state.user_speaking = False",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_user_speech_at', new_value=self._clock(), cause='TurnTakingEngine.user_stopped_speaking')))": "self.state.last_user_speech_at = self._clock()",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='last_aura_speech_at', new_value=self._clock(), cause='TurnTakingEngine.aura_emitted')))": "self.state.last_aura_speech_at = self._clock()",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='scene_energy', new_value=max(0.0, min(1.0, energy)), cause='TurnTakingEngine.update_scene_energy')))": "self.state.scene_energy = max(0.0, min(1.0, energy))",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='pending_repair', new_value=True, cause='TurnTakingEngine.request_repair')))": "self.state.pending_repair = True",
            "get_task_tracker().create_task(get_state_gateway().mutate(StateMutationRequest(key='pending_repair', new_value=False, cause='TurnTakingEngine.consume_repair')))": "self.state.pending_repair = False",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _validate_files(self, files: List[Path]) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", *[str(path) for path in files]],
                capture_output=True,
                text=True,
                timeout=self.validation_timeout_s,
                cwd=str(self.root),
            )
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
                "files": [str(path.relative_to(self.root)) for path in files],
            }
        except Exception as exc:
            record_degradation("structural_improver", exc)
            return {"ok": False, "stderr": str(exc)}

    def _append_ledger(self, summary: Dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **summary}
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False


def get_structural_improver(root: Optional[Path | str] = None) -> StructuralImprover:
    if root is None:
        try:
            from core.config import config

            root = getattr(config.paths, "project_root", Path.cwd())
        except Exception:
            root = Path.cwd()
    return StructuralImprover(root)

