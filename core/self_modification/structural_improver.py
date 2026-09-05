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
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.StructuralImprover")


@dataclass(frozen=True)
class StructuralIssue:
    file_path: str
    line: int
    kind: str
    message: str
    severity: float = 0.5
    repairable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuralRepairResult:
    issue: StructuralIssue
    changed: bool
    success: bool
    message: str
    validation: dict[str, Any] = field(default_factory=dict)


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
        ledger_path: Path | None = None,
        validation_timeout_s: int = 60,
    ):
        self.root = Path(root).resolve()
        self.validation_timeout_s = max(10, int(validation_timeout_s))
        self.ledger_path = ledger_path or (self.root / "data" / "structural_improvements.jsonl")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def scan(self, *, max_files: int = 2000) -> list[StructuralIssue]:
        issues: list[StructuralIssue] = []
        scanned = 0
        for path in self._iter_python_files():
            if scanned >= max_files:
                break
            scanned += 1
            try:
                issues.extend(self._scan_file(path))
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("structural_improver", exc)
                logger.debug("Structural scan skipped %s: %s", path, exc)
        return sorted(issues, key=lambda i: (-i.severity, i.file_path, i.line))

    def find_and_fix(
        self,
        *,
        max_repairs: int = 3,
        kinds: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        allowed = set(kinds or [])
        issues = [
            issue for issue in self.scan()
            if issue.repairable and (not allowed or issue.kind in allowed)
        ]
        results: list[StructuralRepairResult] = []
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

            self._write_source(path, repaired, reason=f"repair:{issue.kind}")
            validation = self._validate_files([path])
            if not validation.get("ok", False):
                if self._restore_original(path, original, validation=validation):
                    return StructuralRepairResult(issue, True, False, "validation failed; rolled back", validation)
                return StructuralRepairResult(
                    issue,
                    True,
                    False,
                    "validation failed; rollback failed",
                    validation,
                )

            return StructuralRepairResult(issue, True, True, "repair applied", validation)
        except (OSError, ConnectionError, TimeoutError) as exc:
            record_degradation("structural_improver", exc)
            validation: dict[str, Any] = {}
            self._restore_original(path, original, validation=validation)
            return StructuralRepairResult(
                issue,
                repaired != original,
                False,
                f"{type(exc).__name__}: {exc}",
                validation,
            )

    def _write_source(self, path: Path, text: str, *, reason: str) -> None:
        """Write one of Aura's own source files, through the canonical gateway.

        Both of this module's writes used to call ``atomic_write_text``
        directly. That is the primitive the gateway wraps, so these — the
        only writes in the system that modify Aura's *own source* without a
        model in the loop — were the ones skipping the governance check,
        the ownership record, and the write ledger. The repair itself may
        well be authorized by the enclosing RSI operation; that is a
        different question from whether the write is accounted for.

        The scope is declared as internal maintenance because it is: a
        deterministic repair of a known defect class in the local checkout,
        with validation and rollback around it. Without the scope the live
        runtime refuses the write as a governance violation, which is the
        correct default for anything writing to source.
        """
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope(
            f"structural_improver.{reason}",
            constraints={
                "repository_root": str(self.root),
                "target": str(path),
                "validated_and_rolled_back": True,
            },
        ):
            get_file_write_gateway().write_text(
                path,
                text,
                encoding="utf-8",
                source=f"self_modification.structural_improver.{reason}",
            )

    def _restore_original(
        self,
        path: Path,
        original: str,
        *,
        validation: dict[str, Any] | None = None,
    ) -> bool:
        try:
            self._write_source(path, original, reason="rollback")
            return True
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            if validation is not None:
                validation["rollback_error"] = f"{type(exc).__name__}: {exc}"
            record_degradation(
                "structural_improver",
                exc,
                action=f"rollback failed for {path}",
            )
            logger.error("Structural improver rollback failed for %s: %s", path, exc)
            return False

    def _iter_python_files(self) -> Iterable[Path]:
        for path in self.root.rglob("*.py"):
            rel_parts = path.relative_to(self.root).parts
            rel_string = "/".join(rel_parts)
            if any(part in self.EXCLUDED_PARTS for part in rel_parts):
                continue
            if any(excluded in rel_string for excluded in self.EXCLUDED_PARTS if "/" in excluded):
                continue
            yield path

    def _scan_file(self, path: Path) -> list[StructuralIssue]:
        rel = str(path.relative_to(self.root))
        text = path.read_text(encoding="utf-8")
        issues: list[StructuralIssue] = []

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

        marker = 'raise ' + 'Not' + 'ImplementedError("Aura Pass'
        if marker in text:
            issues.append(StructuralIssue(
                file_path=rel,
                line=self._first_line_containing(text, marker),
                kind="unimplemented_stub",
                message="audited pass marker still raises an unimplemented runtime error",
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

    def _validate_files(self, files: list[Path]) -> dict[str, Any]:
        try:
            result = get_subprocess_gateway().run(
                [sys.executable, "-m", "py_compile", *[str(path) for path in files]],
                capture_output=True,
                timeout=self.validation_timeout_s,
                cwd=str(self.root),
                read_only=True,
                source="self_modification.structural_improver.py_compile",
                accelerator_capability="none",
            )
            return {
                "ok": result.returncode == 0,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
                "files": [str(path.relative_to(self.root)) for path in files],
            }
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
            record_degradation("structural_improver", exc)
            return {"ok": False, "stderr": str(exc)}

    def _append_ledger(self, summary: dict[str, Any]) -> None:
        payload = {"timestamp": time.time(), **summary}
        get_file_write_gateway().append_text(
            self.ledger_path,
            json.dumps(payload, sort_keys=True, default=str) + "\n",
            source="self_modification.structural_improver.ledger",
        )

    def _is_within_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False


def get_structural_improver(root: Path | str | None = None) -> StructuralImprover:
    if root is None:
        try:
            from core.config import config

            root = getattr(config.paths, "project_root", Path.cwd())
        except (ImportError, AttributeError, RuntimeError):
            root = Path.cwd()
    return StructuralImprover(root)
