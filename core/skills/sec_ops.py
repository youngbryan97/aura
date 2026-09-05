from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill


_SECURITY_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("high", "dynamic_execution", re.compile(r"\b(?:eval|exec)\s*\(", re.IGNORECASE)),
    ("high", "shell_execution", re.compile(r"\b(?:os\.system|subprocess\.(?:Popen|run|call))\b")),
    ("medium", "unsafe_deserialization", re.compile(r"\b(?:pickle\.loads?|yaml\.load)\s*\(")),
    ("medium", "secret_literal", re.compile(r"(?i)\b(?:password|api[_-]?key|token|secret)\s*=\s*['\"][^'\"]{6,}")),
    ("medium", "raw_network_request", re.compile(r"\b(?:requests|httpx)\.(?:get|post|put|delete|request)\s*\(")),
    ("low", "broad_exception", re.compile(r"\bexcept\s+(?:Exception|BaseException)\b")),
)

_SKIP_DIRS = {".git", ".pytest_cache", ".venv", "__pycache__", "node_modules", "venv"}


class SecOpsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str = Field("status", description="status|audit_code|inspect_path|nmap_scan|red_team")
    path: str = "."
    target: str = "localhost"
    max_files: int = Field(64, ge=1, le=500)
    max_bytes_per_file: int = Field(262_144, ge=1024, le=2_000_000)
    external_authorized: bool = False

    @model_validator(mode="before")
    @classmethod
    def _unwrap_params(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("params"), dict):
            merged = dict(value)
            nested = dict(merged.pop("params"))
            nested.update(merged)
            return nested
        return value


class SecOpsSkill(BaseSkill):
    name = "sec_ops"
    description = "Run bounded local security posture checks and refuse active external testing without authorization."
    input_model = SecOpsInput
    effect_scope = "read_only"
    metabolic_cost = 1
    timeout_seconds = 10.0

    def _allowed_roots(self) -> list[Path]:
        home = Path.home()
        return [
            Path.cwd(),
            Path(os.environ.get("AURA_ROOT", "")).expanduser() if os.environ.get("AURA_ROOT") else Path.cwd(),
            Path(tempfile.gettempdir()),
            home / "Desktop",
            home / "Documents",
            home / "Downloads",
        ]

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path or ".")).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve(strict=False)
        for root in self._allowed_roots():
            allowed = root.expanduser().resolve(strict=False)
            try:
                if os.path.commonpath([str(allowed), str(resolved)]) == str(allowed):
                    return resolved
            except (OSError, ValueError):
                continue
        raise ValueError("Security audit path is outside Aura's allowed local inspection roots.")

    @staticmethod
    def _iter_files(target: Path, *, max_files: int) -> tuple[list[Path], int]:
        if target.is_file():
            return [target], 0
        if not target.exists():
            return [], 0
        files: list[Path] = []
        skipped = 0
        for path in target.rglob("*"):
            if len(files) >= max_files:
                skipped += 1
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                skipped += 1
                continue
            try:
                if not path.is_file() or path.is_symlink():
                    continue
            except OSError:
                skipped += 1
                continue
            if path.suffix.lower() not in {".py", ".sh", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml"}:
                skipped += 1
                continue
            files.append(path)
        return files, skipped

    @staticmethod
    def _hash_sample(path: Path, *, max_bytes: int) -> tuple[str, int]:
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while total < max_bytes:
                chunk = handle.read(min(65_536, max_bytes - total))
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
        return digest.hexdigest(), total

    def _audit_file(self, path: Path, *, max_bytes: int) -> dict[str, Any]:
        sha256, sampled = self._hash_sample(path, max_bytes=max_bytes)
        text = path.read_bytes()[:max_bytes].decode("utf-8", errors="ignore")
        findings: list[dict[str, Any]] = []
        for severity, rule_id, pattern in _SECURITY_PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                findings.append(
                    {
                        "severity": severity,
                        "rule_id": rule_id,
                        "line": line,
                        "evidence": match.group(0)[:120],
                    }
                )
        return {
            "path": str(path),
            "sha256": sha256,
            "sampled_bytes": sampled,
            "truncated": path.stat().st_size > sampled,
            "findings": findings,
        }

    def _status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "available",
            "summary": "SecOps is available for bounded local audit_code checks; active network/red-team actions are authorization-gated.",
            "capabilities": ["audit_code", "inspect_path", "status"],
            "blocked_without_authorization": ["nmap_scan", "red_team"],
        }

    async def execute(self, params: SecOpsInput | dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = SecOpsInput(**params)
        context = context or {}
        action = params.action.strip().lower()

        if action == "status":
            return self._status()

        if action in {"nmap_scan", "red_team", "external_scan"} and not (
            params.external_authorized or context.get("external_security_authorized")
        ):
            return {
                "ok": False,
                "status": "blocked",
                "error": "Active external security testing requires explicit authorization and ExternalIOGateway routing.",
                "target": params.target,
            }

        if action not in {"audit_code", "inspect_path", "scan_local"}:
            return {
                "ok": False,
                "status": "invalid_action",
                "error": f"Unsupported SecOps action: {params.action}",
                "supported_actions": ["status", "audit_code", "inspect_path"],
            }

        try:
            target = self._resolve_path(params.path)
            files, skipped = self._iter_files(target, max_files=params.max_files)
            if not files and not target.exists():
                return {"ok": False, "status": "not_found", "error": f"Audit target not found: {target}"}

            reports: list[dict[str, Any]] = []
            for path in files:
                try:
                    reports.append(self._audit_file(path, max_bytes=params.max_bytes_per_file))
                except (OSError, ValueError) as exc:
                    skipped += 1
                    record_degradation(
                        "sec_ops",
                        exc,
                        severity="warning",
                        action="skipped unreadable file during local security audit",
                    )
            findings = [
                {"path": report["path"], **finding}
                for report in reports
                for finding in report.get("findings", [])
            ]
            return {
                "ok": True,
                "status": "completed",
                "summary": (
                    "Bandit-compatible local security audit completed: "
                    f"{len(reports)} files inspected, {len(findings)} findings."
                ),
                "target": str(target),
                "files_scanned": len(reports),
                "files_skipped": skipped,
                "findings": findings,
                "report": {"files": reports, "finding_count": len(findings)},
            }
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "sec_ops",
                exc,
                severity="warning",
                action="returned bounded security audit failure",
            )
            return {"ok": False, "status": "failed", "error": str(exc)}
