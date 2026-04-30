"""Local shadow workspaces for architecture candidates."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.architect.config import ASAConfig
from core.architect.errors import ShadowWorkspaceError
from core.architect.models import RefactorPlan
from core.architect.refactor_planner import plan_to_dict
from core.runtime.atomic_writer import atomic_write_bytes, atomic_write_text


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass(frozen=True)
class ShadowRun:
    run_id: str
    shadow_root: str
    artifact_dir: str
    plan_id: str
    changed_files: tuple[str, ...]
    command_results: tuple[CommandResult, ...] = ()
    candidate_files: dict[str, str] = field(default_factory=dict)


class ShadowWorkspaceManager:
    """Create isolated source copies and apply plans only inside them."""

    def __init__(self, config: ASAConfig | None = None):
        self.config = config or ASAConfig.from_env()
        self.runs_dir = self.config.artifacts / "shadow_runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def create(self, plan: RefactorPlan) -> ShadowRun:
        run_id = f"asa-run-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        shadow_root = Path(tempfile.mkdtemp(prefix=f"{run_id}-"))
        artifact_dir = self.runs_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._copy_tree(self.config.repo_root, shadow_root)
        changed = self.apply_plan(plan, shadow_root)
        candidate_files = self._snapshot_candidates(shadow_root, changed, artifact_dir)
        run = ShadowRun(
            run_id=run_id,
            shadow_root=str(shadow_root),
            artifact_dir=str(artifact_dir),
            plan_id=plan.id,
            changed_files=tuple(changed),
            candidate_files=candidate_files,
        )
        self._write_manifest(run, plan)
        self.cleanup_old_runs()
        return run

    def apply_plan(self, plan: RefactorPlan, shadow_root: Path) -> list[str]:
        changed: list[str] = []
        for step in plan.steps:
            if step.operation == "replace_file":
                if step.new_content is None:
                    raise ShadowWorkspaceError(f"replace_file step lacks content: {step.id}")
                target = shadow_root / step.target_path
                target.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(target, step.new_content)
                changed.append(step.target_path)
            elif step.operation == "proposal":
                proposals = shadow_root / ".aura_architect" / "proposals"
                proposals.mkdir(parents=True, exist_ok=True)
                atomic_write_text(proposals / f"{plan.id}.json", json.dumps(plan_to_dict(plan), indent=2, sort_keys=True, default=str))
            elif step.operation == "quarantine_symbol":
                source = shadow_root / step.target_path
                if not source.exists():
                    raise ShadowWorkspaceError(f"quarantine target missing: {step.target_path}")
                manifest_dir = shadow_root / ".aura_architect" / "quarantine" / plan.id
                manifest_dir.mkdir(parents=True, exist_ok=True)
                atomic_write_bytes(manifest_dir / "original.py", source.read_bytes())
            else:
                raise ShadowWorkspaceError(f"unsupported refactor operation: {step.operation}")
        return list(dict.fromkeys(changed))

    def run_command(
        self,
        run: ShadowRun,
        command: tuple[str, ...] | list[str],
        *,
        timeout: float | None = None,
    ) -> CommandResult:
        cwd = Path(run.shadow_root)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(cwd)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        start = time.monotonic()
        cmd = tuple(str(part) for part in command)
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or self.config.shadow_timeout,
                check=False,
            )
            result = CommandResult(
                command=cmd,
                cwd=str(cwd),
                exit_code=proc.returncode,
                stdout=proc.stdout[-6000:],
                stderr=proc.stderr[-6000:],
                duration_s=round(time.monotonic() - start, 4),
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command=cmd,
                cwd=str(cwd),
                exit_code=-1,
                stdout=(exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
                duration_s=round(time.monotonic() - start, 4),
                timed_out=True,
            )
        self._append_command_result(run, result)
        return result

    def load_run(self, run_id: str) -> ShadowRun:
        manifest_path = self.runs_dir / run_id / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        commands = tuple(CommandResult(**item) for item in payload.get("command_results", ()))
        return ShadowRun(
            run_id=str(payload["run_id"]),
            shadow_root=str(payload["shadow_root"]),
            artifact_dir=str(payload["artifact_dir"]),
            plan_id=str(payload["plan_id"]),
            changed_files=tuple(payload.get("changed_files", ())),
            command_results=commands,
            candidate_files=dict(payload.get("candidate_files", {})),
        )

    def cleanup_old_runs(self) -> None:
        runs = sorted((path for path in self.runs_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
        for old in runs[self.config.retain_shadow_runs:]:
            shutil.rmtree(old, ignore_errors=True)

    def cleanup_shadow_root(self, run: ShadowRun) -> None:
        root = Path(run.shadow_root)
        if root.exists() and root.name.startswith("asa-run-"):
            shutil.rmtree(root, ignore_errors=True)

    def _copy_tree(self, source: Path, destination: Path) -> None:
        for child in source.iterdir():
            rel = child.relative_to(source).as_posix()
            if self.config.is_excluded(rel):
                continue
            dest = destination / child.name
            if child.is_dir():
                shutil.copytree(child, dest, ignore=self._ignore, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dest)

    def _ignore(self, directory: str, names: list[str]) -> set[str]:
        base = Path(directory)
        ignored: set[str] = set()
        for name in names:
            try:
                rel = (base / name).relative_to(self.config.repo_root).as_posix()
            except ValueError:
                rel = name
            if self.config.is_excluded(rel) or self.config.is_excluded(name):
                ignored.add(name)
        return ignored

    def _snapshot_candidates(self, shadow_root: Path, changed: list[str], artifact_dir: Path) -> dict[str, str]:
        candidate_root = artifact_dir / "candidate"
        candidate_files: dict[str, str] = {}
        for rel in changed:
            src = shadow_root / rel
            if not src.exists():
                continue
            dest = candidate_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(dest, src.read_bytes())
            candidate_files[rel] = str(dest)
        return candidate_files

    def _write_manifest(self, run: ShadowRun, plan: RefactorPlan) -> None:
        artifact_dir = Path(run.artifact_dir)
        payload = asdict(run)
        payload["plan"] = plan_to_dict(plan)
        atomic_write_text(artifact_dir / "manifest.json", json.dumps(payload, indent=2, sort_keys=True, default=str))

    def _append_command_result(self, run: ShadowRun, result: CommandResult) -> None:
        artifact_dir = Path(run.artifact_dir)
        results_path = artifact_dir / "commands.jsonl"
        existing = ""
        if results_path.exists():
            existing = results_path.read_text(encoding="utf-8")
        line = json.dumps(asdict(result), sort_keys=True, default=str)
        atomic_write_text(results_path, existing + line + "\n")
        manifest_path = artifact_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        command_results = payload.get("command_results", [])
        command_results.append(asdict(result))
        payload["command_results"] = command_results
        atomic_write_text(manifest_path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def python_executable() -> str:
    return sys.executable or "python3"
