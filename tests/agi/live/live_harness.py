from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

@dataclass
class LiveRunResult:
    ok: bool
    stdout: str
    stderr: str
    artifacts_dir: Path
    returncode: int

class LiveAuraHarness:
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.temp_dirs: list[Path] = []

    def create_isolated_copy(self) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="aura-live-"))
        self.temp_dirs.append(tmp)
        target = tmp / "repo"

        def custom_ignore(directory, contents):
            ignored = set()
            dir_path = Path(directory).resolve()
            repo_path = self.repo_root.resolve()
            
            # Top-level ignores
            if dir_path == repo_path:
                top_level_ignores = {
                    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".pyre",
                    ".aura", ".aura_architect", ".aura_runtime", ".aura_snapshots", ".claude",
                    "dist", "models", "models_gguf", "archive", "dev_archive", "artifacts",
                    "logs", "storage", "test_vdb", "experiments", "benchmarks", "dev_rebuild_and_launch.sh",
                    "boot_log.txt", "test_out.txt", "test_out_2.txt", "nohup.out", "training"
                }
                ignored.update(top_level_ignores & set(contents))
            
            # Sub-directory specific ignores inside data/
            try:
                rel_dir = dir_path.relative_to(repo_path)
                if rel_dir.parts and rel_dir.parts[0] == "data":
                    if len(rel_dir.parts) == 1:
                        # Direct children of data/
                        for name in contents:
                            if name != "memory" and not name.endswith(".json") and not name.endswith(".jsonl"):
                                ignored.add(name)
            except ValueError:
                pass
            
            # General extensions and directory names to ignore anywhere
            for name in contents:
                if name in (".git", "__pycache__"):
                    ignored.add(name)
                elif name.endswith((".db", ".db-shm", ".db-wal", ".sqlite3", ".sqlite", ".zip", ".obj", ".png", ".icns")):
                    ignored.add(name)
                    
            return list(ignored)

        shutil.copytree(self.repo_root, target, ignore=custom_ignore)
        return target

    def run_command(
        self,
        repo: Path,
        args: list[str],
        *,
        timeout_s: int = 300,
        env: dict[str, str] | None = None,
    ) -> LiveRunResult:
        artifacts = repo / "artifacts" / "agi_live"
        artifacts.mkdir(parents=True, exist_ok=True)

        # Ensure we use the current sys.executable to execute python scripts inside the copy
        if args and (args[0] in (".venv/bin/python", "python", "python3")):
            args[0] = sys.executable

        run_env = os.environ.copy()
        run_env.update(env or {})
        run_env["AURA_ARTIFACTS_DIR"] = str(artifacts)
        run_env["AURA_STRICT_RUNTIME"] = "1"
        run_env["AURA_AGI_LIVE_TEST"] = "1"
        
        # Ensure python path includes the isolated repo copy
        run_env["PYTHONPATH"] = str(repo)

        proc = subprocess.run(
            args,
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=run_env,
        )

        return LiveRunResult(
            ok=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            artifacts_dir=artifacts,
            returncode=proc.returncode,
        )

    def cleanup(self):
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception:
                    pass
