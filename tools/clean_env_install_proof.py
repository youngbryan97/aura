#!/usr/bin/env python3
"""Clean-environment install proof — reproducible from a clean clone.

The v1.0 gate: does the repository install and import in a PRISTINE
environment, not just on the dev box where everything is already warm?
This proof:

1. Exports HEAD with `git archive` into a temp dir — exactly the tracked
   files a fresh `git clone` would give (no .venv, no __pycache__, no
   untracked artifacts).
2. Creates a fresh virtualenv inside it.
3. Installs requirements/core.txt + requirements/dev.txt from scratch.
4. Imports the runtime skeleton in that clean interpreter — catching any
   core module that eagerly imports an optional ML extra (a real
   reproducibility bug).
5. Runs `pytest --collect-only` on a lightweight, core-only test set to
   prove tests are importable in the clean env.

PASS = clean checkout + clean install + skeleton imports + collection,
all green. Bounded and self-cleaning (the temp dir is removed on exit).

Usage:
    python tools/clean_env_install_proof.py [--keep] [--timeout 1200]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "current" / "proof_steps"

# Runtime-skeleton modules that requirements/core.txt must be sufficient
# to import. If any eagerly pulls an ML extra (mlx, torch, faster-whisper),
# importing it here fails — that is a real clean-clone reproducibility bug.
_SKELETON_IMPORTS = (
    "core.config",
    "core.container",
    "core.runtime.atomic_writer",
    "core.runtime.thread_inspector",
    "core.runtime.network_gateway",
    "core.conversation.self_claim_verifier",
    "core.runtime.desktop_objective_intent",
    "interface.auth",
)

# Lightweight, import-clean tests that should collect with core+dev only.
_COLLECT_TARGETS = (
    "tests/test_self_claim_verifier.py",
    "tests/test_thread_inspector.py",
    "tests/test_desktop_objective_intent.py",
)


def _run(cmd: list[str], *, cwd: Path, timeout: float, env_path: Path | None = None) -> dict:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout
        )
        return {
            "cmd": " ".join(cmd[:6]),
            "rc": proc.returncode,
            "secs": round(time.time() - started, 1),
            "tail": (proc.stdout or "")[-600:] + (proc.stderr or "")[-1200:],
        }
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd[:6]), "rc": 124, "secs": round(time.time() - started, 1), "tail": "TIMEOUT"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"cmd": " ".join(cmd[:6]), "rc": 1, "secs": 0.0, "tail": str(exc)}


def run_proof(*, keep: bool, timeout: float) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="aura_clean_env_"))
    checkout = tmp / "checkout"
    checkout.mkdir()
    steps: list[dict] = []
    try:
        # 1. Pristine tracked-files-only checkout of HEAD.
        archive = subprocess.run(
            ["git", "archive", "HEAD"], cwd=str(PROJECT_ROOT),
            capture_output=True, timeout=300,
        )
        if archive.returncode != 0:
            return {"passed": False, "stage": "git_archive", "error": archive.stderr.decode()[:500], "steps": steps}
        untar = subprocess.run(
            ["tar", "-x", "-C", str(checkout)], input=archive.stdout,
            capture_output=True, timeout=300,
        )
        steps.append({"step": "git_archive_extract", "rc": untar.returncode,
                      "files": sum(1 for _ in checkout.rglob("*.py"))})
        if untar.returncode != 0:
            return {"passed": False, "stage": "extract", "error": untar.stderr.decode()[:500], "steps": steps}

        # 2. Fresh venv.
        venv = checkout / ".venv"
        steps.append({"step": "venv", **_run([sys.executable, "-m", "venv", str(venv)], cwd=checkout, timeout=120)})
        pip = venv / "bin" / "pip"
        py = venv / "bin" / "python"

        # 3. Install core + dev from scratch.
        steps.append({"step": "pip_upgrade", **_run([str(pip), "install", "-U", "pip", "wheel"], cwd=checkout, timeout=240)})
        steps.append({"step": "pip_core", **_run([str(pip), "install", "-r", "requirements/core.txt"], cwd=checkout, timeout=timeout)})
        steps.append({"step": "pip_dev", **_run([str(pip), "install", "-r", "requirements/dev.txt"], cwd=checkout, timeout=timeout)})

        # 4. Skeleton imports in the clean interpreter.
        import_src = "import importlib,sys\n" + "\n".join(
            f"importlib.import_module({m!r})" for m in _SKELETON_IMPORTS
        ) + "\nprint('IMPORTS_OK')\n"
        steps.append({"step": "skeleton_imports",
                      **_run([str(py), "-c", import_src], cwd=checkout, timeout=180)})

        # 5. Collection on the core-only test subset.
        steps.append({"step": "pytest_collect",
                      **_run([str(py), "-m", "pytest", "--collect-only", "-q", *_COLLECT_TARGETS],
                             cwd=checkout, timeout=180)})

        install_ok = all(
            s.get("rc") == 0 for s in steps
            if s["step"] in {"venv", "pip_upgrade", "pip_core", "pip_dev"}
        )
        imports_ok = steps[-2].get("rc") == 0 and "IMPORTS_OK" in steps[-2].get("tail", "")
        collect_ok = steps[-1].get("rc") == 0
        passed = bool(install_ok and imports_ok and collect_ok)
        return {
            "passed": passed,
            "install_ok": install_ok,
            "imports_ok": imports_ok,
            "collect_ok": collect_ok,
            "checkout": str(checkout) if keep else "(removed)",
            "steps": steps,
        }
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Keep the temp checkout for inspection.")
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-pip-install timeout (s).")
    args = parser.parse_args(argv)

    verdict = run_proof(keep=args.keep, timeout=args.timeout)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "clean_env_install_proof.json").write_text(
        json.dumps(verdict, indent=2, default=str)
    )
    for s in verdict.get("steps", []):
        print(f"  [{s.get('rc')}] {s['step']} ({s.get('secs', '?')}s)")
    print(
        ("CLEAN ENV INSTALL PROOF PASSED" if verdict["passed"] else "CLEAN ENV INSTALL PROOF FAILED")
        + f" install={verdict.get('install_ok')} imports={verdict.get('imports_ok')} collect={verdict.get('collect_ok')}"
    )
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
