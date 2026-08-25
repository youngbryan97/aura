#!/usr/bin/env python3
"""Build and verify Aura's relocatable Rust-optional skill distribution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.skills.discovery import build_skill_catalog, default_skill_roots  # noqa: E402

_REQUIRED_WHEEL_MEMBERS = (
    "aura_main.py",
    "core/identity_prompt.txt",
    "core/skills/discovery.py",
    "interface/static/index.html",
    "skills/os_automation.py",
)
_FORBIDDEN_WHEEL_PREFIXES = (
    "archive/",
    "artifacts/",
    "data/",
    "models/",
    "rust_extensions/",
    "tests/",
)
_NATIVE_SUFFIXES = (".dylib", ".pyd", ".so")
_CHILD_PROBE = r"""
import importlib.util
import json
from pathlib import Path

import core.skills.discovery as discovery

root = Path(discovery.__file__).resolve().parents[2]
catalog = discovery.build_skill_catalog(
    roots=discovery.default_skill_roots(root),
    try_rust=True,
)
print(json.dumps({
    "accepted": [item.to_dict() for item in catalog.accepted],
    "backend": catalog.backend,
    "discovery_file": discovery.__file__,
    "excluded": [item.to_dict() for item in catalog.excluded],
    "issues": [item.to_dict() for item in catalog.issues],
    "native_extension_available": importlib.util.find_spec("aura_m1_ext") is not None,
    "parity_status": catalog.parity_status,
    "root": str(root),
    "source_file_count": catalog.source_file_count,
}, sort_keys=True))
"""


def _run(argv: list[str], *, cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "AURA_LOG_DIR": str(cwd / "logs"),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONPATH": "",
        }
    )
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        check=True,
        text=True,
        timeout=timeout,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: setuptools stages an sdist into "<distribution>-<version>/".
_SDIST_STAGING_RE = re.compile(r"[A-Za-z0-9_.]+-\d[\w.]*")


def _remove_build_staging(before: set[Path], failures: list[str]) -> None:
    """Delete what the sdist build staged in the source tree, and say so.

    Only the staging shapes are removed — the versioned source directory, the
    egg-info, and build/. Anything else new is reported rather than deleted:
    an audit that tidies away a change it did not make would hide it.
    """
    for entry in sorted(set(ROOT.iterdir()) - before):
        name = entry.name
        staged = (
            name.endswith(".egg-info")
            or name == "build"
            or (entry.is_dir() and _SDIST_STAGING_RE.fullmatch(name) is not None)
        )
        if not staged:
            failures.append(f"audit_left_an_unexpected_path:{name}")
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as exc:
            failures.append(f"audit_could_not_clean_up:{name}:{exc}")


def _catalog_payload() -> dict[str, Any]:
    catalog = build_skill_catalog(
        roots=default_skill_roots(ROOT),
        try_rust=False,
    )
    return {
        "accepted": [item.to_dict() for item in catalog.accepted],
        "excluded": [item.to_dict() for item in catalog.excluded],
        "issues": [item.to_dict() for item in catalog.issues],
        "source_file_count": catalog.source_file_count,
    }


def audit_skill_portability() -> dict[str, Any]:
    """Prove a clean sdist/wheel install discovers skills without Rust."""

    failures: list[str] = []
    source_catalog = _catalog_payload()
    # An audit must leave the tree as it found it. `build --sdist
    # --no-isolation` stages into `<name>-<version>/` beside pyproject.toml
    # regardless of --outdir, and 109MB of unpacked source was left in the
    # repository root — enough to make every later evidence capture refuse
    # with "proof source tree is dirty".
    before_root = set(ROOT.iterdir())
    with tempfile.TemporaryDirectory(prefix="aura-skill-portability-") as raw_tmp:
        tmp = Path(raw_tmp)
        sdist_dir = tmp / "sdist"
        wheel_dir = tmp / "wheel"
        venv_dir = tmp / "venv"
        sdist_dir.mkdir()
        wheel_dir.mkdir()

        try:
            _run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--no-isolation",
                    "--outdir",
                    str(sdist_dir),
                    str(ROOT),
                ],
                cwd=tmp,
            )
            sdist = next(sdist_dir.glob("*.tar.gz"))
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-cache-dir",
                    "--no-deps",
                    "--wheel-dir",
                    str(wheel_dir),
                    str(sdist),
                ],
                cwd=tmp,
            )
            wheel = next(wheel_dir.glob("*.whl"))
            _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=tmp)
            venv_python = venv_dir / "bin" / "python"
            _run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--no-deps",
                    str(wheel),
                ],
                cwd=tmp,
            )
            probe = _run([str(venv_python), "-c", _CHILD_PROBE], cwd=tmp)
            installed = json.loads(probe.stdout)
        except (
            OSError,
            StopIteration,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as exc:
            return {
                "failures": [f"build_or_install_failed:{type(exc).__name__}:{exc}"],
                "ok": False,
                "schema": "aura.skill_portability_audit.v1",
            }

        _remove_build_staging(before_root, failures)

        with ZipFile(wheel) as archive:
            members = set(archive.namelist())
            wheel_metadata_name = next(
                name for name in members if name.endswith(".dist-info/WHEEL")
            )
            wheel_metadata = archive.read(wheel_metadata_name).decode("utf-8")

        missing_members = sorted(set(_REQUIRED_WHEEL_MEMBERS) - members)
        forbidden_members = sorted(
            name
            for name in members
            if name.startswith(_FORBIDDEN_WHEEL_PREFIXES) or "/tests/" in name
        )
        native_members = sorted(name for name in members if name.endswith(_NATIVE_SUFFIXES))
        if missing_members:
            failures.append(f"required_wheel_members_missing:{missing_members}")
        if forbidden_members:
            failures.append(f"forbidden_wheel_members_present:{forbidden_members[:20]}")
        if native_members:
            failures.append(f"python_wheel_embeds_native_extension:{native_members}")
        if "Root-Is-Purelib: true" not in wheel_metadata:
            failures.append("wheel_not_declared_purelib")
        if "Tag: py3-none-any" not in wheel_metadata:
            failures.append("wheel_not_platform_independent")
        if installed.get("native_extension_available") is not False:
            failures.append("clean_install_unexpectedly_found_native_extension")
        if installed.get("backend") != "python":
            failures.append(f"clean_install_wrong_backend:{installed.get('backend')}")
        if installed.get("parity_status") != "unavailable":
            failures.append(
                f"clean_install_wrong_parity_status:{installed.get('parity_status')}"
            )
        if installed.get("accepted") != source_catalog["accepted"]:
            failures.append("clean_install_accepted_catalog_diverged")
        if installed.get("excluded") != source_catalog["excluded"]:
            failures.append("clean_install_excluded_catalog_diverged")
        if installed.get("issues") != source_catalog["issues"]:
            failures.append("clean_install_catalog_issues_diverged")
        if installed.get("source_file_count") != source_catalog["source_file_count"]:
            failures.append("clean_install_source_file_count_diverged")
        installed_root = Path(str(installed.get("root", ""))).resolve()
        if not installed_root.is_relative_to(venv_dir.resolve()):
            failures.append("clean_install_imported_checkout_instead_of_wheel")

        return {
            "build": {
                "member_count": len(members),
                "native_member_count": len(native_members),
                "sdist": sdist.name,
                "wheel": wheel.name,
                "wheel_sha256": _sha256(wheel),
                "wheel_size_bytes": wheel.stat().st_size,
            },
            "clean_install": {
                "accepted_count": len(installed.get("accepted") or ()),
                "backend": installed.get("backend"),
                "discovery_file": installed.get("discovery_file"),
                "excluded_count": len(installed.get("excluded") or ()),
                "issue_count": len(installed.get("issues") or ()),
                "native_extension_available": installed.get("native_extension_available"),
                "parity_status": installed.get("parity_status"),
                "source_file_count": installed.get("source_file_count"),
            },
            "failures": failures,
            "ok": not failures,
            "schema": "aura.skill_portability_audit.v1",
            "source": {
                "accepted_count": len(source_catalog["accepted"]),
                "excluded_count": len(source_catalog["excluded"]),
                "issue_count": len(source_catalog["issues"]),
                "source_file_count": source_catalog["source_file_count"],
            },
        }


def main() -> int:
    result = audit_skill_portability()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
