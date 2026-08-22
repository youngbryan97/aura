"""Build, verify, and expose the provenance of a launched Aura desktop runtime.

The installed ``Aura.app`` is intentionally a thin launcher over live source.
That makes source identity a runtime contract: the app must prove which root,
commit, and exact dirty workspace state it was built to launch before cleanup
or process replacement can occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.subprocess_gateway import get_subprocess_gateway

LAUNCH_PROVENANCE_SCHEMA = "aura.launch_provenance.v1"
EXPECTED_BUNDLE_ID = "com.aura.desktop"
RUNTIME_SHELL_ASSETS = (
    "interface/static/index.html",
    "interface/static/design_tokens.css",
    "interface/static/motion_design.css",
    "interface/static/error_banner.css",
    "interface/static/aura.css",
    "interface/static/presence_design.css",
    "interface/static/vendor/vis-network.min.js",
    "interface/static/error_banner.js",
    "interface/static/sound_design.js",
    "interface/static/perf_collector.js",
    "interface/static/aura.js",
    "interface/static/manifest.json",
    "interface/static/service-worker.js",
    "interface/static/icon.svg",
    "interface/static/icon-192.png",
    "interface/static/icon-512.png",
    "interface/static/aura_avatar.svg",
    "interface/static/vendor/fonts/fredoka-variable-latin.woff2",
    "interface/static/vendor/fonts/ibm-plex-mono-400-latin.woff2",
    "interface/static/vendor/fonts/ibm-plex-mono-500-latin.woff2",
    "interface/static/vendor/fonts/ibm-plex-mono-600-latin.woff2",
    "interface/static/voice-processor.js",
)
# Revision-addressed icons are safe for shared immutable caching. Every other
# shell byte remains private to the authenticated desktop runtime.
RUNTIME_SHELL_PUBLIC_ASSETS = frozenset(
    {
        "interface/static/icon.svg",
        "interface/static/icon-192.png",
        "interface/static/icon-512.png",
    }
)
if not RUNTIME_SHELL_PUBLIC_ASSETS.issubset(RUNTIME_SHELL_ASSETS):
    raise RuntimeError("public runtime shell assets must belong to the signed shell")

_SOURCE_PATHS = (
    "aura_main.py",
    "aura_cleanup.py",
    "main_daemon.py",
    "launch_aura.sh",
    "build_app.sh",
    "pyproject.toml",
    "requirements.txt",
    "requirements_hardened.txt",
    "requirements_lock.txt",
    "aura",
    "autonomy_engine",
    "cloud",
    "config",
    "core",
    "executors",
    "infrastructure",
    "integration",
    "interface",
    "llm",
    "memory",
    "native",
    "optimizer",
    "proof_kernel",
    "rust_extensions",
    "scoping",
    "scripts",
    "security",
    "senses",
    "skills",
    "storage",
    "utils",
)
_SOURCE_CACHE_TTL_S = 2.0
_SOURCE_CACHE_LOCK = threading.Lock()
_SOURCE_CACHE: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
_BUNDLE_CACHE_LOCK = threading.Lock()
_BUNDLE_CACHE: dict[str, tuple[int, dict[str, Any]]] = {}
_RECOVERABLE_ERRORS = (
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def runtime_shell_request_path(relative: str) -> str:
    """Map one signed source asset to its canonical HTTP request path."""

    normalized = str(relative or "").strip()
    prefix = "interface/"
    if (
        not normalized.startswith(prefix)
        or normalized.startswith("/")
        or "\\" in normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
    ):
        raise ValueError(f"runtime shell asset path is invalid: {relative!r}")
    return "/" + normalized.removeprefix(prefix)


def _run_git(root: Path, arguments: Sequence[str], *, timeout: float = 3.0) -> str:
    completed = get_subprocess_gateway().run(
        ["git", "-C", str(root), *arguments],
        timeout=timeout,
        read_only=True,
        capture_output=True,
        source="runtime_launch_provenance.git",
        accelerator_capability="none",
    )
    if completed.returncode != 0:
        detail = str(completed.stderr or completed.stdout or "git command failed").strip()
        raise RuntimeError(detail[:500])
    return str(completed.stdout or "")


def _git_identity(root: Path) -> dict[str, str]:
    canonical_root = (
        Path(_run_git(root, ("rev-parse", "--show-toplevel")).strip()).expanduser().resolve()
    )
    commit = _run_git(canonical_root, ("rev-parse", "HEAD")).strip()
    branch_result = get_subprocess_gateway().run(
        ["git", "-C", str(canonical_root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout=3.0,
        read_only=True,
        capture_output=True,
        source="runtime_launch_provenance.git_branch",
        accelerator_capability="none",
    )
    branch = (
        str(branch_result.stdout or "").strip() if branch_result.returncode == 0 else "DETACHED"
    )
    if len(commit) != 40 or any(character not in "0123456789abcdefABCDEF" for character in commit):
        raise RuntimeError("git HEAD did not resolve to a full commit SHA")
    return {
        "source_root": str(canonical_root),
        "commit_sha": commit.lower(),
        "branch": branch,
    }


def _status_paths(status_output: str) -> list[str]:
    records = status_output.split("\0")
    paths: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4 or record[2] != " ":
            continue
        status = record[:2]
        path = record[3:]
        if path:
            paths.append(path)
        if any(marker in status for marker in ("R", "C")) and index < len(records):
            prior_path = records[index]
            index += 1
            if prior_path:
                paths.append(prior_path)
    return sorted(set(paths))


def _hash_workspace_state(root: Path, status_output: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(b"aura-workspace-state-v1\0")
    digest.update(status_output.encode("utf-8", errors="surrogateescape"))
    paths = _status_paths(status_output)
    for relative in paths:
        raw_candidate = root / relative
        candidate = raw_candidate.resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            digest.update(b"outside-root\0" + relative.encode("utf-8", errors="surrogateescape"))
            continue
        encoded_path = relative.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if raw_candidate.is_symlink():
            target = os.readlink(raw_candidate)
            digest.update(b"symlink\0" + target.encode("utf-8", errors="surrogateescape"))
        elif candidate.is_file():
            digest.update(b"file\0")
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"missing\0")
    return {
        "workspace_state_sha256": digest.hexdigest(),
        "source_dirty": bool(status_output),
        "source_change_count": len(paths),
        "source_changed_paths": paths[:128],
        "source_changed_paths_truncated": len(paths) > 128,
    }


def runtime_shell_assets_digest(asset_bytes: Mapping[str, bytes]) -> str:
    """Hash a complete, already-captured browser shell snapshot."""

    expected = set(RUNTIME_SHELL_ASSETS)
    observed = set(asset_bytes)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"runtime shell snapshot is incomplete (missing={missing}, extra={extra})"
        )
    digest = hashlib.sha256()
    digest.update(b"aura-runtime-shell-assets-v1\0")
    for relative in RUNTIME_SHELL_ASSETS:
        content = asset_bytes[relative]
        if not isinstance(content, bytes):
            raise RuntimeError(f"runtime shell snapshot contains non-bytes: {relative}")
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def capture_runtime_shell_assets(root: str | Path) -> tuple[str, dict[str, bytes]]:
    """Capture the exact browser shell bytes without following filesystem links."""

    canonical_root = Path(root).expanduser().resolve(strict=True)
    captured: dict[str, bytes] = {}
    root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    root_descriptor = os.open(canonical_root, root_flags)
    try:
        for relative in RUNTIME_SHELL_ASSETS:
            parts = Path(relative).parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                raise RuntimeError(f"runtime shell asset path is invalid: {relative}")
            descriptor = os.dup(root_descriptor)
            try:
                for index, part in enumerate(parts):
                    is_final = index == len(parts) - 1
                    flags = (
                        os.O_RDONLY
                        | getattr(os, "O_CLOEXEC", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    if not is_final:
                        flags |= getattr(os, "O_DIRECTORY", 0)
                    try:
                        next_descriptor = os.open(part, flags, dir_fd=descriptor)
                    except OSError as exc:
                        raise RuntimeError(
                            f"runtime shell asset traverses a symlink or invalid path: {relative}"
                        ) from exc
                    os.close(descriptor)
                    descriptor = next_descriptor
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise RuntimeError(f"runtime shell asset is not a file: {relative}")
                chunks: list[bytes] = []
                remaining = before.st_size
                while remaining > 0:
                    chunk = os.read(descriptor, min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(
                            f"runtime shell asset truncated while reading: {relative}"
                        )
                    chunks.append(chunk)
                    remaining -= len(chunk)
                after = os.fstat(descriptor)
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if identity_before != identity_after:
                    raise RuntimeError(f"runtime shell asset changed while read: {relative}")
                content = b"".join(chunks)
                if len(content) != after.st_size:
                    raise RuntimeError(f"runtime shell asset read was incomplete: {relative}")
                captured[relative] = content
            finally:
                os.close(descriptor)
    finally:
        os.close(root_descriptor)
    return runtime_shell_assets_digest(captured), captured


def runtime_shell_assets_sha256(root: str | Path) -> str:
    """Hash the exact browser shell set pinned into a signed Aura.app."""

    digest, _assets = capture_runtime_shell_assets(root)
    return digest


def _workspace_state_uncached(root: Path) -> dict[str, Any]:
    output = _run_git(
        root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *_SOURCE_PATHS,
        ),
        timeout=8.0,
    )
    return _hash_workspace_state(root, output)


def _workspace_state(root: Path, *, commit_sha: str) -> dict[str, Any]:
    from core.runtime.flags import FlagKind, declare

    manifest_override = str(
        declare(
            "AURA_LAUNCH_MANIFEST_PATH",
            kind=FlagKind.STRING,
            default="",
            description="Override path for the launch manifest (packaging/tests)",
            owner="core.runtime.launch_provenance",
        ).value()
    )
    cache_key = (str(root), commit_sha, manifest_override)
    now = time.monotonic()
    with _SOURCE_CACHE_LOCK:
        cached = _SOURCE_CACHE.get(cache_key)
        if cached is not None and now - cached[0] < _SOURCE_CACHE_TTL_S:
            return dict(cached[1])
    result = _workspace_state_uncached(root)
    with _SOURCE_CACHE_LOCK:
        _SOURCE_CACHE[cache_key] = (time.monotonic(), dict(result))
    return result


def collect_source_identity(root: str | Path) -> dict[str, Any]:
    """Capture a stable, exact identity for a direct source runtime.

    Installed-app launches already carry a signed manifest. Direct developer
    launches still need an exact commit/workspace/shell identity when a proof
    artifact claims which source produced a result.
    """

    canonical_input = Path(root).expanduser().resolve()
    identity_before = _git_identity(canonical_input)
    canonical_root = Path(identity_before["source_root"])
    workspace_before = _workspace_state_uncached(canonical_root)
    shell_before = runtime_shell_assets_sha256(canonical_root)
    identity_after = _git_identity(canonical_root)
    workspace_after = _workspace_state_uncached(canonical_root)
    shell_after = runtime_shell_assets_sha256(canonical_root)
    if (
        identity_before != identity_after
        or workspace_before["workspace_state_sha256"]
        != workspace_after["workspace_state_sha256"]
        or shell_before != shell_after
    ):
        raise RuntimeError("Aura source changed while runtime identity was captured")
    return {
        **identity_after,
        **workspace_after,
        "shell_assets_sha256": shell_after,
    }


def build_launch_manifest(
    root: str | Path,
    *,
    version: str,
    launcher_source: str | Path,
) -> dict[str, Any]:
    canonical_input = Path(root).expanduser().resolve()
    identity_before = _git_identity(canonical_input)
    canonical_root = Path(identity_before["source_root"])
    shell_before = runtime_shell_assets_sha256(canonical_root)
    workspace_before = _workspace_state_uncached(canonical_root)
    launcher_path = Path(launcher_source).expanduser().resolve()
    launcher_bytes = launcher_path.read_bytes()
    identity_after = _git_identity(canonical_root)
    workspace_after = _workspace_state_uncached(canonical_root)
    shell_after = runtime_shell_assets_sha256(canonical_root)
    if (
        identity_before != identity_after
        or workspace_before["workspace_state_sha256"] != workspace_after["workspace_state_sha256"]
        or shell_before != shell_after
    ):
        raise RuntimeError("Aura source changed while the signed launch manifest was captured")
    return {
        "schema": LAUNCH_PROVENANCE_SCHEMA,
        "generated_at_unix": time.time(),
        "version": str(version),
        **identity_after,
        **workspace_after,
        "shell_assets_sha256": shell_after,
        "launcher_source": str(launcher_path.relative_to(canonical_root)),
        "launcher_source_sha256": hashlib.sha256(launcher_bytes).hexdigest(),
        "bundle_identifier": EXPECTED_BUNDLE_ID,
    }


def write_launch_manifest(
    output: str | Path,
    *,
    root: str | Path,
    version: str,
    launcher_source: str | Path,
) -> dict[str, Any]:
    manifest = build_launch_manifest(root, version=version, launcher_source=launcher_source)
    atomic_write_text(
        output,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("launch manifest must contain a JSON object")
    return payload


def _bundle_for_executable(executable: Path) -> Path | None:
    parts = executable.resolve(strict=False).parts
    for index, part in enumerate(parts):
        if part.endswith(".app"):
            return Path(*parts[: index + 1])
    return None


def _manifest_belongs_to_executable(manifest_path: Path, executable: Path) -> bool:
    bundle = _bundle_for_executable(executable)
    if bundle is None:
        return False
    expected = bundle / "Contents" / "Resources" / "aura-launch-provenance.json"
    return manifest_path.resolve(strict=False) == expected.resolve(strict=False)


def validate_launch_source(
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    canonical_input = Path(root).expanduser().resolve()
    launched_from_app = _truthy(environment.get("AURA_LAUNCHED_FROM_APP"))
    if not launched_from_app:
        return {
            "schema": LAUNCH_PROVENANCE_SCHEMA,
            "required": False,
            "launch_mode": "direct",
            "source_verified": True,
            "verified": False,
            "source_root": str(canonical_input),
            "issues": [],
        }

    issues: list[str] = []
    manifest_path_text = str(environment.get("AURA_LAUNCH_MANIFEST_PATH") or "").strip()
    executable_text = str(environment.get("AURA_LAUNCH_APP_EXECUTABLE") or "").strip()
    expected_root = str(environment.get("AURA_LAUNCH_EXPECTED_ROOT") or "").strip()
    expected_commit = str(environment.get("AURA_LAUNCH_EXPECTED_COMMIT") or "").strip().lower()
    expected_branch = str(environment.get("AURA_LAUNCH_EXPECTED_BRANCH") or "").strip()
    expected_workspace = (
        str(environment.get("AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256") or "").strip().lower()
    )
    expected_bundle_id = str(environment.get("AURA_LAUNCH_BUNDLE_ID") or "").strip()

    required_values = {
        "manifest_path": manifest_path_text,
        "app_executable": executable_text,
        "expected_root": expected_root,
        "expected_commit": expected_commit,
        "expected_branch": expected_branch,
        "expected_workspace_sha256": expected_workspace,
        "bundle_identifier": expected_bundle_id,
    }
    for name, value in required_values.items():
        if not value:
            issues.append(f"missing_{name}")

    manifest: dict[str, Any] = {}
    manifest_path = Path(manifest_path_text).expanduser() if manifest_path_text else None
    executable = Path(executable_text).expanduser() if executable_text else None
    if manifest_path is not None:
        try:
            manifest = _load_manifest(manifest_path)
        except _RECOVERABLE_ERRORS as exc:
            issues.append(f"manifest_unreadable:{type(exc).__name__}")
    if manifest and manifest.get("schema") != LAUNCH_PROVENANCE_SCHEMA:
        issues.append("manifest_schema_mismatch")
    if executable is not None and manifest_path is not None:
        if not executable.is_file():
            issues.append("app_executable_missing")
        if not _manifest_belongs_to_executable(manifest_path, executable):
            issues.append("manifest_outside_app_bundle")

    actual: dict[str, Any] = {}
    try:
        identity = _git_identity(canonical_input)
        actual.update(identity)
        actual.update(
            _workspace_state(Path(identity["source_root"]), commit_sha=identity["commit_sha"])
        )
    except _RECOVERABLE_ERRORS as exc:
        issues.append(f"source_identity_unavailable:{type(exc).__name__}")

    # IDENTITY — which checkout this bundle belongs to. These cannot drift
    # through ordinary work: a mismatch means the app is pointed at a different
    # or moved workspace, which is exactly the condition that must never be
    # allowed to run cleanup or process replacement. Hard failure.
    identity_comparisons = {
        "source_root": (
            str(Path(expected_root).expanduser().resolve()) if expected_root else "",
            actual.get("source_root"),
        ),
        "bundle_identifier": (expected_bundle_id, EXPECTED_BUNDLE_ID),
    }
    for field, (expected, observed) in identity_comparisons.items():
        if expected and str(expected) != str(observed or ""):
            issues.append(f"{field}_mismatch")
        manifest_value = manifest.get(field) if manifest else None
        if expected and str(expected) != str(manifest_value or ""):
            issues.append(f"manifest_{field}_mismatch")

    # FRESHNESS — how far the workspace has moved since the bundle was built.
    # These are MEASURED here, live, and reported; they are not a verdict.
    #
    # They used to be identity: the manifest pinned commit_sha and
    # workspace_state_sha256 at build time and any difference failed the whole
    # check, so every commit and every edit made the installed app "unverified"
    # and launch_aura.sh refused to start with "Rebuild the installed app".
    # Aura commits to her own repository, so that made staleness the normal
    # state and a human rebuild the only exit — the software could not keep
    # itself current by construction.
    #
    # Baking a snapshot of mutable state was never what made the check safe.
    # The safety property is "this bundle belongs to this checkout", which
    # source_root and the code signature carry. What the commit hash adds is a
    # freshness FACT, and a fact measured at launch is strictly more accurate
    # than one copied from build time — the recorded commit now always
    # describes the code that is actually running.
    drift: list[str] = []
    freshness_comparisons = {
        "commit_sha": (expected_commit, actual.get("commit_sha")),
        "branch": (expected_branch, actual.get("branch")),
        "workspace_state_sha256": (expected_workspace, actual.get("workspace_state_sha256")),
    }
    for field, (expected, observed) in freshness_comparisons.items():
        if expected and str(expected) != str(observed or ""):
            drift.append(field)

    # Identity findings decide whether this signed bundle belongs to this
    # checkout. Freshness findings are still issues a health consumer must be
    # able to see, but they are non-blocking because Aura.app intentionally
    # launches live source that can advance after the binary was signed.
    source_verified = not issues
    freshness_issues = [f"source_revision_drift:{field}" for field in drift]
    source_current = source_verified and not drift
    return {
        "schema": LAUNCH_PROVENANCE_SCHEMA,
        "required": True,
        "launch_mode": "signed_app",
        "source_verified": source_verified,
        "verification_scope": "bundle_identity",
        # True when the bundle was built from exactly this workspace state.
        # Informational: the workspace moving on is normal, not a fault.
        "source_current": source_current,
        "freshness_status": (
            "current" if source_current else "drifted" if source_verified else "unverified"
        ),
        "source_drift": sorted(set(drift)),
        "verified": False,
        "issues": sorted(set(issues + freshness_issues)),
        "manifest_path": str(manifest_path or ""),
        "app_executable": str(executable or ""),
        "expected": {
            "source_root": expected_root,
            "commit_sha": expected_commit,
            "branch": expected_branch,
            "workspace_state_sha256": expected_workspace,
            "bundle_identifier": expected_bundle_id,
        },
        "actual": actual,
        "manifest": manifest,
    }


def _strict_bundle_verification(executable: Path) -> dict[str, Any]:
    bundle = _bundle_for_executable(executable)
    if bundle is None or not bundle.is_dir():
        return {"ok": False, "reason": "app_bundle_missing", "bundle_path": ""}
    revision_material: list[tuple[int, int]] = []
    for path in (
        executable,
        bundle / "Contents" / "Info.plist",
        bundle / "Contents" / "_CodeSignature" / "CodeResources",
        bundle / "Contents" / "Resources" / "aura-launch-provenance.json",
    ):
        try:
            stat = path.stat()
            revision_material.append((int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            revision_material.append((0, 0))
    revision = hash(tuple(revision_material))
    key = str(bundle.resolve(strict=False))
    with _BUNDLE_CACHE_LOCK:
        cached = _BUNDLE_CACHE.get(key)
        if cached is not None and cached[0] == revision:
            return dict(cached[1])
    try:
        completed = get_subprocess_gateway().run(
            ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(bundle)],
            timeout=3.0,
            read_only=True,
            capture_output=True,
            source="runtime_launch_provenance.codesign_verify",
            accelerator_capability="none",
        )
        result = {
            "ok": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "bundle_path": str(bundle),
            "detail": str(completed.stderr or completed.stdout or "").strip()[:500],
        }
    except _RECOVERABLE_ERRORS as exc:
        result = {
            "ok": False,
            "bundle_path": str(bundle),
            "reason": f"{type(exc).__name__}: {exc}",
        }
    with _BUNDLE_CACHE_LOCK:
        _BUNDLE_CACHE[key] = (revision, dict(result))
    return result


def collect_runtime_launch_provenance(
    root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = validate_launch_source(root, env=env)
    if not source.get("required"):
        return source

    executable_text = str(source.get("app_executable") or "").strip()
    executable = (
        Path(executable_text).expanduser() if executable_text else Path("/__aura_missing_app__")
    )
    try:
        from core.security.native_desktop_bridge import native_desktop_bridge_identity

        native_identity = native_desktop_bridge_identity(executable=executable)
    except _RECOVERABLE_ERRORS as exc:
        native_identity = {
            "resident_running": False,
            "code_signature": {"available": False},
            "error": f"{type(exc).__name__}: {exc}",
        }
    strict_verification = _strict_bundle_verification(executable)
    signature = native_identity.get("code_signature", {})
    if not isinstance(signature, dict):
        signature = {}
    signature_valid = bool(
        signature.get("available")
        and signature.get("stable_tcc_identity")
        and signature.get("identifier") == EXPECTED_BUNDLE_ID
    )
    resident_running = bool(native_identity.get("resident_running"))
    issues = list(source.get("issues", []))
    if not resident_running:
        issues.append("resident_app_not_running")
    if not signature_valid:
        issues.append("app_signature_unverified")
    if not strict_verification.get("ok"):
        issues.append("strict_bundle_verification_failed")
    verified = bool(
        source.get("source_verified")
        and resident_running
        and signature_valid
        and strict_verification.get("ok")
    )
    return {
        **source,
        "verified": verified,
        "issues": sorted(set(str(issue) for issue in issues if str(issue))),
        "resident_bridge": native_identity,
        "strict_bundle_verification": strict_verification,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aura launch provenance tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser("emit", help="write an Aura.app source manifest")
    emit.add_argument("--root", required=True)
    emit.add_argument("--output", required=True)
    emit.add_argument("--version", required=True)
    emit.add_argument("--launcher-source", required=True)

    preflight = subparsers.add_parser("preflight", help="verify app-pinned source before cleanup")
    preflight.add_argument("--root", required=True)

    inspect = subparsers.add_parser("inspect", help="print the current runtime launch evidence")
    inspect.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "emit":
        manifest = write_launch_manifest(
            args.output,
            root=args.root,
            version=args.version,
            launcher_source=args.launcher_source,
        )
        print(json.dumps(manifest, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = collect_runtime_launch_provenance(args.root)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("verified") else 2
    result = collect_runtime_launch_provenance(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (not result.get("required") or result.get("verified")) else 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXPECTED_BUNDLE_ID",
    "LAUNCH_PROVENANCE_SCHEMA",
    "RUNTIME_SHELL_ASSETS",
    "RUNTIME_SHELL_PUBLIC_ASSETS",
    "build_launch_manifest",
    "capture_runtime_shell_assets",
    "collect_source_identity",
    "collect_runtime_launch_provenance",
    "runtime_shell_assets_digest",
    "runtime_shell_assets_sha256",
    "runtime_shell_request_path",
    "validate_launch_source",
    "write_launch_manifest",
]
