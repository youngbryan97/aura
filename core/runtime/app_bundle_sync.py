"""Keep the installed Aura.app in step with the workspace it launches.

``Aura.app`` is deliberately a thin launcher over live source, so almost
nothing in it can go stale: Python, assets and configuration are read from the
repository at run time. Exactly one artifact is genuinely compiled — the Swift
launcher binary built from ``scripts/AuraLauncher.swift``. When that source
changes, the installed binary is old code, and nothing in the system noticed.

Everything else that USED to require a rebuild was not staleness at all. The
launch manifest pinned ``commit_sha`` and ``workspace_state_sha256`` at build
time and the launch path refused to start on any difference. Aura commits to
her own repository, so "the workspace has moved" is her steady state, and the
only exit was a human running ``build_app.sh`` again. See
:mod:`core.runtime.launch_provenance`, where those fields became measured
facts rather than a verdict.

What remains is this: the launcher binary must track its source, and Aura must
do that for herself rather than depend on anyone noticing. The rule this module
enforces is that a launcher binary is never quietly older than the source it
was built from.

Replacing a bundle that a launcher is currently executing from is the one thing
this will not do. When the resident app is running, the rebuilt bundle is left
staged in ``dist/`` and installed at the next opportunity — the same shape as
any other application auto-update, where the new version takes effect on the
next start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

#: Where a built bundle is installed when it is not the resident app.
DEFAULT_RESIDENT_PATH = Path("/Applications/Aura.app")

#: How long a launcher rebuild may take before it is abandoned. A compile that
#: overruns must never hold up a boot.
BUILD_TIMEOUT_S = 300.0

_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, TypeError, ValueError)


def _manifest_path(bundle: Path) -> Path:
    return bundle / "Contents" / "Resources" / "aura-launch-provenance.json"


def _read_manifest(bundle: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_manifest_path(bundle).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def resident_bundle_path(env: dict[str, str] | None = None) -> Path:
    """The bundle this process was launched from, or the default install path."""
    environment = os.environ if env is None else env
    executable = str(environment.get("AURA_LAUNCH_APP_EXECUTABLE") or "").strip()
    if executable:
        for parent in Path(executable).resolve(strict=False).parents:
            if parent.suffix == ".app":
                return parent
    return DEFAULT_RESIDENT_PATH


def bundle_is_running(bundle: Path) -> bool:
    """Whether a process is currently executing from ``bundle``.

    Used only to decide whether an install is safe right now — never to decide
    whether a rebuild should happen.
    """
    executable = bundle / "Contents" / "MacOS" / "aura-launcher"
    try:
        completed = get_subprocess_gateway().run(
            ["pgrep", "-f", str(executable)],
            timeout=5.0,
            read_only=True,
            capture_output=True,
            source="runtime_app_bundle_sync.pgrep",
            accelerator_capability="none",
        )
    except _RECOVERABLE_ERRORS:
        # Unable to tell means unable to prove it is safe.
        return True
    return completed.returncode == 0 and bool(str(completed.stdout or "").strip())


def launcher_drift(root: Path, bundle: Path) -> dict[str, Any]:
    """Whether ``bundle``'s launcher binary was built from the current source."""
    source = root / "scripts" / "AuraLauncher.swift"
    live_digest = _sha256_file(source)
    manifest = _read_manifest(bundle)
    built_digest = str(manifest.get("launcher_source_sha256") or "").strip().lower()
    binary = bundle / "Contents" / "MacOS" / "aura-launcher"
    return {
        "bundle": str(bundle),
        "bundle_present": bundle.is_dir(),
        "launcher_binary_present": binary.is_file(),
        "launcher_source_sha256": live_digest,
        "built_from_sha256": built_digest,
        # No recorded digest means the bundle predates the field: rebuild, since
        # "cannot tell" is not "current".
        "stale": bool(
            bundle.is_dir()
            and live_digest
            and (not built_digest or built_digest != live_digest)
        ),
    }


def launcher_currency() -> dict[str, Any]:
    """Whether the compiled launcher matches its source. Read-only, never raises.

    ``launcher_drift`` needs a root and a bundle and is the thing that decides
    whether to compile. This is the observable half: something any health
    surface can call without knowing where the repo is and without the risk of
    starting a build.

    It exists because the detector could not fire on the path users take.
    ``sync_app_bundle`` is invoked from ``launch_aura.sh``; the normal launch
    is ``Aura.app`` -> ``spawnAuraProcess`` -> ``aura_main.py``, which never
    sources that script. So the installed launcher went six days stale with
    the whole companion-mode surface missing from the binary, and no surface
    anywhere said a word. A condition nothing reports is a condition nobody
    can act on.
    """
    report: dict[str, Any] = {"schema": "aura.launcher_currency.v1"}
    try:
        root = Path(__file__).resolve().parents[2]
        drift = launcher_drift(root, resident_bundle_path())
        # A rebuild that is staged and never installed is the same silence in
        # a later place. install_staged_bundle refuses while the resident app
        # is running — correctly, because replacing a bundle underneath the
        # process executing from it is how a signed app loses its TCC identity
        # — and the runtime only exists WHILE it is running. So on the
        # double-click path the staged build waits for a launch_aura.sh run
        # that may never come. Say so, and say what clears it.
        staged_digest = str(
            _read_manifest(root / "dist" / "Aura.app").get("launcher_source_sha256") or ""
        ).strip().lower()
        pending = bool(
            staged_digest
            and staged_digest != drift["built_from_sha256"]
            and staged_digest == drift["launcher_source_sha256"]
        )
        report.update(
            {
                "bundle": drift["bundle"],
                "bundle_present": drift["bundle_present"],
                "stale": drift["stale"],
                "built_from_sha256": drift["built_from_sha256"][:12],
                "launcher_source_sha256": drift["launcher_source_sha256"][:12],
                "staged_install_pending": pending,
                # Deliberately the exact command rather than "relaunch".
                # Relaunching by double-click starts the OLD launcher, which
                # has no staged-install step, so the staged build would sit
                # there forever and this field would be a false promise —
                # which is the same defect as the silence it replaces.
                "clears_by": (
                    "with Aura quit, run: python -m core.runtime.app_bundle_sync "
                    "--root . --install-staged   (launch_aura.sh does this "
                    "automatically; double-clicking Aura.app does not)"
                )
                if pending
                else "",
                # The consequence, stated in the terms a reader cares about:
                # a stale launcher is not a cosmetic lag, it is UI and
                # behaviour that the running executable does not contain.
                "consequence": (
                    "installed launcher predates its source; features added to "
                    "scripts/AuraLauncher.swift are absent from the running app "
                    "until it is rebuilt"
                )
                if drift["stale"]
                else "",
            }
        )
    except _RECOVERABLE_ERRORS as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        # Unable to tell is not the same as current, and must not read as it.
        report["stale"] = None
    return report


def sync_app_bundle(
    root: str | Path,
    *,
    resident: str | Path | None = None,
    allow_install: bool | None = None,
) -> dict[str, Any]:
    """Rebuild and install the launcher when its source has moved on.

    Returns a receipt describing what was found and what was done. Never
    raises: this runs on the boot path, and a launcher that could not be
    refreshed must not prevent Aura from starting.
    """
    started = time.monotonic()
    root_path = Path(root).expanduser().resolve()
    bundle = Path(resident).expanduser() if resident else resident_bundle_path()
    receipt: dict[str, Any] = {
        "schema": "aura.app_bundle_sync.v1",
        "root": str(root_path),
        "action": "none",
        "installed": False,
        "at": time.time(),
    }

    try:
        drift = launcher_drift(root_path, bundle)
        receipt.update(drift)
        if not drift["bundle_present"]:
            receipt["action"] = "skipped"
            receipt["reason"] = "no installed bundle to keep current"
            return receipt
        if not drift["stale"]:
            receipt["action"] = "current"
            receipt["reason"] = "launcher binary matches its source"
            return receipt

        build_script = root_path / "scripts" / "bundle_app.sh"
        if not build_script.is_file():
            receipt["action"] = "unavailable"
            receipt["reason"] = "scripts/bundle_app.sh is missing"
            return receipt

        running = bundle_is_running(bundle)
        install_here = (not running) if allow_install is None else bool(allow_install)
        environment = dict(os.environ)
        if install_here:
            environment["AURA_INSTALL_PATH"] = str(bundle)
        else:
            environment.pop("AURA_INSTALL_PATH", None)

        with local_internal_governed_scope(
            "runtime_app_bundle_sync.build",
            domain="self_modification",
            constraints={
                "root": str(root_path),
                "bundle": str(bundle),
                "install_requested": install_here,
            },
        ):
            completed = get_subprocess_gateway().run(
                [str(build_script)],
                cwd=str(root_path),
                timeout=BUILD_TIMEOUT_S,
                capture_output=True,
                env=environment,
                source="runtime_app_bundle_sync.build",
                accelerator_capability="none",
            )
        receipt["build_returncode"] = completed.returncode
        if completed.returncode != 0:
            receipt["action"] = "failed"
            # Build output can name paths and identities; keep the tail only.
            receipt["reason"] = str(completed.stderr or completed.stdout or "")[-400:]
            record_degradation(
                "app_bundle_sync",
                RuntimeError("launcher rebuild failed"),
                action="left the previous launcher binary in place",
                extra={"returncode": completed.returncode},
                enforce_failure_policy=False,
            )
            return receipt

        if install_here:
            receipt["action"] = "rebuilt_and_installed"
            receipt["installed"] = True
        else:
            # Staged: the resident launcher is executing from this bundle right
            # now, so it is replaced at the next start rather than underneath a
            # running process.
            receipt["action"] = "rebuilt_staged"
            receipt["staged_at"] = str(root_path / "dist" / "Aura.app")
            receipt["reason"] = "resident app is running; install deferred to next launch"
        return receipt
    except _RECOVERABLE_ERRORS as exc:
        receipt["action"] = "failed"
        receipt["reason"] = f"{type(exc).__name__}: {exc}"
        record_degradation(
            "app_bundle_sync",
            exc,
            action="could not evaluate or refresh the installed launcher",
            enforce_failure_policy=False,
        )
        return receipt
    finally:
        receipt["duration_s"] = round(time.monotonic() - started, 3)


async def keep_launcher_current(root: str | Path | None = None) -> dict[str, Any]:
    """Install a staged launcher, then stage a rebuild if the source moved.

    The runtime-side entry point, and the reason it exists: ``sync_app_bundle``
    was called only from ``launch_aura.sh``, and the ordinary launch path does
    not run it. ``Aura.app`` spawns ``aura_main.py`` directly and reaches the
    shell script only through ``requiresProtectedFolderFallback()``. So the
    one artifact that genuinely goes stale had a repair that the common path
    could not reach, and the installed launcher sat six days behind its source
    with a whole feature missing from the binary.

    Calling it from the runtime fixes that by construction: every launch path
    ends at this process, whatever started it.

    Both halves run in a worker thread. ``sync_app_bundle`` shells out to a
    Swift compile with a five-minute ceiling, and a boot must never wait on a
    compiler.
    """
    import asyncio

    receipt: dict[str, Any] = {"schema": "aura.launcher_currency_task.v1"}
    if os.environ.get("AURA_TESTING"):
        # A test run must not compile or replace anything on the host.
        receipt["action"] = "skipped"
        receipt["reason"] = "AURA_TESTING"
        return receipt

    root_path = await asyncio.to_thread(
        lambda: (
            Path(root).expanduser().resolve()
            if root
            else Path(__file__).resolve().parents[2]
        )
    )
    try:
        # Cheap and read-only. Doing this first means the overwhelmingly
        # common case — the launcher is current — costs one hash and no
        # thread, rather than a compile that exits immediately.
        drift = await asyncio.to_thread(
            launcher_drift, root_path, resident_bundle_path()
        )
        if not drift["bundle_present"]:
            receipt["action"] = "skipped"
            receipt["reason"] = "no installed bundle"
            return receipt
        receipt["installed_staged"] = await asyncio.to_thread(
            install_staged_bundle, root_path
        )
        if not drift["stale"]:
            receipt["action"] = "current"
            return receipt
        receipt["sync"] = await asyncio.to_thread(sync_app_bundle, root_path)
        receipt["action"] = str(receipt["sync"].get("action") or "unknown")
        return receipt
    except _RECOVERABLE_ERRORS as exc:
        receipt["action"] = "failed"
        receipt["reason"] = f"{type(exc).__name__}: {exc}"
        record_degradation(
            "app_bundle_sync",
            exc,
            action="installed launcher may be older than its source",
            enforce_failure_policy=False,
        )
        return receipt


def install_staged_bundle(
    root: str | Path, *, resident: str | Path | None = None
) -> dict[str, Any]:
    """Install a previously staged ``dist/Aura.app`` when it is safe to do so.

    Called at the start of a launch, before the resident bundle matters, so an
    update built during the previous session takes effect now.
    """
    root_path = Path(root).expanduser().resolve()
    staged = root_path / "dist" / "Aura.app"
    bundle = Path(resident).expanduser() if resident else resident_bundle_path()
    receipt: dict[str, Any] = {
        "schema": "aura.app_bundle_install.v1",
        "staged": str(staged),
        "resident": str(bundle),
        "installed": False,
    }
    try:
        if not staged.is_dir() or not bundle.is_dir():
            receipt["reason"] = "nothing staged"
            return receipt
        staged_digest = str(_read_manifest(staged).get("launcher_source_sha256") or "")
        resident_digest = str(_read_manifest(bundle).get("launcher_source_sha256") or "")
        if not staged_digest or staged_digest == resident_digest:
            receipt["reason"] = "resident launcher already matches the staged build"
            return receipt
        if bundle_is_running(bundle):
            receipt["reason"] = "resident app is running; not replacing it underneath"
            return receipt
        with local_internal_governed_scope(
            "runtime_app_bundle_sync.install_staged",
            domain="self_modification",
            constraints={
                "staged": str(staged),
                "bundle": str(bundle),
                "source_digest": staged_digest,
                "replaced_digest": resident_digest,
            },
        ):
            shutil.rmtree(bundle)
            shutil.copytree(staged, bundle, symlinks=True)
        receipt["installed"] = True
        return receipt
    except _RECOVERABLE_ERRORS as exc:
        receipt["reason"] = f"{type(exc).__name__}: {exc}"
        record_degradation(
            "app_bundle_sync",
            exc,
            action="could not install the staged launcher bundle",
            enforce_failure_policy=False,
        )
        return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keep Aura.app's launcher current")
    parser.add_argument("--root", required=True)
    parser.add_argument("--resident", default=None)
    parser.add_argument(
        "--install-staged",
        action="store_true",
        help="install a bundle staged by a previous run, when safe",
    )
    args = parser.parse_args(argv)
    if args.install_staged:
        receipt = install_staged_bundle(args.root, resident=args.resident)
    else:
        receipt = sync_app_bundle(args.root, resident=args.resident)
    print(json.dumps(receipt, sort_keys=True))
    # A launcher that could not be refreshed is not a boot failure.
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "bundle_is_running",
    "install_staged_bundle",
    "launcher_drift",
    "resident_bundle_path",
    "sync_app_bundle",
]
