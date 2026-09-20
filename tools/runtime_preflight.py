"""Runtime-environment preflight: will the desktop app open cleanly right now?

`make doctor` proves the CODE is healthy; this proves the RUNTIME
ENVIRONMENT is — disk headroom, RAM, port state, model artifacts, .env
secret presence, lockfile drift, log-sink writability. The distinction
matters because every historical "won't boot" incident that wasn't a code
defect was one of these (missing .env, no disk, stale model path, port
squatter).

Deliberately dependency-light at import time: the canonical resource observer
is loaded only when resource checks run. A preflight that cannot report an
environment failure when the environment is broken is useless — that is
exactly when it is needed.

Fast (<5s), offline, loads no models, never prints secret values.
Exit 0 = ready (WARNs allowed); exit 1 = at least one FAIL;
--strict promotes WARN to FAIL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OK = "OK"
INFO = "INFO"
WARN = "WARN"
FAIL = "FAIL"

_STATUS_ICON = {OK: "✅", INFO: "ℹ️ ", WARN: "⚠️ ", FAIL: "❌"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _resolve_resource_observer(observer: Any | None = None) -> tuple[Any | None, str]:
    if observer is not None:
        return observer, ""
    try:
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from core.runtime.resource_observation import get_resource_observer

        return get_resource_observer(), ""
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _run_read_only_command(
    argv: list[str],
    *,
    root: Path,
    source: str,
) -> tuple[Any | None, str]:
    try:
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        completed = get_subprocess_gateway().run(
            argv,
            cwd=root,
            timeout=10,
            read_only=True,
            source=source,
            accelerator_capability="auto",
        )
        return completed, ""
    except (
        AttributeError,
        ImportError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def _observation_source(observation: Any) -> str:
    provenance = getattr(observation, "provenance", None)
    source = getattr(provenance, "source", "unknown")
    return str(getattr(source, "value", source) or "unknown")


def check_python(expected: tuple[int, int] = (3, 12)) -> Check:
    v = sys.version_info
    if (v.major, v.minor) == expected:
        return Check("python", OK, f"{v.major}.{v.minor}.{v.micro} at {sys.executable}")
    return Check(
        "python",
        FAIL,
        f"{v.major}.{v.minor} found, {expected[0]}.{expected[1]} required "
        "(native extensions: mlx, grpc)",
    )


def check_disk(
    path: Path,
    fail_gb: float = 10.0,
    warn_gb: float = 30.0,
    usage_fn: Callable[[Path], Any] | None = None,
    *,
    observer: Any | None = None,
) -> Check:
    if usage_fn is not None:
        try:
            free_bytes = int(usage_fn(path).free)
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            return Check("disk", FAIL, f"cannot stat {path}: {exc}")
        source = "injected"
    else:
        resource_observer, error = _resolve_resource_observer(observer)
        if resource_observer is None:
            return Check("disk", FAIL, f"resource observer unavailable: {error}")
        try:
            observation = resource_observer.disk(str(path))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return Check("disk", FAIL, f"disk observation failed: {type(exc).__name__}:{exc}")
        if not bool(getattr(observation, "available", False)):
            detail = str(getattr(observation, "error", "") or "unknown")
            return Check("disk", FAIL, f"disk observation unavailable: {detail}")
        free_bytes = int(getattr(observation, "free_bytes", 0) or 0)
        source = _observation_source(observation)

    free_gb = free_bytes / 1e9
    detail = f"{free_gb:.1f}GB free on volume of {path} (source={source})"
    if free_gb < fail_gb:
        return Check("disk", FAIL, detail + f" (< {fail_gb:.0f}GB floor — model loads and "
                     "SQLite WAL writes will fail)")
    if free_gb < warn_gb:
        return Check("disk", WARN, detail + f" (< {warn_gb:.0f}GB comfort threshold)")
    return Check("disk", OK, detail)


def check_ram(
    warn_available_gb: float = 8.0,
    *,
    observer: Any | None = None,
) -> Check:
    resource_observer, error = _resolve_resource_observer(observer)
    if resource_observer is None:
        return Check("ram", FAIL, f"resource observer unavailable: {error}")
    try:
        observation = resource_observer.memory()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return Check("ram", FAIL, f"memory observation failed: {type(exc).__name__}:{exc}")
    if not bool(getattr(observation, "available", False)):
        detail = str(getattr(observation, "error", "") or "unknown")
        return Check("ram", FAIL, f"memory observation unavailable: {detail}")

    available_gb = int(getattr(observation, "available_bytes", 0) or 0) / 1e9
    total_gb = int(getattr(observation, "total_bytes", 0) or 0) / 1e9
    source = _observation_source(observation)
    detail = f"{available_gb:.1f}GB available of {total_gb:.0f}GB (source={source})"
    if available_gb < warn_available_gb:
        return Check(
            "ram",
            WARN,
            detail + " — a cold 32B Cortex load wants ~24GB; boot will degrade to "
            "smaller lanes until memory frees",
        )
    return Check("ram", OK, detail)


def check_port(port: int = 8000, host: str = "127.0.0.1") -> Check:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            pass
    except (ConnectionRefusedError, TimeoutError, OSError):
        return Check("port", OK, f":{port} free")
    return Check(
        "port",
        INFO,
        f":{port} already serving — a live Aura instance is likely running "
        "(launcher cleanup phase handles handover)",
    )


def check_env_file(root: Path) -> Check:
    env_file = root / ".env"
    if not env_file.exists():
        return Check(
            "env-file",
            WARN,
            f"no .env at {env_file} — GUI and server share AURA_API_TOKEN via this "
            f"file; recovery ring: ~/.aura/backups/env/",
        )
    try:
        content = env_file.read_text()
    except OSError as exc:
        return Check("env-file", FAIL, f".env unreadable: {exc}")
    if not re.search(r"^AURA_API_TOKEN=.+", content, flags=re.MULTILINE):
        return Check(
            "env-file",
            WARN,
            ".env present but AURA_API_TOKEN missing — desktop GUI may fail to "
            "authenticate",
        )
    return Check("env-file", OK, ".env present with AURA_API_TOKEN")


def check_models(root: Path) -> Check:
    models_dir = root / "models"
    manifest = root / "training" / "fused-model" / "active.json"
    if not models_dir.is_dir():
        return Check(
            "models",
            FAIL,
            f"{models_dir} missing — no local weights; every cognition lane "
            "would run cloud/reflex-only",
        )
    if not manifest.exists():
        return Check("models", INFO, "models/ present; no fused-model manifest "
                     "(registry defaults apply)")
    try:
        data = json.loads(manifest.read_text())
        active = str(data.get("active_model_path") or "").strip()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return Check("models", WARN, f"fused-model manifest unreadable: {exc} "
                     "(registry falls back to defaults)")
    if not active:
        return Check("models", WARN, "fused-model manifest has no active_model_path "
                     "(registry falls back to defaults)")
    if not Path(active).exists():
        return Check(
            "models",
            WARN,
            f"manifest points at missing artifact {active} — registry falls back "
            "to defaults; a weight promotion may have been lost",
        )
    return Check("models", OK, f"active fused model resolves: {Path(active).name}")


_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^ \\;]+)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_lock_drift(lock_path: Path, max_listed: int = 8) -> Check:
    if not lock_path.exists():
        return Check("lock-drift", INFO, f"no lockfile at {lock_path} — skipped")
    pins: dict[str, str] = {}
    try:
        for line in lock_path.read_text().splitlines():
            m = _PIN_RE.match(line.strip())
            if m:
                pins[_normalize(m.group(1))] = m.group(2)
    except OSError as exc:
        return Check("lock-drift", WARN, f"lockfile unreadable: {exc}")
    if not pins:
        return Check("lock-drift", WARN, f"no pins parsed from {lock_path.name}")

    from importlib import metadata

    installed: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            installed[_normalize(name)] = dist.version

    drift = {
        name: (pin, installed[name])
        for name, pin in pins.items()
        if name in installed and installed[name] != pin
    }
    missing = sorted(name for name in pins if name not in installed)
    if drift:
        sample = ", ".join(
            f"{n} lock={p} installed={i}" for n, (p, i) in sorted(drift.items())[:max_listed]
        )
        more = f" (+{len(drift) - max_listed} more)" if len(drift) > max_listed else ""
        return Check(
            "lock-drift",
            WARN,
            f"{len(drift)} of {len(pins)} pins drifted from installed versions: "
            f"{sample}{more}",
        )
    detail = f"all {len(pins)} lock pins that are installed match"
    if missing:
        detail += f"; {len(missing)} pinned-but-not-installed (optional extras)"
    return Check("lock-drift", OK, detail)


def check_logs_writable(log_dir: Path) -> Check:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        probe = log_dir / ".preflight_probe"
        probe.write_text("")
        probe.unlink()
    except OSError as exc:
        return Check("logs", FAIL, f"log sink {log_dir} not writable: {exc}")
    return Check("logs", OK, f"log sink {log_dir} writable")


def check_git(root: Path) -> Check:
    head, head_error = _run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "--short", "HEAD"],
        root=root,
        source="maintenance_tooling:runtime_preflight.git_head",
    )
    if head is None:
        return Check("git", INFO, f"git probe failed: {head_error}")
    if head.returncode != 0:
        return Check("git", INFO, "not a git checkout — source provenance unavailable")

    dirty, dirty_error = _run_read_only_command(
        ["git", "-c", "core.fsmonitor=false", "status", "--porcelain"],
        root=root,
        source="maintenance_tooling:runtime_preflight.git_status",
    )
    if dirty is None:
        return Check("git", INFO, f"git status probe failed: {dirty_error}")
    n_dirty = len([line for line in dirty.stdout.splitlines() if line.strip()])
    detail = f"commit {head.stdout.strip()}"
    if n_dirty:
        detail += f", {n_dirty} dirty files (running code differs from the commit)"
    return Check("git", INFO, detail)


def run_all(root: Path, port: int = 8000) -> list[Check]:
    log_dir = Path(os.getenv("AURA_LOG_DIR") or Path.home() / ".aura" / "logs")
    return [
        check_python(),
        check_disk(root),
        check_ram(),
        check_port(port),
        check_env_file(root),
        check_models(root),
        check_lock_drift(root / "requirements_lock.txt"),
        check_logs_writable(log_dir),
        check_git(root),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--strict", action="store_true", help="WARN counts as FAIL")
    args = parser.parse_args(argv)

    checks = run_all(args.root, port=args.port)
    failing = {FAIL, WARN} if args.strict else {FAIL}
    verdict = FAIL if any(c.status in failing for c in checks) else OK

    if args.json:
        print(json.dumps({"verdict": verdict, "checks": [asdict(c) for c in checks]},
                         indent=2))
    else:
        for c in checks:
            print(f"  {_STATUS_ICON[c.status]} {c.name:<12} {c.detail}")
        icon = _STATUS_ICON[OK] if verdict == OK else _STATUS_ICON[FAIL]
        print(f"{icon} preflight: {'ready' if verdict == OK else 'NOT ready'}")
    return 0 if verdict == OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
