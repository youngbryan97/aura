"""Default safe skeletal Aura boot harness for ASA ghost runs."""
from __future__ import annotations

import asyncio
import json
import os
import py_compile
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


def main() -> int:
    started = time.monotonic()
    os.environ.setdefault("AURA_ASA_GHOST_BOOT", "1")
    os.environ.setdefault("AURA_SAFE_BOOT_DESKTOP", "1")
    os.environ.setdefault("AURA_DISABLE_LLM_WARMUP", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    repo_root = Path.cwd()
    results: dict[str, Any] = {"repo_root": str(repo_root), "checks": {}}
    py_compile.compile(str(repo_root / "aura_main.py"), doraise=True) if (repo_root / "aura_main.py").exists() else None
    results["checks"]["aura_main_compile"] = True

    from core.config import config

    config.skeletal_mode = True
    if hasattr(config, "soma"):
        config.soma.enabled = False
    config.features.mycelium_visualizer = False
    config.features.autonomous_impulses = False

    from core.container import ServiceContainer
    from core.service_registration import register_all_services

    register_all_services(is_proxy=True)
    results["checks"]["service_registration_proxy"] = True

    import aura_main

    boot_shim = _BootShim()
    aura_main._register_runtime_singletons(boot_shim)
    governor = ServiceContainer.get("architecture_governor", default=None)
    alias = ServiceContainer.get("autonomous_architecture_governor", default=None)
    if governor is None or alias is None or governor is not alias:
        raise RuntimeError("architecture governor singleton/alias unavailable after skeletal boot wiring")
    results["checks"]["architecture_governor_singleton"] = True

    from core.runtime.service_manifest import SERVICE_MANIFEST, verify_manifest

    architecture_role = SERVICE_MANIFEST["architecture_governor"]
    manifest_violations = verify_manifest(
        {
            "architecture_governor": governor,
            "autonomous_architecture_governor": alias,
        },
        manifest={"architecture_governor": architecture_role},
    )
    if manifest_violations:
        raise RuntimeError(f"architecture governor manifest violation: {manifest_violations!r}")
    results["checks"]["architecture_manifest"] = True

    probe_results = asyncio.run(_run_temp_gateway_probes())
    if not all(item["ok"] for item in probe_results):
        raise RuntimeError(f"safe gateway probes failed: {probe_results!r}")
    results["checks"]["gateway_boot_probes"] = probe_results
    results["duration_s"] = round(time.monotonic() - started, 4)
    print(json.dumps(results, sort_keys=True, default=str))
    return 0


async def _run_temp_gateway_probes() -> list[dict[str, Any]]:
    from core.runtime.boot_probes import probe_memory_write_read, probe_state_mutate_read

    with tempfile.TemporaryDirectory(prefix="aura-asa-boot-") as tmp:
        root = Path(tmp)
        memory = await probe_memory_write_read(tmp_root=root / "memory")
        state = await probe_state_mutate_read(tmp_root=root / "state")
    return [asdict(memory), asdict(state)]


class _BootShim:
    output_gate = None


if __name__ == "__main__":
    raise SystemExit(main())
