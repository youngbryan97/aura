#!/usr/bin/env python3
"""Decide whether the frozen bundle may still launch, without loading anything.

A bundle is a promise about what will run. Between freezing it and starting the
campaign, the source can move, the active checkpoint can be swapped again, or
the window can stop matching the model. Each of those makes the campaign a
different experiment from the one that was reviewed, and none of them announces
itself: the run simply proceeds and produces numbers about something else.

This is the gate that turns each into a refusal, on CPU, in about a second.
Run it immediately before handing the bundle to the campaign lifecycle owner,
and again on every resume -- a resume after a day of unrelated development is
exactly when the source closure has moved underneath the journal.

Exit code is non-zero on any drift, and every finding names the specific thing
that changed rather than reporting the bundle as generically stale.

    python tools/preflight_27b_recurrence_campaign.py BUNDLE
    python tools/preflight_27b_recurrence_campaign.py BUNDLE --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.learning.hybrid_recurrence_geometry import (  # noqa: E402
    LayerGeometry,
    window_alignment_errors,
)
from tools.prepare_27b_recurrence_campaign import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    _sha,
    _sha_file,
)

#: Headroom over the checkpoint's own bytes, for KV, optimizer state, and the
#: host's own working set. Measured against CP566's residency rather than
#: chosen: a 15 GB checkpoint ran a 300-decode campaign on this host with the
#: desktop up, and the campaign needs room for training state on top.
_RAM_HEADROOM_BYTES = 12 * 1024**3

#: Export, adapters, journals and receipts for one campaign.
_DISK_HEADROOM_BYTES = 20 * 1024**3


def _resource_findings(bundle: dict[str, Any], model: Path) -> list[dict[str, str]]:
    """Refuse a launch the host cannot finish.

    An out-of-memory kill mid-training loses the residency and leaves a partial
    journal that a resume must then decide about. Refusing up front is cheaper
    than deciding whether half a campaign is scientifically resumable.
    """
    findings: list[dict[str, str]] = []
    weight_bytes = int(bundle.get("target_checkpoint", {}).get("weight_bytes") or 0)
    required = weight_bytes + _RAM_HEADROOM_BYTES

    try:
        from core.runtime.mlx_memory_guard import host_pressure

        pressure = host_pressure()
    except (ImportError, OSError, RuntimeError, ValueError):
        pressure = {}
    # host_pressure reports gigabytes and separates free from reclaimable,
    # because macOS "Pages free" excludes pages the kernel hands back on
    # demand. Both count toward what a load can actually use.
    free_gb = pressure.get("free_gb")
    reclaimable_gb = pressure.get("reclaimable_gb")
    if isinstance(free_gb, (int, float)) and isinstance(reclaimable_gb, (int, float)):
        available = (float(free_gb) + float(reclaimable_gb)) * 1024**3
        if available < required:
            findings.append(
                {
                    "kind": "insufficient_ram",
                    "detail": (
                        f"{available / 1024**3:.1f} GiB free plus reclaimable, "
                        f"{required / 1024**3:.1f} GiB needed"
                    ),
                }
            )
        elif pressure.get("under_pressure"):
            findings.append(
                {
                    "kind": "host_under_memory_pressure",
                    "detail": (
                        "reasons: "
                        + ", ".join(str(r) for r in pressure.get("pressure_reasons", []))
                    ),
                }
            )
    else:
        findings.append(
            {
                "kind": "ram_unmeasured",
                "detail": "host pressure could not be read; refusing rather than guessing",
            }
        )

    # Through the observer, like every other resource reading in the tree. A
    # direct shutil.disk_usage answers with no provenance and no record that
    # the reading happened, which is what the ownership audit is for.
    from core.runtime.resource_observation import get_resource_observer

    usage = get_resource_observer().disk(
        model.parent if model.exists() else Path.home()
    )
    if not usage.available:
        findings.append(
            {"kind": "disk_unmeasured", "detail": usage.error or "disk unreadable"}
        )
        return findings
    if usage.free_bytes < _DISK_HEADROOM_BYTES:
        findings.append(
            {
                "kind": "insufficient_disk",
                "detail": (
                    f"{usage.free_bytes / 1024**3:.1f} GiB free, "
                    f"{_DISK_HEADROOM_BYTES / 1024**3:.0f} GiB needed for export"
                ),
            }
        )
    return findings


def _ownership_findings() -> list[dict[str, str]]:
    """Refuse while another process owns the model lane.

    Two campaigns on one 64 GB host is the failure that takes the machine down,
    and the lane controller already knows who holds it.
    """
    try:
        from core.runtime.model_lane_control import get_model_lane_controller

        controller = get_model_lane_controller()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return [
            {
                "kind": "lane_ownership_unmeasured",
                "detail": f"could not read the model lane controller: {exc}",
            }
        ]
    try:
        owners = list(controller.observed_owners())
    except (AttributeError, OSError, RuntimeError, ValueError):
        owners = []
    live = [
        owner
        for owner in owners
        if getattr(owner, "alive", True) and getattr(owner, "owner_id", None)
    ]
    return [
        {
            "kind": "model_lane_already_owned",
            "detail": f"{getattr(owner, 'owner_id', 'unknown')} holds the lane",
        }
        for owner in live
    ]


def _evidence_namespace_findings(bundle: dict[str, Any]) -> list[dict[str, str]]:
    """Refuse a campaign aimed at the namespace holding the old verdict."""
    try:
        from core.learning.recovery_package_identity import (
            evidence_namespace_errors,
        )
    except ImportError:
        return []
    paths = list(bundle.get("evidence_paths") or [])
    if not paths:
        return []
    return [
        {"kind": "stale_evidence_root", "detail": error}
        for error in evidence_namespace_errors(paths)
    ]


def check(bundle: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def fail(kind: str, detail: str) -> None:
        findings.append({"kind": kind, "detail": detail})

    if bundle.get("schema") != CAMPAIGN_SCHEMA:
        fail("schema_unrecognised", str(bundle.get("schema")))
        return findings

    body = {key: value for key, value in bundle.items() if key != "campaign_sha256"}
    if _sha(body) != bundle.get("campaign_sha256"):
        fail("bundle_tampered", "campaign_sha256 does not cover the bundle")
        return findings

    for relative, expected in sorted(bundle.get("source_freeze", {}).items()):
        path = REPO_ROOT / relative
        if not path.exists():
            fail("source_missing", relative)
        elif _sha_file(path) != expected:
            fail("source_drifted", relative)

    for relative, expected in sorted(bundle.get("portable_tissue", {}).items()):
        path = REPO_ROOT / relative
        if not path.exists():
            fail("tissue_missing", relative)
        elif _sha_file(path) != expected:
            fail("tissue_drifted", relative)

    descriptor = bundle.get("target_checkpoint", {})
    model = Path(str(descriptor.get("path", "")))
    config_path = model / "config.json"
    if not config_path.exists():
        fail("checkpoint_absent", str(model))
        return findings
    if _sha_file(config_path) != descriptor.get("config_sha256"):
        fail("checkpoint_config_drifted", str(config_path))
    for name, key in (
        ("model.safetensors.index.json", "weights_index_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("aura_fusion_provenance.json", "fusion_provenance_sha256"),
    ):
        expected = descriptor.get(key)
        path = model / name
        if expected is None:
            continue
        if not path.exists():
            fail("checkpoint_file_absent", name)
        elif _sha_file(path) != expected:
            fail("checkpoint_file_drifted", name)

    # The campaign is defined against one checkpoint. If the runtime has since
    # been pointed at another, launching would measure the wrong model and the
    # receipts would name the right one.
    manifest_path = _install_root() / "training/fused-model/active.json"
    if manifest_path.exists():
        try:
            active = json.loads(manifest_path.read_text()).get("active_model_path")
        except (OSError, ValueError):
            active = None
        if active and Path(active).resolve() != model.resolve():
            fail("active_model_moved", f"runtime now points at {active}")

    findings.extend(_resource_findings(bundle, model))
    findings.extend(_ownership_findings())
    findings.extend(_evidence_namespace_findings(bundle))

    geometry = LayerGeometry.from_config(json.loads(config_path.read_text()))
    committed = bundle.get("recurrence_layer_mapping", {})
    if list(geometry.attention_layers()) != committed.get("attention_layer_indices"):
        fail(
            "attention_layout_changed",
            "the checkpoint's attention layers are not the ones committed",
        )
    window = committed.get("window") or [0, 0]
    alignment = window_alignment_errors(geometry, int(window[0]), int(window[1]))
    for error in alignment:
        fail("window_misaligned", error)

    return findings


def _install_root() -> Path:
    if (REPO_ROOT / "training/fused-model/active.json").exists():
        return REPO_ROOT
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return REPO_ROOT
    return Path(common).parent if common else REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    bundle = json.loads(args.bundle.read_text())
    findings = check(bundle)
    report = {
        "schema": "aura.rlc.27b_campaign_preflight.v1",
        "bundle": str(args.bundle),
        "campaign_sha256": bundle.get("campaign_sha256"),
        "may_launch": not findings,
        "findings": findings,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True))

    if findings:
        for finding in findings:
            print(f"  {finding['kind']:28s} {finding['detail']}")
        print(f"\n{len(findings)} finding(s); the bundle may not launch as frozen.")
        return 1
    print(
        f"OK: {len(bundle.get('source_freeze', {}))} source files, "
        f"{len(bundle.get('portable_tissue', {}))} tissues, and the checkpoint "
        "all match the frozen bundle."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
