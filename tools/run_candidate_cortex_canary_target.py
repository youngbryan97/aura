#!/usr/bin/env python3
"""Execute one candidate-cortex canary inside the detached target process."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_training import (  # noqa: E402
    CANARY_HOST_METRICS_SCHEMA,
    CandidateCortexTrainingError,
    build_canary_command,
    canonical_json_bytes,
    document_sha256,
    load_and_verify_plan,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from core.runtime.resource_observation import (  # noqa: E402
    ResourceObserver,
    get_resource_observer,
)


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with local_internal_governed_scope(
        "candidate_cortex_canary.host_metrics", domain="file_write"
    ):
        get_file_write_gateway().write_bytes_if_absent(
            path,
            canonical_json_bytes(metrics) + b"\n",
            mode=0o600,
            source="candidate_cortex_canary.host_metrics",
        )


def _sample_host(
    stop: threading.Event,
    state: dict[str, Any],
    observer: ResourceObserver | None = None,
) -> None:
    resource_observer = observer or get_resource_observer()
    while True:
        memory = resource_observer.memory(
            root_pid=os.getpid(),
            include_process_tree=False,
        )
        state["sample_count"] += 1
        state["min_available_bytes"] = min(
            state["min_available_bytes"],
            int(memory.available_bytes) if memory.available else 0,
        )
        state["max_used_percent"] = max(
            state["max_used_percent"],
            float(memory.percent) if memory.available else 100.0,
        )
        state["max_process_rss_bytes"] = max(
            state["max_process_rss_bytes"],
            int(memory.process_rss_bytes),
        )
        if stop.wait(0.5):
            return


def _mlx_arguments(command: tuple[str, ...]) -> Any:
    from mlx_lm import lora

    parser = lora.build_parser()
    values = vars(parser.parse_args(list(command[4:])))
    config_path = values.get("config")
    if config_path:
        import yaml

        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = yaml.load(handle, lora.yaml_loader)
        if not isinstance(config, dict):
            raise CandidateCortexTrainingError("mlx_lora_config_invalid")
        for key, value in config.items():
            if values.get(key) is None:
                values[key] = value
    for key, value in lora.CONFIG_DEFAULTS.items():
        if values.get(key) is None:
            values[key] = value
    return types.SimpleNamespace(**values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_and_verify_plan(args.run_root, verify_full_model=True)
    command = build_canary_command(plan)
    adapter_root = Path(str(plan["paths"]["canary_adapter_root"])).resolve(strict=True)
    if any(adapter_root.iterdir()):
        raise CandidateCortexTrainingError("canary_adapter_root_not_empty")
    metrics_path = Path(str(plan["paths"]["canary_host_metrics"])).resolve(strict=False)
    if metrics_path.exists() or metrics_path.is_symlink():
        raise CandidateCortexTrainingError("canary_host_metrics_already_exists")

    started_at = time.time()
    started_monotonic = time.monotonic()
    sample_state: dict[str, Any] = {
        "sample_count": 0,
        "min_available_bytes": 2**63 - 1,
        "max_used_percent": 0.0,
        "max_process_rss_bytes": 0,
    }
    stop = threading.Event()
    observer = get_resource_observer()
    sampler = threading.Thread(
        target=_sample_host,
        args=(stop, sample_state, observer),
        name="candidate-cortex-host-sampler",
        daemon=True,
    )
    sampler.start()
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        with standalone_model_lane(
            owner_id=f"candidate-cortex-canary:{plan['run_id']}",
            model_path=str(plan["model"]["canonical_path"]),
            purpose="train",
            priority=100,
            preemptible=False,
            require_exclusive=True,
            allow_owner_eviction=True,
            metadata={
                "tool": "run_candidate_cortex_canary_target",
                "plan_sha256": plan["plan_sha256"],
            },
        ):
            from mlx_lm import lora

            lora.run(_mlx_arguments(command))
    finally:
        stop.set()
        sampler.join(timeout=2.0)
        finished_at = time.time()
        metrics = {
            "schema": CANARY_HOST_METRICS_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
            "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
            "training_command_sha256": document_sha256(list(command)),
            "target_pid": os.getpid(),
            "started_at_unix": started_at,
            "finished_at_unix": finished_at,
            "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
            **sample_state,
        }
        _write_metrics(metrics_path, metrics)
        print(json.dumps({"host_metrics": metrics}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
