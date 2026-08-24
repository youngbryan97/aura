#!/usr/bin/env python3
"""Bounded host RAM/Metal and atomic model-lane pressure proof.

This proof intentionally does not boot Aura or load model weights. It uses a
small, capped RAM allocation and a small MLX allocation to verify that the
production observer reports real pressure and that physical accelerator work
occurs only behind a durable model-lane reservation. All evidence is labelled
``live_pressure``; ordinary host or simulated observers are rejected.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import gc
import json
import os
import platform
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.lane_admission import lane_budget_gb  # noqa: E402
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import (  # noqa: E402
    LaneClaim,
    LaneTransactionState,
    ModelLaneController,
    ProcessIdentity,
)
from core.runtime.model_runtime_assignment import (  # noqa: E402
    ModelRuntimeAssignment,
    issue_unqualified_model_runtime_assignment,
    locator_identity,
)
from core.runtime.receipts import ReceiptStore  # noqa: E402
from core.runtime.resource_observation import (  # noqa: E402
    HostResourceObserver,
    ObservationSource,
    ResourceObservation,
    assert_live_pressure_observer,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

MIB = 1024**2
GIB = 1024**3


@dataclass(frozen=True)
class SafetyLimits:
    max_memory_percent: float
    min_available_bytes: int
    min_disk_free_bytes: int
    max_thermal_level: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_memory_percent": self.max_memory_percent,
            "min_available_bytes": self.min_available_bytes,
            "min_disk_free_bytes": self.min_disk_free_bytes,
            "max_thermal_level": self.max_thermal_level,
        }


def _git_commit() -> str:
    try:
        completed = get_subprocess_gateway().run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            timeout=5.0,
            check=False,
            read_only=True,
            offline_tooling=True,
            source="proof_tooling:live_resource_pressure_proof.git_commit",
            accelerator_capability="none",
        )
    except (OSError, RuntimeError, ValueError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _safety_violations(
    snapshot: ResourceObservation,
    *,
    limits: SafetyLimits,
) -> list[str]:
    violations: list[str] = []
    if not snapshot.memory.available:
        violations.append(f"memory_observation_unavailable:{snapshot.memory.error}")
    elif snapshot.memory.percent >= limits.max_memory_percent:
        violations.append(
            f"memory_percent:{snapshot.memory.percent:.1f}>={limits.max_memory_percent:.1f}"
        )
    if snapshot.memory.available_bytes < limits.min_available_bytes:
        violations.append(
            f"available_memory:{snapshot.memory.available_bytes}<{limits.min_available_bytes}"
        )
    if not snapshot.disk.available:
        violations.append(f"disk_observation_unavailable:{snapshot.disk.error}")
    elif snapshot.disk.free_bytes < limits.min_disk_free_bytes:
        violations.append(
            f"disk_free:{snapshot.disk.free_bytes}<{limits.min_disk_free_bytes}"
        )
    if not snapshot.thermal.available:
        violations.append(f"thermal_observation_unavailable:{snapshot.thermal.detail}")
    elif snapshot.thermal.level > limits.max_thermal_level:
        violations.append(
            f"thermal_level:{snapshot.thermal.level}>{limits.max_thermal_level}"
        )
    if not snapshot.compute.available:
        violations.append(f"compute_observation_unavailable:{snapshot.compute.error}")
    return violations


def _sample(
    observer: HostResourceObserver,
    *,
    label: str,
    limits: SafetyLimits,
) -> dict[str, Any]:
    snapshot = observer.snapshot(path=ROOT)
    return {
        "label": label,
        "captured_at": time.time(),
        "observation": snapshot.to_dict(),
        "safety_violations": _safety_violations(snapshot, limits=limits),
    }


def _touch_ram(size_bytes: int) -> bytearray:
    allocation = bytearray(size_bytes)
    for offset in range(0, len(allocation), 4096):
        allocation[offset] = 1
    return allocation


def _allocate_mlx(size_bytes: int) -> tuple[Any, Any]:
    import mlx.core as mx

    element_count = max(1, size_bytes // 4)
    allocation = mx.ones((element_count,), dtype=mx.float32)
    mx.eval(allocation)
    return mx, allocation


def _release_mlx(mx: Any, allocations: dict[str, Any], key: str) -> None:
    allocations.pop(key, None)
    gc.collect()
    metal = getattr(mx, "metal", None)
    clear_cache = getattr(mx, "clear_cache", None) or getattr(metal, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    synchronize = getattr(mx, "synchronize", None)
    if callable(synchronize):
        synchronize()


def _cpu_burn(stop: threading.Event) -> None:
    value = 1
    while not stop.is_set():
        value = ((value * 1_103_515_245) + 12_345) & 0x7FFFFFFF


def _run_reservation_race(
    *,
    root: Path,
    observer: HostResourceObserver,
    budget_gb: float,
) -> dict[str, Any]:
    store = ReceiptStore(root / "receipts")

    def process_alive(identity: ProcessIdentity) -> bool:
        return int(identity.pid) == os.getpid()

    first = ModelLaneController(
        state_path=root / "model_lane.json",
        receipt_store=store,
        process_alive=process_alive,
        process_discovery=None,
        observer=observer,
    )
    second = ModelLaneController(
        state_path=root / "model_lane.json",
        receipt_store=store,
        process_alive=process_alive,
        process_discovery=None,
        observer=observer,
    )
    claims = (
        LaneClaim(
            owner_id="live-pressure-race-a",
            model_path="/proof/auxiliary-a",
            request_gb=budget_gb * 0.60,
            request_id="live-pressure-race-a",
            runtime_assignment=issue_unqualified_model_runtime_assignment(
                model_path="/proof/auxiliary-a",
                purpose="serve",
                authority_source="live_resource_pressure_proof",
            ),
        ),
        LaneClaim(
            owner_id="live-pressure-race-b",
            model_path="/proof/auxiliary-b",
            request_gb=budget_gb * 0.60,
            request_id="live-pressure-race-b",
            runtime_assignment=issue_unqualified_model_runtime_assignment(
                model_path="/proof/auxiliary-b",
                purpose="serve",
                authority_source="live_resource_pressure_proof",
            ),
        ),
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            decisions = list(
                pool.map(
                    lambda pair: pair[0].reserve_sync(pair[1]),
                    ((first, claims[0]), (second, claims[1])),
                )
            )
        admitted = [decision for decision in decisions if decision.admitted]
        refused = [decision for decision in decisions if not decision.admitted]
        snapshot = first.snapshot()
        for decision in admitted:
            first.cancel_sync(decision, reason="bounded_race_proof_complete")
        return {
            "passed": len(admitted) == 1 and len(refused) == 1,
            "decisions": [decision.to_dict() for decision in decisions],
            "snapshot_before_cleanup": snapshot,
            "no_capacity_double_spend": (
                float(snapshot["committed_gb"]) + float(snapshot["reserved_gb"])
                <= float(snapshot["budget_gb"])
            ),
        }
    finally:
        store.close()


async def _run_physical_lane_sequence(
    *,
    root: Path,
    observer: HostResourceObserver,
    accelerator_bytes: int,
    limits: SafetyLimits,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    budget_gb = lane_budget_gb(observer=observer)
    store = ReceiptStore(root / "receipts")
    controller = ModelLaneController(
        state_path=root / "model_lane.json",
        receipt_store=store,
        process_alive=lambda identity: identity.pid == os.getpid(),
        process_discovery=None,
        observer=observer,
    )
    process = observer.process(os.getpid())
    if process is None:
        store.close()
        raise RuntimeError("live proof process identity unavailable")
    identity = ProcessIdentity(process.pid, process.create_time)
    allocations: dict[str, Any] = {}
    mx: Any | None = None
    trainer_decision = None
    candidate_decision = None
    events: list[str] = []
    event_times: dict[str, float] = {}
    observations: list[Any] = []
    baseline_accelerator = observer.accelerator().active_bytes

    def mark(event: str) -> None:
        events.append(event)
        event_times[event] = time.monotonic()

    try:
        trainer_claim = LaneClaim(
            owner_id="live-pressure-trainer",
            model_path="/proof/trainer",
            request_gb=budget_gb * 0.55,
            purpose="train",
            request_id="live-pressure-trainer",
            runtime_assignment=issue_unqualified_model_runtime_assignment(
                model_path="/proof/trainer",
                purpose="train",
                authority_source="live_resource_pressure_proof",
            ),
            metadata={"in_process_model_owner": True, "bounded_live_proof": True},
        )
        trainer_decision = controller.reserve_sync(trainer_claim)
        if not trainer_decision.ready_to_spawn:
            raise RuntimeError(f"trainer reservation not ready: {trainer_decision.reason}")
        mark("trainer_reservation_ready")
        mx, allocations["trainer"] = _allocate_mlx(accelerator_bytes)
        mark("trainer_allocation_created")
        trainer_loaded = _sample(
            observer,
            label="trainer_allocation_loaded",
            limits=limits,
        )
        samples.append(trainer_loaded)
        trainer_active = int(
            trainer_loaded["observation"]["accelerator"]["active_bytes"]
        )
        if trainer_active - baseline_accelerator < accelerator_bytes * 0.50:
            raise RuntimeError(
                "accelerator allocation was not observed at the requested scale"
            )
        trainer_decision = controller.commit_sync(
            trainer_decision,
            process=identity,
            observed_gb=trainer_active / GIB,
        )
        observations = controller.owner_observations()

        candidate_claim = LaneClaim(
            owner_id="live-pressure-cortex",
            model_path="/proof/Aura-32B-cortex",
            request_gb=budget_gb * 0.55,
            request_id="live-pressure-cortex",
            runtime_assignment=ModelRuntimeAssignment.issue(
                model_path="/proof/Aura-32B-cortex",
                artifact_identity=locator_identity("/proof/Aura-32B-cortex"),
                artifact_identity_kind="canonical_locator_sha256",
                artifact_identity_exact=False,
                role="cortex",
                purpose="serve",
                authority_source="live_resource_pressure_proof",
            ),
            metadata={"in_process_model_owner": True, "bounded_live_proof": True},
        )
        candidate_decision = controller.reserve_sync(candidate_claim)
        if candidate_decision.state is not LaneTransactionState.EVICTING:
            raise RuntimeError(
                f"candidate did not require eviction: {candidate_decision.state.value}"
            )
        if candidate_decision.ready_to_spawn:
            raise RuntimeError("candidate became ready before required eviction")
        mark("candidate_reserved_eviction_required")

        async def evict(owner: Any, _reason: str) -> bool:
            nonlocal observations
            mark(f"eviction_started:{owner.owner_id}")
            _release_mlx(mx, allocations, "trainer")
            observations = [
                item for item in observations if item.owner_id != owner.owner_id
            ]
            mark(f"eviction_physical_release:{owner.owner_id}")
            return True

        async def reclaim(_claim: LaneClaim) -> bool:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                active = observer.accelerator().active_bytes
                if active <= baseline_accelerator + accelerator_bytes * 0.25:
                    mark("eviction_reclamation_observed")
                    return True
                await asyncio.sleep(0.05)
            return False

        candidate_decision = await controller.prepare(
            candidate_decision,
            evict=evict,
            observe=lambda: list(observations),
            reclaim=reclaim,
            timeout_s=8.0,
        )
        if not candidate_decision.ready_to_spawn:
            raise RuntimeError(f"candidate preparation failed: {candidate_decision.reason}")
        mark("candidate_ready_after_reclamation")

        mx, allocations["candidate"] = _allocate_mlx(accelerator_bytes)
        mark("candidate_allocation_created")
        candidate_loaded = _sample(
            observer,
            label="candidate_allocation_loaded",
            limits=limits,
        )
        samples.append(candidate_loaded)
        candidate_active = int(
            candidate_loaded["observation"]["accelerator"]["active_bytes"]
        )
        candidate_decision = controller.commit_sync(
            candidate_decision,
            process=identity,
            observed_gb=candidate_active / GIB,
        )
        mark("candidate_committed")
        snapshot = controller.snapshot()
        coverage = store.coverage_stats()
        expected_order = (
            events.index("candidate_reserved_eviction_required")
            < next(index for index, item in enumerate(events) if item.startswith("eviction_started:"))
            < events.index("eviction_reclamation_observed")
            < events.index("candidate_ready_after_reclamation")
            < events.index("candidate_allocation_created")
            < events.index("candidate_committed")
        )
        ready_to_load_gap_s = (
            event_times["candidate_allocation_created"]
            - event_times["candidate_ready_after_reclamation"]
        )
        cold_gap_bounded = 0.0 <= ready_to_load_gap_s <= 0.25
        return {
            "passed": bool(
                expected_order
                and cold_gap_bounded
                and candidate_decision.state is LaneTransactionState.COMMITTED
                and not candidate_loaded["safety_violations"]
            ),
            "events": events,
            "event_times_monotonic": event_times,
            "candidate": candidate_decision.to_dict(),
            "snapshot": snapshot,
            "receipt_coverage": coverage,
            "required_eviction_preceded_candidate_load": expected_order,
            "ready_to_load_gap_s": round(ready_to_load_gap_s, 6),
            "cold_gap_bounded": cold_gap_bounded,
            "no_overcommit": (
                float(snapshot["committed_gb"]) + float(snapshot["reserved_gb"])
                <= float(snapshot["budget_gb"])
            ),
        }
    finally:
        if candidate_decision is not None and candidate_decision.state is LaneTransactionState.COMMITTED:
            controller.release_owner_sync(
                candidate_decision.owner_id,
                fencing_token=candidate_decision.fencing_token,
                reason="bounded_live_pressure_complete",
            )
        if trainer_decision is not None and trainer_decision.state is LaneTransactionState.COMMITTED:
            controller.release_owner_sync(
                trainer_decision.owner_id,
                fencing_token=trainer_decision.fencing_token,
                reason="bounded_live_pressure_cleanup",
            )
        if mx is not None:
            _release_mlx(mx, allocations, "trainer")
            _release_mlx(mx, allocations, "candidate")
        store.close()


async def run_proof(args: argparse.Namespace) -> dict[str, Any]:
    observer = HostResourceObserver(
        source=ObservationSource.LIVE_PRESSURE,
        scenario_id=f"bounded-live-pressure:{os.getpid()}:{int(time.time())}",
    )
    assert_live_pressure_observer(observer)
    limits = SafetyLimits(
        max_memory_percent=float(args.max_memory_percent),
        min_available_bytes=int(args.min_available_gb * GIB),
        min_disk_free_bytes=int(args.min_disk_free_gb * GIB),
        max_thermal_level=int(args.max_thermal_level),
    )
    samples: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    error = ""
    ram: bytearray | None = None
    cpu_stop = threading.Event()
    cpu_threads: list[threading.Thread] = []
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="aura-live-pressure-") as temp_dir:
        state_root = Path(temp_dir)
        try:
            baseline = _sample(observer, label="baseline", limits=limits)
            samples.append(baseline)
            checks["baseline_within_safety_envelope"] = not baseline["safety_violations"]
            process_table = observer.process_table()
            checks["process_table_available"] = process_table.available
            checks["live_pressure_source"] = (
                observer.provenance.source is ObservationSource.LIVE_PRESSURE
                and observer.provenance.qualifies_as_live_pressure
            )
            if not all(checks.values()):
                raise RuntimeError("live pressure preflight failed")

            ram = _touch_ram(int(args.ram_mb * MIB))
            ram_loaded = _sample(observer, label="ram_allocation_loaded", limits=limits)
            samples.append(ram_loaded)
            baseline_rss = int(baseline["observation"]["memory"]["process_rss_bytes"])
            loaded_rss = int(ram_loaded["observation"]["memory"]["process_rss_bytes"])
            checks["ram_pressure_observed"] = loaded_rss - baseline_rss >= args.ram_mb * MIB * 0.50
            checks["ram_pressure_within_safety_envelope"] = not ram_loaded["safety_violations"]

            lane = await _run_physical_lane_sequence(
                root=state_root / "physical_lane",
                observer=observer,
                accelerator_bytes=int(args.accelerator_mb * MIB),
                limits=limits,
                samples=samples,
            )
            checks["physical_lane_sequence"] = bool(lane["passed"])
            checks["required_eviction_before_candidate_load"] = bool(
                lane["required_eviction_preceded_candidate_load"]
            )
            checks["physical_lane_no_overcommit"] = bool(lane["no_overcommit"])
            checks["physical_lane_cold_gap_bounded"] = bool(
                lane["cold_gap_bounded"]
            )

            race = _run_reservation_race(
                root=state_root / "reservation_race",
                observer=observer,
                budget_gb=lane_budget_gb(observer=observer),
            )
            checks["concurrent_reservation_single_winner"] = bool(race["passed"])
            checks["race_no_capacity_double_spend"] = bool(
                race["no_capacity_double_spend"]
            )

            for index in range(max(0, min(4, int(args.cpu_workers)))):
                thread = threading.Thread(
                    target=_cpu_burn,
                    args=(cpu_stop,),
                    name=f"live-pressure-cpu-{index}",
                    daemon=False,
                )
                thread.start()
                cpu_threads.append(thread)
            deadline = time.monotonic() + max(0.5, min(15.0, float(args.duration_s)))
            hold_safe = True
            while time.monotonic() < deadline:
                sample = _sample(observer, label="bounded_hold", limits=limits)
                samples.append(sample)
                if sample["safety_violations"]:
                    hold_safe = False
                    break
                await asyncio.sleep(0.25)
            checks["bounded_hold_within_safety_envelope"] = hold_safe
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}:{exc}"
            lane = locals().get("lane", {})
            race = locals().get("race", {})
        finally:
            cpu_stop.set()
            for thread in cpu_threads:
                thread.join(timeout=1.0)
            ram = None
            gc.collect()
            await asyncio.sleep(0.25)
            recovered = _sample(observer, label="recovered", limits=limits)
            samples.append(recovered)
            checks["recovered_within_safety_envelope"] = not recovered["safety_violations"]

    passed = bool(checks) and all(checks.values()) and not error
    return {
        "schema": "aura.live_resource_pressure_proof.v1",
        "passed": passed,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "duration_s": round(time.time() - started, 3),
        "git_commit": _git_commit(),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "resource_observation": observer.provenance.to_dict(),
        "limits": limits.to_dict(),
        "configured_pressure": {
            "ram_mb": args.ram_mb,
            "accelerator_mb": args.accelerator_mb,
            "cpu_workers": args.cpu_workers,
            "duration_s": args.duration_s,
        },
        "checks": checks,
        "samples": samples,
        "physical_lane": lane,
        "reservation_race": race,
        "error": error,
        "claim_supported": "bounded_live_resource_and_atomic_lane_pressure",
        "claim_not_supported": [
            "model_quality",
            "full_model_eviction_latency",
            "multi_hour_soak",
            "24_72_hour_reliability",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ram-mb", type=int, default=128)
    parser.add_argument("--accelerator-mb", type=int, default=64)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--max-memory-percent", type=float, default=88.0)
    parser.add_argument("--min-available-gb", type=float, default=6.0)
    parser.add_argument("--min-disk-free-gb", type=float, default=20.0)
    parser.add_argument("--max-thermal-level", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.ram_mb = max(16, min(512, int(args.ram_mb)))
    args.accelerator_mb = max(16, min(512, int(args.accelerator_mb)))
    args.cpu_workers = max(0, min(4, int(args.cpu_workers)))
    verdict = asyncio.run(run_proof(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.out,
        json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n",
    )
    print(
        "LIVE_RESOURCE_PRESSURE="
        f"{'PASS' if verdict['passed'] else 'FAIL'} "
        f"source={verdict['resource_observation']['source']} "
        f"duration_s={verdict['duration_s']} out={args.out}"
    )
    if verdict["error"]:
        print(f"error={verdict['error']}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
