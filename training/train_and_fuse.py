#!/usr/bin/env python3
"""End-to-end LoRA training + fuse + auto-pickup pipeline.

What this script does, in one run:

1. Optionally rebuild the training dataset (build_dataset_v3) so the
   personality + architecture corpus is fresh.
2. Run the LoRA fine-tune (mlx_lm.lora) with the existing
   training/finetune_lora.py hyperparameters.
3. Fuse the resulting adapter into the base model with mlx_lm.fuse,
   producing a new versioned directory under training/fused-model/.
4. Write training/fused-model/active.json — a small manifest that Aura's
   model_registry reads on boot to pick up the newest fused model
   automatically. No .env edit required.
5. Verify the new model loads, then atomically swap the manifest.

After this script finishes, restarting Aura will use the new weights.
The previous fused model directory is kept (under a versioned name) so
you can roll back by editing active.json or pointing AURA_LLM__MLX_MODEL_PATH.

Usage:
    python training/train_and_fuse.py
    python training/train_and_fuse.py --skip-dataset      # reuse existing data
    python training/train_and_fuse.py --skip-train        # only fuse + publish
    python training/train_and_fuse.py --tag mythos-v1     # name this run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

TRAINING_DIR = Path(__file__).parent
REPO_DIR = TRAINING_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from core.brain.llm.model_artifact_profile import (  # noqa: E402
    build_model_artifact_descriptor,
    get_model_artifact_profile,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.model_lane_control import (  # noqa: E402
    LaneClaim,
    estimate_model_job_footprint_gb,
)
from core.runtime.model_runtime_assignment import (  # noqa: E402
    issue_unqualified_model_runtime_assignment,
)
from core.runtime.resource_observation import (  # noqa: E402
    ResourceObserver,
    get_resource_observer,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from training.model_basis import (  # noqa: E402
    TrainingModelBasis,
    assert_adapter_matches_basis,
    resolve_training_model_basis,
)

DATA_DIR = TRAINING_DIR / "data"
ADAPTER_DIR = TRAINING_DIR / "adapters" / "aura-personality"
CRSM_DELTA_DATA_DIR = DATA_DIR / "crsm_delta"
CRSM_DELTA_MANIFEST = DATA_DIR / "crsm_delta_manifest.json"
FUSED_BASE_DIR = TRAINING_DIR / "fused-model"
ACTIVE_MANIFEST = FUSED_BASE_DIR / "active.json"
CRSM_DATASET = REPO_DIR / "data" / "synthetic_training" / "lora_dataset.jsonl"
CRSM_INTEGRATION_MANIFEST = DATA_DIR / "crsm_integration_manifest.json"

TRAINING_COMMAND_TIMEOUT_S = float(os.environ.get("AURA_TRAINING_COMMAND_TIMEOUT_S", "86400"))
_GIB = 1024**3
_LIVE_AURA_CMD_MARKERS = (
    "aura_main.py",
    "interface/server.py",
    "core/brain/llm/mlx_worker.py",
    "tools/live_boot_proof.py",
    "tools/visible_journal_demo_proof.py",
)
_DELEGATED_GOVERNANCE_ENV_KEYS = (
    "AURA_DELEGATED_GOVERNANCE_RECEIPT_ID",
    "AURA_DELEGATED_GOVERNANCE_DOMAIN",
    "AURA_DELEGATED_GOVERNANCE_SOURCE",
    "AURA_DELEGATED_AUTHORITY_INTENT_ID",
    "AURA_DELEGATED_GOVERNANCE_PARENT_PID",
)
_INHERITED_MODEL_LANE_ENV_KEYS = (
    "AURA_MODEL_LANE_INHERITED_OWNER_ID",
    "AURA_MODEL_LANE_INHERITED_REQUEST_ID",
    "AURA_MODEL_LANE_INHERITED_MODEL_PATH",
    "AURA_MODEL_LANE_INHERITED_PURPOSE",
    "AURA_MODEL_LANE_DELEGATION_TOKEN",
)


def delegated_governance_provenance() -> dict[str, str]:
    """Return the parent authority identifiers carried into this worker."""
    receipt_id = str(os.getenv("AURA_DELEGATED_GOVERNANCE_RECEIPT_ID", "")).strip()
    intent_id = str(os.getenv("AURA_DELEGATED_AUTHORITY_INTENT_ID", "")).strip()
    domain = str(os.getenv("AURA_DELEGATED_GOVERNANCE_DOMAIN", "")).strip()
    source = str(os.getenv("AURA_DELEGATED_GOVERNANCE_SOURCE", "")).strip()
    if not receipt_id and not intent_id:
        return {}
    return {
        "will_receipt_id": receipt_id,
        "executive_intent_id": intent_id,
        "domain": domain,
        "source": source,
    }


def get_default_base_model() -> Path:
    """Return the promoted local Cortex artifact after exact identity validation."""
    return resolve_training_model_basis().path


def enforce_live_delegated_authority(*, crsm_delta: bool, tag: str) -> None:
    """Require the live scheduler's parent-bound receipt for its exact lane."""
    if not crsm_delta or tag != "crsm-closeout" or not _env_flag("AURA_LAUNCHED_FROM_APP"):
        return
    provenance = delegated_governance_provenance()
    expected_parent = str(os.getppid())
    supplied_parent = str(os.getenv("AURA_DELEGATED_GOVERNANCE_PARENT_PID", "")).strip()
    if (
        not provenance.get("will_receipt_id")
        or not provenance.get("executive_intent_id")
        or provenance.get("domain") != "semantic_weight_update"
        or provenance.get("source") != "system_maintenance:crsm_closure"
        or supplied_parent != expected_parent
    ):
        raise SystemExit(
            "live CRSM closeout requires a parent-bound semantic_weight_update authority receipt"
        )


def consume_inherited_pipeline_lane() -> bool:
    """Consume the outer pipeline delegation before its short token expires."""
    values = {
        key: str(os.getenv(key, "")).strip()
        for key in _INHERITED_MODEL_LANE_ENV_KEYS
    }
    if not any(values.values()):
        return False
    if not all(values.values()):
        raise SystemExit("incomplete inherited model-lane delegation")
    from core.runtime.model_lane_control import acquire_standalone_model_lane

    lease = acquire_standalone_model_lane(
        owner_id="crsm-closeout-pipeline-worker",
        model_path=values["AURA_MODEL_LANE_INHERITED_MODEL_PATH"],
        purpose=values["AURA_MODEL_LANE_INHERITED_PURPOSE"],
        request_gb=0.1,
        metadata={"pipeline": "train_and_fuse", "delegated_worker": True},
    )
    if not lease.inherited:
        lease.release(reason="unexpected_non_inherited_pipeline_lane")
        raise SystemExit("delegated pipeline failed to consume its inherited model lane")
    return True


def _run(
    cmd: list[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    model_job: bool = False,
    model_lane_claim: LaneClaim | None = None,
    source: str = "training_tooling:train_and_fuse",
) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    gateway = get_subprocess_gateway()
    effective_env = dict(env) if env is not None else None
    if os.getenv("AURA_GOVERNANCE_MODE", "").strip() == "delegated_subprocess":
        effective_env = dict(os.environ if effective_env is None else effective_env)
        for key in _DELEGATED_GOVERNANCE_ENV_KEYS:
            effective_env.pop(key, None)
        effective_env["AURA_GOVERNANCE_MODE"] = "delegated_subprocess_child"
        effective_env["AURA_REQUIRE_GOVERNANCE"] = "0"
        if not (model_job or model_lane_claim is not None):
            for key in _INHERITED_MODEL_LANE_ENV_KEYS:
                effective_env.pop(key, None)
    run_kwargs = {
        "cwd": REPO_DIR,
        "env": effective_env,
        "timeout": timeout if timeout is not None else TRAINING_COMMAND_TIMEOUT_S,
        "capture_output": False,
        "offline_tooling": True,
        "source": source,
    }
    if model_job or model_lane_claim is not None:
        result = gateway.run_model_blocking(
            cmd,
            **run_kwargs,
            model_lane_claim=model_lane_claim,
        )
    else:
        result = gateway.run(cmd, **run_kwargs, accelerator_capability="auto")
    return result.returncode


def _training_lane_claim(base_model: Path, *, source: str) -> LaneClaim:
    timeout = TRAINING_COMMAND_TIMEOUT_S
    return LaneClaim(
        owner_id=f"training:{source}:{os.getpid()}:{time.time_ns()}",
        model_path=str(base_model),
        request_gb=estimate_model_job_footprint_gb(str(base_model), purpose="train"),
        purpose="train",
        priority=80,
        preemptible=True,
        reservation_ttl_s=timeout + 30.0,
        owner_lease_ttl_s=timeout + 30.0,
        runtime_assignment=issue_unqualified_model_runtime_assignment(
            model_path=str(base_model),
            purpose="train",
            authority_source=source,
        ),
        metadata={"source": source, "pipeline": "train_and_fuse"},
    )


def build_dataset() -> None:
    builder = TRAINING_DIR / "build_dataset_v3.py"
    if not builder.exists():
        print(f"  Dataset builder not found at {builder}; skipping.")
        return
    rc = _run([sys.executable, str(builder)], source="training_tooling:build_dataset")
    if rc != 0:
        sys.exit(f"Dataset build failed (exit {rc}).")


def train_lora(
    *,
    base_model: Path,
    model_basis: TrainingModelBasis,
    resume: bool = False,
) -> None:
    finetune = TRAINING_DIR / "finetune_lora.py"
    if not finetune.exists():
        sys.exit(f"finetune_lora.py not found at {finetune}.")
    # Pass the exact parent-bound basis into the child trainer. The child
    # revalidates the descriptor before it can write or resume an adapter.
    env = os.environ.copy()
    env["AURA_LORA_BASE_MODEL"] = str(base_model)
    env["AURA_LORA_EXPECTED_BASE_DESCRIPTOR_SHA256"] = model_basis.descriptor_sha256
    if resume:
        assert_adapter_matches_basis(ADAPTER_DIR, model_basis)
    cmd = [sys.executable, str(finetune)]
    if resume:
        cmd.append("--resume")
    print(f"  AURA_LORA_BASE_MODEL={base_model}", flush=True)
    rc = _run(
        cmd,
        env=env,
        model_lane_claim=_training_lane_claim(
            base_model,
            source="training_tooling:train_lora",
        ),
        source="training_tooling:train_lora",
    )
    if rc != 0:
        sys.exit(f"LoRA fine-tune failed (exit {rc}).")


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _jsonl_file_stats(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as fh:
        for raw in fh:
            lines += 1
            digest.update(raw)
    stat = path.stat()
    return {
        "path": str(path),
        "lines": lines,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": digest.hexdigest(),
    }


def _example_key(example: dict[str, Any]) -> str:
    try:
        messages = example.get("messages") if isinstance(example, dict) else None
        if not isinstance(messages, list):
            return ""
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            content = " ".join(str(message.get("content") or "").split()).lower()
            parts.append(f"{role}:{content}")
        return "\n".join(parts)
    except (AttributeError, TypeError, ValueError):
        return ""


def _reservoir_sample_jsonl(
    path: Path,
    *,
    count: int,
    rng: random.Random,
    exclude_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample retention examples without loading the full corpus into memory."""
    if count <= 0 or not path.exists():
        return []
    exclude_keys = exclude_keys or set()
    sample: list[dict[str, Any]] = []
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(example, dict) or not isinstance(example.get("messages"), list):
                continue
            key = _example_key(example)
            if not key or key in exclude_keys:
                continue
            seen += 1
            if len(sample) < count:
                sample.append(example)
                continue
            idx = rng.randrange(seen)
            if idx < count:
                sample[idx] = example
    return sample


def _write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")


def _selection_digest(examples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(json.dumps(example, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_crsm_delta_dataset(
    *,
    output_dir: Path = CRSM_DELTA_DATA_DIR,
    max_crsm_examples: int | None = None,
    retention_examples: int | None = None,
    valid_fraction: float = 0.1,
    seed: int = 20260628,
) -> dict[str, Any]:
    """Create a bounded, provenance-rich dataset for CRSM incremental LoRA.

    This is deliberately not a marker shortcut. It extracts the eligible CRSM
    captures through the same production gate as the full corpus, adds a small
    retention sample from the existing training data, writes a standalone
    MLX-compatible dataset, and records hashes proving exactly what trained.
    """
    from training.build_dataset_v3 import build_crsm_experience_examples

    max_crsm_examples = (
        _env_int("AURA_CRSM_DELTA_MAX_EXAMPLES", 600, minimum=1, maximum=5000)
        if max_crsm_examples is None
        else max(1, int(max_crsm_examples))
    )
    retention_examples = (
        _env_int("AURA_CRSM_DELTA_RETENTION_EXAMPLES", 512, minimum=0, maximum=5000)
        if retention_examples is None
        else max(0, int(retention_examples))
    )
    rng = random.Random(seed)

    crsm_examples, crsm_manifest = build_crsm_experience_examples(
        CRSM_DATASET,
        max_examples=max_crsm_examples,
    )
    if not crsm_examples:
        sys.exit("CRSM delta dataset build failed: no eligible CRSM captures after safety filtering.")

    crsm_keys = {_example_key(example) for example in crsm_examples}
    retention_pool = _reservoir_sample_jsonl(
        DATA_DIR / "train.jsonl",
        count=retention_examples,
        rng=rng,
        exclude_keys=crsm_keys,
    )
    if len(retention_pool) < retention_examples:
        retention_pool.extend(
            _reservoir_sample_jsonl(
                DATA_DIR / "valid.jsonl",
                count=retention_examples - len(retention_pool),
                rng=rng,
                exclude_keys=crsm_keys | {_example_key(example) for example in retention_pool},
            )
        )

    selected = [*crsm_examples, *retention_pool]
    rng.shuffle(selected)
    valid_count = max(1, min(len(selected) - 1, int(round(len(selected) * valid_fraction))))
    valid = selected[:valid_count]
    train = selected[valid_count:]

    if not train or not valid:
        sys.exit("CRSM delta dataset build failed: train/valid split would be empty.")

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, valid)

    manifest = {
        **crsm_manifest,
        "builder": "training/train_and_fuse.py:build_crsm_delta_dataset",
        "delta_mode": True,
        "seed": seed,
        "retention_examples": len(retention_pool),
        "selection_sha256": _selection_digest(selected),
        "output": {
            "builder": "training/train_and_fuse.py",
            "total_examples": len(selected),
            "crsm_examples": len(crsm_examples),
            "retention_examples": len(retention_pool),
            "train": _jsonl_file_stats(train_path),
            "valid": _jsonl_file_stats(valid_path),
        },
    }
    CRSM_DELTA_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CRSM_DELTA_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("\nBuilt CRSM delta dataset:")
    print(json.dumps(manifest["output"], indent=2, sort_keys=True))
    return manifest


def _latest_adapter_file(adapter_dir: Path = ADAPTER_DIR) -> Path | None:
    primary = adapter_dir / "adapters.safetensors"
    if primary.exists():
        return primary
    checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
    return checkpoints[-1] if checkpoints else None


def build_crsm_delta_train_command(
    *,
    base_model: Path,
    data_dir: Path,
    adapter_dir: Path,
    resume_adapter_file: Path,
    iters: int,
    max_seq_length: int,
    lora_config_path: Path,
    save_every: int | None = None,
    steps_per_eval: int | None = None,
    steps_per_report: int = 10,
) -> list[str]:
    save_every = save_every or max(25, min(100, iters))
    steps_per_eval = steps_per_eval or max(25, min(100, iters))
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        str(base_model),
        "--train",
        "--data",
        str(data_dir),
        "--adapter-path",
        str(adapter_dir),
        "--resume-adapter-file",
        str(resume_adapter_file),
        "--iters",
        str(iters),
        "--num-layers",
        "-1",
        "--batch-size",
        "1",
        "--learning-rate",
        "5e-6",
        "--save-every",
        str(save_every),
        "--steps-per-eval",
        str(steps_per_eval),
        "--steps-per-report",
        str(steps_per_report),
        "--max-seq-length",
        str(max_seq_length),
        "--grad-checkpoint",
        "-c",
        str(lora_config_path),
    ]


def train_crsm_delta_lora(
    *,
    base_model: Path,
    model_basis: TrainingModelBasis,
    data_dir: Path = CRSM_DELTA_DATA_DIR,
    adapter_dir: Path | None = None,
    iters: int | None = None,
    max_seq_length: int | None = None,
) -> Path:
    """Run a real bounded LoRA update from current CRSM captures."""
    assert_adapter_matches_basis(ADAPTER_DIR, model_basis)
    resume_adapter = _latest_adapter_file(ADAPTER_DIR)
    if resume_adapter is None:
        sys.exit(f"CRSM delta training failed: no source adapter found under {ADAPTER_DIR}.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    adapter_dir = adapter_dir or (ADAPTER_DIR.parent / f"aura-personality-crsm-delta-{timestamp}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    lora_config_path = ADAPTER_DIR / "lora_config.yaml"
    if not lora_config_path.exists():
        lora_config_path = ADAPTER_DIR / "lora_config.json"
    if not lora_config_path.exists():
        sys.exit(f"CRSM delta training failed: missing LoRA config under {ADAPTER_DIR}.")

    iters = (
        _env_int("AURA_CRSM_DELTA_ITERS", 600, minimum=25, maximum=5000)
        if iters is None
        else max(1, int(iters))
    )
    max_seq_length = (
        _env_int("AURA_CRSM_DELTA_MAX_SEQ_LENGTH", 2048, minimum=512, maximum=4096)
        if max_seq_length is None
        else max(128, int(max_seq_length))
    )
    cmd = build_crsm_delta_train_command(
        base_model=base_model,
        data_dir=data_dir,
        adapter_dir=adapter_dir,
        resume_adapter_file=resume_adapter,
        iters=iters,
        max_seq_length=max_seq_length,
        lora_config_path=lora_config_path,
    )
    atomic_write_text(
        adapter_dir / "training_config.json",
        json.dumps(
            {
                "schema": "aura.personality_lora.training.v2",
                "model": str(base_model),
                "training_basis": model_basis.to_record(),
                "mode": "crsm_delta",
                "iters": iters,
                "total_iterations": iters,
                "max_seq_length": max_seq_length,
                "source_adapter_file": str(resume_adapter.resolve(strict=True)),
                "source_adapter_sha256": _file_sha256(resume_adapter),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rc = _run(
        cmd,
        model_job=True,
        model_lane_claim=_training_lane_claim(
            base_model,
            source="training_tooling:crsm_delta_lora",
        ),
        source="training_tooling:crsm_delta_lora",
    )
    if rc != 0:
        sys.exit(f"CRSM delta LoRA fine-tune failed (exit {rc}).")
    if not (adapter_dir / "adapters.safetensors").exists():
        sys.exit(f"CRSM delta LoRA fine-tune ended without {adapter_dir / 'adapters.safetensors'}.")
    return adapter_dir


def _model_size_tag(base_model: Path) -> str:
    """Derive a human-readable tag from measured parameters when available."""
    profile = get_model_artifact_profile(str(base_model))
    if profile.measured and profile.total_parameters > 0:
        billions = profile.total_parameters / 1_000_000_000
        rounded = round(billions)
        if abs(billions - rounded) < 0.1:
            return f"{rounded}B"
        return f"{billions:.1f}B".replace(".", "_")
    name = base_model.name.lower()
    for size in (
        "72b",
        "32b",
        "27b",
        "14b",
        "8b",
        "7b",
        "3b",
        "1.5b",
        "0.5b",
    ):
        if size in name:
            return size.upper().replace(".", "_")
    return "model"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _default_training_headroom_gb(
    base_model: Path,
    *,
    skip_train: bool,
) -> tuple[float, float]:
    """Return measured RAM and disk floors for a train/fuse transaction."""
    profile = get_model_artifact_profile(str(base_model))
    class_floor = {
        "72b": ((44.0, 220.0), (32.0, 160.0)),
        "32b": ((28.0, 110.0), (20.0, 90.0)),
        "14b": ((18.0, 60.0), (12.0, 40.0)),
        "7b": ((18.0, 60.0), (12.0, 40.0)),
    }.get(profile.size_class, ((12.0, 40.0), (8.0, 25.0)))
    floor_memory, floor_disk = class_floor[1 if skip_train else 0]
    if not profile.measured or profile.weight_gb <= 0.0:
        return floor_memory, floor_disk

    loaded_gb = profile.weight_gb + max(1.0, profile.weight_gb * 0.25)
    fuse_peak_gb = max(loaded_gb + 6.0, loaded_gb * 2.25)
    train_peak_gb = max(loaded_gb + 4.0, loaded_gb * 1.8)
    transaction_peak_gb = fuse_peak_gb if skip_train else max(train_peak_gb, fuse_peak_gb)
    measured_memory = transaction_peak_gb + 2.0
    measured_disk = profile.weight_gb * 4.0 + 10.0
    return max(floor_memory, measured_memory), max(floor_disk, measured_disk)


def _live_aura_processes(
    *,
    observer: ResourceObserver | None = None,
) -> list[dict[str, Any]]:
    table = (observer or get_resource_observer()).process_table()
    if not table.available:
        raise RuntimeError(f"process_table_unavailable:{table.error or 'unknown'}")
    current_pid = os.getpid()
    found: list[dict[str, Any]] = []
    for process in table.processes:
        if process.pid == current_pid:
            continue
        cmd = " ".join(process.cmdline)
        if any(marker in cmd for marker in _LIVE_AURA_CMD_MARKERS):
            found.append(
                {"pid": process.pid, "name": process.name, "cmdline": cmd[:500]}
            )
    return found


def training_preflight(
    *,
    base_model: Path,
    skip_train: bool,
    crsm_delta: bool = False,
    observer: ResourceObserver | None = None,
) -> dict[str, Any]:
    observer = observer or get_resource_observer()
    size_tag = _model_size_tag(base_model)
    default_min_available_gb, default_min_free_disk_gb = _default_training_headroom_gb(
        base_model,
        skip_train=skip_train,
    )
    min_available_gb = _env_float("AURA_TRAINING_MIN_AVAILABLE_GB", default_min_available_gb)
    min_free_disk_gb = _env_float("AURA_TRAINING_MIN_FREE_DISK_GB", default_min_free_disk_gb)
    max_memory_percent = _env_float("AURA_TRAINING_MAX_MEMORY_PERCENT", 82.0)

    blockers: list[str] = []
    memory_observation = observer.memory()
    memory: dict[str, Any] = {
        "available_gb": None,
        "percent": None,
        "observation": memory_observation.provenance.to_dict(),
    }
    if not memory_observation.available:
        blockers.append(f"memory_probe_failed:{memory_observation.error or 'unavailable'}")
    else:
        available_gb = memory_observation.available_bytes / _GIB
        percent = float(memory_observation.percent)
        memory.update(
            {"available_gb": round(available_gb, 2), "percent": round(percent, 1)}
        )
        if available_gb < min_available_gb:
            blockers.append(
                f"available_memory:{available_gb:.1f}GB < required {min_available_gb:.1f}GB"
            )
        if percent > max_memory_percent:
            blockers.append(f"memory_pressure:{percent:.1f}% > {max_memory_percent:.1f}%")

    disk_path = FUSED_BASE_DIR if FUSED_BASE_DIR.exists() else FUSED_BASE_DIR.parent
    disk_observation = observer.disk(disk_path)
    free_disk_gb = disk_observation.free_bytes / _GIB if disk_observation.available else 0.0
    if not disk_observation.available:
        blockers.append(f"disk_probe_failed:{disk_observation.error or 'unavailable'}")
    elif free_disk_gb < min_free_disk_gb:
        blockers.append(
            f"free_disk:{free_disk_gb:.1f}GB < required {min_free_disk_gb:.1f}GB"
        )

    live_processes: list[dict[str, Any]] = []
    if not _env_flag("AURA_TRAINING_ALLOW_LIVE_AURA"):
        try:
            live_processes = _live_aura_processes(observer=observer)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            blockers.append(f"process_table_probe_failed:{type(exc).__name__}:{exc}")
    if live_processes:
        blockers.append(f"live_aura_processes:{len(live_processes)}")

    return {
        "passed": not blockers,
        "mode": (
            "crsm_delta_train_fuse_publish"
            if crsm_delta and not skip_train
            else "fuse_publish"
            if skip_train
            else "train_fuse_publish"
        ),
        "base_model": str(base_model),
        "size": size_tag,
        "requirements": {
            "min_available_gb": min_available_gb,
            "max_memory_percent": max_memory_percent,
            "min_free_disk_gb": min_free_disk_gb,
            "block_live_aura": not _env_flag("AURA_TRAINING_ALLOW_LIVE_AURA"),
        },
        "memory": memory,
        "disk": {
            "path": str(disk_path),
            "free_gb": round(free_disk_gb, 2),
            "observation": disk_observation.provenance.to_dict(),
        },
        "resource_observation": observer.provenance.to_dict(),
        "live_aura_processes": live_processes,
        "blockers": blockers,
    }


def enforce_training_preflight(*, base_model: Path, skip_train: bool, crsm_delta: bool = False) -> dict[str, Any]:
    report = training_preflight(base_model=base_model, skip_train=skip_train, crsm_delta=crsm_delta)
    print("\nTraining preflight:")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        sys.exit("Training preflight failed: " + "; ".join(report["blockers"]))
    return report


def fuse_adapter(
    *,
    base_model: Path,
    model_basis: TrainingModelBasis,
    tag: str,
    adapter_dir: Path = ADAPTER_DIR,
) -> Path:
    """mlx_lm fuse base_model + adapter → versioned fused-model dir."""
    if not (adapter_dir / "adapters.safetensors").exists():
        sys.exit(
            f"No adapter found at {adapter_dir}/adapters.safetensors — "
            "run training first or pass --skip-train only after a previous train."
        )

    assert_adapter_matches_basis(adapter_dir, model_basis)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    size_tag = _model_size_tag(base_model)
    fused_name = (
        f"Aura-{size_tag}-{tag}-{timestamp}" if tag
        else f"Aura-{size_tag}-{timestamp}"
    )
    fused_path = FUSED_BASE_DIR / fused_name
    fused_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nFusing → {fused_path}")
    rc = _run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(base_model),
            "--adapter-path",
            str(adapter_dir),
            "--save-path",
            str(fused_path),
        ],
        timeout=1800,
        model_job=True,
        source="training_tooling:fuse_adapter",
    )
    if rc != 0:
        sys.exit(f"Fuse failed (exit {rc}).")
    if not fused_path.exists() or not any(fused_path.iterdir()):
        sys.exit(f"Fuse claimed success but {fused_path} is empty.")
    return fused_path


def verify_load(fused_path: Path) -> None:
    """Smoke-test: tokenize one prompt to confirm the fused model is loadable."""
    print(f"\nVerifying fused model loads: {fused_path}")
    code = (
        "import contextlib\n"
        "import os\n"
        "import sys\n"
        "from core.runtime.model_lane_control import standalone_model_lane\n"
        "from mlx_lm import load\n"
        "model_path = sys.argv[1]\n"
        "lane = contextlib.nullcontext() if os.getenv('AURA_MODEL_LANE_PARENT_ACCOUNTED') == '1' else standalone_model_lane(\n"
        "    owner_id='verify-fused-model', model_path=model_path,\n"
        "    purpose='benchmark', preemptible=True)\n"
        "with lane:\n"
        "    model, tok = load(model_path)\n"
        "    ids = tok.encode('Hello')\n"
        "    print(f'OK: tokenized {len(ids)} tokens, vocab_size={tok.vocab_size}')\n"
    )
    timeout = 600.0
    claim = LaneClaim(
        owner_id=f"training:verify-fused:{os.getpid()}:{time.time_ns()}",
        model_path=str(fused_path),
        request_gb=estimate_model_job_footprint_gb(
            str(fused_path),
            purpose="benchmark",
        ),
        purpose="benchmark",
        priority=50,
        preemptible=True,
        reservation_ttl_s=timeout + 30.0,
        owner_lease_ttl_s=timeout + 30.0,
        runtime_assignment=issue_unqualified_model_runtime_assignment(
            model_path=str(fused_path),
            purpose="benchmark",
            authority_source="training_tooling:verify_fused_model",
        ),
        metadata={"source": "training_tooling:verify_fused_model"},
    )
    rc = _run(
        [sys.executable, "-c", code, str(fused_path)],
        timeout=timeout,
        model_lane_claim=claim,
        source="training_tooling:verify_fused_model",
    )
    if rc != 0:
        sys.exit(f"Verification load failed (exit {rc}).")


def publish_manifest(fused_path: Path, *, tag: str, base_model: Path) -> None:
    """Atomically write active.json so Aura's next boot uses the new model.

    The manifest now includes the base-model size so downstream RAM-aware
    routing (model_registry, inference_gate) can branch on it without
    re-parsing the directory name."""
    FUSED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor = build_model_artifact_descriptor(fused_path)
    manifest = {
        "active_model_path": str(fused_path),
        "artifact_descriptor": descriptor,
        "fused_at": int(time.time()),
        "tag": tag or "",
        "size": _model_size_tag(base_model),
        "base_model": str(base_model),
        "schema_version": 3,
    }
    governance = delegated_governance_provenance()
    if governance:
        manifest["governance"] = governance
    atomic_write_text(
        ACTIVE_MANIFEST,
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nWrote active manifest: {ACTIVE_MANIFEST}")
    print(json.dumps(manifest, indent=2))
    print(
        "\nNext Aura boot will use this fused model automatically. "
        "If AURA_LLM__MLX_MODEL_PATH is set in .env it still wins — "
        "remove or update that line to let the manifest drive."
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def mark_crsm_loop_consumed_after_training(
    fused_path: Path,
    *,
    manifest_path: Path | None = None,
    source: str = "training.train_and_fuse",
    required: bool = False,
) -> bool:
    """Close the CRSM→LoRA monitor only after real train/fuse evidence exists."""
    def _reject(reason: str) -> bool:
        message = f"CRSM loop not marked consumed: {reason}"
        if required:
            raise SystemExit(message)
        print(f"\n{message}")
        return False

    manifest_path = manifest_path or CRSM_INTEGRATION_MANIFEST
    if not CRSM_DATASET.exists():
        return _reject(f"capture dataset is missing: {CRSM_DATASET}")
    manifest = _read_json(manifest_path)
    if not manifest:
        return _reject(f"missing {manifest_path}; rebuild the dataset before training")

    try:
        dataset_state = _jsonl_file_stats(CRSM_DATASET)
    except OSError as exc:
        return _reject(f"capture dataset identity unavailable: {type(exc).__name__}: {exc}")

    source_lines = int(manifest.get("source_lines", 0) or 0)
    source_size = int(manifest.get("source_size", -1) or -1)
    source_sha256 = str(manifest.get("source_sha256") or "")
    accepted = int(manifest.get("accepted", 0) or 0)
    rejected = max(0, source_lines - accepted)

    if source_lines <= 0:
        return _reject("integration manifest saw no source captures")
    if (
        not source_sha256
        or source_sha256 != str(dataset_state.get("sha256") or "")
        or source_size != int(dataset_state.get("size", -2) or -2)
        or source_lines != int(dataset_state.get("lines", -2) or -2)
    ):
        return _reject("capture dataset identity changed after dataset selection")
    rejected_by_reason = dict(manifest.get("rejected_by_reason") or {})
    if int(rejected_by_reason.get("over_max_examples", 0) or 0) > 0:
        return _reject("eligible captures exceeded the bounded dataset selection and remain untrained")

    from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

    governance = delegated_governance_provenance()
    marker_payload = {
        "model_path": str(fused_path),
        "lines_consumed": source_lines,
        "accepted_lines": accepted,
        "rejected_lines": rejected,
        "manifest_path": str(manifest_path),
        "source": source,
    }
    if governance:
        marker_payload.update(
            {
                "governance_receipt_id": governance.get("will_receipt_id"),
                "authority_intent_id": governance.get("executive_intent_id"),
            }
        )
    marked = get_crsm_loop_monitor().mark_dataset_consumed(**marker_payload)
    if marked is not True:
        raise SystemExit("CRSM training completed, but the final consumed marker was not committed.")
    print(
        "\nMarked CRSM captures handled after successful train/fuse: "
        f"{accepted} trained, {rejected} retired."
    )
    return True


def record_crsm_delta_training_state(
    *,
    adapter_dir: Path,
    fused_path: Path,
    manifest_path: Path = CRSM_DELTA_MANIFEST,
    iters: int | None = None,
    max_seq_length: int | None = None,
) -> None:
    """Persist operator-visible evidence for the bounded CRSM delta run."""
    state_path = ADAPTER_DIR / "training_state.json"
    try:
        state = _read_json(state_path)
        manifest = _read_json(manifest_path)
        output = dict(manifest.get("output") or {})
        payload = {
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "adapter_path": str(adapter_dir),
            "fused_model_path": str(fused_path),
            "manifest_path": str(manifest_path),
            "source_lines": int(manifest.get("source_lines", 0) or 0),
            "accepted": int(manifest.get("accepted", 0) or 0),
            "rejected": max(0, int(manifest.get("source_lines", 0) or 0) - int(manifest.get("accepted", 0) or 0)),
            "retention_examples": int(output.get("retention_examples", 0) or 0),
            "train_sha256": (dict(output.get("train") or {})).get("sha256"),
            "valid_sha256": (dict(output.get("valid") or {})).get("sha256"),
            "iters": iters,
            "max_seq_length": max_seq_length,
            "status": "fused_published_marker_ready",
        }
        governance = delegated_governance_provenance()
        if governance:
            payload["governance"] = governance
        state["crsm_delta"] = payload
        atomic_write_text(
            state_path,
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"failed to record CRSM delta training state: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--mark-crsm-consumed",
        action="store_true",
        help=(
            "Mark CRSM captures consumed after fuse/publish even when --skip-train "
            "is used. Intended for run_unattended resume_training.py, where the "
            "training step happened in a separate process before this fuse call."
        ),
    )
    parser.add_argument(
        "--crsm-delta",
        action="store_true",
        help=(
            "Run a bounded real LoRA update over the current CRSM captures plus "
            "retention examples, fuse it, publish it, and then mark CRSM consumed."
        ),
    )
    parser.add_argument(
        "--crsm-delta-iters",
        type=int,
        default=None,
        help="Override AURA_CRSM_DELTA_ITERS for the bounded CRSM LoRA update.",
    )
    parser.add_argument(
        "--crsm-delta-max-examples",
        type=int,
        default=None,
        help="Maximum eligible CRSM examples to include in the bounded delta dataset.",
    )
    parser.add_argument(
        "--crsm-delta-retention-examples",
        type=int,
        default=None,
        help="Retention examples sampled from the existing corpus for the bounded delta dataset.",
    )
    parser.add_argument(
        "--crsm-delta-max-seq-length",
        type=int,
        default=None,
        help="Override AURA_CRSM_DELTA_MAX_SEQ_LENGTH for the bounded CRSM LoRA update.",
    )
    parser.add_argument(
        "--base-model",
        default=str(os.environ.get("AURA_LORA_BASE_MODEL", "")).strip() or None,
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    enforce_live_delegated_authority(crsm_delta=args.crsm_delta, tag=args.tag)

    model_basis = resolve_training_model_basis(args.base_model)
    base_model = model_basis.path
    inherited_pipeline_lane = consume_inherited_pipeline_lane()
    if _env_flag("AURA_TRAINING_ALLOW_LIVE_AURA") and _env_flag("AURA_LAUNCHED_FROM_APP"):
        if not inherited_pipeline_lane or not delegated_governance_provenance():
            sys.exit(
                "live Aura training coexistence requires delegated governance and model-lane ownership"
            )

    print("=" * 60)
    print("  AURA TRAIN → FUSE → PUBLISH PIPELINE")
    print("=" * 60)
    print(f"  base_model: {base_model}")
    print(f"  adapter:    {ADAPTER_DIR}")
    print(f"  output dir: {FUSED_BASE_DIR}")
    print(f"  tag:        {args.tag or '(none)'}")
    print("=" * 60)

    if args.crsm_delta and args.skip_train:
        sys.exit("--crsm-delta requires a real training step; refusing --skip-train shortcut.")

    if not _env_flag("AURA_TRAINING_BYPASS_PREFLIGHT"):
        enforce_training_preflight(
            base_model=base_model,
            skip_train=args.skip_train,
            crsm_delta=args.crsm_delta,
        )
    else:
        print("\nTraining preflight bypassed by AURA_TRAINING_BYPASS_PREFLIGHT=1.")
    if args.preflight_only:
        print("\nPreflight-only mode complete; no dataset, training, fuse, or publish actions executed.")
        return

    adapter_dir = ADAPTER_DIR
    crsm_marker_manifest = CRSM_INTEGRATION_MANIFEST
    crsm_marker_source = "training.train_and_fuse"
    crsm_delta_adapter_dir: Path | None = None

    if args.crsm_delta:
        build_crsm_delta_dataset(
            max_crsm_examples=args.crsm_delta_max_examples,
            retention_examples=args.crsm_delta_retention_examples,
        )
        adapter_dir = train_crsm_delta_lora(
            base_model=base_model,
            model_basis=model_basis,
            data_dir=CRSM_DELTA_DATA_DIR,
            iters=args.crsm_delta_iters,
            max_seq_length=args.crsm_delta_max_seq_length,
        )
        crsm_delta_adapter_dir = adapter_dir
        crsm_marker_manifest = CRSM_DELTA_MANIFEST
        crsm_marker_source = "training.train_and_fuse.crsm_delta"
    elif not args.skip_dataset:
        build_dataset()
    if not args.crsm_delta and not args.skip_train:
        train_lora(
            base_model=base_model,
            model_basis=model_basis,
            resume=args.resume,
        )
    fused_path = fuse_adapter(
        base_model=base_model,
        model_basis=model_basis,
        tag=args.tag,
        adapter_dir=adapter_dir,
    )
    verify_load(fused_path)
    publish_manifest(fused_path, tag=args.tag, base_model=base_model)
    if args.crsm_delta and crsm_delta_adapter_dir is not None:
        record_crsm_delta_training_state(
            adapter_dir=crsm_delta_adapter_dir,
            fused_path=fused_path,
            manifest_path=CRSM_DELTA_MANIFEST,
            iters=args.crsm_delta_iters or _env_int("AURA_CRSM_DELTA_ITERS", 600, minimum=25, maximum=5000),
            max_seq_length=(
                args.crsm_delta_max_seq_length
                or _env_int("AURA_CRSM_DELTA_MAX_SEQ_LENGTH", 2048, minimum=512, maximum=4096)
            ),
        )
    # The consumed marker is the public transaction commit point. All durable
    # model and training-state artifacts must exist before it can close the loop.
    if args.crsm_delta or not args.skip_train or args.mark_crsm_consumed:
        mark_crsm_loop_consumed_after_training(
            fused_path,
            manifest_path=crsm_marker_manifest,
            source=crsm_marker_source,
            required=args.crsm_delta,
        )


if __name__ == "__main__":
    main()
