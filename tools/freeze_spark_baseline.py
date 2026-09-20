#!/usr/bin/env python3
"""Freeze the SPARK-004 baseline bundle before the treatment changes.

Builds one immutable, hash-bound, Ed25519-signed bundle over the resident
checkpoint, tokenizer/config behavior bundle, adapters, decoding contract,
task-generator identity, control manifests, resource envelope,
randomization policy, and the current vanilla/RLC measurements.  The
bundle is the fixed "before" that SPARK-069 admission and SPARK-070/071
comparisons reference; drift against it is detectable by independent
re-verification (`verify` subcommand) without trusting this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.frontier_tasks import (  # noqa: E402
    CURRENT_EXCLUDED_TRAINING_FAMILIES,
    CURRENT_REGISTRY_VERSION,
    FRONTIER_DOMAINS,
)
from core.brain.llm.latent_cortex.frozen_baseline import (  # noqa: E402
    CONTROL_MANIFEST_DIR,
    MEASUREMENT_DIR,
    FrozenBaselineError,
    build_frozen_baseline_certificate,
    publish_frozen_baseline_bundle,
    sign_frozen_baseline_certificate,
    verify_frozen_baseline,
    verify_frozen_baseline_model,
    verify_frozen_baseline_sources,
)
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
    model_behavior_bundle_identity,
    personality_bundle_identity,
    runtime_environment_identity,
)
from core.learning.recurrence_curriculum import (  # noqa: E402
    RECURRENCE_TRAINING_FAMILIES,
)
from core.runtime.resource_observation import get_resource_observer  # noqa: E402

DEFAULT_MODEL = "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
DEFAULT_EXECUTION_SPEC = (
    "config/latent_cortex/resident_32b_recurrent_grpo_execution_spec.json"
)
DEFAULT_ENVELOPE_MANIFEST = (
    "config/latent_cortex/resident_32b_recurrent_grpo_cp305_preregistration.json"
)
DEFAULT_OUTPUT_ROOT = "artifacts/closeout/latent_cortex/spark004_frozen_baseline"
DEFAULT_SIGNING_KEY = "~/.aura/trust/contamination_audit_ed25519_private.pem"
DEFAULT_TRUST_PUBLIC_KEY = "~/.aura/trust/contamination_audit_ed25519_public.pem"

# Preregistered seeds already in force for the current campaign generation
# (tools/prepare_resident_recurrent_grpo_campaign.py) and the intrinsic
# accuracy gate (artifacts cp227): the baseline binds them so any later run
# that silently changes them is detectable.
TRAINING_SEED = 2026072102
ACCURACY_GATE_EVAL_SEED = 424242

# Worker decode defaults from core/brain/llm/latent_cortex/worker_handler.py.
DECODE_MAX_TOKENS = 512
DECODE_TEMPERATURE = 0.0
DECODE_TOP_P = 1.0

TASK_GENERATOR_SOURCES = (
    "core/brain/llm/latent_cortex/frontier_tasks.py",
    "core/brain/llm/latent_cortex/task_verifiers.py",
    "core/learning/recurrence_curriculum.py",
    "tools/train_grpo.py",
)
RANDOMIZATION_SOURCES = (
    "core/brain/llm/latent_cortex/branches.py",
    "core/brain/llm/latent_cortex/schedules.py",
)
DEFAULT_MEASUREMENTS = (
    (
        "cp227_intrinsic_accuracy_gate",
        "paired",
        "artifacts/closeout/latent_cortex/cp227_accuracy_gate/accuracy_gate.json",
    ),
    (
        "cp227_accuracy_gate_detached_receipt",
        "paired",
        "artifacts/closeout/latent_cortex/cp227_accuracy_gate/detached_receipt.json",
    ),
    (
        "cp305_post_training_controller_verdict",
        "rlc",
        "artifacts/closeout/latent_cortex/cp305_resident_32b_recurrent_grpo/"
        "post-training/controller_verdict.json",
    ),
    (
        "cp305_grpo_training_receipt",
        "rlc",
        "artifacts/closeout/latent_cortex/cp305_resident_32b_recurrent_grpo/"
        "training/grpo_receipt.json",
    ),
)
_MAX_SUMMARY_ITEMS = 24


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


def _shallow_summary(document: Mapping[str, object]) -> dict[str, object]:
    """Keep small scalar facts (one nested level) for the certificate."""

    summary: dict[str, object] = {}
    for key in sorted(document):
        if len(summary) >= _MAX_SUMMARY_ITEMS:
            break
        value = document[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            if not isinstance(value, str) or len(value) <= 300:
                summary[key] = value
            continue
        if isinstance(value, Mapping):
            nested = {
                nested_key: nested_value
                for nested_key, nested_value in sorted(value.items())[
                    :_MAX_SUMMARY_ITEMS
                ]
                if isinstance(nested_key, str)
                and (
                    isinstance(nested_value, (int, float, bool))
                    or (
                        isinstance(nested_value, str) and len(nested_value) <= 120
                    )
                    or (
                        isinstance(nested_value, Mapping)
                        and all(
                            isinstance(leaf, (int, float, bool, str))
                            for leaf in nested_value.values()
                        )
                        and len(nested_value) <= _MAX_SUMMARY_ITEMS
                    )
                )
            }
            if nested:
                summary[key] = nested
    return summary


def _source_records(repo: Path, relative_paths: tuple[str, ...]) -> list[dict]:
    records = []
    for relative in relative_paths:
        payload = (repo / relative).read_bytes()
        records.append(
            {
                "repo_path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    return records


def _relative_to_repo(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path)


def _freeze(arguments: argparse.Namespace) -> int:
    repo = REPO_ROOT
    git_commit = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not arguments.allow_dirty:
        return _fail(
            "worktree has tracked modifications; commit first so the baseline "
            "binds a real commit (or pass --allow-dirty for rehearsals)"
        )

    model_root = Path(arguments.model).expanduser()
    if not model_root.is_absolute():
        model_root = repo / model_root
    if not model_root.is_dir():
        return _fail(f"model checkpoint directory not found: {model_root}")

    spec_path = Path(arguments.execution_spec).expanduser()
    if not spec_path.is_absolute():
        spec_path = repo / spec_path
    execution_spec = json.loads(spec_path.read_text())

    envelope_path = Path(arguments.envelope_manifest).expanduser()
    if not envelope_path.is_absolute():
        envelope_path = repo / envelope_path
    envelope_document = json.loads(envelope_path.read_text())
    declared_envelope = envelope_document.get("training", {}).get(
        "resource_envelope"
    )
    if not isinstance(declared_envelope, Mapping) or not declared_envelope:
        return _fail(f"no training.resource_envelope in {envelope_path}")

    control_dir = repo / "config" / "latent_cortex"
    manifest_paths = sorted(control_dir.glob("*.json"))
    if not manifest_paths:
        return _fail(f"no control manifests under {control_dir}")

    measurement_arguments = arguments.measurement or [
        f"{name}={role}={path}" for name, role, path in DEFAULT_MEASUREMENTS
    ]
    measurement_inputs: list[tuple[str, str, Path]] = []
    for entry in measurement_arguments:
        try:
            name, role, raw_path = entry.split("=", 2)
        except ValueError:
            return _fail(f"measurement must be name=role=path: {entry!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            candidate = repo / path
            path = candidate if candidate.exists() else (
                Path(arguments.measurement_root).expanduser() / raw_path
            )
        if not path.is_file():
            return _fail(f"measurement receipt not found: {raw_path}")
        measurement_inputs.append((name, role, path))

    print(f"hashing checkpoint weights under {model_root} ...", flush=True)
    checkpoint_identity = full_weight_checkpoint_identity(model_root)
    behavior_identity = model_behavior_bundle_identity(model_root)
    print(
        f"checkpoint fingerprint {checkpoint_identity['fingerprint'][:16]}... "
        f"({checkpoint_identity['files']} weight files)",
        flush=True,
    )

    file_payloads: dict[str, bytes] = {}
    control_manifests = []
    for path in manifest_paths:
        payload = path.read_bytes()
        bundle_path = f"{CONTROL_MANIFEST_DIR}/{path.name}"
        control_manifests.append(
            {
                "name": path.stem,
                "bundle_path": bundle_path,
                "source_path": _relative_to_repo(path, repo),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
        file_payloads[bundle_path] = payload

    measurements = []
    for name, role, path in measurement_inputs:
        payload = path.read_bytes()
        document = json.loads(payload)
        if not isinstance(document, Mapping):
            return _fail(f"measurement receipt is not an object: {path}")
        bundle_path = f"{MEASUREMENT_DIR}/{name}.json"
        measurements.append(
            {
                "name": name,
                "role": role,
                "bundle_path": bundle_path,
                "source_path": _relative_to_repo(path, repo),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "summary": _shallow_summary(document),
            }
        )
        file_payloads[bundle_path] = payload

    observer = get_resource_observer()
    memory = observer.memory(include_process_tree=False)
    compute = observer.compute()
    if not memory.available or not compute.available:
        return _fail("host resource observation unavailable for frozen baseline")

    material = {
        "schema": "aura.latent_cortex.spark_frozen_baseline.v1",
        "baseline_id": arguments.baseline_id
        or f"spark004-baseline-{time.strftime('%Y%m%d-%H%M%S')}-{git_commit[:8]}",
        "purpose": (
            "SPARK-004 frozen pre-treatment baseline for SPARK-069 admission "
            "and SPARK-070/071 falsification comparisons"
        ),
        "frozen_at_unix": int(time.time()),
        "git_commit": git_commit,
        "worktree_clean": not dirty,
        "environment": {
            "runtime": runtime_environment_identity(),
            "observed_physical_memory_bytes": int(memory.total_bytes),
            "observed_cpu_count": max(1, int(compute.cpu_count)),
        },
        "model": {
            "path": str(model_root.resolve()),
            "checkpoint": checkpoint_identity,
            "behavior_bundle": behavior_identity,
        },
        "adapters": {
            "personality": personality_bundle_identity(
                arguments.personality_adapter
            ),
            "attached_at_baseline": sorted(arguments.attached_adapter or []),
        },
        "decoding": {
            "execution_spec": execution_spec,
            "execution_spec_sha256": hashlib.sha256(
                json.dumps(
                    execution_spec,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "sampling": {
                "decode_max_tokens": DECODE_MAX_TOKENS,
                "decode_temperature": DECODE_TEMPERATURE,
                "decode_top_p": DECODE_TOP_P,
            },
        },
        "task_generators": {
            "registry_version": CURRENT_REGISTRY_VERSION,
            "frontier_domains": list(FRONTIER_DOMAINS),
            "excluded_training_families": list(
                CURRENT_EXCLUDED_TRAINING_FAMILIES
            ),
            "recurrence_training_families": list(RECURRENCE_TRAINING_FAMILIES),
            "sources": _source_records(repo, TASK_GENERATOR_SOURCES),
        },
        "control_manifests": control_manifests,
        "resource_envelope": {"declared": dict(declared_envelope)},
        "randomization": {
            "training_seed": TRAINING_SEED,
            "slot_seed": int(execution_spec.get("slot_seed", 0)),
            "eval_seed": ACCURACY_GATE_EVAL_SEED,
            "seed_policy": "fixed_preregistered_seeds",
            "sources": _source_records(repo, RANDOMIZATION_SOURCES),
        },
        "measurements": measurements,
    }

    certificate = build_frozen_baseline_certificate(material)

    signature = None
    trusted_public_key_pem = None
    signing_key_path = Path(arguments.signing_key).expanduser()
    if signing_key_path.is_file():
        from cryptography.hazmat.primitives import serialization

        private_key = serialization.load_pem_private_key(
            signing_key_path.read_bytes(), password=None
        )
        signature = sign_frozen_baseline_certificate(
            certificate, private_key=private_key
        )
        public_key_path = Path(arguments.trust_public_key).expanduser()
        if public_key_path.is_file():
            trusted_public_key_pem = public_key_path.read_bytes()
    elif arguments.require_signature:
        return _fail(f"signing key not found: {signing_key_path}")

    output_root = Path(arguments.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = repo / output_root
    publish_frozen_baseline_bundle(
        output_root,
        certificate=certificate,
        file_payloads=file_payloads,
        signature=signature,
    )
    verified = verify_frozen_baseline(
        output_root, trusted_public_key_pem=trusted_public_key_pem
    )
    verify_frozen_baseline_sources(verified, repo_root=repo)
    if not arguments.skip_model_verify:
        print("re-hashing checkpoint for independent model verification ...", flush=True)
        verify_frozen_baseline_model(verified, model_root=model_root)

    print(f"baseline_id: {verified['baseline_id']}")
    print(f"certificate_sha256: {verified['certificate_sha256']}")
    print(f"bundle: {output_root}")
    print(f"signed: {signature is not None}")
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    root = Path(arguments.bundle).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    trusted = None
    if arguments.trust_public_key:
        public_key_path = Path(arguments.trust_public_key).expanduser()
        if not public_key_path.is_file():
            return _fail(f"trusted public key not found: {public_key_path}")
        trusted = public_key_path.read_bytes()
    certificate = verify_frozen_baseline(root, trusted_public_key_pem=trusted)
    if arguments.verify_sources:
        verify_frozen_baseline_sources(certificate, repo_root=REPO_ROOT)
    if arguments.verify_model:
        verify_frozen_baseline_model(certificate)
    print(f"baseline_id: {certificate['baseline_id']}")
    print(f"certificate_sha256: {certificate['certificate_sha256']}")
    print("verified: true")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="build and seal the baseline")
    freeze.add_argument("--model", default=DEFAULT_MODEL)
    freeze.add_argument("--execution-spec", default=DEFAULT_EXECUTION_SPEC)
    freeze.add_argument("--envelope-manifest", default=DEFAULT_ENVELOPE_MANIFEST)
    freeze.add_argument("--measurement", action="append", default=None)
    freeze.add_argument(
        "--measurement-root",
        default=str(Path("~/.aura/live-source").expanduser()),
        help="fallback root for repo-relative measurement paths",
    )
    freeze.add_argument("--personality-adapter", default=None)
    freeze.add_argument("--attached-adapter", action="append", default=None)
    freeze.add_argument("--baseline-id", default=None)
    freeze.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    freeze.add_argument("--signing-key", default=DEFAULT_SIGNING_KEY)
    freeze.add_argument("--trust-public-key", default=DEFAULT_TRUST_PUBLIC_KEY)
    freeze.add_argument("--require-signature", action="store_true")
    freeze.add_argument("--allow-dirty", action="store_true")
    freeze.add_argument("--skip-model-verify", action="store_true")
    freeze.set_defaults(handler=_freeze)

    verify = subparsers.add_parser("verify", help="re-verify a sealed bundle")
    verify.add_argument("bundle")
    verify.add_argument("--trust-public-key", default=None)
    verify.add_argument("--verify-sources", action="store_true")
    verify.add_argument("--verify-model", action="store_true")
    verify.set_defaults(handler=_verify)

    arguments = parser.parse_args()
    try:
        return arguments.handler(arguments)
    except FrozenBaselineError as error:
        return _fail(f"frozen baseline rejected: {error.code}")
    except subprocess.CalledProcessError as error:
        return _fail(f"git failed: {error.stderr.strip()}")


if __name__ == "__main__":
    raise SystemExit(main())
