#!/usr/bin/env python3
"""tools/measure_fusion_channel.py — does her substrate reach the forward pass?

Operationally: loads a model, installs the real affective steering hooks on it,
runs `core.consciousness.fusion_probe`, and writes a certificate for the
smallest alpha that holds. The worker reads that certificate to decide whether
the channel may be open on a turn somebody is waiting for; without one it stays
shut.

Why the distribution and not only the text
------------------------------------------
The A/B this replaces sampled two strings and compared bytes. Under greedy
decoding the argmax survives a real change in the logits, so the strings matched
and the channel was recorded as dead while it was working. The probe keeps the
text comparison and stops it being the only evidence.

Usage:

    tools/measure_fusion_channel.py --sweep 0.05,0.1,0.2,0.3
    tools/measure_fusion_channel.py --model-path /Users/you/.aura/models/some-model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")
os.environ.setdefault("AURA_LOG_DIR", "/tmp/aura_fusion_probe_logs")


def _snapshot_dir(repository_id: str) -> Path:
    """The local checkpoint `load()` just read, so the identity is of real bytes."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repository_id, local_files_only=True)).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument(
        "--model-path",
        default="",
        help="local checkpoint directory; defaults to the HF snapshot for --model",
    )
    parser.add_argument(
        "--sweep",
        default="0.1,0.2",
        help="comma-separated alphas; the certificate is written for the smallest that holds",
    )
    parser.add_argument("--steps", type=int, default=24, help="decode steps per probe")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    arguments = parser.parse_args()

    from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor
    from core.consciousness.affective_steering import AffectiveSteeringEngine
    from core.consciousness.fusion_certificate import write_certificate
    from core.consciousness.fusion_probe import measure_fusion
    from core.runtime.model_lane_control import standalone_model_lane

    checkpoint = (
        Path(arguments.model_path) if arguments.model_path else _snapshot_dir(arguments.model)
    )
    # Held for the whole probe. This loads a second model on a machine whose
    # resident cortex already holds about twenty gigabytes, and the lane is
    # what stops the two of them meeting.
    with standalone_model_lane(
        owner_id="fusion-channel-probe",
        model_path=str(checkpoint),
        purpose="measurement",
        request_gb=_weight_gigabytes(checkpoint),
        metadata={"tool": "measure_fusion_channel", "model": arguments.model},
    ):
        return _measure(
            arguments,
            checkpoint,
            build_model_artifact_descriptor,
            AffectiveSteeringEngine,
            measure_fusion,
            write_certificate,
        )


def _weight_gigabytes(checkpoint: Path, floor: float = 1.0) -> float:
    """How much the checkpoint's weights weigh, so the lane can price it."""
    total = sum(
        path.stat().st_size
        for pattern in ("*.safetensors", "*.npz", "*.gguf")
        for path in checkpoint.glob(pattern)
    )
    return max(floor, round(total / 1e9, 2))


def _measure(
    arguments,
    checkpoint: Path,
    build_model_artifact_descriptor,
    steering_engine,
    measure_fusion,
    write_certificate,
) -> int:
    from mlx_lm import load

    print(f"loading {arguments.model}", flush=True)
    model, tokenizer = load(arguments.model)
    descriptor = build_model_artifact_descriptor(checkpoint, repository_id=arguments.model)
    digest = str(descriptor["descriptor_sha256"])
    print(f"checkpoint {checkpoint}\nidentity {digest[:16]}", flush=True)

    engine = steering_engine()
    attached = engine.attach(
        model, tokenizer, alpha=0.0, model_path=checkpoint, model_identity=descriptor
    )
    hooks = list(engine._hooks) if attached else []
    if not hooks:
        print("steering did not attach; nothing to certify", file=sys.stderr)
        return 2
    print(f"attached {len(hooks)} hook(s) at layers {[h._layer_idx for h in hooks]}", flush=True)

    def report(certificate) -> None:
        print(
            f"alpha {certificate.alpha}: shift {certificate.distribution_shift:.5f} "
            f"control {certificate.control_shift:.5f} "
            f"separation {certificate.state_separation:.3f} "
            f"changed {certificate.prompts_that_change}/{certificate.prompts} "
            f"margin {certificate.margin_delta:+.4f} "
            f"noise-margin {certificate.control_margin_delta:+.4f} -> "
            f"{'holds' if certificate.holds else certificate.why_not()}",
            flush=True,
        )

    alphas = [float(part) for part in arguments.sweep.split(",") if part.strip()]
    certificates = measure_fusion(
        model,
        tokenizer,
        hooks,
        engine.set_alpha,
        model_identity=digest,
        model_name=arguments.model,
        alphas=alphas,
        steps=arguments.steps,
        seed=arguments.seed,
        runner="tools/measure_fusion_channel.py",
        on_result=report,
    )

    holding = [certificate for certificate in certificates if certificate.holds]
    if not holding:
        print("no alpha earned a certificate; the channel stays shut", flush=True)
        if certificates and not arguments.no_write:
            write_certificate(min(certificates, key=lambda item: item.alpha), root=REPO)
        return 1

    smallest = min(holding, key=lambda item: item.alpha)
    print(json.dumps(smallest.as_json(), indent=2), flush=True)
    if not arguments.no_write:
        print(f"wrote {write_certificate(smallest, root=REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
