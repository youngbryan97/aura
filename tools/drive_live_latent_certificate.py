#!/usr/bin/env python
"""Drive one compound latent turn through the LIVE desktop app → certificate.

Sends a real user message to the installed app's /api/chat (desktop surface),
extracts the latent-cortex trace + episode receipt from the live response,
grades the PASS conditions the CP-series certificates use, and writes
artifacts/current/cp<NN>_live_latent_turn.json.

Owner-operated, bounded, and honest: a turn that fell back, truncated, or
failed any receipt contract writes verdict FAIL with the reasons — never a
silent pass. Run only when the live app is up and idle.

  .venv/bin/python tools/drive_live_latent_certificate.py --checkpoint 119 \
      [--message "..."] [--timeout 240]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.output_quality import (  # noqa: E402
    evaluate_latent_output,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

# NOT self-referential: a prompt about her own processes routes through the
# self-process grounding contract, which excludes the latent lane by design
# (CP119 first attempt proved this). Compound engineering question, 4 facets.
DEFAULT_MESSAGE = (
    "Compare optimistic and pessimistic locking for a hot task queue, choose "
    "which one you would use in a single-host async runtime, explain why, and "
    "verify your choice with one concrete failure scenario."
)


def _git_head() -> str:
    try:
        completed = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            timeout=10,
            read_only=True,
            source="proof_tooling:live_latent_certificate_git_head",
            accelerator_capability="none",
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _evaluate_live_response(
    body: dict[str, Any],
    *,
    message: str,
    exact_commit: str,
    http_status: int,
) -> dict[str, Any]:
    """Independently grade the exact API payload against the CP119 contract."""

    contract = _mapping(body.get("live_turn_contract"))
    receipt = _mapping(contract.get("latent_cortex_receipt"))
    runtime_identity = _mapping(receipt.get("runtime_identity"))
    response_text = str(body.get("response") or "")
    objective_sha256 = _sha256_text(message)
    response_sha256 = _sha256_text(response_text)
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    def require(name: str, condition: Any) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            reasons.append(name)

    require("http_200", http_status == 200)
    require("live_turn_contract_present", bool(contract))
    for key in (
        "full_mind_path",
        "authentic_cognitive_reply",
        "foreground_model_generation_consumed",
        "single_owner_model_generation_proven",
        "latent_cortex_selected",
        "latent_cortex_attempted",
        "latent_cortex_succeeded",
        "latent_cortex_identity_bound",
        "latent_cortex_path_proven",
        "latent_cortex_raw_output_quality_proven",
        "latent_cortex_final_output_quality_proven",
        "latent_cortex_public_output_quality_proven",
        "latent_cortex_output_quality_proven",
        "final_requested_output_contract_proven",
    ):
        require(key, contract.get(key) is True)
    require(
        "exactly_one_foreground_model_generation",
        contract.get("foreground_model_generation_count") == 1,
    )
    for key in (
        "latent_cortex_fallback_used",
        "bounded_contract_used",
        "legacy_fallback_used",
    ):
        require(f"{key}_false", contract.get(key) is False)
    require(
        "latent_response_path",
        contract.get("response_path") == "cognitive_engine_latent_cortex",
    )
    require(
        "no_missing_full_mind_proofs",
        contract.get("full_mind_missing_proofs") in (None, []),
    )

    require("receipt_complete", receipt.get("last_stage") == "complete")
    from core.brain.llm.latent_cortex.runtime_integrity import (
        runtime_integrity_safe,
    )

    worker_identity = receipt.get("worker_identity")
    require("worker_identity_present", isinstance(worker_identity, dict))
    require(
        "measured_runtime_integrity",
        runtime_integrity_safe(
            receipt.get("runtime_integrity"),
            require_worker=True,
            expected_episode_id=str(receipt.get("episode_id") or ""),
            expected_input_tokens_sha256=str(
                receipt.get("input_tokens_sha256") or ""
            ),
            expected_worker_identity=(
                worker_identity if isinstance(worker_identity, dict) else None
            ),
            expected_fast_weights_applied=(
                receipt.get("fast_weights_applied") is True
            ),
            expected_fast_weights_attach_attempted=(
                receipt.get("fast_weights_attach_attempted") is True
            ),
            expected_checkpoint_fingerprint=str(
                receipt.get("checkpoint_fingerprint") or ""
            ),
            expected_checkpoint_method=str(
                receipt.get("checkpoint_fingerprint_method") or ""
            ),
            expected_checkpoint_file_count=receipt.get(
                "checkpoint_file_count"
            ),
        ),
    )
    require(
        "checkpoint_params_unchanged_telemetry",
        receipt.get("params_unchanged") is True,
    )
    require(
        "decode_complete",
        receipt.get("decode_termination")
        in {
            "eos",
            "token_limit",
            "token_limit_sentence_grace",
            "wall_reserve_sentence_grace",
        },
    )
    require("decode_bridge_applied", receipt.get("decode_bridge_applied") is True)
    require(
        "virtual_width_executed",
        type(receipt.get("n_branches")) is int and receipt.get("n_branches", 0) >= 2,
    )
    require(
        "recurrence_executed",
        type(receipt.get("steps_taken")) is int and receipt.get("steps_taken", 0) >= 2,
    )
    require("latent_optimization_applied", receipt.get("latent_opt_applied") is True)
    require(
        "latent_optimization_step_recorded",
        type(receipt.get("latent_opt_steps")) is int
        and receipt.get("latent_opt_steps", 0) >= 1,
    )
    require("fast_weights_applied", receipt.get("fast_weights_applied") is True)
    require(
        "fast_weights_erased_telemetry",
        receipt.get("fast_weights_erased") is True,
    )

    require("runtime_identity_bound", runtime_identity.get("identity_bound") is True)
    require("signed_app_launch", runtime_identity.get("launch_mode") == "signed_app")
    require(
        "installed_app_verified",
        runtime_identity.get("installed_app_required") is True
        and runtime_identity.get("installed_app_verified") is True,
    )
    require("source_verified", runtime_identity.get("source_verified") is True)
    require(
        "exact_source_commit",
        bool(exact_commit)
        and runtime_identity.get("source_commit") == exact_commit,
    )
    require("source_main", runtime_identity.get("source_branch") == "main")
    require(
        "source_clean",
        runtime_identity.get("source_dirty") is False
        and runtime_identity.get("source_change_count") == 0,
    )
    require(
        "desktop_bundle_identity",
        runtime_identity.get("bundle_identifier") == "com.aura.desktop",
    )
    require("runtime_identity_issues_empty", runtime_identity.get("issues") == [])
    require(
        "resident_32b_logical_identity",
        type(receipt.get("worker_model_parameter_count")) is int
        and receipt.get("worker_model_parameter_count", 0) >= 20_000_000_000
        and receipt.get("worker_model_parameter_count_basis")
        == "architecture_config_logical",
    )

    raw_quality = _mapping(receipt.get("output_quality"))
    final_quality = _mapping(contract.get("latent_cortex_final_output_quality"))
    public_quality = _mapping(contract.get("latent_cortex_public_output_quality"))
    for label, quality in (
        ("raw", raw_quality),
        ("final", final_quality),
        ("public", public_quality),
    ):
        require(
            f"{label}_quality_receipt_valid",
            quality.get("schema") == "aura.latent_output_quality.v1"
            and quality.get("policy") == "resident_latent_product_quality_v1"
            and quality.get("passed") is True
            and quality.get("reasons") == []
            and quality.get("objective_sha256") == objective_sha256,
        )
    require(
        "public_quality_binds_exact_api_bytes",
        public_quality.get("text_sha256") == response_sha256,
    )

    def require_transition(
        name: str,
        before_quality: dict[str, Any],
        after_quality: dict[str, Any],
        *,
        direct_match_key: str,
        chain_key: str,
    ) -> None:
        before_hash = before_quality.get("text_sha256")
        after_hash = after_quality.get("text_sha256")
        chain = _mapping(contract.get(chain_key))
        direct = bool(
            before_hash
            and before_hash == after_hash
            and contract.get(direct_match_key) is True
        )
        mutated = bool(
            before_hash
            and after_hash
            and before_hash != after_hash
            and chain.get("passed") is True
            and type(chain.get("chain_length")) is int
            and chain.get("chain_length", 0) > 0
        )
        require(name, direct or mutated)

    require_transition(
        "raw_to_final_text_bound",
        raw_quality,
        final_quality,
        direct_match_key="latent_cortex_raw_final_quality_hash_match",
        chain_key="latent_cortex_raw_final_mutation_chain",
    )
    require_transition(
        "final_to_public_text_bound",
        final_quality,
        public_quality,
        direct_match_key="latent_cortex_final_public_quality_hash_match",
        chain_key="latent_cortex_final_public_mutation_chain",
    )
    require_transition(
        "raw_to_public_text_bound",
        raw_quality,
        public_quality,
        direct_match_key="latent_cortex_raw_public_quality_hash_match",
        chain_key="latent_cortex_output_mutation_chain",
    )

    independent_quality = evaluate_latent_output(
        response_text,
        generated_tokens=receipt.get("decode_generated_tokens"),
        termination=receipt.get("decode_termination"),
        objective=message,
    )
    require(
        "independent_public_regrade_passed",
        independent_quality.get("passed") is True
        and independent_quality.get("text_sha256") == response_sha256
        and independent_quality.get("objective_sha256") == objective_sha256,
    )
    require(
        "single_owner_route_contract",
        contract.get("response_path") == "cognitive_engine_latent_cortex"
        and contract.get("latent_cortex_selected") is True
        and contract.get("latent_cortex_attempted") is True
        and contract.get("latent_cortex_succeeded") is True
        and contract.get("latent_cortex_fallback_used") is False,
    )

    return {
        "checks": checks,
        "fail_reasons": reasons,
        "contract": contract,
        "latent_receipt": receipt,
        "independent_public_regrade": independent_quality,
        "response_sha256": response_sha256,
        "objective_sha256": objective_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    session_id = f"cp{args.checkpoint}-live-latent"
    payload = json.dumps(
        {"message": args.message, "session_id": session_id}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{args.host}/api/chat",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Aura-Surface": "desktop",
        },
        method="POST",
    )
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "artifacts" / "current" / f"cp{args.checkpoint}_live_latent_turn.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exact_commit = _git_head()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            body = {"response": "", "transport_error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - operator tool: report, don't mask
        elapsed = time.monotonic() - started
        certificate = {
            "schema": "aura.live_latent_certificate.v2",
            "checkpoint": args.checkpoint,
            "exact_commit": exact_commit,
            "request": {"message": args.message, "session_id": session_id},
            "http": {
                "status": None,
                "elapsed_s": round(elapsed, 3),
                "error_class": type(exc).__name__,
                "error": str(exc)[:500],
            },
            "fail_reasons": ["live_transport_failed"],
            "verdict": "FAIL",
            "generated_at": time.time(),
        }
        out_path.write_text(json.dumps(certificate, indent=1, sort_keys=True))
        print(f"verdict=FAIL elapsed={elapsed:.1f}s reasons=['live_transport_failed']")
        print(f"certificate -> {out_path}")
        return 2
    elapsed = time.monotonic() - started
    evaluation = _evaluate_live_response(
        body,
        message=args.message,
        exact_commit=exact_commit,
        http_status=status,
    )
    reasons = list(evaluation["fail_reasons"])
    contract = dict(evaluation["contract"])
    receipt = dict(evaluation["latent_receipt"])
    text = str(body.get("response") or "")

    verdict = "PASS" if not reasons else "FAIL"
    certificate = {
        "schema": "aura.live_latent_certificate.v2",
        "checkpoint": args.checkpoint,
        "exact_commit": exact_commit,
        "request": {"message": args.message, "session_id": session_id},
        "http": {"status": status, "elapsed_s": round(elapsed, 3)},
        "response": {
            "response": text[:4000],
            "response_sha256": evaluation["response_sha256"],
            "status": body.get("status"),
            "full_mind_path": contract.get("full_mind_path"),
            "authentic_cognitive_reply": contract.get("authentic_cognitive_reply"),
            "latent_cortex_succeeded": contract.get("latent_cortex_succeeded"),
            "latent_cortex_identity_bound": contract.get("latent_cortex_identity_bound"),
            "latent_cortex_output_quality_proven": contract.get(
                "latent_cortex_output_quality_proven"
            ),
        },
        "acceptance_checks": evaluation["checks"],
        "independent_public_regrade": evaluation["independent_public_regrade"],
        "live_turn_contract": contract,
        "latent_receipt": receipt,
        "fail_reasons": reasons,
        "verdict": verdict,
        "generated_at": time.time(),
    }
    out_path.write_text(json.dumps(certificate, indent=1, sort_keys=True))
    print(f"verdict={verdict} elapsed={elapsed:.1f}s reasons={reasons or 'none'}")
    print(f"certificate -> {out_path}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
