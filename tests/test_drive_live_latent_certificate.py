from __future__ import annotations

from subprocess import CompletedProcess

from tests.fixtures.rlc_runtime_integrity import attach_bound_runtime_integrity
from tools import drive_live_latent_certificate as certificate


class _Gateway:
    def __init__(self, result: CompletedProcess[str] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def run(self, argv: list[str], **kwargs: object) -> CompletedProcess[str]:
        self.calls.append((argv, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_git_head_uses_bounded_read_only_subprocess_gateway(monkeypatch) -> None:
    gateway = _Gateway(CompletedProcess(["git"], 0, "abc123\n", ""))
    monkeypatch.setattr(certificate, "get_subprocess_gateway", lambda: gateway)

    assert certificate._git_head() == "abc123"
    assert gateway.calls == [
        (
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
            {
                "cwd": certificate.REPO_ROOT,
                "capture_output": True,
                "timeout": 10,
                "read_only": True,
                "source": "proof_tooling:live_latent_certificate_git_head",
                "accelerator_capability": "none",
            },
        )
    ]


def test_git_head_fails_honestly_when_probe_cannot_run(monkeypatch) -> None:
    gateway = _Gateway(RuntimeError("probe denied"))
    monkeypatch.setattr(certificate, "get_subprocess_gateway", lambda: gateway)

    assert certificate._git_head() == ""


def _passing_live_body(commit: str) -> tuple[dict[str, object], str]:
    answer = (
        "Optimistic locking lets workers proceed concurrently and rejects a stale "
        "version at commit, while pessimistic locking grants one worker an exclusive "
        "queue lease before mutation. I would choose pessimistic locking for a hot "
        "single-host async task queue because contention is expected and one bounded "
        "owner prevents duplicate execution. To verify that choice, cancel worker A "
        "after it acquires the lease, advance the lease timeout, and assert worker B "
        "resumes exactly once while A's stale fencing token cannot publish. That "
        "concrete failure scenario tests recovery, at-most-once publication, and the "
        "main tradeoff: predictable serialization in exchange for less wasted work."
    )
    quality = certificate.evaluate_latent_output(
        answer,
        generated_tokens=120,
        termination="eos",
        objective=certificate.DEFAULT_MESSAGE,
    )
    assert quality["passed"] is True
    empty_chain = {
        "schema": "aura.text_mutation_chain.v1",
        "passed": True,
        "chain_length": 0,
        "reasons": [],
    }
    contract: dict[str, object] = {
        "full_mind_path": True,
        "authentic_cognitive_reply": True,
        "foreground_model_generation_consumed": True,
        "foreground_model_generation_count": 1,
        "single_owner_model_generation_proven": True,
        "latent_cortex_selected": True,
        "latent_cortex_attempted": True,
        "latent_cortex_succeeded": True,
        "latent_cortex_identity_bound": True,
        "latent_cortex_path_proven": True,
        "latent_cortex_raw_output_quality_proven": True,
        "latent_cortex_final_output_quality_proven": True,
        "latent_cortex_public_output_quality_proven": True,
        "latent_cortex_output_quality_proven": True,
        "final_requested_output_contract_proven": True,
        "latent_cortex_fallback_used": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "response_path": "cognitive_engine_latent_cortex",
        "full_mind_missing_proofs": [],
        "latent_cortex_final_output_quality": dict(quality),
        "latent_cortex_public_output_quality": dict(quality),
        "latent_cortex_raw_final_quality_hash_match": True,
        "latent_cortex_final_public_quality_hash_match": True,
        "latent_cortex_raw_public_quality_hash_match": True,
        "latent_cortex_raw_final_mutation_chain": dict(empty_chain),
        "latent_cortex_final_public_mutation_chain": dict(empty_chain),
        "latent_cortex_output_mutation_chain": dict(empty_chain),
        "latent_cortex_receipt": {
            "last_stage": "complete",
            "params_unchanged": True,
            "decode_termination": "eos",
            "decode_generated_tokens": 120,
            "decode_bridge_applied": True,
            "n_branches": 2,
            "steps_taken": 2,
            "latent_opt_applied": True,
            "latent_opt_steps": 1,
            "fast_weights_applied": True,
            "fast_weights_erased": True,
            "worker_model_parameter_count": 32_000_000_000,
            "worker_model_parameter_count_basis": "architecture_config_logical",
            "output_quality": dict(quality),
            "runtime_identity": {
                "identity_bound": True,
                "launch_mode": "signed_app",
                "installed_app_required": True,
                "installed_app_verified": True,
                "source_verified": True,
                "source_commit": commit,
                "source_branch": "main",
                "source_dirty": False,
                "source_change_count": 0,
                "bundle_identifier": "com.aura.desktop",
                "issues": [],
            },
        },
    }
    # fc635dc92 added measured runtime integrity to the certificate: the
    # receipt must now carry a worker identity and an integrity proof bound to
    # this episode's tokens and checkpoint. The passing fixture has to satisfy
    # the contract it certifies, or the test proves only that an incomplete
    # receipt fails.
    attach_bound_runtime_integrity(contract["latent_cortex_receipt"])
    return {
        "response": answer,
        "status": "cognitive_engine",
        "live_turn_contract": contract,
    }, answer


def test_live_response_certificate_requires_complete_exact_app_contract() -> None:
    commit = "a" * 40
    body, _answer = _passing_live_body(commit)

    result = certificate._evaluate_live_response(
        body,
        message=certificate.DEFAULT_MESSAGE,
        exact_commit=commit,
        http_status=200,
    )

    assert result["fail_reasons"] == []
    assert all(result["checks"].values())


def test_live_response_certificate_rejects_stale_app_and_unbound_public_bytes() -> None:
    commit = "a" * 40
    body, _answer = _passing_live_body(commit)
    body["response"] = f"{body['response']} altered after certification"

    result = certificate._evaluate_live_response(
        body,
        message=certificate.DEFAULT_MESSAGE,
        exact_commit="b" * 40,
        http_status=200,
    )

    assert "exact_source_commit" in result["fail_reasons"]
    assert "public_quality_binds_exact_api_bytes" in result["fail_reasons"]


def test_live_response_certificate_rejects_duplicate_foreground_generation() -> None:
    commit = "a" * 40
    body, _answer = _passing_live_body(commit)
    contract = body["live_turn_contract"]
    assert isinstance(contract, dict)
    contract["foreground_model_generation_count"] = 2
    contract["single_owner_model_generation_proven"] = False

    result = certificate._evaluate_live_response(
        body,
        message=certificate.DEFAULT_MESSAGE,
        exact_commit=commit,
        http_status=200,
    )

    assert "single_owner_model_generation_proven" in result["fail_reasons"]
    assert "exactly_one_foreground_model_generation" in result["fail_reasons"]


def test_live_response_certificate_independently_rejects_forged_cp118_style_pass() -> None:
    commit = "a" * 40
    body, _answer = _passing_live_body(commit)
    malformed = (
        f"<request>{certificate.DEFAULT_MESSAGE}</request> "
        "Both approaches process work."
    )
    forged_quality = {
        "schema": "aura.latent_output_quality.v1",
        "policy": "resident_latent_product_quality_v1",
        "passed": True,
        "text_sha256": certificate._sha256_text(malformed),
        "objective_sha256": certificate._sha256_text(certificate.DEFAULT_MESSAGE),
        "reasons": [],
    }
    body["response"] = malformed
    contract = body["live_turn_contract"]
    assert isinstance(contract, dict)
    contract["latent_cortex_final_output_quality"] = dict(forged_quality)
    contract["latent_cortex_public_output_quality"] = dict(forged_quality)
    receipt = contract["latent_cortex_receipt"]
    assert isinstance(receipt, dict)
    receipt["output_quality"] = dict(forged_quality)

    result = certificate._evaluate_live_response(
        body,
        message=certificate.DEFAULT_MESSAGE,
        exact_commit=commit,
        http_status=200,
    )

    assert "independent_public_regrade_passed" in result["fail_reasons"]
    assert result["independent_public_regrade"]["passed"] is False


def test_transport_failure_writes_durable_fail_certificate(
    monkeypatch, tmp_path
) -> None:
    out_path = tmp_path / "transport-failure.json"
    monkeypatch.setattr(certificate, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        certificate.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    monkeypatch.setattr(
        certificate.sys,
        "argv",
        [
            "drive_live_latent_certificate.py",
            "--checkpoint",
            "119",
            "--out",
            str(out_path),
        ],
    )

    assert certificate.main() == 2
    artifact = certificate.json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["verdict"] == "FAIL"
    assert artifact["fail_reasons"] == ["live_transport_failed"]
    assert artifact["http"]["error_class"] == "OSError"


def test_git_head_rejects_nonzero_git_result(monkeypatch) -> None:
    gateway = _Gateway(CompletedProcess(["git"], 128, "", "not a repository"))
    monkeypatch.setattr(certificate, "get_subprocess_gateway", lambda: gateway)

    assert certificate._git_head() == ""
