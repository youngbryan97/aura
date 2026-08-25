"""CP126 841bf5f7: privileged output contracts must carry authority.

The worker selected its prompt builder, sampling regime, validator and
output normalizer straight from booleans on the IPC job, verifying no
principal and no allowed combination. Anything that could put a dict on the
request queue could therefore change how output was produced and judged.

This is not claimed as a security perimeter — a component able to forge a
job already runs inside the parent. It buys provenance (the selection names
who made it), confused-deputy resistance (a copied or over-eager job dict no
longer silently changes validation), and tamper evidence (a job mutated
after signing fails instead of taking effect).
"""
from __future__ import annotations

import ast
import inspect

import pytest

from core.brain.llm.contract_authority import (
    AUTH_FIELD,
    EXCLUSIVE_CONTRACT_FIELDS,
    PRIVILEGED_CONTRACT_FIELDS,
    authority_receipt,
    exclusive_conflict,
    new_contract_key,
    selected_privileged_fields,
    sign_job,
    verify_job,
)


@pytest.fixture
def key() -> bytes:
    return new_contract_key()


def _job(**extra):
    job = {"id": "req-1", "action": "generate"}
    job.update(extra)
    return job


class TestKeysArePerSpawn:
    def test_keys_are_unique(self):
        assert new_contract_key() != new_contract_key()

    def test_keys_are_long_enough(self):
        assert len(new_contract_key()) >= 32


class TestPrivilegeDetection:
    def test_an_ordinary_job_selects_nothing(self):
        assert selected_privileged_fields(_job()) == []

    def test_each_contract_is_privileged(self):
        for field in PRIVILEGED_CONTRACT_FIELDS:
            if field == "schema":
                continue
            assert selected_privileged_fields(_job(**{field: True})) == [field]

    def test_a_false_flag_is_not_a_selection(self):
        assert selected_privileged_fields(_job(proof_evaluation_contract=False)) == []

    def test_schema_is_privileged_only_when_present(self):
        assert selected_privileged_fields(_job(schema={})) == []
        assert selected_privileged_fields(_job(schema={"type": "object"})) == ["schema"]

    def test_selection_order_is_stable(self):
        job = _job(operator_evidence_contract=True, strict_answer_contract=True)
        assert selected_privileged_fields(job) == selected_privileged_fields(dict(job))


class TestExclusiveContracts:
    def test_one_contract_is_not_a_conflict(self):
        assert exclusive_conflict(_job(strict_answer_contract=True)) == []

    def test_two_contracts_conflict(self):
        conflict = exclusive_conflict(
            _job(strict_answer_contract=True, proof_evaluation_contract=True),
        )
        assert len(conflict) == 2

    def test_every_exclusive_field_is_privileged(self):
        for field in EXCLUSIVE_CONTRACT_FIELDS:
            assert field in PRIVILEGED_CONTRACT_FIELDS

    def test_a_contradiction_is_refused_even_when_signed(self, key):
        signed = sign_job(
            _job(strict_answer_contract=True, strict_value_contract=True),
            key,
            principal="latent_cortex",
        )
        # A signature over a contradiction still leaves no single contract
        # to honour.
        assert verify_job(signed, key).startswith("ambiguous_output_contract")


class TestSigningAndVerification:
    def test_an_ordinary_job_needs_no_authority(self, key):
        assert verify_job(_job(), key) == ""

    def test_an_ordinary_job_is_not_stamped(self, key):
        job = sign_job(_job(), key, principal="p")
        assert AUTH_FIELD not in job

    def test_a_signed_selection_verifies(self, key):
        signed = sign_job(
            _job(proof_evaluation_contract=True), key, principal="latent_cortex",
        )
        assert verify_job(signed, key) == ""

    def test_an_unsigned_selection_is_refused(self, key):
        refusal = verify_job(_job(proof_evaluation_contract=True), key)
        assert refusal.startswith("unauthenticated_contract_selection")

    def test_a_forged_signature_is_refused(self, key):
        job = _job(proof_evaluation_contract=True)
        job[AUTH_FIELD] = {"principal": "attacker", "signature": "0" * 64}
        assert verify_job(job, key).startswith("invalid_contract_authority")

    def test_a_signature_from_another_key_is_refused(self, key):
        signed = sign_job(_job(proof_evaluation_contract=True), new_contract_key(), principal="p")
        assert verify_job(signed, key).startswith("invalid_contract_authority")

    def test_a_signature_cannot_be_lifted_to_another_job(self, key):
        signed = sign_job(_job(proof_evaluation_contract=True), key, principal="p")
        lifted = dict(signed)
        lifted["id"] = "req-2"
        assert verify_job(lifted, key).startswith("invalid_contract_authority")

    def test_adding_a_field_after_signing_is_refused(self, key):
        signed = sign_job(_job(proof_evaluation_contract=True), key, principal="p")
        signed["capability_inventory_contract"] = True
        assert verify_job(signed, key).startswith("invalid_contract_authority")

    def test_changing_a_structured_contract_after_signing_is_refused(self, key):
        signed = sign_job(
            _job(unified_recurrent_shadow_contract={"request_sha256": "a" * 64}),
            key,
            principal="shadow_probe",
        )
        signed["unified_recurrent_shadow_contract"]["request_sha256"] = "b" * 64

        assert verify_job(signed, key).startswith("invalid_contract_authority")

    def test_changing_the_principal_after_signing_is_refused(self, key):
        signed = sign_job(_job(proof_evaluation_contract=True), key, principal="p")
        signed[AUTH_FIELD]["principal"] = "someone_else"
        assert verify_job(signed, key).startswith("invalid_contract_authority")

    def test_an_incomplete_authority_is_refused(self, key):
        job = _job(proof_evaluation_contract=True)
        job[AUTH_FIELD] = {"principal": "p"}
        assert verify_job(job, key).startswith("incomplete_contract_authority")

    def test_a_non_dict_authority_is_refused(self, key):
        job = _job(proof_evaluation_contract=True)
        job[AUTH_FIELD] = "trust me"
        assert verify_job(job, key).startswith("unauthenticated_contract_selection")


class TestKeylessWorkersDoNotEnforce:
    def test_no_key_means_no_enforcement(self):
        # A bare harness cannot sign, so enforcing would refuse everything.
        assert verify_job(_job(proof_evaluation_contract=True), None) == ""

    def test_consistency_is_still_enforced_without_a_key(self):
        assert verify_job(
            _job(strict_answer_contract=True, proof_evaluation_contract=True), None,
        ).startswith("ambiguous_output_contract")

    def test_signing_without_a_key_is_a_no_op(self):
        job = sign_job(_job(proof_evaluation_contract=True), None, principal="p")
        assert AUTH_FIELD not in job


class TestReceipts:
    def test_a_signed_selection_reports_its_principal(self, key):
        signed = sign_job(
            _job(proof_evaluation_contract=True), key, principal="latent_cortex",
        )
        receipt = authority_receipt(signed)
        assert receipt["principal"] == "latent_cortex"
        assert receipt["authenticated"] is True
        assert receipt["privileged_fields"] == ["proof_evaluation_contract"]

    def test_an_unsigned_selection_reports_unauthenticated(self):
        receipt = authority_receipt(_job(proof_evaluation_contract=True))
        assert receipt["authenticated"] is False
        assert receipt["principal"] == ""


class TestWiring:
    def test_the_worker_loop_accepts_a_key(self):
        from core.brain.llm import mlx_worker

        sig = inspect.signature(mlx_worker._mlx_worker_loop)
        assert "contract_key" in sig.parameters
        # Keyword default keeps the spawn arity backward compatible.
        assert sig.parameters["contract_key"].default is None

    def test_the_worker_verifies_before_generating(self):
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker)
        assert "_verify_contract_authority(job, contract_key)" in source

    def test_a_missing_authority_module_still_blocks_contradictions(self):
        from core.brain.llm import mlx_worker

        source = inspect.getsource(mlx_worker._verify_contract_authority)
        assert "ambiguous_output_contract" in source
        assert "except ImportError" in source

    def test_the_client_generates_a_key_per_spawn(self):
        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        assert "self._contract_key = new_contract_key()" in source

    def test_the_client_passes_the_key_to_the_child(self):
        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        block = source.split("target=_mlx_worker_loop", 1)[1][:400]
        assert "self._contract_key," in block

    @staticmethod
    def _shutdown_sentinel(node) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    def test_every_submission_site_authorizes(self):
        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        tree = ast.parse(source)
        submitted_jobs = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Direct synchronous queue submission.
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "put"
                and node.args
                and ast.unparse(node.func.value).endswith(("_req_q", "request_queue"))
            ):
                if not self._shutdown_sentinel(node.args[0]):
                    submitted_jobs.append(node.args[0])
            # Async paths pass queue.put and the job to run_io_bound.
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "run_io_bound"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "put"
                and ast.unparse(node.args[0].value).endswith("_req_q")
            ):
                submitted_jobs.append(node.args[1])

        # `_req_q.put(None)` is the worker's shutdown sentinel, not a job: it
        # carries no action and no principal, and there is nothing to
        # authorise. Every real payload still has to cross _authorize_job.
        assert submitted_jobs
        assert all(
            isinstance(job, ast.Call)
            and isinstance(job.func, ast.Attribute)
            and job.func.attr == "_authorize_job"
            for job in submitted_jobs
        ), "every MLX request-queue submission must cross _authorize_job"

    def test_each_principal_is_named(self):
        from core.brain.llm import mlx_client

        source = inspect.getsource(mlx_client)
        for principal in (
            "mlx_client.generate",
            "mlx_client.latent_reason",
            "mlx_client.health_probe",
            "mlx_client.structured_request",
            "mlx_client.expert_adapter",
        ):
            assert f'principal="{principal}"' in source


class TestRoundTripThroughBothSides:
    def test_action_state_runtime_is_a_privileged_worker_contract(self, key):
        from core.brain.llm import mlx_client, mlx_worker

        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client._contract_key = key
        job = client._authorize_job(
            _job(action_state_runtime={"schema": "signed-public-frame"}),
            principal="mlx_client.latent_reason",
        )

        assert "action_state_runtime" in selected_privileged_fields(job)
        assert mlx_worker._verify_contract_authority(job, key) == ""

    def test_client_signature_verifies_in_the_worker(self, key):
        from core.brain.llm import mlx_client, mlx_worker

        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client._contract_key = key

        job = client._authorize_job(
            _job(proof_evaluation_contract=True), principal="mlx_client.generate",
        )
        assert mlx_worker._verify_contract_authority(job, key) == ""

    def test_an_unsigned_job_is_refused_by_the_worker(self, key):
        from core.brain.llm import mlx_worker

        refusal = mlx_worker._verify_contract_authority(
            _job(proof_evaluation_contract=True), key,
        )
        assert refusal.startswith("unauthenticated_contract_selection")

    def test_a_different_workers_key_does_not_verify(self, key):
        from core.brain.llm import mlx_client, mlx_worker

        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client._contract_key = key
        job = client._authorize_job(
            _job(proof_evaluation_contract=True), principal="mlx_client.generate",
        )
        assert mlx_worker._verify_contract_authority(job, new_contract_key())
