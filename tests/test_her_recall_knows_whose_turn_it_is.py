"""The job her worker receives names whose turn it is, so her one-shot memory is consulted at all.

MEASURED 10 November 2026: the worker refuses recall to a job that names nobody (``binding_for_job``), and the
client's request named nobody. Her log held 251 openings of her store and not one installed recall.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.brain.nonparametric_binding import LOCAL_OWNER_PRINCIPAL, binding_for_job, principal_of_the_turn


def _request(**kwargs):
    from core.brain.llm.mlx_client import _build_the_generation_request

    return _build_the_generation_request(
        _bridge_get=lambda _name, fallback: fallback,
        adaptive_suggested_max_tokens=256,
        contract_generation_floor=0,
        generation_max_tokens=256,
        hard_output_token_ceiling=None,
        kwargs=kwargs,
        prompt="question",
        req_id="request-1",
        requested_output_contract={},
        self=SimpleNamespace(_job_seq_counter=7, temp=0.7, top_p=0.9, max_tokens=512),
    )


def test_a_turn_that_names_nobody_is_the_local_operators_and_the_worker_binds_it():
    job = _request()
    assert job["principal"] == LOCAL_OWNER_PRINCIPAL
    binding = binding_for_job(job, source_id="foreground_recall")
    assert binding is not None and binding.principal == LOCAL_OWNER_PRINCIPAL


def test_only_an_authenticated_principal_changes_whose_turn_it_is():
    assert _request(authenticated_principal="visitor-7")["principal"] == "visitor-7"
    # A plain claim on the caller's mapping would read someone else's entries.
    assert _request(principal="someone-else")["principal"] == LOCAL_OWNER_PRINCIPAL
    assert principal_of_the_turn({"authenticated_principal": "  "}) == LOCAL_OWNER_PRINCIPAL
    assert principal_of_the_turn(None) == LOCAL_OWNER_PRINCIPAL
