"""A receipt must be bound to the request it claims to answer.

CP126 f22c4ed8: the facade validated request_payload_sha256 only as a
64-character hex string. It never hashed the actual question, messages,
config or budget — so a receipt could describe a different request entirely
and still pass. The digest proved the worker could format a string.

The facade cannot recompute it: mlx_client hashes WIRE-NORMALIZED config,
budget and runtime controls plus six more fields, and duplicating that
normalization here would drift and reject valid receipts on the live path.
The client already binds against the payload it actually sent, so the
facade's job is to confirm the binding HAPPENED.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import LatentCortexService


def _errors(**kwargs):
    receipt = {"request_payload_sha256": "a" * 64}
    receipt.update(kwargs.pop("receipt", {}))
    return LatentCortexService._receipt_contract_errors(
        receipt, {}, None, None, ..., "general", **kwargs
    )


def test_a_matching_binding_raises_no_request_error():
    errors = _errors(expected_request_payload_sha256="a" * 64)

    assert "request_payload_identity_mismatch" not in errors
    assert "request_payload_identity_unbound" not in errors


def test_a_receipt_for_a_different_request_is_refused():
    errors = _errors(expected_request_payload_sha256="b" * 64)

    assert "request_payload_identity_mismatch" in errors


def test_an_unbound_receipt_is_reported_not_waved_through():
    """This is the defect: a shape check standing in for a binding."""
    errors = _errors()

    assert "request_payload_identity_unbound" in errors


def test_a_malformed_digest_is_still_refused():
    errors = LatentCortexService._receipt_contract_errors(
        {"request_payload_sha256": "not-a-digest"}, {}, None, None, ..., "general",
        expected_request_payload_sha256="a" * 64,
    )

    assert "request_payload_identity_unproven" in errors


def test_a_missing_digest_is_still_refused():
    errors = LatentCortexService._receipt_contract_errors(
        {}, {}, None, None, ..., "general", expected_request_payload_sha256="a" * 64,
    )

    assert "request_payload_identity_unproven" in errors


def test_the_client_publishes_the_digest_it_bound():
    """Without this the facade has nothing to confirm against."""
    import inspect

    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client)
    assert '"request_payload_sha256_bound": expected_request_sha256' in source


def test_the_client_still_refuses_a_mismatch_itself():
    """The facade's check is a second line, not a replacement for the
    client's own binding against the payload it sent."""
    import inspect

    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client)
    assert 'identity_errors.append("request_payload_sha256_mismatch")' in source


@pytest.mark.parametrize("evidence", [{}, {"baseline_text": "private", "baseline_tokens": [1]}, None])
def test_client_success_mapping_preserves_private_validation_evidence(evidence):
    import ast
    from pathlib import Path

    source = Path("core/brain/llm/mlx_client.py").read_text()
    tree = ast.parse(source)
    mappings = [
        node for node in ast.walk(tree) if isinstance(node, ast.Dict)
        and any(isinstance(key, ast.Constant) and key.value == "request_payload_sha256_bound"
                for key in node.keys)
    ]
    assert len(mappings) == 1
    values = [value for key, value in zip(mappings[0].keys, mappings[0].values)
              if isinstance(key, ast.Constant) and key.value == "answer_replacement_private"]
    assert len(values) == 1
    expression = ast.Expression(body=values[0])
    # noqa: S307 — the test reads one expression out of the real source and
    # evaluates it, so what is asserted is the shipped mapping rather than
    # a copy of it that could drift.
    actual = eval(  # noqa: S102,S307
        compile(expression, "client-success-mapping", "eval"),  # noqa: S102
        {"res": {"answer_replacement_private": evidence}},
    )
    assert actual is evidence


def test_the_facade_does_not_recompute_the_digest():
    """Duplicating the wire normalization would drift and reject valid
    receipts on the live path."""
    import inspect

    from core.brain import latent_cortex_service

    source = inspect.getsource(latent_cortex_service.LatentCortexService.deep_reason)
    assert "latent_request_payload_sha256(" not in source
