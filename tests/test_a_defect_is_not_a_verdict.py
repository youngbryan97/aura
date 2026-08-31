"""A crash in a thinking step must not read as a decision about the action.

Live 2026-08-30: a type bug inside the episode client became
``client_error:AttributeError``, which became ``episode_integrity_refusal:...``,
which the external-execution coordinator may never bypass. So a browser task
that had cleared every authority gate was refused before dispatch, and the
refusal looked exactly like caution.
"""

from __future__ import annotations

import io
import logging

from core.brain.preaction_cortex import _availability_failure
from core.runtime.errors import record_degradation


def test_a_python_defect_lets_the_action_proceed_without_the_rehearsal():
    """The same posture the cortex being absent already has."""
    for kind in ("AttributeError", "IndexError", "KeyError", "NotImplementedError"):
        assert _availability_failure(f"client_error:{kind}") is not None


def test_a_deliberate_refusal_still_stops_the_action():
    """Reaching a judgement requires the episode to have run, and a refusal is
    raised on purpose carrying what was wrong."""
    for kind in ("ValueError", "RuntimeError"):
        assert _availability_failure(f"client_error:{kind}") is None
    assert _availability_failure("proof_missing") is None
    assert _availability_failure("episode_integrity_refusal:unknown") is None


def test_the_raise_site_does_not_change_the_verdict():
    """The reason now carries where it came from, which must not reclassify it."""
    where = "core.brain.llm.mlx_client:latent_reason_async:10429"
    assert _availability_failure(f"client_error:AttributeError@{where}") is not None
    assert _availability_failure(f"client_error:ValueError@{where}") is None


def test_the_transport_failures_it_already_admitted_still_pass():
    for reason in (
        "client_error:OSError",
        "client_error:TimeoutError",
        "no_resident_model",
        "generation_gate_busy",
    ):
        assert _availability_failure(reason) is not None


def test_a_contained_exception_says_where_it_was_raised():
    """The record carried the traceback; the log line never did, so a contained
    fault read as its message and nothing else."""
    caught = io.StringIO()
    handler = logging.StreamHandler(caught)
    logger = logging.getLogger("Aura.Errors")
    logger.addHandler(handler)
    try:

        def where_it_happens():
            return [].get("x")  # type: ignore[attr-defined]

        try:
            where_it_happens()
        except AttributeError as exc:
            record_degradation("demo", exc, severity="warning", action="test")
    finally:
        logger.removeHandler(handler)
    said = caught.getvalue()
    assert "where_it_happens" in said
    assert "raised at" in said


def test_never_reaching_action_selection_is_not_disagreeing_about_it():
    """Four fields are written at the very END of a full episode. An episode
    that stopped earlier has a receipt without them, and that was read as the
    worker's policy CONTRADICTING the host's — which is ineligible for bypass,
    so a browser task that had cleared every authority gate never pressed a
    key. Nothing was claimed, so nothing can disagree."""
    assert _availability_failure("no_action_policy_to_check") is not None
    assert (
        _availability_failure("no_action_policy_to_check:ValueError: ended early")
        is not None
    )


def test_a_policy_that_really_disagrees_still_stops_the_action():
    """The distinction only means anything if the other half still holds."""
    assert _availability_failure("runtime_action_policy_receipt_mismatch") is None
    assert (
        _availability_failure(
            "runtime_action_policy_receipt_mismatch:ValueError: fields differ"
        )
        is None
    )
