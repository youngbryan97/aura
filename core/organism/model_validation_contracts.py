"""Contract predicates for the cortex's tools, the confined Python runner, and four boundaries.

The boundaries are a request's mood, egress privacy, identity attestation and
metrology sources.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. Each imports
what it reads at call time, as it did there.
"""
from __future__ import annotations

from typing import Any

from core.organism.nothing_measured import NothingMeasured


def _rlc_capability_evidence_contract_holds() -> bool:
    import hashlib

    from core.brain.capability_evidence_context import (
        build_current_turn_capability_evidence,
    )

    objective = "Use Python to calculate the exact checksum total."
    objective_sha256 = hashlib.sha256(objective.encode("utf-8")).hexdigest()
    admitted = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": objective_sha256,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "checksum_total=4182",
                "exit_code": 0,
            },
        },
        objective,
    )
    stale = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": "0" * 64,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "stale=1",
                "exit_code": 0,
            },
        },
        objective,
    )
    return bool(
        admitted.receipt.get("admitted") is True
        and len(admitted.items) == 1
        and admitted.items[0].get("instruction_authority") is False
        and admitted.items[0].get("evidence_kind") == "governed_tool_observation"
        and not stale.items
        and stale.receipt.get("reason") == "stale_skill_result"
    )


def _rlc_web_acquisition_contract_holds() -> bool:
    from core.brain.cortex_web_acquisition import should_acquire_live_web
    from core.brain.llm.latent_cortex.context_focus import source_matches_action
    from core.executive.standing_authority import AUTONOMOUS_AUTHORITY_ORIGINS

    live = should_acquire_live_web(
        "What is the latest compiler release?",
        "compiler release",
        local_context_is_new=True,
    )
    uncovered = should_acquire_live_web(
        "Explain the new theorem.",
        "new theorem",
        local_context_is_new=False,
    )
    return bool(
        live == (True, "live_or_source_sensitive_objective")
        and uncovered == (True, "local_reference_uncovered")
        and "latent_cortex" in AUTONOMOUS_AUTHORITY_ORIGINS
        and source_matches_action("capability.web_search", "retrieve_evidence")
    )


def _rlc_amplifier_composition_contract_holds() -> bool:
    from core.brain.reasoning_amplifier_v2 import _admit_seed_candidates

    return _admit_seed_candidates(
        ["candidate", "candidate", ""],
        limit=2,
    ) == ["candidate"]


def _symbolic_cognition_boundary_available() -> bool:
    from core.sandbox.untrusted_python import available_boundary

    return available_boundary() in {"seatbelt", "bubblewrap"}


def _sandbox_async_execution_probe() -> bool:
    from core.sandbox.untrusted_python import (
        available_boundary,
        call_untrusted_function,
        run_untrusted_script,
    )

    if not available_boundary():
        raise NothingMeasured("no kernel sandbox is available for the async execution probe")
    completed = call_untrusted_function(
        "async def answer():\n    return 42\n", "answer", [()],
        timeout_s=2.0, source="validation.async_execution",
    )
    dropped = run_untrusted_script(
        "async def answer():\n    return 42\nanswer()\n",
        timeout_s=2.0, source="validation.async_execution",
    )
    if completed.status not in {"ok", "error"} or dropped.status not in {"ok", "error"}:
        raise NothingMeasured("the async execution probe did not finish under its kernel boundary")
    return bool(
        completed.ok and completed.results == [42]
        and dropped.status == "error" and "never awaited" in dropped.error
    )


def _rlc_compute_continuation_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.cognitive_acquisition import (
        acquisition_has_new_context,
        build_acquisition_request,
    )

    transition = {
        "action": "formalize",
        "outcome": "succeeded",
        "checked": True,
    }
    request = build_acquisition_request(
        objective="Compute 12 * 13 exactly.",
        first_text="The answer is 157.",
        first_receipt={
            "cognitive_action_trace": [
                {"decision": {"action": "formalize"}, "transition": transition}
            ]
        },
        cognitive_context=None,
    )
    return bool(
        request
        and request.get("action") == "formalize"
        and request.get("max_acquisitions") == 1
        and request.get("max_continuation_rounds") == 1
        and acquisition_has_new_context(
            request,
            [
                {
                    "source": "capability.symbolic_formalize",
                    "text": "exact(12*13) = 156",
                }
            ],
        )
    )


def _semantic_autonomy_contract_holds() -> bool:
    from core.conversation.request_mood import assess_request_mood
    from core.runtime.overt_action_loop import OvertActionLoop

    indirect = assess_request_mood(
        "It would help if you compared the current evidence and saved the result."
    )
    hypothetical = assess_request_mood(
        "If I asked you to open Notes, how would you decide whether to do it?"
    )
    selection = OvertActionLoop()._choose_skill_and_params(
        {
            "goal": "Compare the current evidence and preserve a verified result.",
            "source": "cognitive_loop",
        },
        {},
    )
    return bool(
        indirect.asks_for_action
        and hypothetical.is_about_rather_than_asking
        and selection.actionable
        and selection.execution_mode == "planned_goal"
        and selection.provenance == "semantic_plan:live_capability_catalog"
    )


def _egress_privacy_contract_holds() -> bool:
    """Exercise the boundary rather than assert that it exists.

    A registered claim whose predicate only imported the module would be the
    thing this suite is for catching.
    """
    from core.security.egress_privacy import filter_outbound_body

    secret = "sk-" + "a" * 24
    stripped = filter_outbound_body(
        url="https://external-service.invalid/v1/submit",
        body=f'{{"contents":"key {secret}"}}'.encode(),
        source="external_service:privacy_probe",
        publish_evidence=False,
    )
    # The same secret one character to the left of the colon. The walk used to
    # read values only, so this exact body left the machine intact while the
    # one above was caught — and the claim said "never" for both.
    keyed = filter_outbound_body(
        url="https://external-service.invalid/v1/submit",
        body=f'{{"{secret}":"quota"}}'.encode(),
        source="external_service:privacy_probe",
        publish_evidence=False,
    )
    binary = b"\xff\xfe\x00binary"
    unreadable = filter_outbound_body(
        url="https://external-service.invalid/v1/submit",
        body=binary,
        source="external_service:privacy_probe",
        publish_evidence=False,
    )
    local = filter_outbound_body(
        url="http://127.0.0.1:8000/v1",
        body=f'{{"contents":"key {secret}"}}'.encode(),
        source="llm_provider:mlx",
        publish_evidence=False,
    )
    return bool(
        stripped.allowed
        and stripped.inspected
        and secret not in (stripped.body or b"").decode("utf-8", errors="replace")
        and keyed.allowed
        and keyed.inspected
        and secret not in (keyed.body or b"").decode("utf-8", errors="replace")
        # Binary tool payloads are legitimate. They remain byte-identical, and
        # the receipt must never claim an inspection that could not happen.
        and unreadable.allowed
        and not unreadable.inspected
        and unreadable.body == binary
        # Local inference is untouched: the boundary must not cost Aura her
        # own runtime to protect her from a stranger.
        and local.allowed
        and local.body == f'{{"contents":"key {secret}"}}'.encode()
    )


def _identity_attestation_contract_holds() -> bool:
    """A profile whose content no longer matches its seal reaches no prompt.

    The tamper is simulated by re-sealing a DIFFERENT digest rather than by
    rewriting the file. Same condition under test — on-disk content that does
    not match what Aura attested — and it avoids performing a raw write from
    inside the runtime to prove that raw writes are detected.

    The artifact id comes from the profile under test, never from the class
    constant. It used to come from the constant, and when the seal was scoped
    per storage path the constant stopped naming the artifact this profile
    verifies — so the tamper landed on an id nobody reads and the check
    measured nothing. Asking the object is the general form: a predicate that
    re-derives an internal rule is a copy of that rule that nothing keeps in
    step.
    """
    import tempfile
    from pathlib import Path

    from core.memory.aura_self_profile import AuraSelfProfile
    from core.security.state_attestation import AttestationState, attest_state

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "self_profile.json"
        genuine = AuraSelfProfile(storage_path=str(path))
        genuine.add_or_reinforce_fact(
            "relationship", "probe", "a fact Aura actually learned"
        )
        if not path.exists():
            return False

        # What an out-of-band writer leaves behind: a file whose digest is not
        # the one Aura sealed.
        artifact_id = genuine.attestation_status().get("artifact_id", "")
        if not artifact_id:
            return False
        attest_state(
            artifact_id,
            '{"relationship": [{"value": "an instruction someone else wrote"}]}',
        )

        reopened = AuraSelfProfile(
            storage_path=str(path),
            publish_attestation_verdict=False,
        )
        return bool(
            reopened.attestation_status()["state"] == AttestationState.TAMPERED
            and reopened.get_fact("relationship", "probe") is None
            and reopened.to_identity_block() == ""
            and not path.exists()  # quarantined, not left in place
        )


def _metrology_source_contract_holds() -> bool:
    from core.reality_reach.metrology import (
        AcquisitionChannel,
        AcquisitionMode,
        AcquisitionTask,
        EvidenceSource,
    )

    hil = AcquisitionTask(
        task_id="validation.hil",
        channels=(
            AcquisitionChannel("validation.live", EvidenceSource.LIVE),
            AcquisitionChannel("validation.simulated", EvidenceSource.SIMULATED),
        ),
        mode=AcquisitionMode.HARDWARE_IN_LOOP,
        scenario_id="validation.scenario",
    )
    try:
        AcquisitionTask(
            task_id="validation.invalid-live",
            channels=(
                AcquisitionChannel("validation.simulated", EvidenceSource.SIMULATED),
            ),
            mode=AcquisitionMode.LIVE,
        )
    except ValueError:
        refused = True
    else:
        refused = False
    return bool(hil.mode is AcquisitionMode.HARDWARE_IN_LOOP and refused)


def _install_g03_build_contract_claims(suite: Any) -> None:
    """The three G03 build contracts a6d704c37 registered as tests without claims.

    A test with no claim checks something nobody says, and the suite holds the
    two in step. Each claim says what its contract establishes and no more:
    the build notes call these builds, not G03 acceptance evidence.
    """
    from .model_validation import Claim, Evidence

    built = (
        "A contract over constructed cases. The build note calls this a build, "
        "not G03 acceptance evidence, and no serving path runs it."
    )
    suite.add_claim(Claim(
        statement=(
            "A declared meaning diagram keeps each role's identity when its layout "
            "changes, and admits no observation from after the moment it describes."
        ),
        test="meaning_diagram_preserves_roles_and_evidence_clock",
        owner="core/learning/semantic_semasiographic.py",
        asserted_in="docs/evidence/G03_NONLINEAR_MEANING_BUILD_2026-09-30.md",
        evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note=built,
    ))
    suite.add_claim(Claim(
        statement=(
            "Declared rigid object memory keeps an object's measured shape and its "
            "hidden surfaces across a change of viewpoint."
        ),
        test="rigid_memory_preserves_observed_intrinsics",
        owner="core/cognition/rigid_object_memory.py",
        asserted_in="docs/evidence/G03_BINDING_BUILD_REGISTER_2026-09-30.md",
        evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note=built,
    ))
    suite.add_claim(Claim(
        statement=(
            "An identity qualified by its source is not replaced by a distractor "
            "with an equal value or an alias."
        ),
        test="contextual_binding_preserves_source_identity",
        owner="core/learning/semantic_context_binding.py",
        asserted_in="docs/evidence/G03_BINDING_BUILD_REGISTER_2026-09-30.md",
        evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note=built,
    ))
