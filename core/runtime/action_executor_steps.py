"""The steps executing one action is assembled from.

Lifted whole out of `action_executor`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
from typing import Any


class _RunsTheActionSteps:
    """Lifted whole out of ActionExecutor; see action_executor.py."""

    @staticmethod
    async def _execute_part_1(
        action_name: str,
        action_summary: Any,
        domain: Any,
        expectation_contract: Any,
        external_execute_coordinator: Any,
        external_execution_offer: Any,
        external_execution_transaction: Any,
        preaction_thread: Any,
    ) -> Any:
        if (
            external_execution_offer is not None
            and external_execution_transaction.get("state") == "DECIDED"
        ):
            preaction_thread.rehearsal = {
                "schema": "aura.preaction_cortex.v1",
                "phase": "rehearsal",
                "action_name": action_name,
                "domain": domain.value,
                "ran": False,
                "skip_reason": "durable_external_execution_decision_reused",
            }
        else:
            rehearsal = await preaction_thread.rehearse(
                action_summary=action_summary,
                expectation_objective=expectation_contract.objective,
            )
            if external_execution_offer is not None:
                if rehearsal.get("ran") is True:
                    external_execution_transaction = await asyncio.to_thread(
                        external_execute_coordinator.record_handoff,
                        offer=external_execution_offer,
                        handoff=rehearsal.get("external_execution_handoff") or {},
                        cognitive_action_trace=(
                            preaction_thread.external_execution_trace()
                        ),
                        readiness=(
                            preaction_thread.external_execution_readiness()
                        ),
                        model_output=(
                            preaction_thread.external_execution_model_output()
                        ),
                        action_policy_evidence=(
                            preaction_thread.external_action_policy_evidence()
                        ),
                        executors=(
                            preaction_thread.external_action_executors()
                        ),
                        action_policy_receipt=(
                            preaction_thread.external_action_policy_receipt()
                        ),
                        runtime_operation=(
                            preaction_thread.external_runtime_operation()
                        ),
                    )
                else:
                    # Named as the class it belongs to. A rehearsal
                    # that recorded no reason did not run, which is an
                    # availability failure — and calling it anything
                    # else made the executor's own fallback ineligible
                    # for the bypass it was reaching for, refusing the
                    # action outright.
                    skip_reason = str(
                        rehearsal.get("skip_reason")
                        or "availability_failure:rehearsal_unavailable"
                    )
                    external_execution_transaction = await asyncio.to_thread(
                        external_execute_coordinator.record_bypass,
                        offer=external_execution_offer,
                        reason=skip_reason,
                    )
        return external_execution_transaction

    @staticmethod
    def _execute_part_2(
        action_id: str | None,
        action_name: str,
        exc: Any,
        expectation_contract: Any,
        external_execution_transaction: Any,
        request_digest: Any,
        will_receipt_id: Any,
    ) -> dict[str, Any]:
        from .action_executor import (
            SkillStatus,
            _raise_site,
            record_degradation,
        )

        record_degradation(
            "action_executor.external_execution",
            exc,
            action=(
                f"refused {action_name} before effect dispatch because "
                "its external execution transaction could not be proven"
            ),
            severity="degraded",
            enforce_failure_policy=False,
        )
        failure_result = {
            "ok": False,
            "status": SkillStatus.FAILED_RECOVERABLE.value,
            # The type is not the reason.
            #
            # "external_execution_preparation_failed:ValueError" is
            # what the person is shown when an action refuses before
            # it starts, and it names none of the dozen things that
            # raise ValueError in preparation. Live 2026-08-31 it was
            # shown three times over a single afternoon for three
            # different causes. What the exception says, and where it
            # was raised, are both already known here.
            "error": (
                "external_execution_preparation_failed:"
                f"{type(exc).__name__}: {exc}"
                f" [raised at {_raise_site(exc)}]"
            ),
            "will_receipt_id": will_receipt_id,
            "action_expectation": expectation_contract.to_dict(),
            "action_id": action_id,
            "request_digest": request_digest,
            "transport_succeeded": False,
            "effect_verified": False,
            "retry_safe": not isinstance(exc, ValueError),
            "manual_reconciliation_required": False,
            "external_execution_transaction": dict(
                external_execution_transaction
            ),
        }
        return failure_result

    @staticmethod
    async def _execute_begin_task(
        external_execute_coordinator: Any,
        external_execution_offer: Any,
        external_execution_transaction: Any,
        will_receipt_id: Any,
    ) -> tuple[str, Any]:
        from .action_executor import (
            _abandon_external_dispatch,
            _external_dispatch_task_id,
            get_task_tracker,
        )

        begin_task = get_task_tracker().create_task(
            asyncio.to_thread(
                external_execute_coordinator.begin_dispatch,
                external_execution_offer,
                authorization_receipt_id=will_receipt_id,
                task_id=_external_dispatch_task_id(),
            )
        )
        try:
            external_execution_transaction = await asyncio.shield(
                begin_task
            )
        except asyncio.CancelledError:
            external_execution_transaction = await asyncio.shield(
                begin_task
            )
            cancelled_owner = (
                external_execution_transaction.get("dispatch_owner")
                or {}
            )
            await _abandon_external_dispatch(
                coordinator=external_execute_coordinator,
                offer=external_execution_offer,
                dispatch_attempt_id=str(
                    cancelled_owner.get("attempt_id") or ""
                ),
                effect_may_have_occurred=False,
                reason="cancelled_before_effect_dispatch",
            )
            raise
        external_dispatch_attempt_id = str(
            (
                external_execution_transaction.get("dispatch_owner")
                or {}
            ).get("attempt_id")
            or ""
        )
        if not external_dispatch_attempt_id:
            raise ValueError(
                "external execution dispatch intent lacks an owner token"
            )
        return external_dispatch_attempt_id, external_execution_transaction

    @staticmethod
    def _execute_part_4(action_name: str, dispatch_result: Any, exc: Any) -> dict[str, Any]:
        from .action_executor import (
            SkillStatus,
            logger,
            record_degradation,
        )

        record_degradation(
            "action_executor",
            exc,
            action=f"recorded failed action transaction for {action_name}",
        )
        logger.error("Error executing action %s: %s", action_name, exc, exc_info=True)
        transport_may_have_succeeded = dispatch_result.get("ok") is True
        result = {
            "ok": False,
            "status": SkillStatus.FAILED_RECOVERABLE.value,
            "error": str(exc),
            "effect_verified": False,
            "transport_succeeded": transport_may_have_succeeded,
            "verification_evidence": {
                "observation": {
                    "effect_verified": False,
                    "reason": (
                        "verification_exception_after_transport"
                        if transport_may_have_succeeded
                        else "execution_exception"
                    ),
                    "error_type": type(exc).__qualname__,
                }
            },
        }
        return result

    @staticmethod
    async def _execute_part_5(
        action_name: str,
        external_dispatch_attempt_id: str,
        external_dispatch_heartbeat: Any,
        external_execute_coordinator: Any,
        external_execution_offer: Any,
        external_execution_transaction: Any,
        post_receipt: Any,
        result: Any,
    ) -> tuple[str, bool]:
        from .action_executor import (
            SkillStatus,
            _await_external_closure,
            _complete_external_execution_transaction,
            _stop_external_dispatch_heartbeat,
        )

        if external_execution_offer is not None:
            result.update(
                {
                    "receipt_persisted": False,
                    "post_action_receipt_pending": True,
                    "post_action_receipt_attempt_id": post_receipt.receipt_id,
                    "_post_action_recovery_contract": post_receipt.to_dict(),
                }
            )
        await _await_external_closure(
            _complete_external_execution_transaction(
                coordinator=external_execute_coordinator,
                offer=external_execution_offer,
                transaction=external_execution_transaction,
                dispatch_attempt_id=external_dispatch_attempt_id,
                result=result,
                action_name=action_name,
            ),
            coordinator=external_execute_coordinator,
            offer=external_execution_offer,
            dispatch_attempt_id=external_dispatch_attempt_id,
            heartbeat=external_dispatch_heartbeat,
            result=result,
            cancellation_reason="cancelled_during_external_completion",
        )
        await _stop_external_dispatch_heartbeat(external_dispatch_heartbeat)
        result.pop("_post_action_recovery_contract", None)
        final_transport_succeeded = result.get("transport_succeeded") is True
        final_status = str(
            result.get("status")
            or SkillStatus.FAILED_RECOVERABLE.value
        )
        return final_status, final_transport_succeeded

    @staticmethod
    def _execute_final_error_msg(
        final_effect_verified: bool,
        final_status: Any,
        final_transport_succeeded: Any,
        post_receipt: Any,
        result: Any,
    ) -> tuple[Any, Any]:
        from .action_executor import (
            PostActionReceipt,
            _stable_digest,
        )

        final_error_msg = (
            str(result.get("error") or "")
            if not result.get("ok", False)
            else ""
        )
        if (
            final_transport_succeeded != post_receipt.transport_succeeded
            or final_status != post_receipt.status
            or final_effect_verified != post_receipt.effect_verified
            or final_error_msg != post_receipt.error_status
        ):
            post_receipt = PostActionReceipt(
                **{
                    **post_receipt.to_dict(),
                    "output_hash": _stable_digest(result),
                    "status": final_status,
                    "effect_verified": final_effect_verified,
                    "error_status": final_error_msg,
                    "transport_succeeded": final_transport_succeeded,
                    "retry_safe": bool(result.get("retry_safe", False)),
                    "manual_reconciliation_required": bool(
                        result.get(
                            "manual_reconciliation_required",
                            False,
                        )
                    ),
                }
            )
        return final_error_msg, post_receipt

    @staticmethod
    async def _execute_part_7(
        action_name: str,
        external_execute_coordinator: Any,
        external_execution_offer: Any,
        persisted_post_receipt: Any,
        post_receipt: Any,
        receipt_store: Any,
        result: Any,
    ) -> None:
        from .action_executor import (
            _ACTION_EXECUTOR_RECOVERABLE_ERRORS,
            record_degradation,
        )

        result["post_action_receipt_id"] = post_receipt.receipt_id
        result["post_action_output_hash"] = post_receipt.output_hash
        result["receipt_persisted"] = True
        result["post_action_receipt_pending"] = False
        if external_execute_coordinator is not None and external_execution_offer is not None:
            try:
                linked = await asyncio.to_thread(
                    external_execute_coordinator.link_post_action_receipt,
                    offer=external_execution_offer,
                    persisted_receipt=persisted_post_receipt.to_dict(),
                    receipt_store=receipt_store,
                )
                result["external_execution_transaction"] = linked
                result["external_execution_receipt_linked"] = True
            except _ACTION_EXECUTOR_RECOVERABLE_ERRORS as exc:
                record_degradation(
                    "action_executor.external_execution",
                    exc,
                    action=(
                        "preserved completed effect and durable post-action "
                        f"receipt after transaction-link failure for {action_name}"
                    ),
                    severity="degraded",
                    enforce_failure_policy=False,
                )
                result["external_execution_receipt_linked"] = False

