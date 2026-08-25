"""The parent side of the endogenous language pathway, in one place.

Four hooks: put z_Aura on an outbound job, fold the worker's receipt into the
parent's health view, pair a reply with the state that produced it, and absorb
what the turn concluded back into that state.

They live here rather than in ``mlx_client`` because that file is fifteen
thousand lines and the size ratchet exists to stop it being sixteen. The
client calls four names; everything they do is here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger("Aura.EndogenousClientHooks")


def observe_endogenous_receipt(response: Mapping[str, Any]) -> None:
    """Fold the worker's endogenous receipt into the parent's health view.

    The bias is applied in the worker process, so without this the parent
    could report a pathway as wired while every generation refused it.
    """
    try:
        receipt = response.get("endogenous_bias")
        if not receipt:
            return
        from core.brain.llm.endogenous_decode import observe_receipt

        observe_receipt(receipt)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("endogenous receipt not observed: %s", exc)

def absorb_endogenous_outcome(response: Mapping[str, Any]) -> None:
    """Fold what the turn concluded back into the state that asked for it.

    Without this arrow a conclusion evaporates when the response is
    emitted and the next turn starts from a state that learned nothing.
    Off by default because it changes live dynamics: set
    AURA_ENDOGENOUS_ABSORB=1 to close the loop.
    """
    try:
        if not response.get("text"):
            return
        if os.environ.get("AURA_ENDOGENOUS_ABSORB", "").strip().lower() not in {
            "1",
            "true",
            "on",
            "yes",
        }:
            return
        from core.brain.llm.endogenous_absorption import absorb, outcome_from_response

        receipt = absorb(outcome_from_response(response))
        if not receipt.accepted:
            logger.debug("endogenous absorption did not land: %s", receipt.reason)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("endogenous absorption unavailable: %s", exc)

def record_endogenous_pair(response: Mapping[str, Any]) -> None:
    """Store the state that produced this reply, beside the reply.

    Only terminal frames carry text, and only a request this parent still
    holds a state for can be paired, so an unmatched frame writes nothing
    rather than pairing a reply with the wrong state.
    """
    try:
        text = response.get("text")
        if not text:
            return
        from core.brain.llm.endogenous_pair_recorder import record_response

        record_response(str(response.get("id") or ""), str(text))
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
        logger.debug("endogenous pair not recorded: %s", exc)

def attach_endogenous_state(
    req: dict[str, Any],
    *,
    model_path: str = "",
    override: Mapping[str, Any] | None = None,
) -> None:
    """Put z_Aura on an outbound generation job.

    Fail-open by construction: a turn that cannot assemble a state is a turn
    that generates the way it always did. What must never happen is a job
    carrying a state that looks live and is not, so a state whose channels
    are all absent is left off the job rather than shipped as zeros.
    """
    try:
        from core.brain.llm.endogenous_decode import JOB_STATE_KEY
        from core.brain.llm.endogenous_state import assemble_state
        from core.brain.llm.endogenous_vocab_head import (
            MIN_COVERAGE,
            alpha_from_env,
        )

        alpha = alpha_from_env()
        if alpha <= 0.0:
            return
        if override:
            # An experiment supplies the state directly. Assembling a fresh one
            # here would silently discard the intervention and turn every arm
            # of a causal battery into its own control.
            payload = dict(override)
        else:
            state = assemble_state()
            if state.coverage < MIN_COVERAGE:
                return
            payload = state.to_payload()
        req[JOB_STATE_KEY] = payload
        req["endogenous_alpha"] = alpha

        # Hold the state until the reply lands, so the corpus the head is
        # fitted on is her own state paired with her own words rather than
        # two records that have to be joined by timestamp later.
        from core.brain.llm.endogenous_pair_recorder import remember_pending

        remember_pending(
            str(req.get("id") or ""),
            payload,
            lane=str(req.get("serving_lane") or ""),
            model=str(model_path or ""),
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("endogenous state not attached to job: %s", exc)


__all__ = [
    "absorb_endogenous_outcome",
    "attach_endogenous_state",
    "observe_endogenous_receipt",
    "record_endogenous_pair",
]
