"""Bind procedural obligations to the action and observation that can answer them.

Pixel evidence stays private to the action path; procedure receipts contain only
detached JSON data. Input delivery, screen response and a specific effect are
separate facts, including when a plan is replaced while the input is in flight.
"""
from __future__ import annotations

import time
from typing import Any

from core.runtime.skill_contract import PredicateOperator, PredicateState, SemanticPredicate
from core.verify.invariants import invariant


def screen_evidence(observation: dict[str, Any], clickable: Any = ()) -> dict[str, Any]:
    """A small detached observation for deterministic procedure evaluation."""
    return {
        "text": str(observation.get("text") or ""),
        "clickable": [str(move) for move in clickable],
        "surface_id": str(observation.get("surface_id") or observation.get("scoped_to") or ""),
        "capture_epoch": str(observation.get("_capture_epoch") or ""),
        "capture_at": observation.get("capture_at"),
        "bounds": list(observation.get("bounds") or ()),
        "regions": [{key: region.get(key) for key in ("text", "x", "y", "width", "height", "center_x", "center_y")}
                    for region in (*tuple(observation.get("layout") or ()), *tuple(observation.get("shapes") or ()))
                    if isinstance(region, dict)],
    }


def transfer_effects() -> tuple[SemanticPredicate, ...]:
    return (SemanticPredicate(
        predicate_id="object_reached_destination", evidence_path="after.placement.state",
        operator=PredicateOperator.EQUALS, expected="verified",
        repair_hint="Observe the source and destination again; repair this transfer before continuing.",
    ),)


def bind_step(run: Any) -> None:
    """Freeze the procedure instances and their own step binding before dispatch."""
    from core.agency.putting_things_in_place import what_is_carried
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    before = run.pending.get("observed_before") or {}
    effects = transfer_effects() if what_is_carried(run.key) else ()
    bindings = []
    procedures = [procedure for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None))
                  if procedure is not None]
    if any(procedure.pending_step is not None for procedure in procedures):
        raise RuntimeError("the earlier delivered requirement still needs its observation")
    try:
        for procedure in procedures:
            attempt = procedure.begin(run.key, before, effects=effects)
            if attempt is not None:
                bindings.append((procedure, attempt))
                run.pending["step_bindings"] = tuple(bindings)
    except BaseException:
        abandon_bound_step(run.pending, run.key)
        raise
    run.pending["step_bindings"] = tuple(bindings)
    run.pending["step_settled"] = False
    run.pending["verification_started"] = time.monotonic()
    run.pending["verification_epochs"] = set()


def abandon_bound_step(pending: dict[str, Any], move: str) -> None:
    """Release owned obligations after refused or uncertain delivery, without success credit."""
    try:
        for procedure, attempt in pending.get("step_bindings", ()):
            if procedure.pending_step is attempt:
                try:
                    procedure.tried(move, None, before=attempt.before.evidence, after={}, step_key=attempt.step_key)
                except (RuntimeError, TypeError, ValueError) as exc:
                    from core.runtime.errors import record_degradation

                    record_degradation("screen_pursuit", exc, severity="info",
                                       action="released an unmeasured procedure attempt after failed delivery")
                finally:
                    if procedure.pending_step is attempt:
                        procedure.pending_step = None
    finally:
        pending["step_bindings"] = ()
        pending["step_settled"] = True


def artifact_ready_for(move: str, can_do: Any = None) -> bool:
    """A construction task's execution controls need currently measured work, including unmatched controls."""
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.cognition.procedure_binding import consumes_artifact

    label = what_is_clicked(move)
    guide = THE_GUIDE.get()
    procedures = [procedure for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None))
                  if procedure is not None]
    constructing = any(frame.act == "carry" for procedure in procedures for frame in procedure.steps())
    if not constructing or not label or not consumes_artifact(label):
        return True
    measured = getattr(can_do, "placed_count", None)
    return callable(measured) and measured() > 0


@invariant("agency.construction_execution_requires_current_placement", scope="agency",
           owner="core/skills/screen_step_evidence.py", observational=False)
def _construction_readiness_invariant() -> tuple:
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
    from core.cognition.reading_the_rules import Frame, Rules

    step = Frame("Place a component in the workspace", act="carry", thing="component", where="workspace")
    guide = Guide(rules=Rules(frames={step.key: step}), built="An old account says it was built")
    can_do = WhatWorksHere(carried_to={'drag "component" to "workspace"': True})
    token = THE_GUIDE.set(guide)
    try:
        assert not artifact_ready_for('click "Run"', can_do)
        assert not artifact_ready_for('click "Test circuit"', can_do)
        assert artifact_ready_for('click "Library"', can_do)
    finally:
        THE_GUIDE.reset(token)
    return ()


def admissible_step(move: str, *, reaching: Any = (), can_do: Any = None) -> bool:
    """Known later requirements wait for their prerequisites, independent of scores."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if not artifact_ready_for(move, can_do):
        return False
    for procedure in (getattr(guide, "rules", None), getattr(guide, "plan", None)):
        if procedure is None:
            continue
        if procedure.pending_step is not None:
            return False
        step, needed = procedure.step_of(move), procedure.next_step()
        if (step is not None and needed is not None and step.order > needed.order and move not in reaching
                and not procedure.reaches_next(move)):
            return False
    return True


def settle_step(pending: dict[str, Any], observation: dict[str, Any], clickable: Any, can_do: Any = None) -> bool:
    """Answer delivered obligations once, before the screen can propose a new plan."""
    if pending.get("step_settled"):
        return True
    previous = pending.get("deliberation")
    chosen = getattr(previous, "chosen", None)
    if chosen is None:
        return True
    after = screen_evidence(observation, clickable)
    before = pending.get("observed_before") or {}
    intent = pending.get("transfer_intent")
    effects = transfer_effects() if intent is not None else ()
    if intent is not None:
        from core.perception.observed_transfer import verify_transfer

        receipt = verify_transfer(intent, observation)
        pending["transfer_receipt"] = receipt
        if receipt.state is PredicateState.UNKNOWN:
            epochs = pending.setdefault("verification_epochs", set())
            if after.get("capture_epoch") and after["capture_epoch"] != before.get("capture_epoch"):
                epochs.add(after["capture_epoch"])
            if len(epochs) < 2 and time.monotonic() - pending.get("verification_started", 0.0) < 30.0:
                return False
        if can_do is not None:
            can_do.carried(chosen.name, True, receipt=receipt)
        if receipt.state is not PredicateState.UNKNOWN:
            after["placement"] = {"state": "verified" if receipt.state is PredicateState.SATISFIED else "absent"}
    # Response cannot be claimed across a different surface or an old capture.
    fresh = (bool(before.get("capture_epoch"))
             and bool(after.get("capture_epoch"))
             and before.get("capture_epoch") != after.get("capture_epoch")
             and bool(before.get("surface_id")) and before.get("surface_id") == after.get("surface_id")
             and before.get("bounds") == after.get("bounds"))
    before_at, after_at = before.get("capture_at"), after.get("capture_at")
    fresh = fresh and (type(before_at) in (int, float) and type(after_at) in (int, float)
                       and before_at < after_at and after_at > pending.get("dispatch_completed", float("inf")))
    changed = (before.get("text") != after.get("text")
               or before.get("clickable") != after.get("clickable")
               or before.get("regions") != after.get("regions")
               or after.get("placement", {}).get("state") == "verified") if fresh else None
    if intent is not None and pending["transfer_receipt"].state is PredicateState.UNKNOWN:
        changed = None
    elif after.get("placement", {}).get("state") != "verified" and changed is not None:
        from .screen_pursuit_on_a_page import it_was_answered

        changed = it_was_answered(changed, observation)
    for procedure, attempt in pending.get("step_bindings", ()):
        procedure.tried(chosen.name, changed, before=before, after=after,
                        step_key=attempt.step_key, effects=effects)
    pending["step_settled"] = True
    pending["step_after"] = after
    return True


def trial_context(pending: dict[str, Any], observation: dict[str, Any], clickable: Any) -> dict[str, Any]:
    return {"before": pending.get("observed_before") or {},
            "after": pending.get("step_after") or screen_evidence(observation, clickable),
            "transfer": pending.get("transfer_receipt")}
