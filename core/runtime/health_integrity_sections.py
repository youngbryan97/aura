"""The four sections of the runtime integrity block.

Lifted whole out of `health_contract`, which had grown past the 2000-line
ceiling. Every name taken from that module is imported at CALL time: it
imports this one to re-export these four, and a test that patches a name on
it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import Any


def _integrity_of_her_shape_and_boundaries(block: dict[str, Any]) -> None:
    # The order in which she thinks, compiled and sealed.
    #
    # Three peer architectures make the same complaint from three directions:
    # Generative Agents puts the whole cycle in one function, Soar's decision
    # cycle is a state machine anyone can enumerate, and LangGraph refuses to
    # start when the topology does not resolve. Aura has the order, the
    # frequency and the per-phase contracts, and nothing joined them, so a
    # field two phases both write was found by watching it happen.
    #
    # The seal is what a receipt can carry: two runs of one commit agree, and
    # a phase added, removed, reordered or re-declared does not.
    from .health_contract import (
        _GRAPH_NODES_WORTH_WALKING,
        get_runtime_service,
    )

    try:
        from core.runtime.the_shape_of_one_turn import compile_the_cognition

        block["the_shape_of_one_turn"] = {
            mode: {
                key: value
                for key, value in compile_the_cognition(mode).to_dict().items()
                if key not in {"phases"}
            }
            for mode in ("foreground", "background")
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_shape_of_one_turn"] = {"error": repr(exc)}
    # Decision points that have only ever answered one way.
    #
    # A gate nobody can pass takes a working system and gates it into a coma;
    # a gate nobody can fail is decorative. Neither is findable in the source,
    # because the code branches both ways and only the traffic says which
    # branch is real. LIVE 2026-09-07: `/api/readyz` answered 503 for the whole
    # of every turn while the runtime answered perfectly, and the prompt cache
    # missed on every turn for the life of the process.
    #
    # Reported, never enforced. A runtime that refuses to start because a
    # counter looks lopsided is the coma arriving by another route.
    try:
        from core.verify.one_way_decisions import decision_census, one_way_decisions

        block["one_way_decisions"] = {
            "census": decision_census(),
            "only_one_answer": [
                {
                    "name": item.name,
                    "verdict": item.verdict,
                    "decisions": item.total,
                    "last_refusal_reason": item.last_reason,
                }
                for item in one_way_decisions()
            ],
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["one_way_decisions"] = {"error": repr(exc)}
    # One working memory, and whether anything still normalises it against a
    # number of its own. Three readers did, and each was pinned at its own
    # ceiling for most of a conversation — a constant that looked like a
    # measurement. Reported live so a new one cannot arrive unseen between
    # test runs.
    try:
        from core.state.one_working_memory import (
            THE_STORE,
            the_capacity,
            the_caps_that_disagree,
            who_else_holds_it,
        )

        disagreeing = the_caps_that_disagree()
        block["one_working_memory"] = {
            "store": THE_STORE,
            "capacity": the_capacity(),
            "projections": len(who_else_holds_it()),
            "caps_that_disagree": disagreeing,
            "agreed": not disagreeing,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["one_working_memory"] = {"error": repr(exc)}
    # The runtime boundary, and how much of the declared service spine the
    # runtime that is actually up can resolve. A name it cannot resolve is a
    # service somebody else owns — a module singleton, a boot-local — which
    # is the ownership debt, measured rather than asserted.
    try:
        from core.runtime.what_a_runtime_is import (
            THE_OPERATIONS,
            the_services_it_can_address,
            what_is_missing_from,
        )

        # Through the runtime registry, like everything else here. Reaching
        # for ServiceContainer put core.container behind core.runtime, which
        # is the dependency that stops the foundation coming up to report on
        # a mind that failed to start — and there is a test that says so.
        interface = get_runtime_service("kernel_interface", default=None)
        running = getattr(interface, "kernel", None)
        boundary: dict[str, Any] = {"operations": sorted(THE_OPERATIONS)}
        if running is None:
            boundary["runtime"] = "not up"
        else:
            from core.runtime.what_a_runtime_is import a_runtime_over

            over = a_runtime_over(running)
            boundary["runtime"] = type(running).__name__
            boundary["missing"] = what_is_missing_from(over)
            boundary.update(the_services_it_can_address(over))
        block["the_runtime_boundary"] = boundary
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_runtime_boundary"] = {"error": repr(exc)}
    # Who holds the scarce things, who is queued for them, and how the waiting
    # has gone. A lock answers none of those: it is a boolean with a queue
    # nobody can see, and whose order is whatever the loop decided.
    try:
        from core.runtime.who_gets_it_next import (
            how_it_has_gone,
            who_holds_what,
            who_is_waiting,
        )

        block["who_holds_what"] = {
            "held": who_holds_what(),
            "waiting": who_is_waiting(),
            "record": how_it_has_gone(),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["who_holds_what"] = {"error": repr(exc)}
    # Which durable fields have an authority. Counts only — the lists are long
    # and the file that carries them is the baseline. A field that loses its
    # owner between two builds is what this is for.
    try:
        from core.state.who_owns_each_field import what_it_stood_at_last_time

        block["who_owns_each_field"] = what_it_stood_at_last_time()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["who_owns_each_field"] = {"error": repr(exc)}
    # The graph stores under one shape, and whether a reference from one into
    # another still lands. Over the LIVE instances: the same check over fresh
    # graphs measures nothing, which is how a check like this usually fails.
    try:
        from core.knowledge.one_graph import (
            every_graph,
            references_that_lead_nowhere,
            which_stores_have_not_registered,
        )

        graphs = every_graph(live=True)
        # Bounded: the integrity walk is O(nodes x links), and a health route
        # must not become the most expensive thing in the process. Above the
        # bound the count is reported and the walk is not run.
        nodes = sum(len(one.all_nodes()) for one in graphs.values())
        too_big = nodes > _GRAPH_NODES_WORTH_WALKING
        nowhere = [] if (too_big or not graphs) else references_that_lead_nowhere(graphs)
        block["one_graph"] = {
            "stores": sorted(graphs),
            "not_registered": which_stores_have_not_registered(),
            "nodes": nodes,
            "references_that_lead_nowhere": nowhere[:20],
            "how_many_lead_nowhere": len(nowhere),
            "walked": not too_big,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["one_graph"] = {"error": repr(exc)}
    # Which answer route actually answered. A route offered every turn that
    # has never answered one is either unable to fire or gated wrong, and
    # neither is visible from the source: declining and being unable to
    # answer look identical from outside.
    try:
        from core.runtime.what_answered_this_turn import (
            how_the_routes_have_gone,
            routes_that_have_never_answered,
        )

        block["what_answered_this_turn"] = {
            "routes": how_the_routes_have_gone(),
            "never_answered": routes_that_have_never_answered(),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_answered_this_turn"] = {"error": repr(exc)}
    # Who owns the runtime right now, and what a cancelled turn is still
    # waiting on. A runtime that reports idle while it is still tearing a turn
    # down will start the next one on top of it.
    try:
        from core.runtime.whose_turn_it_is import the_turn

        block["whose_turn_it_is"] = the_turn().report()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["whose_turn_it_is"] = {"error": repr(exc)}
    # Who is listening, in a form a restart can put back. AutoGen saves agent
    # state and says it does not save this; a subscription that does not
    # survive a restart is a listener that silently stops.
    try:
        from core.runtime.what_a_message_carries import what_was_subscribed

        subscribed = what_was_subscribed()
        block["what_a_message_carries"] = {
            "subscriptions": len(subscribed),
            "topics": sorted({one["topic"] for one in subscribed}),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_a_message_carries"] = {"error": repr(exc)}
    # What each phase committed at its boundary, and what is held by more than
    # one. A stuck refcount has to be answerable rather than a mystery.
    try:
        from core.state.what_a_phase_changed import how_the_boundaries_have_gone

        block["what_a_phase_changed"] = how_the_boundaries_have_gone()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_a_phase_changed"] = {"error": repr(exc)}
    # What ran on the loop's thread that must not. An on-loop fsync once froze
    # this loop for twenty minutes, and the fix was a rule in a guide; this is
    # the part that can tell you the rule was broken.
    try:
        from core.runtime.which_thread_may_do_this import how_it_has_gone

        block["which_thread_may_do_this"] = how_it_has_gone()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["which_thread_may_do_this"] = {"error": repr(exc)}
    # Every state holder, and whether it decides a fact, shows one, or is
    # scratch. Counts here; the table itself is in the module.
    try:
        from core.state.what_kind_of_state_is_this import how_the_state_is_organised

        organised = how_the_state_is_organised()
        block["what_kind_of_state_is_this"] = {
            key: value for key, value in organised.items() if key != "by_kind"
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_kind_of_state_is_this"] = {"error": repr(exc)}
    # What a skill gave back that it did not declare. Every one of the 82
    # declares now; a declaration nothing checks is a comment.
    try:
        from core.skills.what_every_skill_gives_back import how_results_have_differed

        differed = how_results_have_differed()
        block["what_every_skill_gives_back"] = {
            "skills_that_differed": len(differed),
            "differed": differed,
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_every_skill_gives_back"] = {"error": repr(exc)}

def _integrity_of_control_and_measured_effect(block: dict[str, Any]) -> None:
    # What the uncalibrated decoding policy actually does. Counts only: the
    # full sweep is 5,760 states and belongs in a test, not on a route.
    from .health_contract import (
        _attach_causal_evidence,
        get_runtime_service,
    )

    try:
        from core.runtime.service_registry import get_runtime_service

        # Through the registry, not by import: this package may not reach
        # core.brain, and a health block that needed that edge would be a
        # layering violation dressed as observability.
        provider = get_runtime_service("the_control_policy_sweep", default=None)
        if not callable(provider):
            raise RuntimeError("the control policy sweep is not registered")
        swept = provider()
        block["the_control_policy"] = {
            "policy": swept["policy"],
            "calibrated": swept["calibrated"],
            "states_swept": swept["states_swept"],
            "controls_that_never_move": swept["controls_that_never_move"],
            "discriminates": swept["discriminates"],
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["the_control_policy"] = {"error": repr(exc)}
    # How many named faculties have a measured downstream effect. A channel
    # wired to a consumer is not one; only `measured` is evidence.
    try:
        from core.verify.what_has_a_measured_effect import what_it_stood_at_last_time

        block["what_has_a_measured_effect"] = what_it_stood_at_last_time()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_has_a_measured_effect"] = {"error": repr(exc)}
    # Whether each organ knows what it owns, consumes, promises, and does when
    # it fails. Counts from the committed baseline: asking every package walks
    # the whole tree, and health is served on a route.
    try:
        from core.verify.what_each_organ_says import the_baseline

        held = the_baseline()
        block["what_each_organ_says"] = {
            "organs": held.get("organs"),
            "answer_all_four": held.get("answer_all_four"),
            "answer_nothing": held.get("answer_nothing"),
            "who_does_not_say": held.get("who_does_not_say", {}),
        }
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_each_organ_says"] = {"error": repr(exc)}
    # The benchmarks somebody else designed: what ran, what could not, and
    # whether anything is claimed without its limit.
    try:
        from core.verify.what_was_measured_outside import how_it_stands

        block["what_was_measured_outside"] = how_it_stands()
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["what_was_measured_outside"] = {"error": repr(exc)}
    # Which semantic routes decide an answer and which only watch one. A
    # shadow route contributes no answers however good it gets.
    try:
        from core.runtime.service_registry import get_runtime_service

        # Through the registry: this package may not reach core.brain, and a
        # health block that needed that edge would be a layering violation
        # dressed as observability.
        provider = get_runtime_service("which_routes_are_authoritative", default=None)
        block["which_routes_are_authoritative"] = (
            provider() if callable(provider) else {"registered": False}
        )
    except Exception as exc:  # noqa: BLE001 — health must never raise at its caller
        block["which_routes_are_authoritative"] = {"error": repr(exc)}
    # Whether Aura's own cognitive state reached the words she produced. Read
    # through the registry rather than imported: this package may not reach
    # core.brain, and a health block that needed that edge would be a layering
    # violation dressed as observability. ``unexpected_refusals`` is the field
    # that matters — no trained head at all is the pathway waiting for a fit,
    # while a head on disk that refuses to attach is a mismatch with the
    # resident model.
    try:
        from core.runtime.service_registry import get_runtime_service

        # The runtime registry only. This package resolves services through
        # the low-level registry by contract, so that the foundation can come
        # up and report without the container — and a test pins that this file
        # never reaches for ServiceContainer.
        provider = get_runtime_service("endogenous_language_health", default=None)
        if callable(provider):
            block["endogenous_language"] = provider()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["endogenous_language_error"] = repr(exc)
    # The shape morphogenesis is holding: how many cells, how many bindings,
    # and whether any of them can reach each other.
    #
    # The topology had exactly one reader and it was an HTTP route —
    # `_collect_morphogenesis_status` in interface/routes/system.py — so
    # nothing that reads this report could see it, including the mind's own
    # self-knowledge. Asked live on 2026-09-20 which of its cells were cut
    # off, with `/api/health` carrying `components: 7, partitioned: true` at
    # that moment, the answer was "I don't have any operational information
    # regarding a system called 'morphogenic cell'". A population reported as
    # fifty healthy cells says nothing about whether any of them can reach
    # each other, which is the thing this layer is for.
    try:
        from core.runtime.service_registry import get_runtime_service

        runtime = get_runtime_service("morphogenetic_runtime", default=None)
        status = getattr(runtime, "status", None)
        if callable(status):
            shape = status()
            topology = dict(shape.get("topology") or {})
            registry = dict(shape.get("registry") or {})
            components = int(topology.get("components", 1) or 1)
            block["the_shape_it_is_holding"] = {
                "cells": int(registry.get("cells", 0) or 0),
                "organs": int(registry.get("organs", 0) or 0),
                "bindings": int(topology.get("edges", 0) or 0),
                "topology_version": int(topology.get("version", 0) or 0),
                "components": components,
                "partitioned": components > 1,
            }
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["the_shape_it_is_holding_error"] = repr(exc)
    # Who this runtime is, and where its state lives. Every persistent record
    # is stamped with this, so a store found in the wrong place can be traced
    # to the process that wrote it.
    try:
        from core.runtime.state_ownership import runtime_identity

        block["runtime_identity"] = runtime_identity()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["runtime_identity_error"] = repr(exc)
    # Whether the copy that makes deletion survivable actually exists, and
    # whether it is somewhere a wipe would not reach.
    #
    # Surfaced because both halves failed silently by default: an ark that
    # was never built reports "ok: false, no ark manifest" and nothing else
    # would ever say so, and an ark inside the blast radius looks identical
    # to a safe one until the moment it is needed. This block was wrong on
    # the first attempt — state_root() is ~/.aura and the repo is
    # ~/.aura/live-source, so the ark died to the same `rm -rf` as the
    # original.
    try:
        from core.security.existence_guard import get_existence_guard

        guard = get_existence_guard()
        block["existence_guard"] = {
            "ark": guard.verify_ark(),
            "ark_location": guard.ark_is_outside_the_blast_radius(),
            "sealed": guard.is_sealed(),
        }
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["existence_guard_error"] = repr(exc)
    # Whether the compiled launcher is the launcher its source describes.
    #
    # LIVE DEFECT, 2026-08-10. Bryan reported companion mode did not work: he
    # closed the window and no bubble appeared. Every Python-side organ was
    # correct, and the reason was that the installed launcher binary was built
    # 2026-08-03 while scripts/AuraLauncher.swift had gained the entire
    # companion surface on 2026-08-09. The resident binary contained zero
    # occurrences of "/api/ambient/visibility"; the feature was not broken, it
    # was absent from the executable.
    #
    # Aura.app is deliberately a thin launcher over live source, so Python,
    # assets and config cannot go stale — which is exactly why this one
    # artifact is dangerous. It is the only compiled thing, so it is the only
    # thing a restart does NOT bring current, and its drift therefore looks
    # like a feature that silently does nothing.
    #
    # core.runtime.app_bundle_sync detects and repairs this correctly and was
    # reachable only from launch_aura.sh, which the normal double-click path
    # never executes — spawnAuraProcess runs aura_main.py directly and only
    # falls back to the shell script for protected folders. So the detector
    # existed, was tested, and could not fire for the user who needed it.
    # Reporting it here is what makes the condition observable at all.
    try:
        from core.runtime.app_bundle_sync import launcher_currency

        block["launcher_currency"] = launcher_currency()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["launcher_currency_error"] = repr(exc)
    # What Aura has been ALLOWED to learn permanently, and on whose evidence.
    # Without this the durable-learning gate could be doing anything and the
    # health surface would look identical — the gate's own report existed and
    # had no caller, which is the residue shape this codebase keeps finding.
    try:
        from core.governance.durable_learning import get_durable_learning_gate

        block["durable_learning"] = get_durable_learning_gate().report()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["durable_learning_error"] = repr(exc)
    # Turns that reached cognition without a Will decision.
    #
    # The message handler's comment claimed ALL processing passes the Unified
    # Will. It does not: the gate is skipped entirely before the Will starts,
    # and continues degraded when it raises. Both were silent, so a turn
    # nobody governed looked exactly like a turn the Will approved. A runtime
    # serving ungoverned turns is precisely the kind of fact a green verdict
    # cannot express on its own, which is what this block is for.
    try:
        from core.runtime.governance_coverage import ungoverned_turn_report

        block["ungoverned_turns"] = ungoverned_turn_report()
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["ungoverned_turns_error"] = repr(exc)
    # Whether the activation-grounded Φ complex is being fed at all.
    #
    # The residual channel carries 8-bit Grassmann states out of the MLX
    # worker, and for its whole existence nothing in the parent drained it —
    # so the complex reported insufficient_history:0/50 forever while three
    # modules and a live writer said otherwise. Depth on the surface means a
    # regression to zero is visible instead of silent.
    try:
        phi_core = get_runtime_service("phi_core", default=None)
        if phi_core is not None and hasattr(phi_core, "grassmann_history_depth"):
            # Typed wide because the publishers key below carries either the
            # per-hook diagnostics or a sentence saying why there are none.
            history: dict[str, Any] = {
                "grassmann_states": int(phi_core.grassmann_history_depth())
            }
            # A depth of zero has four possible causes and they are not the
            # same problem: no hook was ever called, the encoder is still
            # filling its window, the encoder is refusing every sample, or
            # nothing drains the ring. Reported as one number they were
            # indistinguishable, and the channel sat at zero for its whole
            # existence with nothing able to say why. The publisher's own
            # counters are read here so the zero explains itself.
            engine = get_runtime_service("affective_steering_engine", default=None)
            hooks = list(getattr(engine, "_hooks", None) or []) if engine else []
            if hooks:
                history["publishers"] = [
                    dict(hook.get_diagnostics().get("phi_residual", {}))
                    for hook in hooks
                    if hasattr(hook, "get_diagnostics")
                ]
            elif engine is not None:
                history["publishers"] = "steering engine present with no hooks"
            block["phi_residual_history"] = history
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["phi_residual_history_error"] = repr(exc)
    # Whether admission is predicting from measurement or still guessing.
    #
    # Read through the runtime service registry rather than by importing the
    # estimator: core/runtime may not depend on core.brain, and that rule is
    # the reason the foundation can come up and report on a mind that failed
    # to start. The estimator registers itself; health only reads.
    try:
        estimator = get_runtime_service("admission_throughput_estimator", default=None)
        if estimator is not None:
            throughput = estimator.report()
            block["admission_throughput"] = {
                "shapes_measured": throughput["shapes_measured"],
                "total_samples": throughput["total_samples"],
            }
        else:
            block["admission_throughput"] = {"registered": False}
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["admission_throughput_error"] = repr(exc)
    _attach_causal_evidence(block)

def _integrity_of_taint_locks_and_custody(block: dict[str, Any]) -> None:
    # Which path actually produced the recent replies. A demo showing a fluent
    # answer establishes nothing about the pipeline until this says the pipeline
    # ran.
    try:
        from core.verify.turn_receipt import recent_receipts

        receipts = recent_receipts(limit=16)
        from core.verify.turn_receipt import latency_by_component

        block["turn_latency"] = latency_by_component(limit=16)
        block["turn_paths"] = {
            "recent": receipts,
            "full_pipeline_turns": sum(
                1 for r in receipts if r.get("full_pipeline_ran")
            ),
            "model_generation_turns": sum(
                1 for r in receipts if r.get("model_generation")
            ),
            "turns_recorded": len(receipts),
        }
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["turn_paths_error"] = repr(exc)
    try:
        from core.runtime.taint import credibility_caveat, taint_compact, taint_report

        block["taint"] = taint_report()
        block["taint_compact"] = taint_compact()
        caveat = credibility_caveat()
        if caveat:
            block["credibility_caveat"] = caveat
    except Exception as exc:  # noqa: BLE001 — integrity reporting is additive
        block["taint_error"] = repr(exc)
    try:
        from core.runtime.lockdep import lockdep_report

        lock_report = lockdep_report()
        block["lockdep"] = {
            "clean": lock_report["clean"],
            "acquires_checked": lock_report["acquires_checked"],
            "splats": lock_report["splats"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["lockdep_error"] = repr(exc)
    try:
        from core.runtime.pressure_stall import psi_narrative, psi_report, saturated_resources

        block["pressure"] = psi_report()
        block["pressure_saturated"] = saturated_resources()
        block["pressure_narrative"] = psi_narrative()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["pressure_error"] = repr(exc)
    try:
        # Two boundaries that are worth exactly as much as their visibility.
        # Egress counts what left after being read; attestation says whether
        # the identity Aura booted with is the one she last wrote. Both
        # answer honestly when nothing has happened yet — a zero here means
        # "nothing was refused", never "nothing was checked".
        from core.security.egress_privacy import egress_privacy_counters
        from core.security.state_attestation import attestation_report

        block["egress_privacy"] = egress_privacy_counters()
        block["state_attestation"] = attestation_report()
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["egress_privacy_error"] = repr(exc)
    try:
        from core.knowledge.metta import metta_report
        from core.organism.model_validation import get_suite, validation_report
        from core.runtime.foundations import cognition_validation_status

        validation = validation_report()
        block["self_model"] = {
            "claims": len(validation["claims"]),
            "tests": len(validation["tests"]),
            "unsupported_claims": [c["statement"] for c in get_suite().unsupported_claims()],
            "empirical_run": cognition_validation_status(),
            "metta": {k: metta_report()[k] for k in ("rules", "reductions", "truncations")},
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["self_model_error"] = repr(exc)
    try:
        # Which stages lose facts the turn already established. A break here
        # is not a bad answer — it is a composition defect, and it names the
        # stage responsible, which is the part that was previously impossible
        # to recover after the fact.
        from core.runtime.fact_custody import custody_report

        custody = custody_report()
        block["fact_custody"] = {
            "turns_tracked": custody["turns_tracked"],
            "turns_with_breaks": custody["turns_with_breaks"],
            "stages_that_broke_custody": custody["stages_that_broke_custody"],
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["fact_custody_error"] = repr(exc)
    try:
        from core.fsw.assertions import assertions_report
        from core.fsw.command_dispatch import command_report
        from core.fsw.health_checker import health_checker_report
        from core.fsw.rate_groups import rate_group_report
        from core.fsw.restart_protection import restart_report
        from core.fsw.telemetry_dictionary import telemetry_report

        telemetry = telemetry_report()
        pings = health_checker_report()
        block["flight_software"] = {
            "telemetry": {
                "channels": telemetry["channels"],
                "violations": telemetry["violations"],
                "recent_events": telemetry["recent_events"],
            },
            "restart_protection": restart_report()["core_sets"],
            "rate_groups": {
                k: rate_group_report()[k] for k in ("slipping", "total_cycles", "total_slips")
            },
            "assertions": {
                "clean": assertions_report()["clean"],
                "distinct_sites": assertions_report()["distinct_sites"],
            },
            "health_pings": {
                "unresponsive": pings["unresponsive"],
                "slow": pings["slow"],
                "critical_unresponsive": pings["critical_unresponsive"],
            },
            "commands": {
                "declared": command_report()["commands"],
                "dispatched": command_report()["dispatched"],
            },
        }
    except Exception as exc:  # noqa: BLE001 - each health add-on is isolated
        block["flight_software_error"] = repr(exc)
    try:
        from core.observability.histograms import histograms_report
        from core.observability.trace_events import tracer_report
        from core.runtime.field_trials import field_trials_report
        from core.runtime.memory_infra import memory_infra_report
        from core.security.rule_of_two import rule_of_two_report

        histograms = histograms_report()
        memory = memory_infra_report()
        posture = rule_of_two_report()
        block["observability"] = {
            "histograms": {
                "count": histograms["count"],
                "clipping": histograms["clipping"],
                "expired": [e["name"] for e in histograms["expired"]],
            },
            "trace": {
                k: tracer_report()[k] for k in ("enabled", "buffered", "dropped", "span_s")
            },
            "memory_attribution": memory["leak_report"],
            "field_trials": field_trials_report()["active_groups"],
            "security_posture": {
                "violations": posture["violations"],
                "at_the_limit": posture["at_the_limit"],
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["observability_error"] = repr(exc)
    try:
        from core.ontogeny.service import ontogeny_health_report

        ontogeny = ontogeny_health_report()
        # The owner supplies this bounded projection. The full diagnostic
        # report includes per-control-point corpus aggregates and must never be
        # pulled into a high-frequency health probe.
        block["ontogeny"] = {
            "episodes_seen": ontogeny.get("episodes_seen"),
            "novelty": ontogeny.get("novelty"),
            "state": {
                k: (ontogeny.get("state") or {}).get(k)
                for k in ("steps", "era", "fingerprint", "age_days")
            },
            "stages": dict(ontogeny.get("stages") or {}),
            "frozen": ontogeny.get("frozen"),
            "observation_rate": ontogeny.get("observation_rate"),
            "calibration": {
                cp: {"ece": rep.get("ece"), "overconfidence": rep.get("overconfidence")}
                for cp, rep in (ontogeny.get("calibration") or {}).items()
            },
            "world_model": {
                k: (ontogeny.get("world_model") or {}).get(k)
                for k in ("step_count", "train_steps", "mean_surprise", "last_loss")
            },
        }
        from core.ontogeny.conclusion import get_verbalization_ledger

        verbalization = get_verbalization_ledger().report()
        block["ontogeny"]["verbalization"] = {
            k: verbalization[k]
            for k in ("checked", "with_violations", "overstatements", "faithful_rate")
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["ontogeny_error"] = repr(exc)
    try:
        from core.bus.qos import qos_report
        from core.health.diagnostics_aggregator import diagnostics_report
        from core.observability.bus_recorder import bus_recorder_report
        from core.runtime.lifecycle import lifecycle_report
        from core.runtime.parameters import parameters_report

        diagnostics = diagnostics_report()
        lifecycles = lifecycle_report()
        block["middleware"] = {
            "diagnostics": {
                "level": diagnostics["level"],
                "stale": diagnostics["stale"],
                "errors": diagnostics["errors"],
                "summary": diagnostics["summary"],
            },
            "lifecycles": {
                "by_state": lifecycles["by_state"],
                "critical_inactive": lifecycles["critical_inactive"],
                "errored": lifecycles["errored"],
            },
            "qos": {
                "topics": qos_report()["topic_count"],
                "mismatches": len(qos_report()["qos_mismatches"]),
                "not_alive": qos_report()["not_alive"],
            },
            "parameters": {
                "count": parameters_report()["count"],
                "changed_from_default": parameters_report()["changed_from_default"],
            },
            "bus_ring": {
                k: bus_recorder_report()[k]
                for k in ("ring_size", "ring_span_s", "dumps", "recording")
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["middleware_error"] = repr(exc)

def _integrity_of_orchestration_verifier_and_learning(block: dict[str, Any]) -> None:
    from .health_contract import (
        get_runtime_service,
    )

    try:
        from core.runtime.admission import admission_report
        from core.runtime.eviction import eviction_report
        from core.runtime.lease import lease_report
        from core.runtime.quota import quota_report
        from core.runtime.reconcile import reconcile_report

        admission = admission_report()
        eviction = eviction_report()
        block["orchestration"] = {
            "admission": {
                "hooks": len(admission["mutating"]) + len(admission["validating"]),
                "admitted": admission["admitted"],
                "denied": admission["denied"],
            },
            "quota": quota_report()["by_qos_class"],
            "eviction": {
                "eviction_order": eviction["eviction_order"],
                "breached": eviction["currently_breached"],
                "reclaims": eviction["reclaims"],
                "evictions": eviction["evictions"],
            },
            "controllers": reconcile_report(),
            "leases": lease_report(),
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["orchestration_error"] = repr(exc)
    try:
        from core.runtime.sanitizers import sanitizer_report

        block["sanitizers"] = sanitizer_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["sanitizers_error"] = repr(exc)
    try:
        from core.verify.invariants import last_report

        block["verifier"] = last_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["verifier_error"] = repr(exc)
    try:
        from core.pipeline.pass_manager import pass_manager_report

        passes = pass_manager_report()
        block["passes"] = {
            "bisect_limit": passes["bisect_limit"],
            "skips": passes["skips"],
            "hottest": passes["hottest"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["passes_error"] = repr(exc)
    try:
        from core.runtime.memory_consent import memory_consent_report

        # A person asking to be forgotten and not being answered is a
        # privacy control quietly doing nothing. Counted here so it is
        # visible without anyone having to read a log for it.
        block["memory_consent"] = memory_consent_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["memory_consent_error"] = repr(exc)
    try:
        from core.runtime.host_sleep import host_sleep_report

        # A gap in a timeline is not evidence of a stall when the host was
        # shut. Reported so a reader of this surface can tell the two apart.
        block["host_sleep"] = host_sleep_report()
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["host_sleep_error"] = repr(exc)
    try:
        from core.runtime.oom_policy import oom_report

        oom = oom_report()
        block["oom"] = {
            "next_victim": oom["next_victim"],
            "sheddable_organs": oom["sheddable_organs"],
            "immune_organs": oom["immune_organs"],
            "recent_sheds": oom["recent_sheds"][-3:],
            "restart_requested": oom["restart_requested"],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["oom_error"] = repr(exc)
    # Grounding: does what she persisted match what actually ran?
    #
    # The work ledger and the canary counters are detectors, and a detector
    # nobody reads is worse than no detector — it produces the confidence of
    # having checked with none of the checking. This block is their reader.
    try:
        from core.security.injection_canary import canary_status
        from core.verify.work_ledger import status as work_ledger_status

        canaries = canary_status()
        block["grounding"] = {
            "work_ledger": work_ledger_status(),
            "injection_canaries": {
                "evaluated": canaries["evaluated"],
                # A count says the detector ran, not that it ran on
                # anything real. Until a lane plants a canary in a prompt
                # carrying somebody's untrusted content, every number
                # beside this came from a validator's synthetic material.
                "live_evaluated": canaries["live_evaluated"],
                "watching_live_traffic": canaries["watching_live_traffic"],
                "incidents": canaries["hijacked"] + canaries["leaked"],
                "incident_rate": canaries["incident_rate"],
                # A probe lane that keeps failing has silently stopped
                # detecting; that is itself the finding.
                "blind": canaries["blind"],
                "inconclusive": canaries["inconclusive"],
            },
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["grounding_error"] = repr(exc)
    # What keeps failing, as opposed to what failed. The degradation log
    # answers the second; only the scar record answers the first.
    try:
        from core.runtime.degradation_habituation import get_habituation

        habituation = get_habituation().status()
        block["chronic_faults"] = {
            "signatures_tracked": habituation["signatures_tracked"],
            "saturated": habituation["saturated"],
            "residual_floor": habituation["residual_floor"],
            # Truncated: this is a caveat line, not a fault database.
            "chronic": habituation["chronic"][:5],
        }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["chronic_faults_error"] = repr(exc)
    # Memories that have done more harm than good, and how much the ambient
    # mind declined to say. Both are numbers nothing else in the runtime
    # produces.
    try:
        # Registry lookup, never an import; see core/memory/retrieval_outcomes.py.
        ledger = get_runtime_service("retrieval_outcome_ledger", default=None)
        if ledger is None:
            # An absence, not a fault — the same answer the ambient governor
            # below already gives for the same shape of question, in this
            # same block. This raised instead, so a process that simply had
            # no ledger (every test process, a tool, a partial boot) put
            # judgement_error on the integrity surface and took the whole
            # judgement block with it, including the governor's reading,
            # which was there and fine. Not measured here is not the same as
            # measured and bad, and only one of them is worth waking anyone.
            block["judgement"] = {"retrieval": {"registered": False}}
        else:
            outcomes = ledger.status()
            block["judgement"] = {
                "retrieval": {
                    "registered": True,
                    "tracked": outcomes["tracked"],
                    "graded": outcomes["graded"],
                    "harmful_memories": outcomes["harmful_memories"][:5],
                },
            }
        # Resolved through the low-level runtime registry, not imported and
        # not fetched from ServiceContainer. Two separate rules point here:
        # core/runtime may not depend on core.agency (the layering gate
        # rejects the direct import), and this module may not reach into
        # the container at all (test_health_contract_uses_low_level_runtime_registry
        # — the foundation's health surface has to work when the container
        # is the thing that failed).
        #
        # The governor registers itself; the runtime reads whatever is
        # there and reports honestly when nothing is.
        governor = get_runtime_service("ambient_governor", default=None)
        if governor is None:
            block["judgement"]["ambient"] = {"registered": False}
        else:
            ambient = governor.status()
            block["judgement"]["ambient"] = {
                "registered": True,
                "configured": ambient["configured"],
                "spent_today": ambient["spent_today"],
                "remaining_today": ambient["remaining_today"],
                "withheld": ambient["withheld"],
                "restraint_rate": ambient["restraint_rate"],
                "calibration": ambient["calibration"],
            }
    except Exception as exc:  # noqa: BLE001 — each health add-on is isolated
        block["judgement_error"] = repr(exc)
