"""Wire the cognitive loop into Aura's live Will (CP244).

The conductor (CP243) runs the full loop for any query. This connects it to
agency through one dedicated pathway: a supervised hook runs the loop over a
query drawn from Aura's current context and offers its trust-labelled result
to Global Workspace. It is her Will invoking the loop in her actual life,
instead of an operator hand-driving a lab pipeline.

The wiring obeys the discipline that has kept the live instance safe all
along -- the same one the halting head follows:

* **On by default, explicitly disableable.** ``AURA_COGNITIVE_LOOP_PATHWAY=0``
  is the kill switch. The live path remains bounded, background-classified,
  and incapable of turning unverified output into action or retained belief.
* **Degrades honestly.** If the memory or LLM organs are not resolvable, the
  loop is not built and the hook proposes nothing; it never fabricates an
  action to look busy. A pathway hook that raises is already caught by
  agency as a degradation, so a failure here cannot break the pulse.
* **Never retains the unverified.** In her real life most queries have no
  oracle. When no verifier organ is available the loop marks its answer
  UNVERIFIED and the learner never fires -- the conductor already enforces
  this, and it is why turning the loop on cannot quietly train Aura on her
  own guesses.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from core.service_names import ServiceNames

COGNITIVE_LOOP_PATHWAY_SCHEMA = "aura.cognitive_loop_pathway.v1"
ENABLE_FLAG = "AURA_COGNITIVE_LOOP_PATHWAY"
logger = logging.getLogger("Aura.Agency.CognitiveLoop")


def _bounded_seconds(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using %.1fs", name, raw, default)
        return default
    if not minimum <= value <= maximum:
        logger.warning("Out-of-range %s=%r; using %.1fs", name, raw, default)
        return default
    return value


# One dedicated pathway avoids duplicate model work through adjacent drives.
TARGET_PATHWAYS = ("cognitive_loop",)
# The loop runs real model inference. A cooldown keeps it from firing on every
# agency pulse and loading a latency-sensitive live instance -- the same
# rate-limit discipline the other autonomous pathways already use.
COOLDOWN_SECONDS = _bounded_seconds(
    "AURA_COGNITIVE_LOOP_COOLDOWN", 180.0, minimum=1.0, maximum=86_400.0
)
# Every external call is bounded: an unbounded generate() could hang the
# agency pulse hook. A whole loop (up to max_attempts generations) must also
# not run away, so the cycle carries its own ceiling.
GENERATE_TIMEOUT_S = _bounded_seconds(
    "AURA_COGNITIVE_LOOP_GENERATE_TIMEOUT", 45.0, minimum=1.0, maximum=300.0
)
CYCLE_TIMEOUT_S = _bounded_seconds(
    "AURA_COGNITIVE_LOOP_CYCLE_TIMEOUT", 150.0, minimum=1.0, maximum=900.0
)
WORKSPACE_TIMEOUT_S = _bounded_seconds(
    "AURA_COGNITIVE_LOOP_WORKSPACE_TIMEOUT", 2.0, minimum=0.05, maximum=30.0
)


def _degrade(exc: BaseException, action: str, *, severity: str = "degraded") -> None:
    """Route a swallowed exception through the canonical degradation channel.

    CLAUDE.md forbids silent catch-alls; every ``except`` here records why it
    fired, at info-level for expected backpressure (timeouts) and degraded
    otherwise, so a live failure is visible rather than hidden behind a
    returned None.
    """
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "cognitive_loop_pathway", exc, severity=severity, action=action,
            enforce_failure_policy=False,
        )
    except Exception as record_exc:  # noqa: BLE001 - the recorder itself failed; logged as the secondary signal
        # The primary recorder may itself be unavailable during boot. Keep a
        # secondary observable signal without raising into the agency pulse.
        logger.warning(
            "Cognitive-loop degradation recorder failed (%s) while handling %s: %s",
            type(record_exc).__name__,
            action,
            type(exc).__name__,
        )


def is_enabled() -> bool:
    # ON by default now (owner's decision). Set AURA_COGNITIVE_LOOP_PATHWAY=0
    # to disable. Safe-by-construction: degrades honestly, cannot break the
    # pulse, and never retains unverified output -- so on-by-default cannot
    # corrupt learning, only spend some inference the cooldown bounds.
    return os.environ.get(ENABLE_FLAG, "1") != "0"


class _RouterDeliberator:
    """Adapts the async llm_router.generate to the loop's deliberator seam."""

    def __init__(self, router: Any) -> None:
        self._router = router

    async def deliberate(self, query: str, material: list[str]) -> str:
        blocks = []
        if material:
            blocks.append("Known context:\n" + "\n".join(f"- {m}" for m in material))
        blocks.append(query)
        blocks.append("Work through it step by step, then give your answer.")
        prompt = "\n\n".join(blocks)
        try:
            # Bounded: an unbounded generate() would hang the pulse hook.
            return await asyncio.wait_for(
                self._router.generate(
                    prompt,
                    origin="cognitive_loop_pathway",
                    purpose="autonomous_internal_deliberation",
                    is_background=True,
                    foreground_request=False,
                ),
                timeout=GENERATE_TIMEOUT_S,
            )
        except TimeoutError as exc:
            _degrade(exc, "cognitive-loop deliberation timed out", severity="warning")
            return ""
        except Exception as exc:
            _degrade(exc, "cognitive-loop deliberation failed")
            return ""


def build_live_loop(container: Any = None) -> Any:
    """Build a CognitiveLoop from live organs, or None if they are missing.

    Resolves memory (retrieval) and the LLM router (deliberation) from the
    ServiceContainer. Returns None -- not a crippled loop -- when the organs
    are unavailable, so the caller proposes nothing rather than fabricating.
    """
    from core.learning.cognitive_loop import CognitiveLoop
    from core.learning.facade_retrieval import FacadeRetrieval
    from core.learning.workspace_producers import (
        RetrievalProducer,
        WorkspaceComposer,
    )

    resolve = _resolver(container)
    router = resolve(ServiceNames.LLM_ROUTER)
    if router is None or not hasattr(router, "generate"):
        return None
    facade = resolve(ServiceNames.MEMORY_FACADE)

    producers = []
    if facade is not None and hasattr(facade, "search_sync"):
        producers.append(RetrievalProducer(FacadeRetrieval(facade)))
    composer = WorkspaceComposer(producers=producers)

    try:
        return CognitiveLoop(
            composer=composer,
            deliberator=_RouterDeliberator(router),
            # No live programmatic verifier for open-ended research: answers
            # are UNVERIFIED and never retained. A real verifier organ can be
            # wired here later, and only then does learning switch on.
            verifier=None,
            max_attempts=1,
        )
    except ValueError as exc:
        _degrade(exc, "cognitive-loop live construction rejected")
        return None


def _resolver(container: Any):
    if container is not None and hasattr(container, "get"):
        return lambda name: container.get(name, default=None)
    from core.container import ServiceContainer

    return lambda name: ServiceContainer.get(name, default=None)


def _derive_query(agency: Any) -> str:
    """Draw a query from Aura's current context -- her monologue or a goal.

    The loop is data-driven: it does not need a hand-written question, it
    picks up whatever Aura is currently attending to. Returns '' when there
    is nothing to reason about, which makes the hook propose nothing.
    """
    monologue = str(getattr(agency, "_current_monologue", "") or "").strip()
    if len(monologue) > 12:
        return monologue[:400]
    goals = getattr(getattr(agency, "state", None), "pending_goals", None) or []
    for goal in goals:
        if not isinstance(goal, dict) or goal.get("status", "pending") != "pending":
            continue
        text = str(
            goal.get("text")
            or goal.get("goal")
            or goal.get("description")
            or goal.get("objective")
            or ""
        ).strip()
        if len(text) > 8:
            return text[:400]
    return ""


async def cognitive_loop_provider(
    *, pathway: str, now: float, idle_seconds: float, agency: Any
) -> dict[str, Any] | None:
    """Schedule one cycle without blocking the agency pulse.

    A completed supervised task is collected on the next pulse. Running work
    returns immediately, preserving AgencyCore's non-blocking contract.
    """
    pending = getattr(agency, "_cognitive_loop_task", None)
    if pending is not None:
        if not isinstance(pending, asyncio.Task):
            _degrade(
                TypeError("cognitive-loop task latch is not an asyncio.Task"),
                "cognitive-loop task latch reset",
            )
            try:
                agency._cognitive_loop_task = None
            # not a failure: a latch that will not clear on this object is one this pathway
            # has nothing more to do with.
            except (AttributeError, TypeError):
                return None
        elif not pending.done():
            return None
        else:
            try:
                action = pending.result()
            # not a failure: a cancelled pending action produced no action, and this reader
            # is not the one that cancelled it.
            except asyncio.CancelledError:
                action = None
            except Exception as exc:
                _degrade(exc, "cognitive-loop supervised task failed")
                action = None
            try:
                agency._cognitive_loop_task = None
            except (AttributeError, TypeError):
                logger.warning("Cognitive-loop completed task latch could not be cleared")
            return action if isinstance(action, dict) else None

    raw_last_run = getattr(agency, "_cognitive_loop_last_run", None)
    if isinstance(raw_last_run, dict):
        # Migrate the pre-CP254 per-pathway clock without losing its bound.
        prior_values = [float(value or 0.0) for value in raw_last_run.values()]
        last_run = max(prior_values, default=0.0)
    elif raw_last_run is None:
        last_run = None
    else:
        last_run = float(raw_last_run)
    if last_run is not None and (
        now <= last_run or now - last_run < COOLDOWN_SECONDS
    ):
        return None
    loop = build_live_loop()
    if loop is None:
        return None
    query = _derive_query(agency)
    if not query:
        return None
    try:
        agency._cognitive_loop_last_run = float(now)
    except (AttributeError, TypeError) as exc:
        _degrade(exc, "cognitive-loop cooldown latch unavailable")
        return None

    cycle = _run_cognitive_loop_cycle(
        loop=loop,
        query=query,
        pathway=pathway,
        agency=agency,
    )
    try:
        from core.utils.task_tracker import get_task_tracker

        task = get_task_tracker().create_task(
            cycle,
            name="agency.cognitive_loop.cycle",
        )
    except Exception as exc:
        cycle.close()
        agency._cognitive_loop_last_run = raw_last_run
        _degrade(exc, "cognitive-loop supervised task creation failed")
        return None
    if not isinstance(task, asyncio.Task):
        cycle.close()
        agency._cognitive_loop_last_run = raw_last_run
        _degrade(
            RuntimeError("task tracker returned no asyncio.Task"),
            "cognitive-loop supervised task creation rejected",
        )
        return None
    try:
        agency._cognitive_loop_task = task
    except (AttributeError, TypeError) as exc:
        task.cancel()
        try:
            agency._cognitive_loop_last_run = raw_last_run
        except (AttributeError, TypeError):
            logger.warning("Cognitive-loop cooldown could not be rolled back")
        _degrade(exc, "cognitive-loop supervised task latch unavailable")
        return None
    return None


async def _run_cognitive_loop_cycle(
    *,
    loop: Any,
    query: str,
    pathway: str,
    agency: Any,
) -> dict[str, Any] | None:
    """Execute the governed background cycle and return an internal action."""

    # Run inside the governed maintenance scope the rest of agency uses for
    # internal model work, and under a whole-cycle timeout so a stuck loop
    # cannot wedge the pulse. A failure degrades to "no proposal", never an
    # exception into the pulse.
    try:
        from core.governance_context import local_internal_governed_scope

        with local_internal_governed_scope(
            "cognitive_loop_pathway", domain="state_mutation"
        ):
            result = await asyncio.wait_for(loop.arun(query), timeout=CYCLE_TIMEOUT_S)
    except TimeoutError as exc:
        _degrade(exc, f"cognitive-loop cycle timed out on {pathway}", severity="warning")
        return None
    except Exception as exc:
        _degrade(exc, f"cognitive-loop cycle failed on {pathway}")
        return None
    if result.answer is None or not str(result.answer).strip():
        return None

    answer = str(result.answer).strip()[:600]
    priority = 0.55 if result.verified else 0.3
    workspace_receipt = await _publish_result_to_workspace(
        answer=answer,
        pathway=pathway,
        verified=bool(result.verified),
        priority=priority,
        loop_receipt=result.to_receipt(),
    )
    receipt = result.to_receipt()
    receipt["workspace"] = workspace_receipt
    try:
        agency._last_cognitive_loop_receipt = receipt
    except (AttributeError, TypeError):
        logger.warning("Cognitive-loop receipt could not be attached to AgencyCore")
    return {
        "type": "internal_reflection",
        "internal_only": True,
        "source": f"{pathway}:cognitive_loop",
        "thought": answer,
        "content": answer,
        "trust": "verified" if result.verified else "unverified_hypothesis",
        "verified": result.verified,
        "attempts": result.attempts,
        "priority": priority,
        "workspace_admitted": bool(workspace_receipt.get("admitted")),
        "receipt": receipt,
    }


async def _publish_result_to_workspace(
    *,
    answer: str,
    pathway: str,
    verified: bool,
    priority: float,
    loop_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Offer the thought to attention without retaining it as a belief."""
    try:
        from core.consciousness.global_workspace import ContentType
        from core.container import ServiceContainer

        workspace = ServiceContainer.get("global_workspace", default=None)
    except (ImportError, AttributeError, RuntimeError) as exc:
        _degrade(exc, "cognitive-loop workspace resolution failed")
        return {"admitted": False, "reason": "workspace_resolution_failed"}
    if workspace is None or not hasattr(workspace, "publish"):
        return {"admitted": False, "reason": "workspace_unavailable"}
    trust = "verified" if verified else "unverified_hypothesis"
    try:
        admitted = await asyncio.wait_for(
            workspace.publish(
                priority=priority,
                source="cognitive_loop_pathway",
                payload={
                    "schema": COGNITIVE_LOOP_PATHWAY_SCHEMA,
                    "pathway": pathway,
                    "trust": trust,
                    "answer": answer,
                    "loop_receipt": loop_receipt,
                    "retained_as_belief": False,
                },
                reason=f"[{trust}] {answer}",
                content_type=ContentType.META,
            ),
            timeout=WORKSPACE_TIMEOUT_S,
        )
    except TimeoutError as exc:
        _degrade(exc, "cognitive-loop workspace publication timed out", severity="warning")
        return {"admitted": False, "reason": "workspace_timeout"}
    except Exception as exc:
        _degrade(exc, "cognitive-loop workspace publication failed")
        return {"admitted": False, "reason": "workspace_error"}
    return {
        "admitted": bool(admitted),
        "reason": "accepted_for_competition" if admitted else "workspace_rejected",
        "trust": trust,
    }


def register_if_enabled(agency: Any) -> dict[str, Any]:
    """Register the loop hook on the target pathways, IF enabled.

    Returns a receipt of what was registered. When the flag is off this
    registers nothing and reports so -- the live instance is unchanged.
    """
    if not is_enabled():
        return {
            "schema": COGNITIVE_LOOP_PATHWAY_SCHEMA,
            "enabled": False,
            "registered": [],
        }
    registered = []
    for pathway in TARGET_PATHWAYS:
        try:
            agency.register_pathway_hook(pathway, cognitive_loop_provider)
            registered.append(pathway)
        except (ValueError, TypeError):
            # Unknown pathway or bad provider -> skip it, do not crash agency.
            continue
    return {
        "schema": COGNITIVE_LOOP_PATHWAY_SCHEMA,
        "enabled": True,
        "registered": registered,
    }


__all__ = [
    "COGNITIVE_LOOP_PATHWAY_SCHEMA",
    "ENABLE_FLAG",
    "TARGET_PATHWAYS",
    "build_live_loop",
    "cognitive_loop_provider",
    "is_enabled",
    "register_if_enabled",
]
