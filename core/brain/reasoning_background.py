"""Register the reasoning background loops on the autonomy conductor.

Two bounded, internal, governed loops complete the flywheel:

* ``reasoning_idle_precompute`` — drains the precompute queue during idle, solving
  verifier-dirty hard problems off the foreground critical path; wins land in the
  solved-cache for instant re-use.
* ``reasoning_self_improve`` — when enough verifier-clean traces accumulate, feeds
  them to the existing governed/validated fine-tune pipe (STaR bootstrap).
* ``learn_kinds_she_meets`` — one step of learning a kind of problem from the
  solved examples she met (core/learning/examples_she_meets.py), in a task of
  its own so a proposal her model takes minutes over holds no other job.

Wiring (one line, where the conductor's defaults are registered)::

    from core.brain.reasoning_background import register_reasoning_jobs
    register_reasoning_jobs(conductor)

Idempotent: registering twice is a no-op. Kept out of the conductor module itself so
this lands without entangling concurrent edits there.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ReasoningBackground")

# Conservative intervals: idle pre-compute is cheap-ish but uses the cortex, and the
# self-improve feed can trigger governed training, so it runs rarely.
_PRECOMPUTE_INTERVAL_S = 180.0
_SELF_IMPROVE_INTERVAL_S = 3600.0
_NONPARAM_INGEST_INTERVAL_S = 1800.0

_registered: set[int] = set()
#: Tokens a live proposal may take: the longest private channel (18,369) plus the
#: longest reply (3,492) of the G09 proposals that wrote a function
#: (docs/evidence/G09_PROCEDURES_PREREGISTRATION_2_2026-10-12.md).
_PROPOSAL_TOKENS = 18_369 + 3_492
_LEARNING: dict[str, Any] = {}
def _flag_on(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "on", "yes", "enabled"}


async def _job_idle_precompute() -> dict[str, Any]:
    from core.brain.reasoning_precompute import idle_precompute_tick

    solved = await idle_precompute_tick(max_items=1, per_item_timeout=90.0)
    return {"ok": True, "solved": solved}


async def _job_self_improve() -> dict[str, Any]:
    from core.brain.reasoning_self_improvement import get_reasoning_self_improvement

    result = await get_reasoning_self_improvement().maybe_improve()
    if not isinstance(result, Mapping):
        raise TypeError("reasoning self-improvement returned a non-mapping result")
    return dict(result)


async def _job_nonparametric_ingest() -> dict[str, Any]:
    """Ingest trusted knowledge through the already-resident Cortex worker.

    Model weights must never be loaded into the orchestrator process.  That old
    path retained roughly one full 32B Metal allocation after ingestion and
    starved the conversational worker.  The worker command reuses the serving
    model, performs one bounded pair, and declines to cold-load a model merely
    for maintenance.
    """
    if not _flag_on("AURA_NONPARAMETRIC_INGEST", "1"):
        return {"status": "disabled"}
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        if get_memory_pressure_snapshot().refuse_heavy_local_generation:
            return {"status": "skipped_memory_pressure"}
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # CP126 60e56c67. A bare pass here sent the job on to resident-model
        # ingestion. The signal exists to refuse heavy local generation, and
        # it was treated as optional in exactly the state where its
        # availability was uncertain — background maintenance is never worth
        # an unmeasured allocation against a resident cortex.
        record_degradation(
            "reasoning_background",
            exc,
            severity="warning",
            action="skipped nonparametric ingestion because memory pressure was unobservable",
        )
        return {
            "status": "skipped_memory_pressure_unobservable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        from core.brain.llm.mlx_client import get_mlx_client

        client = get_mlx_client()
        if not client.is_alive():
            return {
                "status": "skipped_worker_not_resident",
                "spawned_worker": False,
            }
        return await client.ingest_nonparametric_async(
            max_pairs=1,
            scan_limit=16,
            max_positions=48,
            max_sequence_tokens=96,
            timeout_s=10.0,
        )
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_ingest_job", exc)
        return {"status": "error", "error": f"{type(exc).__name__}"}


def _her_embedder() -> tuple[Any, Any]:
    """Her embedder's document and query sides, only from an engine the runtime already holds."""
    import numpy as np

    from core.learning.kinds_by_meaning import TASK
    from core.memory.embedding_runtime import (
        acquire_shared_embedding_engine,
        shared_embedding_runtime_snapshot,
    )

    if not shared_embedding_runtime_snapshot().get("engine_live"):
        return None, None
    engine = _LEARNING.get("embedder")
    if engine is None:
        engine = _LEARNING["embedder"] = acquire_shared_embedding_engine("learn_kinds_she_meets")

    def documents(texts: list[str]) -> Any:
        return np.asarray(engine.embed_batch(texts), dtype=np.float32)

    def queries(texts: list[str]) -> Any:
        return np.asarray([engine.embed_query(text, task=TASK) for text in texts], dtype=np.float32)

    return documents, queries


async def _check_one_she_answered(router: Any) -> dict[str, Any]:
    """Settle one request she answered in chat by independent programs, when it is one they can settle.

    The amplifier notes a problem whose programs agree and whose answer her
    verifier passes as a solved example she met (independent_programs_agreed);
    learn_one_step then learns kinds from those. Her model is asked at
    background priority, so a person's turn comes first.
    """
    from core.brain.executable_reasoning import should_use_executable_reasoning
    from core.brain.reasoning_amplifier_v2 import amplify_turn, is_amplifiable, search_allowance
    from core.learning.examples_she_meets import mark_programs_checked, next_to_check

    item = next_to_check()
    if item is None:
        return {"status": "nothing_to_check"}
    objective = str(item["problem"])
    task_type = is_amplifiable(objective)
    if task_type is None or not should_use_executable_reasoning(objective, task_type=task_type):
        await mark_programs_checked(item["key"], "not_settled_by_programs")
        return {"status": "not_settled_by_programs"}
    # Priced as an induction proposal is: her model writing a program for one problem.
    _one, search_s = search_allowance(objective, task_type, token_cap=_PROPOSAL_TOKENS, sample_budget=3, system=None)
    if search_s <= 0.0:
        return {"status": "no_rate_measured"}

    async def generate(prompt: str, temperature: float) -> str:
        out = await router.think(
            prompt, temperature=temperature, prefer_tier="primary", origin="check_what_she_answered",
            is_background=True, foreground_request=False, allow_cloud_fallback=False,
        )
        return str((out.get("content") or out.get("response") or "") if isinstance(out, dict) else (out or "")).strip()

    result = await amplify_turn(
        objective, generate, task_type=task_type, time_budget_s=search_s, sample_budget=3,
        extra_context={"enable_executable_reasoning": True, "allow_textual_fallback_after_executable": False},
    )
    authority = str(getattr(result.receipt, "promotion_authority", "") or "unsettled")
    await mark_programs_checked(item["key"], authority)
    logger.info("🧠 Checked what she answered by independent programs: %s (%s).", authority, objective[:80])
    return {"status": "checked", "authority": authority}


async def _learn_one_step() -> dict[str, Any]:
    from core.container import ServiceContainer
    from core.learning.examples_she_meets import learn_one_step

    router = ServiceContainer.get("llm_router", default=None)
    if router is None:
        return {"status": "no_router"}
    try:
        checked = await _check_one_she_answered(router)
    except (ImportError, OSError, RuntimeError, AttributeError, KeyError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background", exc, severity="warning",
                           action="left what she answered for the next idle check")
        checked = {"status": "error", "error": type(exc).__name__}

    async def propose_one(request: str) -> str:
        out = await router.think(
            messages=[{"role": "user", "content": request}],
            origin="learn_kinds_she_meets",
            purpose="procedure_induction",
            prefer_tier="primary",
            is_background=True,
            foreground_request=False,
            allow_cloud_fallback=False,
            skip_runtime_payload=True,
            # Induction runs decode greedily (tools/g12_batched.py); so does this.
            temperature=0.0,
            max_tokens=_PROPOSAL_TOKENS,
        )
        return str((out.get("content") or out.get("response") or "") if isinstance(out, dict) else (out or ""))

    try:
        documents, queries = _her_embedder()
        result = await learn_one_step(propose_one, documents=documents, queries=queries)
    except (ImportError, OSError, RuntimeError, AttributeError, KeyError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_learn_in_use", exc, severity="warning",
                           action="left the examples she met for the next idle step")
        return {"status": "error", "error": type(exc).__name__}
    result = {**result, "programs": checked}
    logger.info("🧠 Learning in use: %s", result)
    return result


async def _job_learn_kinds_she_meets() -> dict[str, Any]:
    """Start one step of learning kinds from the examples she met, unless the last is still running."""
    if not _flag_on("AURA_LEARN_KINDS_IN_USE", "1"):
        return {"status": "disabled"}
    running = _LEARNING.get("task")
    if running is not None and not running.done():
        return {"status": "step_running"}
    from core.runtime.task_ownership import create_tracked_task

    _LEARNING["task"] = create_tracked_task(_learn_one_step(), name="learn_kinds_she_meets",
                                            owner="reasoning_background")
    return {"status": "step_started"}


def register_reasoning_jobs(conductor: Any) -> bool:
    """Register the bounded reasoning background loops on an autonomy conductor.

    Returns True if registration happened. Safe to call multiple times.
    """
    if conductor is None:
        return False
    if id(conductor) in _registered:
        return False
    register = getattr(conductor, "register", None)
    if not callable(register):
        return False
    try:
        register(
            "reasoning_idle_precompute",
            _PRECOMPUTE_INTERVAL_S,
            _job_idle_precompute,
            run_immediately=False,
            policy="maintenance",
        )
        register(
            "reasoning_self_improve",
            _SELF_IMPROVE_INTERVAL_S,
            _job_self_improve,
            run_immediately=False,
            policy="research",
        )
        register(
            "learn_kinds_she_meets",
            # Checked as often as idle pre-compute; the research policy decides when she is idle enough.
            _PRECOMPUTE_INTERVAL_S,
            _job_learn_kinds_she_meets,
            run_immediately=False,
            policy="research",
        )
        register(
            "reasoning_nonparametric_ingest",
            _NONPARAM_INGEST_INTERVAL_S,
            _job_nonparametric_ingest,
            run_immediately=False,
            policy="research",
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("reasoning_background_register", exc)
        return False
    _registered.add(id(conductor))
    logger.info(
        "🧠 Reasoning background loops registered (idle pre-compute + self-improve + non-parametric ingest)."
    )
    return True
