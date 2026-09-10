"""Register the reasoning background loops on the autonomy conductor.

Two bounded, internal, governed loops complete the flywheel:

* ``reasoning_idle_precompute`` — drains the precompute queue during idle, solving
  verifier-dirty hard problems off the foreground critical path; wins land in the
  solved-cache for instant re-use.
* ``reasoning_self_improve`` — when enough verifier-clean traces accumulate, feeds
  them to the existing governed/validated fine-tune pipe (STaR bootstrap).

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
