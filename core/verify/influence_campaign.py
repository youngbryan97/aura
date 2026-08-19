"""core/verify/influence_campaign.py — the wire between measuring and knowing.

:mod:`core.verify.causal_influence` can reach a verdict. :mod:`core.verify.influence_probe`
can run the trials that produce one. :mod:`core.verify.influence_receipt` can
report one. Until this module existed, nothing called the second of those
outside a test, so the first had no observations, the third had nothing to
report, and ``channel_is_influential()`` — the function that answers "did this
faculty change the output?" — was defined, exported, and called by zero lines
of code.

The apparatus was complete and unpowered. Every lesion site on the live
generation path, every null-arm refusal, every bootstrap interval existed to
answer a question that was never asked, because asking it required somebody to
run a campaign and nobody did. On a live boot every channel read UNMEASURED
forever, which is honest and also permanent.

This module asks the question. Three parts, and the third is the one that was
actually missing:

**Persistence.** ``InfluenceLedger`` already carried ``as_dict``/``load`` with a
comment explaining that "a ledger that resets every boot never reaches a
verdict, so the samples have to outlive the process that took them". Nothing
called either. Samples are now read at startup and written after each campaign,
so evidence accumulates across restarts instead of dying with the process.

**A bounded campaign.** Trials are expensive — three generations each, two of
them purely to earn the right to believe the third. So a campaign is small,
deadline-bounded, refuses to start without headroom, and runs one channel at a
time. Nothing here loops until something happens.

**Admission.** A probe generates. Generating competes with the person waiting,
and the whole point of the ContextVar-scoped lesion is that a trial must never
degrade a real turn. So the campaign refuses unless the runtime says there is
room, and it takes the cheapest possible answer — no headroom, no campaign, try
later.

What this module deliberately does NOT do is decide anything. It produces
verdicts. Consumers decide. Keeping the producer ignorant of the consequences
is the same separation that keeps :mod:`core.verify.work_ledger` trustworthy
when the audit's patterns are wrong.

Layering: this module measures faculties it must never import. The generator is
injected by the caller. See DEPS.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock
from core.verify.causal_influence import get_influence_ledger
from core.verify.influence_probe import TrialReport, measure_channel
from core.verify.lesion_registry import LesionUnavailable, get_lesion_registry

logger = logging.getLogger("Verify.InfluenceCampaign")

__all__ = [
    "CampaignReport",
    "campaign_admission_reason",
    "ledger_path",
    "load_persisted_ledger",
    "persist_ledger",
    "run_influence_campaign",
]

#: Ledger schema. Bumping this invalidates old samples rather than
#: silently mixing two meanings of the same channel id.
LEDGER_SCHEMA_VERSION = 1

#: Deliberately small. A campaign is 3 generations per trial per channel; at
#: two channels and three trials that is eighteen generations, which is already
#: a meaningful amount of a local model's day. Accumulation across boots is
#: what gets us to a verdict, not one big run.
DEFAULT_TRIALS = 3
DEFAULT_PER_GENERATION_TIMEOUT_S = 90.0
DEFAULT_DEADLINE_S = 900.0

#: Free unified memory below which a campaign will not start. A probe that
#: triggers shedding has measured the shedding, not the faculty.
MIN_FREE_GB_FOR_CAMPAIGN = 8.0


@dataclass
class CampaignReport:
    """What one campaign achieved, and what it refused to do."""

    started_at: float
    elapsed_s: float
    channels_attempted: tuple[str, ...] = ()
    channels_skipped: dict[str, str] = field(default_factory=dict)
    trials: list[TrialReport] = field(default_factory=list)
    refused: str = ""
    persisted_to: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "elapsed_s": round(self.elapsed_s, 3),
            "channels_attempted": list(self.channels_attempted),
            "channels_skipped": dict(self.channels_skipped),
            "trials": [t.as_dict() for t in self.trials],
            "refused": self.refused,
            "persisted_to": self.persisted_to,
        }

    @property
    def ran(self) -> bool:
        return not self.refused and bool(self.trials)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_PERSIST_LOCK = checked_lock("influence_campaign.persist")


def ledger_path() -> Path:
    """Where accumulated influence samples live."""

    from core.runtime.state_ownership import state_root

    return state_root() / "data" / "verify" / "influence_ledger.json"


def load_persisted_ledger() -> bool:
    """Restore accumulated samples. Returns whether anything was loaded.

    Never raises: a missing or corrupt sample file must not stop a boot. The
    cost of failure here is that verdicts start over, which is exactly the
    state this function exists to improve on — so it degrades to that rather
    than to a crash.
    """

    path = ledger_path()
    try:
        if not path.exists():
            return False
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        record_degradation(
            "influence_campaign",
            exc,
            action="started with an empty influence ledger; verdicts restart from zero",
        )
        return False
    try:
        get_influence_ledger().load(payload)
    except (TypeError, ValueError, AttributeError) as exc:
        record_degradation(
            "influence_campaign",
            exc,
            action="ignored an unreadable influence ledger payload",
        )
        return False
    channels = get_influence_ledger().channels()
    if channels:
        logger.info(
            "[INFLUENCE] restored samples for %d channel(s) from %s",
            len(channels),
            path,
        )
    return bool(channels)


async def persist_ledger() -> str:
    """Write accumulated samples through the governed write gateway."""

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = ledger_path()
    payload = get_influence_ledger().as_dict()
    with _PERSIST_LOCK:
        with local_internal_governed_scope(
            "influence_campaign", domain="state_mutation"
        ):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(
                path.parent, source="influence_campaign"
            )
            await gateway.write_json_async(
                path,
                payload,
                schema_version=LEDGER_SCHEMA_VERSION,
                schema_name="influence_ledger",
                source="influence_campaign",
            )
    return str(path)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


def campaign_admission_reason(*, min_free_gb: float = MIN_FREE_GB_FOR_CAMPAIGN) -> str:
    """Why a campaign must not start now, or "" when it may.

    A reason string rather than a bool because "shutting down" and "not enough
    memory" call for different retry behaviour, and reporting them identically
    is how a transient refusal becomes indistinguishable from a permanent one.
    """

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        if is_shutdown_requested():
            return "shutdown_requested"
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "influence_campaign",
            exc,
            severity="debug",
            action="refused a campaign because the shutdown probe failed",
        )
        return "shutdown_probe_unavailable"

    try:
        from core.runtime.resource_observation import get_resource_observer

        observation = get_resource_observer().memory()
        if not observation.available:
            # `available` is a BOOL — "did the observation succeed" — sitting
            # next to `available_bytes`, while the psutil-compat aliases
            # `total`/`free`/`used` ARE byte counts. Reading `.available` as a
            # number yields 1.0 and refuses every campaign forever. Named here
            # because the first draft of this function did exactly that.
            return "memory_observation_unavailable"
        # GiB, because min_free_gb is stated the way the machine is described
        # — a "64GB" Mac holds 68,719,476,736 bytes. Dividing by 1e9 made the
        # measured headroom read about 7% larger than it is, so the gate
        # admitted campaigns with less room than it was told to require, and
        # the failure that buys is memory shedding during the probe.
        free_gb = float(observation.available_bytes) / 1024**3
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        # No observer means no evidence of headroom. Refuse rather than gamble:
        # a probe that triggers shedding has measured the shedding.
        return "memory_observation_unavailable"
    if free_gb < float(min_free_gb):
        return f"insufficient_memory:{free_gb:.1f}GB<{min_free_gb:.1f}GB"
    return ""


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------


async def run_influence_campaign(
    *,
    generate: Callable[[], Awaitable[Any]],
    channels: Sequence[str] | None = None,
    trials: int = DEFAULT_TRIALS,
    per_generation_timeout_s: float = DEFAULT_PER_GENERATION_TIMEOUT_S,
    deadline_s: float = DEFAULT_DEADLINE_S,
    metric: str = "text",
    persist: bool = True,
    min_free_gb: float = MIN_FREE_GB_FOR_CAMPAIGN,
) -> CampaignReport:
    """Run one bounded measurement campaign and accumulate its evidence.

    ``generate`` must hold its input fixed across every call — the probe varies
    exactly one thing and it is the lesion. Callers that cannot promise that
    should not call this: the trial would measure their variation and attribute
    it to the channel.

    Channels default to every channel with a registered lesion, because a
    channel without one cannot be measured at all and asking is just a slower
    way of finding that out.
    """

    started = time.time()
    report = CampaignReport(started_at=started, elapsed_s=0.0)

    refusal = campaign_admission_reason(min_free_gb=min_free_gb)
    if refusal:
        report.refused = refusal
        report.elapsed_s = time.time() - started
        logger.info("[INFLUENCE] campaign not started (%s)", refusal)
        return report

    registry = get_lesion_registry()
    requested = tuple(channels) if channels else registry.channels()
    if not requested:
        report.refused = "no_registered_lesions"
        report.elapsed_s = time.time() - started
        return report

    runnable: list[str] = []
    for name in requested:
        if registry.is_registered(name):
            runnable.append(name)
        else:
            # Recorded, not silently dropped: a channel nothing can lesion is
            # a claim nothing can check, and that is a finding.
            report.channels_skipped[name] = "no_registered_lesion"

    report.channels_attempted = tuple(runnable)
    monotonic_start = time.monotonic()

    for name in runnable:
        remaining = deadline_s - (time.monotonic() - monotonic_start)
        if remaining <= 0:
            report.channels_skipped[name] = "deadline"
            continue
        # Re-check admission between channels. A campaign that started with
        # headroom can outlive it, and the person who arrives mid-campaign
        # must not wait behind a measurement.
        late_refusal = campaign_admission_reason(min_free_gb=min_free_gb)
        if late_refusal:
            report.channels_skipped[name] = late_refusal
            continue
        try:
            report.trials.append(
                await measure_channel(
                    name,
                    generate=generate,
                    trials=trials,
                    per_generation_timeout_s=per_generation_timeout_s,
                    deadline_s=remaining,
                    metric=metric,
                )
            )
        except LesionUnavailable as exc:
            report.channels_skipped[name] = f"lesion_unavailable:{exc}"
        except (RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "influence_campaign",
                exc,
                action=f"abandoned the {name} measurement and continued the campaign",
            )
            report.channels_skipped[name] = f"{type(exc).__name__}"

    if persist and report.trials:
        try:
            report.persisted_to = await persist_ledger()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "influence_campaign",
                exc,
                action="kept this campaign's samples in memory only; they will not survive restart",
            )

    report.elapsed_s = time.time() - started
    logger.info(
        "[INFLUENCE] campaign finished: %d channel(s) measured, %d skipped, %.1fs",
        len(report.trials),
        len(report.channels_skipped),
        report.elapsed_s,
    )
    return report
