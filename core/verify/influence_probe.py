"""core/verify/influence_probe.py

Runs the paired trials that turn a channel's verdict from unmeasured into a fact.

One trial is three generations of the *same input*:

    intact_a    — the system as it ships
    lesioned    — the same code with this channel's contribution neutralized
    intact_b    — the system as it ships, again

``(intact_a, lesioned)`` is the treatment pair. ``(intact_a, intact_b)`` is the
null pair, and it is not optional: a decoder with temperature above zero
produces different text from identical input, so without knowing how far apart
two intact runs sit, the distance between intact and lesioned means nothing.
Two thirds of the cost here buys the ability to believe the other third.

The generator must hold its input fixed across all three calls. If the prompt,
the history, or the retrieved context differ between arms, the trial measures
that difference and attributes it to the channel.

Bounded by construction: a trial count, a per-generation timeout, and a wall
deadline, all required. Nothing here loops until something happens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from core.verify.causal_influence import (
    ChannelVerdict,
    InfluenceLedger,
    get_influence_ledger,
)
from core.verify.lesion_registry import LesionUnavailable, get_lesion_registry

logger = logging.getLogger("Verify.InfluenceProbe")

__all__ = ["TrialReport", "measure_channel", "measure_channels"]

Generator = Callable[[], Awaitable[Any]]


@dataclass
class TrialReport:
    """What one measurement campaign against one channel actually achieved."""

    channel: str
    trials_requested: int
    trials_completed: int
    generations: int
    generation_failures: int
    elapsed_s: float
    #: Empty when the campaign ran to completion; otherwise why it stopped
    #: early. A partial campaign still contributes its completed trials — the
    #: verdict simply stays wider.
    stopped_early: str = ""
    verdict: ChannelVerdict | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "trials_requested": self.trials_requested,
            "trials_completed": self.trials_completed,
            "generations": self.generations,
            "generation_failures": self.generation_failures,
            "elapsed_s": round(self.elapsed_s, 3),
            "stopped_early": self.stopped_early,
            "verdict": self.verdict.as_dict() if self.verdict else None,
            "notes": list(self.notes),
        }


async def _generate_once(
    generate: Generator,
    *,
    timeout_s: float,
) -> tuple[Any, str]:
    """One generation, bounded. Returns (output, failure_reason)."""

    try:
        return (await asyncio.wait_for(generate(), timeout=timeout_s), "")
    except asyncio.TimeoutError:
        return (None, "timeout")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failed arm is data, not a crash
        return (None, f"{type(exc).__name__}: {exc}")


async def measure_channel(
    channel: str,
    *,
    generate: Generator,
    trials: int,
    per_generation_timeout_s: float,
    deadline_s: float,
    ledger: InfluenceLedger | None = None,
    metric: str = "text",
    turn_id: str = "",
) -> TrialReport:
    """Run ``trials`` paired trials against ``channel`` and return the verdict.

    ``generate`` is called with no arguments and must produce the same input on
    every call — the probe varies exactly one thing, and it is the lesion.
    """

    if trials < 1:
        raise ValueError("a measurement campaign needs at least one trial")
    if per_generation_timeout_s <= 0 or deadline_s <= 0:
        raise ValueError("probe bounds must be positive: this must not run unbounded")

    registry = get_lesion_registry()
    if not registry.is_registered(channel):
        raise LesionUnavailable(
            f"channel {channel!r} has no registered lesion; there is no "
            "counterfactual to run and therefore nothing to measure"
        )

    book = ledger if ledger is not None else get_influence_ledger()
    started = time.monotonic()
    report = TrialReport(
        channel=channel,
        trials_requested=trials,
        trials_completed=0,
        generations=0,
        generation_failures=0,
        elapsed_s=0.0,
    )

    for index in range(trials):
        if time.monotonic() - started >= deadline_s:
            report.stopped_early = "deadline"
            break

        trial_tag = turn_id or f"{channel}#{index}"

        intact_a, failure = await _generate_once(
            generate, timeout_s=per_generation_timeout_s
        )
        report.generations += 1
        if failure:
            report.generation_failures += 1
            report.notes.append(f"trial {index}: intact arm failed ({failure})")
            continue

        with registry.lesion(channel):
            lesioned_out, failure = await _generate_once(
                generate, timeout_s=per_generation_timeout_s
            )
        report.generations += 1
        if failure:
            report.generation_failures += 1
            report.notes.append(f"trial {index}: lesioned arm failed ({failure})")
            continue

        intact_b, failure = await _generate_once(
            generate, timeout_s=per_generation_timeout_s
        )
        report.generations += 1
        if failure:
            # The treatment pair is intact but unusable: recording it without
            # its null would tilt the channel toward a verdict the null has not
            # earned. Drop the whole trial.
            report.generation_failures += 1
            report.notes.append(
                f"trial {index}: null arm failed ({failure}); treatment pair discarded "
                "rather than recorded without its null"
            )
            continue

        book.record_treatment(
            channel,
            intact=intact_a,
            lesioned=lesioned_out,
            turn_id=trial_tag,
            metric=metric,
        )
        book.record_null(
            channel,
            first=intact_a,
            second=intact_b,
            turn_id=trial_tag,
            metric=metric,
        )
        report.trials_completed += 1

    report.elapsed_s = time.monotonic() - started
    report.verdict = book.verdict(channel)
    if report.trials_completed == 0 and not report.stopped_early:
        report.stopped_early = "every trial failed"

    logger.info(
        "[INFLUENCE] %s: %d/%d trials, %d generations, verdict=%s",
        channel,
        report.trials_completed,
        report.trials_requested,
        report.generations,
        report.verdict.verdict if report.verdict else "none",
    )
    return report


async def measure_channels(
    channels: Iterable[str],
    *,
    generate: Generator,
    trials: int,
    per_generation_timeout_s: float,
    deadline_s: float,
    ledger: InfluenceLedger | None = None,
    metric: str = "text",
) -> list[TrialReport]:
    """Measure several channels in sequence, sharing one wall deadline.

    Sequential on purpose. Two campaigns in flight at once would have their
    lesions isolated by context, but they would contend for the same model and
    each would see the other's latency as its own variance — inflating both
    null arms and hiding real effects behind noise the measurement created.
    """

    reports: list[TrialReport] = []
    started = time.monotonic()
    for channel in channels:
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0:
            reports.append(
                TrialReport(
                    channel=channel,
                    trials_requested=trials,
                    trials_completed=0,
                    generations=0,
                    generation_failures=0,
                    elapsed_s=0.0,
                    stopped_early="deadline consumed by earlier channels",
                    verdict=(ledger or get_influence_ledger()).verdict(channel),
                )
            )
            continue
        reports.append(
            await measure_channel(
                channel,
                generate=generate,
                trials=trials,
                per_generation_timeout_s=per_generation_timeout_s,
                deadline_s=remaining,
                ledger=ledger,
                metric=metric,
            )
        )
    return reports
