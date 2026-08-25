"""Declared channels for the endogenous language pathway.

Ids are a contract and are never reused. The channel that matters operationally
is ``endogenous.unexpected_refusals``: no head at all is the pathway waiting for
a fit and reads zero, while a head on disk that will not attach is a mismatch
between the artifact and the resident model and shows up here immediately.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EndogenousTelemetry")

CHANNEL_COVERAGE = "endogenous.state_coverage"
CHANNEL_APPLIED_SHARE = "endogenous.bias_applied_share"
CHANNEL_UNEXPECTED = "endogenous.unexpected_refusals"
CHANNEL_CORPUS_TURNS = "endogenous.corpus_turns"
CHANNEL_VERDICT = "endogenous.head_verdict"

#: Rank order for the verdict channel. Position is the value written.
VERDICT_RANKS: tuple[str, ...] = (
    "no_head",
    "no_signal",
    "style_prior",
    "content_bearing",
)

_declared = False


def declare() -> list[str]:
    """Declare the pathway's channels. Idempotent."""
    global _declared
    if _declared:
        return []
    try:
        from core.fsw.telemetry_dictionary import ChannelType, channel
    except ImportError as exc:
        record_degradation(
            "endogenous_telemetry",
            exc,
            severity="debug",
            action="telemetry dictionary unavailable",
        )
        return []

    names: list[str] = []
    for spec in (
        dict(
            identifier=0x1301,
            name=CHANNEL_COVERAGE,
            unit="fraction",
            description="share of z_Aura dimensions a live organ answered for",
            owner="core/brain/llm/endogenous_state.py",
            group="endogenous_language",
            yellow_low=0.10,
            stale_after_s=600.0,
        ),
        dict(
            identifier=0x1302,
            name=CHANNEL_APPLIED_SHARE,
            unit="fraction",
            description="share of generations the vocabulary bias was applied to",
            owner="core/brain/llm/endogenous_decode.py",
            group="endogenous_language",
            stale_after_s=600.0,
        ),
        dict(
            identifier=0x1303,
            name=CHANNEL_UNEXPECTED,
            type=ChannelType.INT,
            unit="count",
            description="refusals that mean a fault rather than an absent artifact",
            owner="core/brain/llm/endogenous_decode.py",
            group="endogenous_language",
            yellow_high=0,
            red_high=5,
            stale_after_s=600.0,
        ),
        dict(
            identifier=0x1304,
            name=CHANNEL_CORPUS_TURNS,
            type=ChannelType.INT,
            unit="count",
            description="recorded turns usable for a fit at the current layout",
            owner="core/brain/llm/endogenous_pair_recorder.py",
            group="endogenous_language",
            stale_after_s=3600.0,
        ),
        dict(
            identifier=0x1305,
            name=CHANNEL_VERDICT,
            type=ChannelType.INT,
            unit="rank",
            description="what the resident head measurably earned the right to claim",
            owner="core/brain/llm/endogenous_readout_training.py",
            group="endogenous_language",
            enum_labels=VERDICT_RANKS,
            stale_after_s=3600.0,
        ),
    ):
        try:
            channel(**spec)
            names.append(str(spec["name"]))
        except (RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "endogenous_telemetry",
                exc,
                severity="warning",
                action=f"channel {spec['name']} not declared",
            )
    _declared = True
    return names


def publish(status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write the current pathway readings. Returns what was written."""
    declare()
    try:
        from core.fsw.telemetry_dictionary import write
    except ImportError:
        return {}

    from core.brain.llm.endogenous_decode import pathway_health

    health = pathway_health()
    written: dict[str, Any] = {}
    verdict = "no_head"
    coverage = 0.0
    if status:
        coverage = float(status.get("state_coverage") or 0.0)
        report = (status.get("head_report") or {}) if isinstance(status, dict) else {}
        candidate = str(report.get("verdict") or "")
        if status.get("head_present") and candidate in VERDICT_RANKS:
            verdict = candidate
        elif status.get("head_present"):
            verdict = "no_signal"

    for name, value in (
        (CHANNEL_COVERAGE, coverage),
        (CHANNEL_APPLIED_SHARE, float(health.get("applied_share") or 0.0)),
        (CHANNEL_UNEXPECTED, int(health.get("unexpected_refusals") or 0)),
        (CHANNEL_VERDICT, VERDICT_RANKS.index(verdict)),
    ):
        try:
            write(name, value)
            written[name] = value
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("telemetry channel %s not written: %s", name, exc)

    try:
        from core.brain.llm.endogenous_pair_recorder import corpus_summary

        turns = int(corpus_summary().get("usable_records") or 0)
        write(CHANNEL_CORPUS_TURNS, turns)
        written[CHANNEL_CORPUS_TURNS] = turns
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("corpus size not written: %s", exc)
    return written


__all__ = [
    "CHANNEL_APPLIED_SHARE",
    "CHANNEL_CORPUS_TURNS",
    "CHANNEL_COVERAGE",
    "CHANNEL_UNEXPECTED",
    "CHANNEL_VERDICT",
    "VERDICT_RANKS",
    "declare",
    "publish",
]
