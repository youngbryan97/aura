"""The influence campaign has an owner, a schedule, and no verdicts.

Sixty-nine services are declared, six can be lesioned, and none carries an
intervention verdict. The apparatus for producing one is complete: a probe
that runs the trials, a ledger that persists across boots, a receipt that
reports the result, and a conductor job that calls all three every hour.

So the interesting question is not what is missing. It is why an hourly job
has produced nothing, and the answer is in its admission bar. The campaign
runs under the research background profile: fifteen minutes of no user
activity, memory below 85%, conversation ready. A machine with a resident 32B
sits above that memory line most of the time, and a machine somebody is
working on is rarely idle for fifteen minutes. Both conditions are right for a
job that spends three generations per trial. Together they are a job that
never runs, and nothing said so — a deferral logged a reason and left no count,
so "no verdicts yet" and "never once admitted" looked identical from outside.

This counts. Every consideration, every deferral with its reason, every run,
and how long since the last verdict. Two things fall out of that which a log
line cannot give:

* whether the bar is ever met on this host, which is a fact about the bar
  rather than about the campaign;
* which condition refuses most often, which is what you would change first.

Nothing here relaxes anything. A campaign that degrades a live turn is worse
than a campaign that never runs, and the point of measuring the refusals is to
stop guessing about which trade is being made.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.WhyTheCampaignDidNotRun")

__all__ = [
    "AConsideration",
    "note_a_consideration",
    "note_a_verdict",
    "how_the_campaign_has_gone",
    "the_bar_has_never_been_met",
    "where_it_is_kept",
    "forget_everything",
]

#: The most considerations kept. A year of hourly deferrals is 8,760 rows and
#: the useful part of it is the counts, which are folded rather than stored.
MOST_KEPT = 200


@dataclass(frozen=True, slots=True)
class AConsideration:
    """One time the job came up, and what happened."""

    at: float
    #: "ran", "deferred", "idle", "unavailable" — the job's own word.
    went: str
    #: Why it was deferred, in the admission layer's words. Empty for a run.
    because: str = ""
    #: The channel it measured, where it ran one.
    channel: str = ""
    #: Whether that run produced a verdict rather than another sample.
    reached_a_verdict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "went": self.went,
            "because": self.because,
            "channel": self.channel,
            "reached_a_verdict": self.reached_a_verdict,
        }


_SEEN: list[AConsideration] = []
_COUNTS: dict[str, int] = {}
_REASONS: dict[str, int] = {}
_LAST_VERDICT_AT = [0.0]
_LOCK = threading.Lock()


def where_it_is_kept() -> Any:
    """The file the record lives in, under whichever state root this process has."""
    from pathlib import Path

    from core.runtime.state_ownership import state_root

    return Path(state_root()) / "influence_campaign_considerations.json"


def _kept_at() -> str:
    """The path as a string, or why it could not be worked out.

    Reporting must not raise. The state root is resolved from the environment
    and a process without one is a real condition, so a report that dies on
    the way to naming a file is a report that hides everything else it knows.
    """
    try:
        return str(where_it_is_kept())
    except Exception as exc:  # noqa: BLE001
        return f"(unknown: {type(exc).__name__})"


def _load_if_needed() -> None:
    if _COUNTS:
        return
    try:
        raw = json.loads(where_it_is_kept().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a record that cannot be read is not a
        # reason to stop the job it is recording. The path itself can raise:
        # a state root that is not resolvable is a real condition here and the
        # first version let it out of note_a_consideration.
        return
    if not isinstance(raw, dict):
        return
    _COUNTS.update({str(k): int(v) for k, v in (raw.get("counts") or {}).items()})
    _REASONS.update({str(k): int(v) for k, v in (raw.get("reasons") or {}).items()})
    _LAST_VERDICT_AT[0] = float(raw.get("last_verdict_at") or 0.0)


def _snapshot() -> str:
    return json.dumps(
        {
            "schema": "aura.influence_campaign.considerations.v1",
            "written_at": time.time(),
            "counts": dict(_COUNTS),
            "reasons": dict(_REASONS),
            "last_verdict_at": _LAST_VERDICT_AT[0],
            "recent": [one.to_dict() for one in _SEEN[-20:]],
        },
        indent=2,
        sort_keys=True,
    )


def _write(body: str) -> None:
    """Put it on disk, outside the lock, and never raise from doing so."""
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        gateway = get_file_write_gateway()
        path = where_it_is_kept()
        gateway.ensure_directory(path.parent, source="influence_campaign")
        gateway.write_text(path, body, source="influence_campaign")
    except Exception as exc:  # noqa: BLE001 — recording must not stop the job
        logger.debug("campaign considerations not written down: %s", exc)


def note_a_consideration(
    went: str,
    *,
    because: str = "",
    channel: str = "",
    reached_a_verdict: bool = False,
) -> AConsideration:
    """Record one time the job came up. Cheap, and called on the job's path."""
    one = AConsideration(
        at=time.time(),
        went=str(went),
        because=str(because),
        channel=str(channel),
        reached_a_verdict=bool(reached_a_verdict),
    )
    with _LOCK:
        _load_if_needed()
        _SEEN.append(one)
        del _SEEN[:-MOST_KEPT]
        _COUNTS[one.went] = _COUNTS.get(one.went, 0) + 1
        if one.because:
            _REASONS[one.because] = _REASONS.get(one.because, 0) + 1
        if one.reached_a_verdict:
            _LAST_VERDICT_AT[0] = one.at
        body = _snapshot()
    _write(body)
    return one


def note_a_verdict(channel: str) -> AConsideration:
    """A run that reached a verdict rather than adding another sample."""
    return note_a_consideration("ran", channel=channel, reached_a_verdict=True)


def the_bar_has_never_been_met() -> bool:
    """Whether the admission bar has admitted the campaign even once.

    True here is a fact about the bar on this host, not about the campaign.
    """
    with _LOCK:
        _load_if_needed()
        return _COUNTS.get("ran", 0) == 0 and sum(_COUNTS.values()) > 0


def how_the_campaign_has_gone() -> dict[str, Any]:
    """For the health report: how often it came up, ran, and why it did not."""
    with _LOCK:
        _load_if_needed()
        counts = dict(sorted(_COUNTS.items()))
        reasons = dict(sorted(_REASONS.items(), key=lambda kv: (-kv[1], kv[0])))
        last = _LAST_VERDICT_AT[0]
        recent = [one.to_dict() for one in _SEEN[-5:]]
    considered = sum(counts.values())
    ran = counts.get("ran", 0)
    return {
        "schema": "aura.influence_campaign.considerations.v1",
        "considered": considered,
        "ran": ran,
        "share_admitted": (ran / considered) if considered else 0.0,
        "counts": counts,
        "refused_because": reasons,
        "refuses_most_often": next(iter(reasons), ""),
        "seconds_since_a_verdict": (time.time() - last) if last else None,
        "the_bar_has_never_been_met": ran == 0 and considered > 0,
        "kept_at": _kept_at(),
        "recent": recent,
    }


def forget_everything() -> None:
    with _LOCK:
        _SEEN.clear()
        _COUNTS.clear()
        _REASONS.clear()
        _LAST_VERDICT_AT[0] = 0.0
