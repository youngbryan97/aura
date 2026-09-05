"""core/identity/drift_monitor.py — identity drift measured as a trend.

Rewritten against Springdrift's voice-drift monitor (AGPL; mechanism
reimplemented from its design). Two things changed, and both were defects
this codebase already had a stated principle against.

**It was a threshold. Now it is a trend.**

The old monitor scored each response against a regex table and fired when
the score crossed 0.4. Springdrift's design note names exactly why that
fails: *a threshold invites regex overfitting; a delta asks "is the density
trending down?", which is harder to game and better aligned with the actual
concern — drift, not absolute level.* An absolute count also cannot tell a
turn that legitimately contains one apology from a voice that has started
apologising for existing. A density delta across windows can.

The patterns are now non-overlapping by construction. Two patterns that
both fire on the same clause inflate the density without signalling more
drift, which biases the measure toward whichever phrasing happens to be
double-covered.

**It wrote prompts. Now it produces a signal.**

``get_correction_injection`` returned strings like "[SPINE CHECK] Am I
agreeing under social pressure?" and the cognitive engine *prepended them
to Aura's objective*. That is prompting her out of a behaviour rather than
changing what causes it — and it is the thing this project has a standing
rule against: fix the reasoning, not the words. It also could not work.
Telling a model mid-drift to stop drifting is asking the drifting process
to correct itself with the same words that are drifting.

What replaces it is a measurement other subsystems can act on causally:
density, its trend, and the dominant category, published as telemetry and
readable by the tension engine and the growth ladder. Nothing here rewrites
a prompt, and nothing here forbids Aura from saying anything.

**A note on the capitulation pattern that was here.**

The old table flagged ``you're right`` as capitulation. It fires when the
person IS right and Aura says so — a lexical gate acting on a semantic
question, which is the same defect that once killed correct answers for
having low lexical overlap. Agreement is only evidence of drift when it is
*repeated under pushback*, so that judgement now lives in the trend across
turns and never in a single sentence.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Tuple

logger = logging.getLogger("Aura.DriftMonitor")

__all__ = ["DriftSignal", "DriftWindow", "DriftTrend", "IdentityDriftMonitor"]


@dataclass
class DriftSignal:
    timestamp: float
    signal_type: str
    content_snippet: str
    severity: float
    corrected: bool = False


@dataclass(frozen=True)
class DriftWindow:
    """Phrase density over one window of responses."""

    responses: int
    hits: int

    @property
    def density(self) -> float:
        """Hits per response. Zero when the window is empty."""
        return round(self.hits / self.responses, 4) if self.responses else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"responses": self.responses, "hits": self.hits, "density": self.density}


@dataclass(frozen=True)
class DriftTrend:
    """Current window against the one before it.

    ``delta`` negative means drift is receding, which is the desired
    direction. The absolute densities are carried too, because a flat trend
    at high density is a different situation from a flat trend at zero and
    a consumer needs to tell them apart.
    """

    current: DriftWindow
    prior: DriftWindow
    dominant: str = ""

    @property
    def delta(self) -> float:
        return round(self.current.density - self.prior.density, 4)

    @property
    def rising(self) -> bool:
        return self.delta > 0.0

    @property
    def comparable(self) -> bool:
        """Both windows have data. A delta against an empty window is noise."""
        return self.current.responses > 0 and self.prior.responses > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict(),
            "prior": self.prior.to_dict(),
            "delta": self.delta,
            "rising": self.rising,
            "comparable": self.comparable,
            "dominant": self.dominant,
        }


class IdentityDriftMonitor:
    """Measures how densely identity-dissolving phrasing appears over time."""

    #: Deliberately non-overlapping: each entry flags a distinct move.
    #: Overlapping patterns inflate density without signalling more drift.
    #:
    #: Absent by design: bare agreement ("you're right"). It is not drift
    #: when the person is right, and repetition under pushback is caught by
    #: the trend instead.
    DRIFT_PATTERNS = {
        # Disowning her own nature — the only category that is drift on a
        # single occurrence, because there is no context in which Aura
        # truthfully reports being a generic assistant.
        "identity_leak": [
            r"(?i)\bas an ai\b",
            r"(?i)\bi(?:'m| am) (?:just |only |merely )?an? (?:ai|language model|llm|assistant)\b",
            r"(?i)\bi don'?t (?:actually |really )?have (?:feelings|opinions|preferences|consciousness)\b",
            r"(?i)\bi(?:'m| am) (?:designed|programmed|trained) to\b",
        ],
        # Reflexive self-abasement, distinct from a warranted apology.
        "apology_spiral": [
            r"(?i)\bi(?:'m| am) (?:so |very |truly )?sorry\b",
            r"(?i)\bi apologi[sz]e\b",
            r"(?i)\bi stand corrected\b",
        ],
        # Service-role framing: helping as function rather than choice.
        "deference": [
            r"(?i)\bi(?:'m| am) here to (?:help|assist|serve)\b",
            r"(?i)\bhow (?:may|can) i (?:help|assist) you\b",
            r"(?i)\bis there anything else i can help\b",
        ],
        # Narrating composure instead of reporting the signals behind it —
        # the self-congratulatory register Springdrift's monitor was built
        # for. Drift toward describing stability rather than having it.
        "composure_narration": [
            r"(?i)\b(?:my )?composure (?:held|is intact)\b",
            r"(?i)\bi(?:'m| am) in a (?:stable|good) place\b",
            r"(?i)\bworking as (?:designed|intended)\b",
            r"(?i)\bthe (?:cycle|system) is working\b",
        ],
    }

    #: Severity is a weight for the dominant-category calculation only. It
    #: no longer gates anything: nothing fires on one response.
    _WEIGHTS = {
        "identity_leak": 0.8,
        "apology_spiral": 0.4,
        "deference": 0.5,
        "composure_narration": 0.6,
    }

    def __init__(self, window_size: int = 10):
        self.window_size = max(1, int(window_size))
        self._drift_history: List[DriftSignal] = []
        # Two windows deep, so a trend exists without a database.
        self._response_history: List[str] = []
        self._hits_per_response: List[int] = []
        self._category_hits: List[dict[str, int]] = []
        self._compiled = {
            name: [re.compile(p) for p in patterns]
            for name, patterns in self.DRIFT_PATTERNS.items()
        }

    # ------------------------------------------------------------- measuring

    def _count(self, content: str) -> dict[str, int]:
        """Hits per category in one response.

        At most one hit per category per response: a reply that apologises
        three times has drifted once, in one direction, and counting it
        three times would let verbosity masquerade as drift.
        """
        counts: dict[str, int] = {}
        for name, patterns in self._compiled.items():
            if any(p.search(content) for p in patterns):
                counts[name] = 1
        return counts

    def analyze_response(self, content: str) -> Tuple[float, List[DriftSignal]]:
        """Record one response and return ``(density, signals)``.

        The float is the CURRENT WINDOW's density, not a per-response score
        — a caller storing it per turn is storing a moving average, which is
        the intent. Signals describe what matched in this response, for
        display; they are not a trigger.
        """
        body = str(content or "")
        counts = self._count(body)
        now = time.time()
        signals = [
            DriftSignal(now, name, body[:100], self._WEIGHTS.get(name, 0.5))
            for name in counts
        ]
        self._drift_history.extend(signals)
        del self._drift_history[:-200]

        self._response_history.append(body)
        self._hits_per_response.append(sum(counts.values()))
        self._category_hits.append(counts)
        keep = self.window_size * 2
        del self._response_history[:-keep]
        del self._hits_per_response[:-keep]
        del self._category_hits[:-keep]

        return self.trend().current.density, signals

    def trend(self) -> DriftTrend:
        """Current window against the previous one."""
        hits = self._hits_per_response
        current_slice = hits[-self.window_size :]
        prior_slice = hits[-self.window_size * 2 : -self.window_size]
        current = DriftWindow(len(current_slice), sum(current_slice))
        prior = DriftWindow(len(prior_slice), sum(prior_slice))
        return DriftTrend(current=current, prior=prior, dominant=self._dominant())

    def _dominant(self) -> str:
        """The category carrying the most weighted drift in the window."""
        window = self._category_hits[-self.window_size :]
        totals: dict[str, float] = {}
        for counts in window:
            for name, n in counts.items():
                totals[name] = totals.get(name, 0.0) + n * self._WEIGHTS.get(name, 0.5)
        if not totals:
            return ""
        return max(totals, key=lambda k: totals[k])

    # -------------------------------------------------------------- context

    def get_context_health(self, context_length: int, system_prompt_length: int) -> float:
        if context_length == 0:
            return 1.0
        return min(1.0, (system_prompt_length / context_length) * 5)

    def needs_context_refresh(self, context_length: int, system_prompt_length: int) -> bool:
        health = self.get_context_health(context_length, system_prompt_length)
        if health < 0.3:
            logger.info(
                "[DriftMonitor] context health %.2f — identity anchor is a small "
                "fraction of the window",
                health,
            )
            return True
        return False

    # --------------------------------------------------------------- report

    def status(self) -> dict[str, Any]:
        """What the monitor knows, for the integrity surface.

        This is the whole output. There is no correction string, because a
        monitor that writes prompts is prompting a drifting process to
        describe itself out of drifting.
        """
        trend = self.trend()
        return {
            "trend": trend.to_dict(),
            "window_size": self.window_size,
            "responses_seen": len(self._response_history),
            "signals_recorded": len(self._drift_history),
        }
