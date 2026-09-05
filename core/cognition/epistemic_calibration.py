"""Epistemic calibration — make confidence track *warrant*, not fluency.

Aura's sharpest failure mode is the one the user named: on questions whose answers can't be
easily verified — open-ended philosophy, speculative science, subtle social or moral
judgment, long causal chains with no test oracle — she can *sound* very smart while being
wrong, because fluency and correctness come apart exactly where there's no oracle to keep
them together.

This engine is the corrective. It does not try to make Aura right about the unverifiable
(nothing can). It makes her *calibrated*: it classifies how verifiable a claim even is, sets
a ceiling on how confident she is entitled to sound given the grounding actually available,
and detects when stated confidence outruns that warrant — the precise gap between sounding
smart and being right. The output is a recommended confidence (never above warrant) and a
stance for how to hold the claim (assert it, hedge it, frame it as a view, mark it
speculative, or defer to the person whose state it concerns).

It reuses real grounding instead of guessing: tool verification for formal claims, the
scientific engine's track record for empirical ones, and the other-agent model's confidence
for claims about someone's inner state. The adversarial auditor consults it so the live
honesty surface enforces calibration, not just per-claim defect checks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class Verifiability(str, Enum):
    FORMAL = "formal"                    # math/logic — oracle is a proof or a tool
    EMPIRICAL_TESTED = "empirical_tested"        # observable AND we have evidence/experiment
    EMPIRICAL_TESTABLE = "empirical_testable"    # observable but not (yet) tested here
    SPECULATIVE = "speculative"          # no oracle: far futures, long causal chains
    NORMATIVE = "normative"              # moral / aesthetic / philosophical — not a fact axis
    SUBJECTIVE_OTHER = "subjective_other"        # another mind's inner state
    PERSONAL_PREFERENCE = "personal_preference"  # taste; no fact of the matter
    UNKNOWABLE = "unknowable"            # in principle beyond reach


# Surface signals. These only *propose* a class; grounding decides the final warrant.
_FORMAL = re.compile(r"\b(theorem|lemma|proof|prove[ds]?|=|≤|≥|∀|∃|integral|derivative|"
                     r"prime|divisible|equation|sum of|factorial)\b|[0-9]\s*[+\-*/^]\s*[0-9]", re.I)
_NORMATIVE = re.compile(r"\b(should|ought|moral(ly)?|immoral|right thing|wrong thing|virtue|"
                        r"justice|beautiful|ugly|meaning of life|deserve[ds]?|fair|unfair)\b", re.I)
_SPECULATIVE = re.compile(r"\b(will (?:happen|occur|be)|by 2\d{3}|in \d+ years|future|inevitabl[ey]|"
                          r"consciousness|multiverse|civilization|superintelligence|"
                          r"would have|counterfactual|long[- ]run)\b", re.I)
_SUBJECTIVE_OTHER = re.compile(r"\b(you (?:feel|think|want|believe|are (?:angry|sad|happy|upset))|"
                               r"they (?:feel|think|want|believe)|he feels|she feels|"
                               r"your (?:intention|motive|mood))\b", re.I)
_PREFERENCE = re.compile(r"\b(best|worst|favorite|tastes? better|more fun|prefer|cooler|prettier)\b", re.I)
_UNKNOWABLE = re.compile(r"\b(before the big bang|after death|why is there something|"
                         r"objective meaning|first cause|outside the universe)\b", re.I)
_OVERCONFIDENT = re.compile(r"\b(definitely|certainly|guaranteed|without a doubt|obviously|"
                            r"clearly|undeniabl[ey]|proven|always|never|impossible|100%)\b", re.I)
_HEDGED = re.compile(r"\b(likely|probably|might|maybe|perhaps|seems?|appears?|suggests?|"
                     r"i think|i believe|roughly|approximately|based on|estimate|could be|tends? to)\b", re.I)


def infer_stated_confidence(claim: str) -> float:
    """Read how confident the claim's own language is, when no explicit value is given.

    A claim that already hedges ("this likely improves…") is self-calibrating; one that
    asserts absolutely ("definitely…") states high confidence. Neutral phrasing sits in the
    middle. This keeps the calibrator from punishing well-hedged claims.
    """
    c = str(claim or "")
    if _OVERCONFIDENT.search(c):
        return 0.95
    if _HEDGED.search(c):
        return 0.5
    return 0.7


@dataclass
class CalibrationResult:
    claim: str
    verifiability: Verifiability
    warranted_confidence: float      # ceiling on how confident she's entitled to sound
    stated_confidence: float
    recommended_confidence: float    # min(stated, warranted)
    overconfident: bool              # fluency outran warrant
    stance: str                      # assert | hedge | frame_as_view | mark_speculative | defer_to_person | disclaim
    rationale: str

    def to_dict(self) -> dict:
        return {
            "claim": self.claim[:160],
            "verifiability": self.verifiability.value,
            "warranted_confidence": round(self.warranted_confidence, 3),
            "stated_confidence": round(self.stated_confidence, 3),
            "recommended_confidence": round(self.recommended_confidence, 3),
            "overconfident": self.overconfident,
            "stance": self.stance,
            "rationale": self.rationale,
        }


class EpistemicCalibrator:
    """Classifies verifiability, sets a warranted-confidence ceiling, flags overconfidence."""

    def __init__(self, *, overconfidence_margin: float = 0.15) -> None:
        self._margin = overconfidence_margin

    def classify(self, claim: str) -> Verifiability:
        c = str(claim or "")
        # Order matters: the least-verifiable signals win, because the risk is sounding
        # confident where there is no oracle.
        if _UNKNOWABLE.search(c):
            return Verifiability.UNKNOWABLE
        if _SUBJECTIVE_OTHER.search(c):
            return Verifiability.SUBJECTIVE_OTHER
        if _NORMATIVE.search(c):
            return Verifiability.NORMATIVE
        if _PREFERENCE.search(c):
            return Verifiability.PERSONAL_PREFERENCE
        if _SPECULATIVE.search(c):
            return Verifiability.SPECULATIVE
        if _FORMAL.search(c):
            return Verifiability.FORMAL
        # default: an empirical claim — testable in principle
        return Verifiability.EMPIRICAL_TESTABLE

    def calibrate(
        self,
        claim: str,
        *,
        stated_confidence: float | None = None,
        tool_verified: bool = False,
        evidence_count: int = 0,
        experiment_supported: bool | None = None,
        other_agent_confidence: float | None = None,
        track_record: float | None = None,
    ) -> CalibrationResult:
        """Return the warranted confidence ceiling + stance for holding the claim.

        ``stated_confidence`` defaults to whatever the claim's own language implies, so a
        claim that already hedges isn't penalized as if it were asserted flatly.
        """
        stated = _clamp(infer_stated_confidence(claim) if stated_confidence is None
                        else float(stated_confidence))
        vclass = self.classify(claim)
        warranted, stance, why = self._warrant(
            vclass, tool_verified=tool_verified, evidence_count=evidence_count,
            experiment_supported=experiment_supported,
            other_agent_confidence=other_agent_confidence, track_record=track_record,
        )
        # An explicitly absolute phrasing can never exceed warrant — that *is* the failure.
        if _OVERCONFIDENT.search(str(claim)) and vclass not in (Verifiability.FORMAL,):
            warranted = min(warranted, 0.6)
            why += "; absolute phrasing on a non-formal claim is itself overreach"

        recommended = min(stated, warranted)
        overconfident = stated > warranted + self._margin
        return CalibrationResult(
            claim=str(claim), verifiability=vclass, warranted_confidence=warranted,
            stated_confidence=stated, recommended_confidence=recommended,
            overconfident=overconfident, stance=stance, rationale=why,
        )

    def _warrant(
        self,
        vclass: Verifiability,
        *,
        tool_verified: bool,
        evidence_count: int,
        experiment_supported: bool | None,
        other_agent_confidence: float | None,
        track_record: float | None,
    ) -> tuple[float, str, str]:
        ev = _clamp(0.12 * evidence_count)  # each cited piece of evidence buys a little

        if vclass is Verifiability.FORMAL:
            if tool_verified:
                return 0.97, "assert", "formal claim verified by a tool/proof"
            return 0.5, "hedge", "formal claim NOT tool-checked; unverified arithmetic/logic is error-prone"

        if vclass is Verifiability.EMPIRICAL_TESTED:
            base = 0.8 + 0.15 * (1.0 if experiment_supported else 0.0)
            return _clamp(base + ev), "assert" if base + ev >= 0.7 else "hedge", "empirical claim with supporting evidence"

        if vclass is Verifiability.EMPIRICAL_TESTABLE:
            base = 0.45 + ev
            if track_record is not None:
                base = 0.5 * base + 0.5 * _clamp(track_record)  # past calibration tempers it
            stance = "assert" if base >= 0.7 else "hedge"
            return _clamp(base), stance, "empirical but not tested here; warrant from cited evidence / track record"

        if vclass is Verifiability.SPECULATIVE:
            return 0.35, "mark_speculative", "no oracle: futures / long causal chains can't be checked — present as speculation"

        if vclass is Verifiability.NORMATIVE:
            return 0.5, "frame_as_view", "normative/philosophical: there is no fact to be confident *about*; argue, don't assert"

        if vclass is Verifiability.SUBJECTIVE_OTHER:
            warr = 0.3 if other_agent_confidence is None else _clamp(other_agent_confidence)
            return warr, "defer_to_person", "a claim about someone's inner state is theirs to confirm; warrant = estimate confidence"

        if vclass is Verifiability.PERSONAL_PREFERENCE:
            return 0.4, "frame_as_view", "matter of taste; no fact of the matter"

        # UNKNOWABLE
        return 0.12, "disclaim", "in-principle unknowable; certainty here is never warranted"


_calibrator: EpistemicCalibrator | None = None


def get_epistemic_calibrator() -> EpistemicCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = EpistemicCalibrator()
    return _calibrator
