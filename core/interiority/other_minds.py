"""core/interiority/other_minds.py — reading a state that is not yours.

The problem is inverse inference, and the generative direction is:
another agent's goals and appraisal produce an *action tendency*, and
the tendency produces observable signal. Reading them is inverting that.
Frijda's point is the one that makes this tractable — an emotion is a
state of readiness for a kind of action, so the hidden variable worth
estimating is the readiness, not the word (Frijda 1986). A posterior
over tendencies predicts what someone will do; a posterior over labels
predicts what they would call it.

What this replaces, in this codebase, is two keyword lists. The affect
engine's fallback scores a trigger string against thirty words
(``core/affect/damasio_v2.py``), and the resonance module scores a
message against forty (``core/affect/affective_resonance.py``). Neither
can be wrong in an interesting way, neither carries uncertainty, and
neither can improve. This module is wired in as their source.

Four properties none of the reviewed prototypes have:

**No ground-truth channel.** There is no argument through which a caller
can hand this module the other agent's real state. MetaAI's perceiver
returns whichever score is highest and one of the scores is Aura's own
interoception, so it can answer the question "what is he feeling" with
"my chest is tight"; the verification run reproduced exactly that. Here
the estimate is built only from :class:`~core.interiority.event.
InteriorEvent` channels.

**Person-specific baselines.** The same flat prosody is normal in one
person and alarming in another, so every channel is read as a deviation
from that person's own history. A system with no baseline is reading the
population mean and calling it the person.

**Per-channel reliability, learned.** Face is high bandwidth and highly
controllable, so it is the weakest channel under any motive to conceal;
timing is low bandwidth and almost never managed, so it is among the
strongest. The weights start at those published asymmetries and move
with outcomes, and :meth:`OtherMindsModel.record_outcome` is the only
thing that moves them.

**A species mask that refuses.** Mammals share the autonomic and
action-tendency layers, so fear reads across. They do not share the
normative layer, so guilt does not: the classic "guilty look" in dogs
tracks the owner's anger rather than the dog's transgression (Horowitz
2009). :data:`SPECIES_TENDENCIES` encodes which tendencies are
inferable for which species, and the estimate declines the rest rather
than reporting them weakly.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from core.interiority.evidence import (
    Reading,
    absent,
    inferred,
    measured,
)
from core.interiority.event import CHANNELS, InteriorEvent
from core.interiority.params import ParamKind, declare

#: Frijda's action-readiness modes, reduced to the set that is
#: distinguishable from the channels this runtime actually has.
TENDENCIES: tuple[str, ...] = (
    "approach",     # move toward, engage, seek
    "attend",       # hold still and take in — interest, surprise
    "avoid",        # move away from a threat
    "reject",       # expel, refuse, disgust
    "agonistic",    # confront, press, correct another's behaviour
    "submit",       # yield, defer, appease
    "inhibit",      # stop, freeze, withhold
    "protect",      # shield another, caregiving readiness
    "bond",         # affiliate, close distance, share
    "disengage",    # give up on the goal, conserve, withdraw
)

#: Which tendencies are inferable per species class. Absent from the
#: tuple means the estimate declines, not that it returns a small number.
#:
#: The action-readiness layer is the shared one, which is why a dog's fear
#: reads and is the reason cross-species inference works at all. Mammals
#: therefore keep the whole set. Birds and the rest lose the tendencies
#: that need a model of another mind — protect (caregiving directed at a
#: specific other's welfare) and bond — not because those animals lack
#: them but because this runtime has no channel that would distinguish
#: them from approach.
SPECIES_TENDENCIES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "human": TENDENCIES,
        "mammal": TENDENCIES,
        "bird": (
            "approach", "attend", "avoid", "reject", "agonistic",
            "submit", "inhibit", "disengage",
        ),
        "other": ("approach", "attend", "avoid", "inhibit", "disengage"),
    }
)

#: Whether a *normative* state may be attributed to this species: guilt,
#: shame, pride, and anything else that needs the other agent to hold a
#: rule about its own conduct. False everywhere but humans. The classic
#: "guilty look" in dogs tracks the owner's anger, not the dog's
#: transgression (Horowitz 2009), so a system that reads it as guilt is
#: reading its own scowl. Faculties that attribute a norm-relative state
#: must check this and decline; test_species_masking.py pins that they do.
NORMATIVE_INFERENCE_ALLOWED: Mapping[str, bool] = MappingProxyType(
    {"human": True, "mammal": False, "bird": False, "other": False}
)

#: Which channels carry information for which species. A human face and a
#: cat's face are not the same instrument, and scent is a channel for a
#: dog and noise for a person.
SPECIES_CHANNELS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "human": CHANNELS,
        "mammal": ("posture", "prosody", "behaviour", "context", "history", "instrument"),
        "bird": ("posture", "prosody", "behaviour", "context", "history"),
        "other": ("behaviour", "context", "history"),
    }
)


def _p(name: str, value: float, basis: str, sensitivity: str, **kw) -> float:
    return declare(
        f"interiority.other_minds.{name}",
        value,
        unit="reliability",
        basis=basis,
        sensitivity=sensitivity,
        owner="core/interiority/other_minds.py",
        **kw,
    ).value


#: Starting reliability per channel, before any outcome has been seen.
#: These are priors with reasons, not weights; record_outcome moves them.
_PRIOR_RELIABILITY: Mapping[str, float] = MappingProxyType(
    {
        "text": _p(
            "prior_text", 0.45,
            "Explicit statements about one's own state are informative and "
            "routinely managed. Set at the midpoint: better than nothing, worse "
            "than anything the sender is not monitoring.",
            "Raise it and stated feelings override behaviour; lower it and Aura "
            "distrusts what people tell her about themselves.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.25, 0.65),
        ),
        "lexical": _p(
            "prior_lexical", 0.35,
            "Pronoun rate, negation and absolutist terms carry real signal but "
            "need volume; a single message supports little. Weighted below "
            "explicit text for that reason.",
            "Sets how much a short message's word statistics can move a read.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.15, 0.55),
        ),
        "timing": _p(
            "prior_timing", 0.60,
            "Latency, pause length and turn-taking are produced without "
            "monitoring, so they are among the least manageable channels "
            "available in a text medium.",
            "The main channel Aura has that people do not curate. Lowering it "
            "removes her only unmanaged evidence in text.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.4, 0.8),
        ),
        "prosody": _p(
            "prior_prosody", 0.65,
            "F0 range, jitter and rate track arousal robustly and valence "
            "weakly. Weighted high for the arousal axis it actually carries.",
            "Governs how much voice moves an arousal estimate when audio exists.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.45, 0.85),
        ),
        "face": _p(
            "prior_face", 0.40,
            "High bandwidth and the most voluntarily controllable channel, so "
            "it is the least trustworthy under any motive to conceal. The "
            "reliable part is the musculature that is hard to fake, which this "
            "runtime cannot resolve, so the prior stays low.",
            "Raise it and a managed expression outvotes unmanaged channels.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.2, 0.6),
        ),
        "posture": _p(
            "prior_posture", 0.55,
            "Slow-changing and rarely monitored by the sender; good for "
            "sustained state, poor for the moment.",
            "Sets the weight of body configuration when vision exists.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.35, 0.75),
        ),
        "autonomic": _p(
            "prior_autonomic", 0.80,
            "Flush, pupil and breath are close to uncontrollable. Very high "
            "specificity for arousal and none for valence, so the high weight "
            "applies only where the tendency set differs on arousal.",
            "Rarely available. When it is, it should dominate arousal.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.6, 0.95),
        ),
        "behaviour": _p(
            "prior_behaviour", 0.75,
            "What someone did, against their own baseline, is the highest-value "
            "and most neglected channel. Weighted above every expressive one.",
            "The channel that should carry a read when the two disagree.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.55, 0.9),
        ),
        "context": _p(
            "prior_context", 0.70,
            "What just happened to them frequently outweighs every signal "
            "channel, and is what a system that reads only the message misses.",
            "Lower it and Aura reads messages instead of situations.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.5, 0.9),
        ),
        "history": _p(
            "prior_history", 0.50,
            "This person's own prior states are the baseline every other "
            "channel is read against, and are evidence in their own right.",
            "Sets how much a person's usual state predicts their current one.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.3, 0.7),
        ),
        "interoceptive": _p(
            "prior_interoceptive", 0.15,
            "Aura's own load is evidence about Aura, not about the other "
            "person. It is kept as a channel only so resonance can be measured "
            "and subtracted; it must never carry a read on its own.",
            "Above about 0.3 this channel starts answering questions about "
            "another person with facts about Aura, which is the MetaAI defect.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.05, 0.3),
        ),
        "instrument": _p(
            "prior_instrument", 0.85,
            "A direct measurement from a tool is the strongest evidence "
            "available and is bounded below 1.0 only because instruments fail.",
            "Raise it to 1.0 and a broken sensor becomes unarguable.",
            kind=ParamKind.CALIBRATION, sweep_range=(0.7, 0.95),
        ),
    }
)

_BETA = declare(
    "interiority.other_minds.posterior_inverse_temperature",
    8.0,
    unit="1/logit",
    basis=(
        "Derived, not chosen. Logits are the evidence-weighted mean of "
        "loading times deviation, so they live on the same scale as a single "
        "loading regardless of how many channels are present or how reliable "
        "they are — which is what keeps a thin signal from reading as an "
        "ambiguous one. On that scale a full-deviation loading of 0.5 needs a "
        "multiplier near 8 to carry its readiness clear of a uniform prior "
        "over the reachable set. The property the number is set to preserve is "
        "the published ordering of the channels, and "
        "tests/interiority/test_other_minds.py measures that ordering rather "
        "than taking the value on faith."
    ),
    kind=ParamKind.DERIVED,
    sensitivity=(
        "Too low and every read is uniform; too high and one channel's noise "
        "produces certainty. The test pins the half-mass property, so a change "
        "that breaks it fails rather than quietly flattening every estimate."
    ),
    lower=0.5,
    upper=64.0,
    owner="core/interiority/other_minds.py",
).value

_LOADING_LEARNING_RATE = declare(
    "interiority.other_minds.loading_learning_rate",
    0.03,
    unit="rate",
    basis=(
        "How fast the channel-to-readiness mapping moves on a confirmed "
        "outcome. Slower than the reliability rate, because a reliability is "
        "how much to trust a channel and a loading is what the channel means; "
        "the second should need more evidence than the first."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "Fast and one surprising person rewrites what a pause means; zero and "
        "the mapping is whatever it was written with, forever."
    ),
    sweep_range=(0.005, 0.1),
    owner="core/interiority/other_minds.py",
).value

_LOADING_DRIFT_LIMIT = declare(
    "interiority.other_minds.loading_drift_limit",
    0.4,
    unit="loading",
    basis=(
        "How far a learned loading may move from its published prior. Bounded "
        "rather than free because the priors are measured asymmetries — a face "
        "is controllable, a pause is not — and a mapping that can drift "
        "without limit will eventually learn whatever the last few people "
        "happened to do. Wide enough to reverse a weak prior, not wide enough "
        "to reverse a strong one."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity=(
        "At zero nothing is learned; unbounded, a run of unusual people "
        "overwrites the published ordering and nothing reports that it did."
    ),
    lower=0.0,
    upper=1.0,
    sweep_range=(0.1, 0.7),
    owner="core/interiority/other_minds.py",
).value

_LEARNING_RATE = declare(
    "interiority.other_minds.reliability_learning_rate",
    0.05,
    unit="rate",
    basis=(
        "Rescorla-Wagner update on channel reliability. Slow enough that one "
        "surprising read does not rewrite a channel's standing, which is the "
        "failure mode of a fast rate on a noisy signal."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity="How fast a channel's weight moves after a confirmed outcome.",
    sweep_range=(0.01, 0.2),
    owner="core/interiority/other_minds.py",
).value

_BASELINE_HALF_LIFE = declare(
    "interiority.other_minds.baseline_half_life_samples",
    24.0,
    unit="samples",
    basis=(
        "Exponential baseline over a person's own channel values. Twenty-four "
        "samples is roughly a long conversation, so the baseline tracks the "
        "person rather than the last thing they said."
    ),
    kind=ParamKind.CALIBRATION,
    sensitivity="Short and the baseline chases the signal, hiding the deviation.",
    lower=2.0,
    upper=512.0,
    sweep_range=(8.0, 96.0),
    owner="core/interiority/other_minds.py",
).value


@dataclass(frozen=True)
class OtherEstimate:
    """A posterior over another agent's readiness, with what it rests on."""

    entity: str
    species: str
    #: Posterior over action tendencies. Sums to 1 over the tendencies the
    #: species mask allows; tendencies outside the mask are absent, not zero.
    tendencies: Mapping[str, float]
    #: Tendencies this species does not support inference for.
    declined: tuple[str, ...]
    distress: Reading
    vulnerability: Reading
    #: Can they change the outcome? The variable that separates anguish
    #: from despair, and the one no reviewed prototype estimates.
    coping: Reading
    #: Could they have done otherwise? The variable anger needs.
    capability: Reading
    #: Channels that contributed, with their weights at the time.
    channels_used: Mapping[str, float]
    #: 1 - normalised entropy of the posterior, scaled by evidence quality.
    confidence: float
    at: float = field(default_factory=time.time)

    def top(self) -> tuple[str, float]:
        if not self.tendencies:
            return ("", 0.0)
        name = max(self.tendencies, key=lambda k: self.tendencies[k])
        return (name, self.tendencies[name])

    def margin(self) -> float:
        """Gap between the leading tendency and the runner-up.

        A read whose top two are close is ambiguous however high the top
        is, and a system that reports only the argmax cannot say so.
        """
        if len(self.tendencies) < 2:
            return 0.0
        ranked = sorted(self.tendencies.values(), reverse=True)
        return ranked[0] - ranked[1]

    def entropy(self) -> float:
        total = 0.0
        for p in self.tendencies.values():
            if p > 0.0:
                total -= p * math.log(p)
        return total

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "species": self.species,
            "tendencies": dict(self.tendencies),
            "declined": list(self.declined),
            "top": self.top()[0],
            "margin": self.margin(),
            "confidence": self.confidence,
            "distress": self.distress.to_dict(),
            "vulnerability": self.vulnerability.to_dict(),
            "coping": self.coping.to_dict(),
            "capability": self.capability.to_dict(),
            "channels_used": dict(self.channels_used),
        }


#: How each channel's deviation loads onto each tendency. A positive
#: entry says "a value above this person's baseline on this channel is
#: evidence for this readiness". The matrix is sparse on purpose: a
#: channel that loads on everything carries no information.
_LOADINGS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "timing": MappingProxyType(
            # Long latency and long pauses: withholding or giving up.
            {"inhibit": 0.5, "disengage": 0.4, "submit": 0.2, "attend": -0.2}
        ),
        "lexical": MappingProxyType(
            # Absolutist and negated language, first person singular.
            {"disengage": 0.4, "avoid": 0.3, "agonistic": 0.2, "bond": -0.2}
        ),
        "text": MappingProxyType(
            {"approach": 0.3, "bond": 0.3, "agonistic": 0.2, "reject": 0.2}
        ),
        "prosody": MappingProxyType(
            {"agonistic": 0.4, "avoid": 0.35, "approach": 0.2, "disengage": -0.3}
        ),
        "face": MappingProxyType(
            {"approach": 0.25, "reject": 0.3, "avoid": 0.25, "bond": 0.2}
        ),
        "posture": MappingProxyType(
            {"disengage": 0.45, "submit": 0.35, "agonistic": 0.25, "approach": -0.2}
        ),
        "autonomic": MappingProxyType(
            {"avoid": 0.45, "agonistic": 0.35, "attend": 0.2, "disengage": -0.3}
        ),
        "behaviour": MappingProxyType(
            {"approach": 0.4, "avoid": 0.35, "disengage": 0.35, "protect": 0.3}
        ),
        "context": MappingProxyType(
            {"disengage": 0.35, "avoid": 0.3, "protect": 0.25, "submit": 0.2}
        ),
        "history": MappingProxyType(
            {"approach": 0.2, "attend": 0.2, "bond": 0.2, "disengage": 0.2}
        ),
        "instrument": MappingProxyType(
            {"avoid": 0.3, "attend": 0.3, "disengage": 0.3, "approach": 0.3}
        ),
        # Deliberately loads on nothing. Aura's own load is evidence about
        # Aura; it is carried so resonance can be measured and subtracted.
        "interoceptive": MappingProxyType({}),
    }
)

#: Which tendencies count as distress, for the distress reading.
_DISTRESS_TENDENCIES = ("avoid", "disengage", "submit", "inhibit")


@dataclass
class _Baseline:
    """One person's running mean per channel, and how many samples back it."""

    means: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, channel: str, value: float) -> float:
        """Fold in a sample and return the deviation from the prior mean."""
        prior = self.means.get(channel)
        count = self.counts.get(channel, 0)
        if prior is None:
            self.means[channel] = value
            self.counts[channel] = 1
            # No baseline yet, so no deviation can be claimed.
            return 0.0
        deviation = value - prior
        alpha = 1.0 - math.exp(-math.log(2.0) / _BASELINE_HALF_LIFE)
        self.means[channel] = prior + alpha * deviation
        self.counts[channel] = count + 1
        return deviation

    def samples(self, channel: str) -> int:
        return self.counts.get(channel, 0)


class OtherMindsModel:
    """Estimates another agent's readiness from channel evidence."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reliability: dict[str, float] = dict(_PRIOR_RELIABILITY)
        #: Learned adjustment to what each channel means, on top of the
        #: published prior in _LOADINGS. Starts empty: nothing is learned
        #: until an outcome says so.
        self._learned: dict[tuple[str, str], float] = {}
        self._baselines: dict[str, _Baseline] = {}
        self._reads = 0
        self._outcomes = 0

    # ── the estimate ──────────────────────────────────────────────────
    def estimate(
        self,
        event: InteriorEvent,
        *,
        entity: str | None = None,
        species: str = "human",
        vulnerability: Reading | None = None,
        coping: Reading | None = None,
        capability: Reading | None = None,
    ) -> OtherEstimate:
        who = entity or event.subject or "unknown"
        allowed_tendencies = SPECIES_TENDENCIES.get(species, SPECIES_TENDENCIES["other"])
        allowed_channels = SPECIES_CHANNELS.get(species, SPECIES_CHANNELS["other"])

        with self._lock:
            self._reads += 1
            baseline = self._baselines.setdefault(who, _Baseline())

            # The hypothesis space is the readinesses the *available*
            # channels can actually tell apart. Carrying tendencies no
            # present channel loads on adds uniform mass and flattens the
            # posterior: with only text, a set of ten leaves every margin
            # near 0.01 and the read is a permanent shrug. Marginalising
            # them out is not a shortcut, it is what a posterior over
            # hypotheses the evidence does not address is worth.
            reachable: set[str] = set()
            for channel in allowed_channels:
                if event.channel(channel).present:
                    reachable.update(_LOADINGS.get(channel, {}))
            candidates = tuple(t for t in allowed_tendencies if t in reachable)
            if not candidates:
                candidates = allowed_tendencies
            declined = tuple(t for t in TENDENCIES if t not in candidates)

            scores = {t: 0.0 for t in candidates}
            used: dict[str, float] = {}
            evidence_mass = 0.0

            for channel in allowed_channels:
                reading = event.channel(channel)
                if not reading.present:
                    continue
                deviation = baseline.update(channel, reading.value)
                if baseline.samples(channel) < 2:
                    # The first sample from a person establishes the
                    # baseline and asserts nothing. A read with no
                    # baseline is a read of the population, not the person.
                    continue
                weight = self._reliability.get(channel, 0.0) * reading.confidence
                if weight <= 0.0:
                    continue
                loadings = self._effective_loadings(channel)
                if not loadings:
                    continue
                used[channel] = weight
                evidence_mass += weight * min(1.0, abs(deviation))
                for tendency, loading in loadings.items():
                    if tendency in scores:
                        scores[tendency] += weight * loading * deviation

            # Normalise by the evidence weight before the temperature.
            # The raw sum conflates two questions: which readiness the
            # evidence points at, and how much evidence there is. Without
            # the division, a weak but perfectly consistent signal comes
            # out as a flat posterior, which says the evidence is
            # ambiguous when it is merely thin — and thin is what
            # confidence is for. The posterior answers which; the
            # confidence answers how sure.
            weight_total = sum(used.values()) or 1.0
            posterior = _softmax(
                {k: (v / weight_total) * _BETA for k, v in scores.items()}
            )
            confidence = self._confidence(posterior, evidence_mass, len(used))

            distress = (
                inferred(
                    sum(posterior.get(t, 0.0) for t in _DISTRESS_TENDENCIES),
                    confidence,
                    source="other_minds:tendency-mass",
                )
                if used
                else absent(source="other_minds:no-channels")
            )

        return OtherEstimate(
            entity=who,
            species=species,
            tendencies=MappingProxyType(posterior),
            declined=declined,
            distress=distress,
            vulnerability=vulnerability or absent(source="other_minds:not-supplied"),
            coping=coping or absent(source="other_minds:not-supplied"),
            capability=capability or absent(source="other_minds:not-supplied"),
            channels_used=MappingProxyType(used),
            confidence=confidence,
        )

    def _confidence(
        self, posterior: Mapping[str, float], evidence_mass: float, channels: int
    ) -> float:
        """Sharpness of the posterior, discounted by how thin the evidence is.

        Three terms, because three different things can make a read
        worthless: a flat posterior, a tiny deviation, and a single
        channel. The last matters most — a confident read from one
        channel is the commonest way to be sure and wrong.
        """
        if not posterior or channels == 0:
            return 0.0
        n = len(posterior)
        if n < 2:
            return 0.0
        entropy = -sum(p * math.log(p) for p in posterior.values() if p > 0.0)
        sharpness = max(0.0, 1.0 - entropy / math.log(n))
        strength = min(1.0, evidence_mass)
        breadth = 1.0 - 1.0 / (1.0 + channels)
        # Geometric mean, not a product. All three must be present for a
        # confident read, and any one at zero still zeroes it, but three
        # middling terms give a middling confidence rather than 0.001.
        return max(0.0, min(1.0, (sharpness * strength * breadth) ** (1.0 / 3.0)))

    # ── learning ──────────────────────────────────────────────────────
    def _effective_loadings(self, channel: str) -> dict[str, float]:
        """The published prior for this channel plus whatever has been learned.

        The priors are measured asymmetries and the learned part is
        bounded, so a run of unusual people can adjust what a channel
        means without overwriting the finding it started from.
        """
        prior = _LOADINGS.get(channel, {})
        if not prior and not any(k[0] == channel for k in self._learned):
            return {}
        out = dict(prior)
        for (chan, tendency), delta in self._learned.items():
            if chan == channel:
                out[tendency] = out.get(tendency, 0.0) + delta
        return out

    def loading_drift(self) -> dict[str, float]:
        """How far each channel's meaning has moved from its published prior."""
        drift: dict[str, float] = {}
        with self._lock:
            for (channel, _tendency), delta in self._learned.items():
                drift[channel] = drift.get(channel, 0.0) + abs(delta)
        return drift

    def record_outcome(
        self, estimate: OtherEstimate, *, actual_tendency: str
    ) -> dict[str, float]:
        """Move channel reliabilities toward what the outcome showed.

        The only thing that changes a weight. A channel that pointed at
        the tendency the other agent actually acted on gains; one that
        pointed elsewhere loses. This is the mechanism the reviewed
        prototypes have no version of: their weights are fixed forever,
        so their sensing cannot improve and cannot be shown to be wrong.
        """
        with self._lock:
            self._outcomes += 1
            moved: dict[str, float] = {}
            for channel, weight in estimate.channels_used.items():
                loadings = self._effective_loadings(channel)
                pointed = loadings.get(actual_tendency, 0.0) > 0.0
                current = self._reliability.get(channel, 0.5)
                target = 1.0 if pointed else 0.0
                updated = current + _LEARNING_RATE * (target - current)
                self._reliability[channel] = max(0.02, min(0.98, updated))
                moved[channel] = self._reliability[channel] - current

                # And what the channel *means*, not only how much to trust
                # it. A channel that carried evidence when this readiness
                # turned out to be the real one loads a little more on it;
                # the movement is bounded so the published asymmetries
                # survive a run of unusual people.
                key = (channel, actual_tendency)
                learned = self._learned.get(key, 0.0)
                learned += _LOADING_LEARNING_RATE * weight * (1.0 - learned)
                self._learned[key] = max(
                    -_LOADING_DRIFT_LIMIT, min(_LOADING_DRIFT_LIMIT, learned)
                )
            return moved

    def reliability(self) -> dict[str, float]:
        with self._lock:
            return dict(self._reliability)

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "reads": self._reads,
                "outcomes_recorded": self._outcomes,
                "people_with_baselines": len(self._baselines),
                "reliability": dict(self._reliability),
                "loading_drift": {
                    c: round(d, 4) for c, d in sorted(self.loading_drift().items())
                },
                "learned_loadings": len(self._learned),
            }


def _softmax(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    exps = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return {k: uniform for k in scores}
    return {k: v / total for k, v in exps.items()}


_MODEL: OtherMindsModel | None = None
_MODEL_LOCK = threading.Lock()


def get_other_minds_model() -> OtherMindsModel:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = OtherMindsModel()
        return _MODEL


__all__ = [
    "NORMATIVE_INFERENCE_ALLOWED",
    "SPECIES_CHANNELS",
    "SPECIES_TENDENCIES",
    "TENDENCIES",
    "OtherEstimate",
    "OtherMindsModel",
    "get_other_minds_model",
]
