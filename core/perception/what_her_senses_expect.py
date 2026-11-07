"""What her eyes and ears expect of what plays, and how far each look and each second of sound departs from it.

The eye goes to what is surprising, where surprise is how far new data moves the watcher's beliefs: Itti and Baldi
(2005) read it at every place of a video from low-level features, as the divergence (KL) of each feature's belief
after a frame from its belief before, at several time scales at once, and 72% of people's gaze shifts on television
and games went to places more surprising than average. Sudden, loud sounds are the brainstem reflex in Juslin's
account of how sound moves people, and the startle is the fastest response a body has.

So each sense here keeps a belief about what it will meet next, per feature, as a Gaussian whose spread grows a
little before each look (the world may have changed) and shrinks with each observation:

    before:  N(mu, s^2 + q)                  q: how fast this time scale forgets
    after:   s'^2 = 1 / (1/(s^2 + q) + 1/r)  r: how noisy the sense is
             mu'  = s'^2 (mu/(s^2 + q) + x/r)
    surprise = KL(after || before), in nats, averaged over features and over the scales

- **Sight:** a small grey copy of the picture where it plays (32 by 18, the glance her eyes already take to see a
  scene change) and a coarser colour copy (8 by 4), at a fast and a slow scale. A cut, a flash, an explosion, a new
  place read high; the same shot held still reads low and lower.
- **Sound:** its loudness each second, in decibels against full scale, from her ear (core/senses/her_ear.py). Its
  surprise as above, and an onset where it rises well above what her ear expected: the startle's stimulus.

What each departure is worth to her, and how it habituates, is core/together/what_moves_her.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Belief", "SightExpectation", "SoundExpectation", "glance_of", "loudness_of"]

#: The scales a sense keeps beliefs at: how much spread each adds before a look, as a share of its noise. The fast one
#: follows what is on now and is surprised by a cut; the slow one remembers the video and is surprised by what is new
#: to all of it.
FAST, SLOW = 1.0, 0.05


@dataclass
class Belief:
    """A Gaussian belief per feature about the next observation, kept at one time scale."""

    noise: float
    forget: float
    mean: Any = None
    spread: Any = None

    def surprise(self, x: Any) -> float:
        """Take in ``x``; the mean divergence of the belief after it from the belief before, in nats."""
        import numpy as np

        x = np.asarray(x, dtype=np.float64).ravel()
        if self.mean is None or self.mean.shape != x.shape:
            self.mean, self.spread = x.copy(), np.full_like(x, self.noise)
            return 0.0
        before_var = self.spread + self.forget * self.noise
        after_var = 1.0 / (1.0 / before_var + 1.0 / self.noise)
        after_mean = after_var * (self.mean / before_var + x / self.noise)
        kl = 0.5 * (np.log(before_var / after_var) + (after_var + (after_mean - self.mean) ** 2) / before_var - 1.0)
        self.mean, self.spread = after_mean, after_var
        return float(np.mean(kl))


def glance_of(picture: bytes) -> Any | None:
    """A picture's features for her eyes' expectation: a 32 by 18 grey copy and an 8 by 4 colour copy, in [0, 1]."""
    import io

    import numpy as np
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(picture))
        grey = np.asarray(image.convert("L").resize((32, 18)), dtype=np.float32).ravel() / 255.0
        colour = np.asarray(image.convert("RGB").resize((8, 4)), dtype=np.float32).ravel() / 255.0
    except (OSError, ValueError):
        return None
    return np.concatenate([grey, colour])


@dataclass
class SightExpectation:
    """What her eyes expect of the next look at what plays."""

    #: How noisy a glance is, as a variance of a feature in [0, 1]: compression and small motion between two looks of
    #: the same shot.
    noise: float = 0.002
    scales: list[Belief] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.scales:
            self.scales = [Belief(noise=self.noise, forget=forget) for forget in (FAST, SLOW)]

    def took(self, glance: Any) -> float | None:
        """Take in one look's glance; how far it departed from what her eyes expected, in nats, or None without one."""
        if glance is None:
            return None
        return sum(belief.surprise(glance) for belief in self.scales) / len(self.scales)


def loudness_of(samples: Any) -> tuple[float, float] | None:
    """A stretch of sound's loudness: its RMS and its peak, in decibels against full scale; None when it is empty."""
    import numpy as np

    samples = np.asarray(samples, dtype=np.float64).ravel()
    if not samples.size:
        return None
    floor = 1e-9
    rms = math.sqrt(float(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return 20.0 * math.log10(max(rms, floor)), 20.0 * math.log10(max(peak, floor))


#: How far above what her ear expected a second's loudness has to rise to be an onset, in decibels: a sudden sound,
#: not speech going on. Speech moves a few decibels from word to word.
ONSET_DB = 10.0
#: Quieter than this is silence, in dBFS: no surprise is read in it, and a rise out of it is read from here.
SILENCE_DB = -70.0
#: How many of the last seconds what this video usually sounds like is read over: their median, which a pause of a
#: few seconds barely moves.
USUAL_OVER_S = 60


@dataclass
class SoundExpectation:
    """What her ear expects of the next second of what plays."""

    #: How noisy a second's loudness is, as a variance in dB^2.
    noise: float = 9.0
    scales: list[Belief] = field(default_factory=list)
    lately: list[float] = field(default_factory=list)
    #: How far the last second rose above what this video usually sounds like, in dB (0 when it did not): what of its
    #: sound stands out, as a swell, a crash or a bang does and speech coming back after a pause does not.
    above_usual: float = 0.0

    def __post_init__(self) -> None:
        if not self.scales:
            self.scales = [Belief(noise=self.noise, forget=forget) for forget in (FAST, SLOW)]

    def usual(self) -> float | None:
        """What this video usually sounds like: the median loudness of its last ``USUAL_OVER_S`` seconds."""
        if not self.lately:
            return None
        ordered = sorted(self.lately)
        return ordered[len(ordered) // 2]

    def heard(self, loudness_db: float) -> tuple[float, float]:
        """Take in one second's loudness; how far it departed from what her ear expected (nats) and how far it rose
        above that expectation as an onset (dB above ``ONSET_DB``, 0 when it did not)."""
        level = max(SILENCE_DB, float(loudness_db))
        # Against the louder of what the last seconds were and what this video usually sounds like: speech coming back
        # after a pause rises far above the pause, and is no startle; a bang rises above the video's usual too.
        fast = self.scales[0].mean
        known = [value for value in (float(fast[0]) if fast is not None else None, self.usual()) if value is not None]
        expected = max(known) if known else level
        usual = self.usual()
        self.above_usual = max(0.0, level - usual) if usual is not None else 0.0
        surprise = sum(belief.surprise([level]) for belief in self.scales) / len(self.scales)
        self.lately = [*self.lately, level][-USUAL_OVER_S:]
        return surprise, max(0.0, level - expected - ONSET_DB)
