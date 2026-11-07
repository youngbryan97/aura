"""What her eyes and ears expect of what plays, and how far each look and each second departs from it: a Gaussian
belief per feature at a fast and a slow time scale, surprise as the divergence of the belief after from the belief
before (Itti and Baldi 2005), and an onset where loudness rises well above what her ear expected (the startle)."""
from __future__ import annotations

import io

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_a_held_shot_settles_and_a_cut_is_a_surprise():
    from core.perception.what_her_senses_expect import SightExpectation

    rng = np.random.default_rng(0)
    eyes = SightExpectation()
    shot = rng.random(672)
    held = [eyes.took(shot + rng.normal(0, 0.01, 672)) for _ in range(8)]
    assert held[0] == 0.0  # the first look sets what is expected
    assert held[-1] < held[1]  # the same shot held: less and less surprising
    assert eyes.took(rng.random(672)) > 10 * max(held[2:])  # a cut
    assert eyes.took(None) is None


def test_speech_has_no_onset_and_a_bang_does():
    from core.perception.what_her_senses_expect import SoundExpectation

    rng = np.random.default_rng(1)
    ears = SoundExpectation()
    speech = [ears.heard(-30 + rng.normal(0, 2)) for _ in range(10)]
    surprise, onset = ears.heard(-8.0)
    assert all(rise == 0.0 for _s, rise in speech)
    assert onset > 5.0 and surprise > 3 * max(s for s, _r in speech[1:])
    assert ears.heard(-200.0)[1] == 0.0  # silence is read from its floor, not as a fall to minus infinity


def test_loudness_and_a_glance_of_a_picture():
    from PIL import Image

    from core.perception.what_her_senses_expect import glance_of, loudness_of

    rms, peak = loudness_of(np.full(16000, 0.1))
    assert rms == pytest.approx(-20.0, abs=0.01) and peak == pytest.approx(-20.0, abs=0.01)
    assert loudness_of([]) is None
    out = io.BytesIO()
    Image.new("RGB", (320, 180), (255, 0, 0)).save(out, format="PNG")
    glance = glance_of(out.getvalue())
    assert glance.shape == (32 * 18 + 8 * 4 * 3,) and 0.0 <= glance.min() <= glance.max() <= 1.0
    assert glance_of(b"not a picture") is None


def test_speech_coming_back_after_a_pause_is_no_startle():
    from core.perception.what_her_senses_expect import SoundExpectation

    ears = SoundExpectation()
    for _ in range(60):
        ears.heard(-28.0)  # a minute of narration
    for _ in range(3):
        ears.heard(-65.0)  # a pause
    assert ears.heard(-27.0)[1] == 0.0 and ears.above_usual < 2.0  # the narration again
    assert ears.heard(-5.0)[1] > 5.0 and ears.above_usual > 20.0  # a bang over it


def test_her_eyes_start_again_when_what_they_look_at_changes_and_a_cut_in_it_still_reads():
    from PIL import Image

    from core.skills.watching_the_screen import OnTheirScreen

    def picture(colour):
        out = io.BytesIO()
        Image.new("RGB", (64, 36), colour).save(out, format="PNG")
        return out.getvalue()

    eyes = object.__new__(OnTheirScreen)
    from core.perception.what_her_senses_expect import SightExpectation

    eyes.expects, eyes.last_sight, eyes._looked_within = SightExpectation(), None, None
    assert eyes._sight_of(picture((10, 10, 10))) == 0.0
    eyes._sight_of(picture((10, 10, 10)))
    assert eyes._sight_of(picture((250, 250, 250)), (0.1, 0.1, 0.9, 0.9)) == 0.0  # where it plays found: a new view
    eyes._sight_of(picture((250, 250, 250)), (0.1, 0.1, 0.9, 0.9))
    assert eyes._sight_of(picture((0, 0, 0)), (0.1, 0.1, 0.9, 0.9)) > 1.0  # a cut within it
