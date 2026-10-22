"""The mask keeps speech timing and leaves the clean recording intact."""
import numpy as np
import pytest

from core.voice.duplex.treatment import treat_voice


def test_clean_treatment_preserves_every_sample_without_mutating_the_input():
    source = np.linspace(-.9, .9, 24000, dtype=np.float32)
    result = treat_voice(source, 24000)
    np.testing.assert_array_equal(result, source)
    assert not np.shares_memory(result, source)


@pytest.mark.parametrize("mode", ["veil", "veil_deep"])
def test_mask_keeps_duration_and_intelligible_band_without_clipping(mode):
    rate = 24000
    time = np.arange(rate) / rate
    source = (.2 * np.sin(2 * np.pi * 180 * time)
              + .15 * np.sin(2 * np.pi * 1500 * time)
              + .08 * np.sin(2 * np.pi * 4500 * time)).astype(np.float32)
    result = treat_voice(source, rate, mode)
    assert result.dtype == np.float32 and result.shape == source.shape
    assert np.max(np.abs(result)) <= .97
    spectrum = np.abs(np.fft.rfft(result))
    assert spectrum[1500] > spectrum[180] * .15
    assert spectrum[4500] > spectrum[1500] * .1
    assert not np.array_equal(result, source)


@pytest.mark.parametrize("source", [np.array([np.nan]), np.ones((2, 2))])
def test_bad_audio_is_rejected(source):
    with pytest.raises(ValueError, match="finite mono"):
        treat_voice(source, 24000, "veil")


def test_heavy_mask_adds_electronic_texture_while_retaining_the_carrier_pitch():
    rate = 24000
    time = np.arange(rate * 2) / rate
    source = (.15 * np.sin(2 * np.pi * 1500 * time)).astype(np.float32)
    clean = treat_voice(source, rate, 'veil')
    heavy = treat_voice(source, rate, 'veil_deep')
    # Discard the filter onset. The retained second is an integer number of
    # carrier and modulation periods, so spectral leakage cannot supply texture.
    normal_spectrum = np.abs(np.fft.rfft(clean[rate:]))
    heavy_spectrum = np.abs(np.fft.rfft(heavy[rate:]))
    assert np.argmax(heavy_spectrum) == 1500
    assert heavy_spectrum[1448] > normal_spectrum[1448] * 20
    assert heavy_spectrum[1552] > normal_spectrum[1552] * 20
    assert heavy_spectrum[1500] > (heavy_spectrum[1448] + heavy_spectrum[1552]) * 3
