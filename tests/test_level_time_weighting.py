import pytest
import pyfar as pf
import numpy as np


@pytest.mark.parametrize("weighting", ["F", "S"])
@pytest.mark.parametrize("amplitude", [1, [1, 1]])
def test_time_weighting_pressure_shape(weighting, amplitude):
    fs = 48000
    impulse = pf.signals.impulse(fs + 1, 0, amplitude, fs)
    weighted = pf.level.time_weighted_sound_pressure(impulse, weighting)

    assert isinstance(weighted, pf.Signal)
    assert impulse.time.shape == weighted.time.shape


@pytest.mark.parametrize("fs", [44100, 48000, 192000])
@pytest.mark.parametrize(("weighting", "expected_decay"),
                         [("F", -34.7), ("S", -4.3)])
def test_time_weighting_pressure_decay(weighting, expected_decay, fs):
    """
    According to DIN EN 61672-1 §5.8.2, after an impulse the level
    should fall of at a rate of 34.7 dB/s for FAST and 4.3 dB/s for the
    SLOW time weighting.
    """
    impulse = pf.signals.impulse(2 * fs, sampling_rate=fs)

    weighted = pf.level.time_weighted_sound_pressure(impulse, weighting)
    levels = 10 * np.log10(weighted.time ** 2)
    diff1 = levels[0][fs] - levels[0][0]
    diff2 = levels[0][fs + 100] - levels[0][100]
    assert abs(diff1 - expected_decay) < 0.1
    assert abs(diff2 - expected_decay) < 0.1


@pytest.mark.parametrize("weighting", ["F", "S"])
def test_time_weighting_pressure_same_level(weighting):
    """The energy of a long 1kHz signal must be identical
    across F, S and eq levels (DIN EN 61672-1 §5.8.3).
    """
    fs = 48000
    duration = 60
    x = pf.signals.sine(1000, fs * duration, sampling_rate=48000)

    L_eq = 10 * np.log10(np.sum(x.time[0]**2) / x.n_samples)
    weighted = pf.level.time_weighted_sound_pressure(x, weighting)
    # compare last sample to the equivalent average,
    # so the exp-filter has enough time to integrate energy
    level = 10 * np.log10(weighted.time[0][-1] ** 2)
    assert abs(L_eq - level) < 0.1


@pytest.mark.parametrize("time_weighting", ["F", "S"])
@pytest.mark.parametrize("freq_weighting", ["A", "C", "Z"])
@pytest.mark.parametrize("num_octave_band_fractions", [1, 2, 3, 6, None])
def test_time_weighted_level(time_weighting, freq_weighting,
                             num_octave_band_fractions):
    """Test that the time weighting convenience function produces
    the same result as applying every step manually.
    """
    s = pf.signals.sine([1000, 2000], 4800, sampling_rate=48000)

    expected = s
    if freq_weighting != "Z":
        expected = pf.dsp.filter.frequency_weighting_filter(s, freq_weighting)
    if num_octave_band_fractions is not None:
        expected = pf.dsp.filter.fractional_octave_bands(
            expected, num_octave_band_fractions)
    expected = pf.level.time_weighted_sound_pressure(expected, time_weighting)

    with np.errstate(divide='ignore'):
        expected = 10 * np.log10(expected.time ** 2)
        result = pf.level.time_weighted_level(s, freq_weighting,
                                              time_weighting,
                                              num_octave_band_fractions,
                                              reference_pressure=1,
                                              replace_zeros=False)

    assert expected.shape == result.shape
    assert np.allclose(expected, result)


def test_time_weighted_level_errors():
    # invalid time weighting
    match = "Unknown 'time_weighting' value: X"
    with pytest.raises(ValueError, match=match):
        pf.level.time_weighted_level(pf.signals.sine(1000, 48000), "A", "X")

    # invalid frequency weighting
    match = "Frequency weighting must be 'A', 'C', or 'Z'"
    with pytest.raises(ValueError, match=match):
        pf.level.time_weighted_level(pf.signals.sine(1000, 48000), "X", "F")

    # invalid num_octave_band_fractions
    match = "Number of octave band fractions must be a " \
            "positive integer or None"
    with pytest.raises(TypeError, match=match):
        pf.level.time_weighted_level(
            pf.signals.sine(1000, 48000), "A", "F", 2.5)
    with pytest.raises(ValueError, match=match):
        pf.level.time_weighted_level(pf.signals.sine(1000, 48000), "A", "F", 0)
    with pytest.raises(ValueError, match=match):
        pf.level.time_weighted_level(
            pf.signals.sine(1000, 48000), "A", "F", -1)

    # invalid signal type
    td = pf.TimeData(np.array([1, 2, 3]), np.array([0, 1, 2]))
    match = "'signal' parameter must be a pyfar.Signal"
    with pytest.raises(TypeError, match=match):
        pf.level.time_weighted_level(np.array([1, 2, 3]), "A", "F")
    with pytest.raises(TypeError, match=match):
        pf.level.time_weighted_level(td, "A", "F")


def test_time_weighted_level_replace_zeros():
    """Test that the replace_zeros parameter correctly replaces zero values
    with the smallest positive number representable in the data.
    """
    s = pf.Signal(np.zeros(1000), sampling_rate=48000)

    levels_replace = pf.level.time_weighted_level(
        s, "Z", "F", replace_zeros=True)
    with pytest.warns(RuntimeWarning,
                      match="divide by zero encountered in log10"):
        levels_no_replace = pf.level.time_weighted_level(
            s, "Z", "F", replace_zeros=False)

    assert np.isfinite(levels_replace[0][100])
    assert levels_no_replace[0][100] == -np.inf
    # since there are only zeros in the signal, all values must be the same
    assert np.all(levels_replace[0] == levels_replace[0][0])
