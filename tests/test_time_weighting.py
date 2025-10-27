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
@pytest.mark.parametrize("weighting,expected_decay", [("F", -34.7),
                                                      ("S", -4.3)])
def test_time_weighting_pressure_decay(weighting, expected_decay, fs):
    """According to DIN EN 61672-1 §5.8.2, after an impulse the level
    should fall of at a rate of 34.7 dB/s for FAST and 4.3 dB/s for the
    SLOW time weighting."""
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
    across F, S and eq levels. (DIN EN 61672-1 §5.8.3)"""
    fs = 48000
    duration = 60
    x = pf.signals.sine(1000, fs * duration, sampling_rate=48000)

    L_eq = 10 * np.log10(np.sum(x.time[0]**2) / x.n_samples) 
    weighted = pf.level.time_weighted_sound_pressure(x, weighting)
    # compare last sample to the equivalent average, 
    # so the exp-filter has enough time to integrate energy
    level = 10 * np.log10(weighted.time[0][-1] ** 2)
    assert abs(L_eq - level) < 0.1
    