import pyfar as pf
import numpy as np

def test_time_weighted_levels_decay():
    fs = 48000
    x = pf.signals.impulse(2 * fs, sampling_rate=fs)

    levels_F = pf.dsp.time_weighted_levels(x, "F")
    diff1_F =  levels_F[0][fs] - levels_F[0][0]
    diff2_F = levels_F[0][fs + 100] - levels_F[0][100]
    F_DECAY = -34.7 # DIN EN 61672-1 §5.8.2
    assert abs(diff1_F - F_DECAY) < 0.1
    assert abs(diff2_F - F_DECAY) < 0.1

    levels_S = pf.dsp.time_weighted_levels(x, "S")
    diff1_S =  levels_S[0][fs] - levels_S[0][0]
    diff2_S = levels_S[0][fs + 100] - levels_S[0][100]
    S_DECAY = -4.3 # DIN EN 61672-1 §5.8.2
    assert abs(diff1_S - S_DECAY) < 0.1
    assert abs(diff2_S - S_DECAY) < 0.1


def test_time_weighting_levels_same_level():
    """
    The energy of a long 1kHz signal must be identical
    across F, S and eq levels.
    """
    fs = 48000
    duration = 60
    x = pf.signals.sine(1000, fs * duration, sampling_rate=48000)

    L_eq = 10 * np.log10(np.sum(x.time[0]**2) / x.n_samples)
    levels_F = pf.dsp.time_weighted_levels(x, "F")
    levels_S = pf.dsp.time_weighted_levels(x, "S")
    # compare last sample to the equivalent average,
    # so the exp-filter has enough time to integrate energy
    assert abs(L_eq - levels_F[0][-1]) < 0.1
    assert abs(L_eq - levels_S[0][-1]) < 0.1
