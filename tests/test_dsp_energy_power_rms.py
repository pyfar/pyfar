import pyfar as pf


def test_energy():
    """
    Test the energy of a Sinewave
    """
    signal = pf.signals.sine(500, 500)
    energy = pf.dsp.energy(signal)
    assert energy == 1


def test_power():
    """
    Test the power of a Sinewave
    """
    signal = pf.signals.sine(500, 500)
    power = pf.dsp.power(signal)
    assert power == 1


def test_rms():
    """
    Test the rms of a Sinewave
    """
    signal = pf.signals.sine(500, 500)
    rms = pf.dsp.rms(signal)
    assert rms == 0.707
