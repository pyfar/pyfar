import pytest

import stub_utils


# Sine stubs
# test_dsp.py
# test_fft.py
# test_plot.py
# test_signal.py
@pytest.fixture
def sine():
    """Sine signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 1000
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
                        frequency,
                        sampling_rate,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)

    return signal


# test_fft.py
@pytest.fixture
def sine_odd():
    """Sine signal stub with static properties,
    odd number of samples.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    frequency = 441
    sampling_rate = 44100
    n_samples = 999
    fft_norm = 'rms'
    cshape = (1,)

    time, freq, frequency = stub_utils.sine_func(
                        frequency,
                        sampling_rate,
                        n_samples,
                        fft_norm,
                        cshape)

    signal = stub_utils.signal_stub(
                        time,
                        freq,
                        sampling_rate,
                        fft_norm)

    return signal


# Impulse stubs
# test_dsp.py
@pytest.fixture
def impulse():
    """Delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    pass


# test_dsp.py
@pytest.fixture
def impulse_group_delay():
    """Delayed delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    pass


# test_dsp.py
@pytest.fixture
def impulse_group_delay_two_channel():
    """Delayed 2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    pass


# test_dsp.py
@pytest.fixture
def impulse_group_delay_two_by_two_channel():
    """Delayed 2-by-2 channel delta impulse signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    group_delay : ndarray
        Group delay of impulse signal
    """
    pass


# test_fft.py
@pytest.fixture
def impulse_rms():
    """Delta impulse signal stub with static properties, rms-FFT normalization.

    Returns
    -------
    signal : Signal
        Stub of impulse signal
    """
    pass
