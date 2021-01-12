import pytest
import numpy as np
import copy

from unittest import mock

from pyfar import Signal
from pyfar import fft
from pyfar.orientations import Orientations


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
    pass


# test_fft.py
@pytest.fixture
def sine_odd():
    """Sine signal stub with static properties.

    Returns
    -------
    signal : Signal
        Stub of sine signal
    """
    pass


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


