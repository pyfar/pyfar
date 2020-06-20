#from unittest import mock
from matplotlib.testing.decorators import image_comparison
import numpy as np
import numpy.testing as npt
import pytest

import haiopy.plot as plot
from haiopy import Signal

@image_comparison(baseline_images=['plot_time'])
def test_plot_time():
    times = np.linspace(0, 1, 1024, endpoint=False)
    square_wave = sgn.square(2 * np.pi * 440 * t)
    test_signal_object = Signal(data=square_wave,
                                sampling_rate=44100,
                                domain='time',
                                signal_type='power')
    plot.plot_time(test_signal_object)
