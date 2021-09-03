#!/usr/bin/env python

import pytest
import numpy as np
import numpy.testing as npt
import pyfar as pf
from pyfar.signals import impulse
import pyfar.signals as pfs

from pyfar.dsp.dsp import deconvolve


def test_input_type_error():
    """Test assertions by passing non Signal-Type"""
    with pytest.raises(TypeError):
        deconvolve('error', impulse(3))
    with pytest.raises(TypeError):
        deconvolve(impulse(3), 'error')


def test_input_sampling_freq_error():
    """Test assertions by passing signals with different sampling frequency"""
    with pytest.raises(ValueError):
        deconvolve(impulse(3, sampling_rate=44100),
                   impulse(3, sampling_rate=48000))


def test_output_type():
    """Test Type of returned Signal and basic deconvolution with impulses"""
    res = deconvolve(impulse(3), impulse(3), freq_range=(1, 22050))
    assert isinstance(res, pf.Signal)
    npt.assert_allclose(impulse(3).time, res.time, rtol=1e-9)


def test_output_length():
    """Test im output length is correct"""
    res = deconvolve(impulse(3), impulse(3), freq_range=(1, 22050))
    assert res.n_samples == 3
    res = deconvolve(impulse(5), impulse(3), freq_range=(1, 22050))
    assert res.n_samples == 5
    res = deconvolve(impulse(3), impulse(5), freq_range=(1, 22050))
    assert res.n_samples == 5


def test_output_sweep():
    """test of flat output frequency response resulting from deconvolving
    a sweep with the same sweep"""
    sweep = pfs.exponential_sweep_time(44100, (100, 10000))
    res = pf.dsp.dsp.deconvolve(sweep,
                                sweep,
                                freq_range=(1, 44100)).freq[0, 1000:-12500]
    npt.assert_allclose(np.ones_like(res), res, atol=1e-9)
