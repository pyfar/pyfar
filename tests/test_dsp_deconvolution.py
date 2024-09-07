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
    with pytest.raises(TypeError,
                       match='system_output has to be of type pyfar.Signal'):
        deconvolve('error', impulse(3))
    with pytest.raises(TypeError,
                       match='system_input has to be of type pyfar.Signal'):
        deconvolve(impulse(3), 'error')


def test_input_sampling_freq_error():
    """Test assertions by passing signals with different sampling frequency"""
    with pytest.raises(ValueError,
                       match='The two signals have different sampling rates!'):
        deconvolve(impulse(3, sampling_rate=44100),
                   impulse(3, sampling_rate=48000))


def test_fft_length_error():
    """Test assertion by passing fft_length shorter than n_samples of
    given Signals"""
    with pytest.raises(ValueError,
                       match="The fft_length can not be shorter than" +
                             "system_output.n_samples."):
        deconvolve(impulse(6, sampling_rate=44100),
                   impulse(5, sampling_rate=44100),
                   fft_length=5)
    with pytest.raises(ValueError,
                       match="The fft_length can not be shorter than" +
                             "system_input.n_samples."):
        deconvolve(impulse(5, sampling_rate=44100),
                   impulse(6, sampling_rate=44100),
                   fft_length=5)


def test_default():
    """Test the default parameters of deconvolution function."""
    res = deconvolve(impulse(6, sampling_rate=44100),
                     impulse(5, sampling_rate=44100),
                     freq_range=(1, 22050))
    assert res.n_samples == 6
    assert res.fft_norm == 'none'
    assert res.sampling_rate == 44100


def test_output_type():
    """Test Type of returned Signal and basic deconvolution with impulses"""
    res = deconvolve(impulse(3), impulse(3), freq_range=(1, 22050))
    assert isinstance(res, pf.Signal)
    npt.assert_allclose(impulse(3).time, res.time, rtol=1e-9)


def test_output_length():
    """Test if output length is correct"""
    res = deconvolve(impulse(3), impulse(3), freq_range=(1, 22050))
    assert res.n_samples == 3
    res = deconvolve(impulse(5), impulse(3), freq_range=(1, 22050))
    assert res.n_samples == 5
    res = deconvolve(impulse(3), impulse(5), freq_range=(1, 22050))
    assert res.n_samples == 5
    res = deconvolve(impulse(3), impulse(5), fft_length=7,
                     freq_range=(1, 22050))
    assert res.n_samples == 7


def test_output_sweep():
    """test of flat output frequency response resulting from deconvolving
    a sweep with the same sweep"""
    sweep = pfs.exponential_sweep_time(44100, (100, 10000))
    res = pf.dsp.dsp.deconvolve(sweep,
                                sweep,
                                freq_range=(1, 44100)).freq[0, 1000:-12500]
    npt.assert_allclose(np.ones_like(res), res, atol=1e-9)


def test_fft_norm():
    """Test the correct call of _match_fft_norm with parameter division=True
    for two example inputs."""
    sig1 = pf.Signal([1, 0, 0, 0], 44100, fft_norm='amplitude')
    res = pf.dsp.dsp.deconvolve(sig1, sig1, freq_range=(1, 44100))
    assert res.fft_norm == 'none'
    sig2 = pf.Signal([1, 0, 0, 0], 44100, fft_norm='none')
    sig3 = pf.Signal([1, 0, 0, 0], 44100, fft_norm='amplitude')
    with pytest.raises(ValueError, match="Either fft_norm_2 "):
        deconvolve(sig2, sig3, freq_range=(1, 44100))


def test_freq_range():
    """Test if freq_range default is correct"""
    res1 = deconvolve(impulse(3), impulse(3), freq_range=None)
    res2 = deconvolve(impulse(3), impulse(3), freq_range=(0, 22050))
    npt.assert_allclose(res1.time, res2.time, rtol=1e-14)
