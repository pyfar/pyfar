import pytest
import numpy as np
import numpy.testing as npt
from pyfar import Signal
import pyfar.dsp.filter as pfilt
import pyfar.classes.filter as pclass


def test_butter(impulse):
    # Uses scipy function. We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.butter(None, 2, 1000, 'lowpass', 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Butterworth lowpass of order 2. "
                             "Cut-off frequency 1000 Hz.")

    # Filter
    x = pfilt.butter(impulse, 2, 1000, 'lowpass')
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.butter(impulse, 2, 1000, 'lowpass', 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.butter(None, 2, 1000, 'lowpass')


def test_cheby1(impulse):
    # Uses scipy function. We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.cheby1(None, 2, 1, 1000, 'lowpass', 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Chebychev Type I lowpass of order 2. "
                             "Cut-off frequency 1000 Hz. "
                             "Pass band ripple 1 dB.")

    # Filter
    x = pfilt.cheby1(impulse, 2, 1, 1000, 'lowpass')
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.cheby1(impulse, 2, 1, 1000, 'lowpass', 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.cheby1(None, 2, 1, 1000, 'lowpass')


def test_cheby2(impulse):
    # Uses scipy function. We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.cheby2(None, 2, 40, 1000, 'lowpass', 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Chebychev Type II lowpass of order 2. "
                             "Cut-off frequency 1000 Hz. "
                             "Stop band attenuation 40 dB.")

    # Filter
    x = pfilt.cheby2(impulse, 2, 40, 1000, 'lowpass')
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.cheby2(impulse, 2, 40, 1000, 'lowpass', 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.cheby2(None, 2, 40, 1000, 'lowpass')


def test_ellip(impulse):
    # Uses scipy function. We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.ellip(None, 2, 1, 40, 1000, 'lowpass', 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Elliptic (Cauer) lowpass of order 2. "
                             "Cut-off frequency 1000 Hz. "
                             "Pass band ripple 1 dB. "
                             "Stop band attenuation 40 dB.")

    # Filter
    x = pfilt.ellip(impulse, 2, 1, 40, 1000, 'lowpass')
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.ellip(impulse, 2, 1, 40, 1000, 'lowpass', 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.ellip(None, 2, 1, 40, 1000, 'lowpass')


def test_bessel(impulse):
    # Uses scipy function. We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.bessel(None, 2, 1000, 'lowpass', 'phase', 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Bessel/Thomson lowpass of order 2 and 'phase' "
                             "normalization. Cut-off frequency 1000 Hz.")

    # Filter
    x = pfilt.bessel(impulse, 2, 1000, 'lowpass', 'phase')
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.bessel(impulse, 2, 1000, 'lowpass', 'phase', 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.bessel(None, 2, 1000, 'lowpass', 'phase')


def test_peq(impulse):
    # Uses third party code.
    # We thus only test the functionality not the results
    # Filter object
    f_obj = pfilt.peq(None, 1000, 10, 2, sampling_rate=44100)
    assert isinstance(f_obj, pclass.FilterIIR)
    assert f_obj.comment == ("Second order parametric equalizer (PEQ) of type "
                             "II with 10 dB gain at 1000 Hz (Quality = 2).")

    # Filter
    x = pfilt.peq(impulse, 1000, 10, 2)
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # test ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.peq(impulse, 1000, 10, 2, sampling_rate=44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.peq(None, 1000, 10, 2)
    # check wrong input arguments
    with pytest.raises(ValueError):
        x = pfilt.peq(impulse, 1000, 10, 2, peq_type='nope')
    with pytest.raises(ValueError):
        x = pfilt.peq(impulse, 1000, 10, 2, quality_warp='nope')


def test_shelve(impulse):
    # Uses third party code.
    # We thus only test the functionality not the results

    shelves = [pfilt.low_shelve, pfilt.high_shelve]
    kinds = ['Low', 'High']

    for shelve, kind in zip(shelves, kinds):
        # Filter object
        f_obj = shelve(None, 1000, 10, 2, sampling_rate=44100)
        assert isinstance(f_obj, pclass.FilterIIR)
        assert f_obj.comment == (f"{kind}-shelve of order 2 and type I with "
                                 "10 dB gain at 1000 Hz.")

        # Filter
        x = shelve(impulse, 1000, 10, 2)
        y = f_obj.process(impulse)
        assert isinstance(x, Signal)
        npt.assert_allclose(x.time, y.time)

        # ValueError
        with pytest.raises(ValueError):
            # pass signal and sampling rate
            x = shelve(impulse, 1000, 10, 2, sampling_rate=44100)
        with pytest.raises(ValueError):
            # pass no signal and no sampling rate
            x = shelve(None, 1000, 10, 2)
        # check wrong input arguments
        with pytest.raises(ValueError):
            x = shelve(impulse, 1000, 10, 2, shelve_type='nope')
        with pytest.raises(ValueError):
            x = shelve(impulse, 1000, 10, 3)


def test_crossover(impulse):
    # Uses scipy function. We thus mostly test the functionality not the
    # results

    # Filter object
    f_obj = pfilt.crossover(None, 2, 1000, 44100)
    assert isinstance(f_obj, pclass.FilterSOS)
    assert f_obj.comment == ("Linkwitz-Riley cross over network of order 2 at "
                             "1000 Hz.")

    # test filter
    x = pfilt.crossover(impulse, 2, 1000)
    y = f_obj.process(impulse)
    assert isinstance(x, Signal)
    npt.assert_allclose(x.time, y.time)

    # test ValueError
    with pytest.raises(ValueError):
        # pass signal and sampling rate
        x = pfilt.crossover(impulse, 2, 1000, 44100)
    with pytest.raises(ValueError):
        # pass no signal and no sampling rate
        x = pfilt.crossover(None, 2, 1000)
    with pytest.raises(ValueError):
        # odd filter order
        x = pfilt.crossover(impulse, 3, 1000)

    # check if frequency response sums to unity for different filter orders
    for order in [2, 4, 6, 8]:
        x = pfilt.crossover(impulse, order, 4000)
        x_sum = np.sum(x.freq, axis=-2).flatten()
        x_ref = np.ones(x.n_bins)
        npt.assert_allclose(x_ref, np.abs(x_sum))

    # check network with multiple cross-over frequencies
    f_obj = pfilt.crossover(None, 2, [100, 10_000], 44100)
    assert f_obj.comment == ("Linkwitz-Riley cross over network of order 2 at "
                             "100, 10000 Hz.")
    x = pfilt.crossover(impulse, 2, [100, 10_000])
