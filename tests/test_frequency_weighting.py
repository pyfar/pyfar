from numpy import testing as npt
import numpy as np
import pyfar
import pyfar.dsp.filter as pffilt
import pytest
import re


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("num_fractions", [1, 3])
@pytest.mark.parametrize("freq_range", [(20, 20000)])
def test_frequency_weighting_constants(weighting, num_fractions, freq_range):
    nominals, iec_weights = pyfar.constants. \
            frequency_weighting_band_corrections(weighting,
                                                 num_fractions,
                                                 freq_range)
    calculated_weights = pyfar.constants. \
            frequency_weighting_curve(weighting, nominals)

    assert isinstance(iec_weights, np.ndarray)
    assert isinstance(calculated_weights, np.ndarray)

    # maybe there is a better way of testing. tolerance is 0.2, because
    # we need 0.1 for rounding the weight and then larger error
    # due to the difference in nominal vs exact frequency being evaluated
    npt.assert_allclose(calculated_weights, iec_weights, atol=0.3)


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("num_fractions", [1, 3])
@pytest.mark.parametrize("freq_range", [(0, 100), (4000, 100_000)])
def test_frequency_weighting_constants_warning(
    weighting, num_fractions, freq_range):

    if num_fractions == 1:
        message = 'octave-band are defined only from 11.2 Hz to 22387.2 Hz'
    else:
        message = ('one-third-octave-band are defined only from 8.91 Hz '
                       'to 22387.2 Hz')

    with pytest.warns(UserWarning, match=message):
        nominals, iec_weights = pyfar.constants. \
            frequency_weighting_band_corrections(weighting,
                                                 num_fractions,
                                                 freq_range)

    calculated_weights = pyfar.constants. \
            frequency_weighting_curve(weighting, nominals)

    assert isinstance(iec_weights, np.ndarray)
    assert isinstance(calculated_weights, np.ndarray)

    # maybe there is a better way of testing. tolerance is 0.2, because
    # we need 0.1 for rounding the weight and then larger error
    # due to the difference in nominal vs exact frequency being evaluated
    npt.assert_allclose(calculated_weights, iec_weights, atol=0.3)


def test_frequency_weighting_band_corrections_errors():
    # invalid weighting
    match = "Allowed literals for weighting are 'A' and 'C'"
    with pytest.raises(ValueError, match=match):
        pyfar.constants.frequency_weighting_band_corrections("B", 1,
                                                             (100, 1000))

    # invalid bands
    match = "num_fractions must be 1 or 3"
    with pytest.raises(ValueError, match=match):
        pyfar.constants.frequency_weighting_band_corrections("A", "octave",
                                                             (100, 1000))


def test_frequeny_weighting_curve_errors():
    # empty frequency range
    match = "Allowed literals for weighting are 'A' and 'C'"
    with pytest.raises(ValueError, match=match):
        pyfar.constants.frequency_weighting_curve("B", [1, 2, 3])


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("base_rate", [48000, 44100, 16000])
@pytest.mark.parametrize("rate_factor", [1, 2, 1/2, 4, 1/4, 8, 1/8])
def test_frequency_weighting_filter_default(weighting, base_rate, rate_factor):
    fs = base_rate * rate_factor
    filt = pffilt.frequency_weighting_filter(None, weighting, sampling_rate=fs)
    assert isinstance(filt, pyfar.FilterSOS)
    is_class_1, _, _ = pffilt.frequency_weighting. \
        _check_filter(filt.sampling_rate, filt.coefficients[0], weighting)
    assert is_class_1


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("fs", [48000, 44100])
@pytest.mark.parametrize("errwgt", [lambda nf: 100**nf, lambda nf: 1 + nf])
def test_frequency_weighting_filter_errwgt(weighting, fs, errwgt):
    without = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs)
    with_errwgt = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs, error_weighting=errwgt)
    # they should produce different results with different weightings
    assert not np.allclose(without.coefficients, with_errwgt.coefficients)


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("fs", [48000, 44100, 16000])
def test_frequency_weighting_filter_errwgt_passthrough(weighting, fs):
    with_none = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs)
    # weights every frequency with 1, so same as without any error weighting
    with_passthrough = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs, error_weighting=lambda _: 1)
    npt.assert_allclose(with_none.coefficients, with_passthrough.coefficients)


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("fs", [48000, 44100])
def test_frequency_weighting_filter_errwgt_recommended(weighting, fs):
    # the function comment mentions lambda nf: 100**nf to produce good
    # results for common sample rates, so make sure that is true
    without = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs)
    with_errwgt = pffilt.frequency_weighting_filter(
        None, weighting, sampling_rate=fs, error_weighting=lambda nf: 100**nf)
    stats_without = pffilt.frequency_weighting._check_filter(
        fs, without.coefficients[0], weighting)
    stats_with = pffilt.frequency_weighting._check_filter(
        fs, with_errwgt.coefficients[0], weighting)
    # assert both are class 1
    assert stats_without[0]
    assert stats_with[0]


def test_frequency_weighting_filter_kwargs():
    without = pffilt.frequency_weighting_filter(None, "A", sampling_rate=44100)
    # jac= is a parameter for scipy's least_squares method
    with_kwarg = pffilt.frequency_weighting_filter(
        None, "A", sampling_rate=44100, jac="3-point")
    assert not np.allclose(without.coefficients, with_kwarg.coefficients)


def test_frequency_weighting_filter_on_signal():
    signal = pyfar.signals.impulse(1000)
    filtered1 = pffilt.frequency_weighting_filter(signal)
    assert isinstance(filtered1, pyfar.Signal)

    fs = signal.sampling_rate
    filt = pffilt.frequency_weighting_filter(None, sampling_rate=fs)
    assert isinstance(filt, pyfar.FilterSOS)
    filtered2 = filt.process(signal)
    assert isinstance(filtered2, pyfar.Signal)
    npt.assert_allclose(filtered1.time, filtered2.time)


def test_frequency_weighting_filter_errors():
    signal = pyfar.signals.impulse(100)

    match = "Allowed literals for weighting are 'A' and 'C'"
    with pytest.raises(ValueError, match=match):
        pffilt.frequency_weighting_filter(signal, "B")

    match = "Either signal or sampling_rate must be none."
    with pytest.raises(ValueError, match=match):
        pffilt.frequency_weighting_filter(signal, "A", sampling_rate=48e3)
    with pytest.raises(ValueError, match=match):
        pffilt.frequency_weighting_filter(None, "A")

    regex = r"The generated A weighting filter is not class 1 compliant.*"
    with pytest.warns(UserWarning, match=re.compile(regex)):
        pffilt.frequency_weighting_filter(signal, "A",
                                          error_weighting=lambda _: 0)


def test_frequency_weighting_filter_caching():
    """
    The caching function should give a new copy of the cached value
    after each call, rather than returning multiple references to the
    same array.
    This is to avoid bugs if the user mutates the array.
    """
    fs = 48000
    # mutating coefficients of one filter should not affect other filters built
    # from the same cached coefficients
    a = pffilt.frequency_weighting_filter(None, "A", sampling_rate=fs)
    b = pffilt.frequency_weighting_filter(None, "A", sampling_rate=fs)
    assert np.all(a.coefficients == b.coefficients)
    # if both share a reference to the same coefficients object,
    # this would mutate both
    a.coefficients[0][0][1] = 10
    assert a.coefficients[0][0][1] != b.coefficients[0][0][1]

    # just to be safe, the private (internal) function should also not expose
    # references to cached arrays, which must not be mutated
    a = pffilt.frequency_weighting. \
        _design_frequency_weighting_filter_cached(fs, "A", 100)
    b = pffilt.frequency_weighting. \
        _design_frequency_weighting_filter_cached(fs, "A", 100)
    assert np.all(a == b)
    a[0][0] = 10
    assert not np.all(a == b)
