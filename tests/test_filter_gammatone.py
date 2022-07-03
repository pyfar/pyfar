import pytest
import numpy as np
import numpy.testing as npt
import pyfar.dsp.filter as filter


def test_gammatone_bands_init_and_getter():

    # initialize filter bank
    GFB = filter.GammatoneBands([0, 22050], resolution=.5)

    # test the getter
    npt.assert_array_equal(GFB.freq_range, [0, 22050])
    assert GFB.resolution == .5
    assert GFB.reference_frequency == 1000
    assert len(GFB.frequencies) == 85
    assert GFB.delay == 0.004

    # test the coefficients against AMT toolbox reference
    assert isinstance(GFB._coefficients, np.ndarray)
    assert len(GFB._coefficients) == 85
    npt.assert_array_almost_equal(
        GFB._coefficients[:2],
        [0.996393927986346 + 1j * 0.000253555298760638,
         0.996192240915680 + 1j * 0.00206860301231624])
    npt.assert_array_almost_equal(
        GFB._coefficients[-2:],
        [-0.699557965676674 + 1j * 0.198561122653194,
         -0.709666728508100 + 1j * 0.0826088418784688])

    assert isinstance(GFB._normalizations, np.ndarray)
    assert len(GFB._normalizations) == 85
    npt.assert_array_almost_equal(
        GFB._normalizations[:2], [3.38183204919764e-10, 4.19495909178277e-10])
    npt.assert_array_almost_equal(
        GFB._normalizations[-2:], [0.0110779622981509, 0.0132955514268072])


def test_erb_frequencies():
    """Test erb_frequencies against reference from the AMT toolbox"""

    frequencies = filter.erb_frequencies([0, 22050], .5)

    # assert type and length
    assert isinstance(frequencies, np.ndarray)
    assert len(frequencies) == 85

    # assert selected values against AMT toolbox reference
    npt.assert_array_almost_equal(
        frequencies[:2], [1.78607762641850, 14.5744473226466])
    npt.assert_array_almost_equal(
        frequencies[-2:], [20108.8703817471, 21236.6440502371])


def test_erb_frequencies_assertions():
    """Test assertions for erb_frequencies"""

    # freq_range must be an array of length 2
    with pytest.raises(ValueError, match="freq_range must be an array"):
        filter.erb_frequencies(1)
    with pytest.raises(ValueError, match="freq_range must be an array"):
        filter.erb_frequencies([1])
    # values freq_range must be increasing
    with pytest.raises(ValueError, match="The first value of freq_range"):
        filter.erb_frequencies([1, 0])
    # resolution must be > 0
    with pytest.raises(ValueError, match="Resolution must be larger"):
        filter.erb_frequencies([0, 1], 0)
