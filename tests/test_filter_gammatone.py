import pytest
import numpy as np
import numpy.testing as npt
import pyfar.dsp.filter as filter


def test_erb_frequencies():
    """Test erb_frequencies against reference from the AMT toolbox"""

    frequencies = filter.erb_frequencies([0, 22050], .5)

    # assert type and length
    assert isinstance(frequencies, np.ndarray)
    assert len(frequencies) == 85

    # assert selected values against AMT reference
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
