import pyfar as pf
import numpy as np
from pathlib import Path
import numpy.testing as npt
import pytest


@pytest.mark.parametrize('exact_center_frequency', [1000, 1000 * 10**.3])
@pytest.mark.parametrize('num_fractions', [1, 3])
@pytest.mark.parametrize('tolerance_class', [1, 2])
def test_octave_band_tolerance(
        exact_center_frequency, num_fractions, tolerance_class):
    """Test values returned by octave_band_tolerance."""

    # load reference data
    filename = Path((
        "tests/references/octave_band_tolerance_"
        f"{int(exact_center_frequency)}Hz_num_fractions{num_fractions}_"
        f"class{tolerance_class}.csv"))
    data = np.loadtxt(filename, delimiter=',')
    frequencies_ref = data[0]
    tolerance_ref = data[1:]

    # generate tolerance using pyfar
    tolerance, frequencies = pf.constants.octave_band_tolerance(
                exact_center_frequency, num_fractions, tolerance_class)

    # actual tests (data written and tested with 2 decimals of precision)
    assert type(tolerance) == np.ndarray
    assert type(frequencies) == np.ndarray
    assert tolerance.shape == (2, 19)
    assert frequencies.shape == (19, )
    npt.assert_almost_equal(frequencies, frequencies_ref, 2)
    npt.assert_almost_equal(tolerance, tolerance_ref, 2)


def test_octave_band_tolerance_errors():
    """Test if all errors are raised as expected."""

    with pytest.raises(ValueError, match='The exact center frequency'):
        pf.constants.octave_band_tolerance([1000], 1, 1)

    with pytest.raises(ValueError, match="num_fractions is 2"):
        pf.constants.octave_band_tolerance(1000, 2, 1)

    with pytest.raises(ValueError, match="tolerance_class is 0"):
        pf.constants.octave_band_tolerance(1000, 1, 0)
