from numpy.testing._private.utils import assert_raises
import pytest
import numpy as np
from numpy import testing as npt

from pyfar.dsp import filter


def test_center_frequencies_iec():
    nominal_octs = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    actual_octs = filter.center_frequencies_fractional_octaves(num_fractions=1)
    actual_octs_nom = actual_octs[0]
    npt.assert_allclose(actual_octs_nom, nominal_octs)

    nominal_thirds = [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
        800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
        12500, 16000, 20000]
    actual_thirds = filter.center_frequencies_fractional_octaves(
        num_fractions=3)
    actual_thirds_nom = actual_thirds[0]
    npt.assert_allclose(actual_thirds_nom, nominal_thirds)

    with pytest.raises(ValueError, match="lower and upper limit"):
        filter.center_frequencies_fractional_octaves(frequency_range=(1,))

    with pytest.raises(ValueError, match="lower and upper limit"):
        filter.center_frequencies_fractional_octaves(frequency_range=(3, 4, 5))

    with pytest.raises(
            ValueError, match="second frequency needs to be higher"):
        filter.center_frequencies_fractional_octaves(
            frequency_range=(8e3, 1e3))

    with pytest.raises(ValueError, match="Number of fractions can only be"):
        filter.center_frequencies_fractional_octaves(5)

    actual_octs = filter.center_frequencies_fractional_octaves(
        num_fractions=1, frequency_range=(100, 4e3))
    actual_octs_nom = actual_octs[0]
    nominal_octs_part = [125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_octs_nom, nominal_octs_part)


def test_fractional_oct_filter_iec():
    sr = 48e3
    order = 2
    expected = np.zeros((3, order, 6))

    expected = np.array([
        [[1.99518917e-03,  3.99037834e-03,  1.99518917e-03,
          1.00000000e+00, -1.89455465e+00,  9.21866028e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.94204953e+00,  9.52106382e-01]],
        [[7.47518158e-03,  1.49503632e-02,  7.47518158e-03,
          1.00000000e+00, -1.74644971e+00,  8.50561709e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.86728468e+00,  9.06300424e-01]],
        [[2.65806645e-02,  5.31613291e-02,  2.65806645e-02,
          1.00000000e+00, -1.34871529e+00,  7.26916714e-01],
         [1.00000000e+00, -2.00000000e+00,  1.00000000e+00,
          1.00000000e+00, -1.67171842e+00,  8.18664740e-01]]])

    actual = filter.filter_fractional_octave_bands(
        sr, 1, freq_range=(1e3, 4e3), order=order)
    np.testing.assert_allclose(actual, expected)
