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
        filter.center_frequencies_fractional_octaves(f_lims=(1,))

    with pytest.raises(ValueError, match="lower and upper limit"):
        filter.center_frequencies_fractional_octaves(f_lims=(3, 4, 5))

    with pytest.raises(
            ValueError, match="second frequency needs to be higher"):
        filter.center_frequencies_fractional_octaves(
            f_lims=(8e3, 1e3))

    actual_octs = filter.center_frequencies_fractional_octaves(
        num_fractions=1, f_lims=(100, 4e3))
    actual_octs_nom = actual_octs[0]
    nominal_octs_part = [125, 250, 500, 1000, 2000, 4000]
    npt.assert_allclose(actual_octs_nom, nominal_octs_part)
