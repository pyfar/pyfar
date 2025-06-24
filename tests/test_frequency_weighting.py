import numpy as np
import pytest
from numpy import testing as npt
import pyfar
import pyfar.dsp.filter as pffilt


def test_frequency_weighting_constants():
    weightings = ["A", "C"]
    band_options = ["octave", "third"]
    freq_ranges = [
        (20, 20000),
        (0, 100),
        (4000, 100_000),
    ]
    for freq_range in freq_ranges:
        for weighting in weightings:
            for bands in band_options:
                _test_frequency_weighting_constants_combination(
                    weighting, bands, freq_range)


def _test_frequency_weighting_constants_combination(
        weighting, bands, freq_range):

    (nominals, iec_weights) = pyfar.constants. \
            frequency_weighting_band_corrections(weighting, bands, freq_range)
    calculated_weights = pyfar.constants. \
            frequency_weighting_curve(weighting, nominals)
            
    # maybe there is a better way of testing. tolerance is 0.2, because
    # we need 0.1 for rounding the weight and then larger error
    # due to the difference in nominal vs exact frequency being evaluated
    npt.assert_allclose(calculated_weights, iec_weights, atol=0.3)


def test_frequency_weighting_filter():
    base_rates = np.array([48000, 44100, 16000])
    test_rates = np.array([
        base_rates,
        base_rates / 2, base_rates * 2,
        base_rates / 4, base_rates * 4,
        base_rates / 8, base_rates * 8,
    ]).flatten().tolist()

    for weighting in ["A", "C"]:
        for fs in test_rates:
            filt = pffilt.frequency_weighting_filter(None,
                                                     weighting,
                                                     sampling_rate=fs)
            is_class_1, _, _ = pffilt.frequency_weighting. \
                _check_filter(filt.sampling_rate,
                              filt.coefficients[0],
                              weighting)
            assert is_class_1
