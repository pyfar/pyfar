from numpy import testing as npt
import pyfar
import pyfar.dsp.filter as pffilt
import pytest


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("bands", ["octave", "third"])
@pytest.mark.parametrize("freq_range", [(20, 20000), (0, 100),
                                        (4000, 100_000)])
def test_frequency_weighting_constants(weighting, bands, freq_range):
    (nominals, iec_weights) = pyfar.constants. \
            frequency_weighting_band_corrections(weighting, bands, freq_range)
    calculated_weights = pyfar.constants. \
            frequency_weighting_curve(weighting, nominals)

    # maybe there is a better way of testing. tolerance is 0.2, because
    # we need 0.1 for rounding the weight and then larger error
    # due to the difference in nominal vs exact frequency being evaluated
    npt.assert_allclose(calculated_weights, iec_weights, atol=0.3)


@pytest.mark.parametrize("weighting", ["A", "C"])
@pytest.mark.parametrize("base_rate", [48000, 44100, 16000])
@pytest.mark.parametrize("rate_factor", [1, 2, 1/2, 4, 1/4, 8, 1/8])
def test_frequency_weighting_filter(weighting, base_rate, rate_factor):
    fs = base_rate * rate_factor
    filt = pffilt.frequency_weighting_filter(None, weighting, sampling_rate=fs)
    is_class_1, _, _ = pffilt.frequency_weighting. \
        _check_filter(filt.sampling_rate, filt.coefficients[0], weighting)
    assert is_class_1

