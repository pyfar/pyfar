import pytest
import numpy as np
import numpy.testing as npt
import pyfar as pf
import pyfar.dsp.filter as filter


def test_gammatone_bands_init_and_getter():
    """
    Initialize the filter bank, test all public properties, test all private
    variables.
    """

    # initialize filter bank
    GFB = filter.GammatoneBands([0, 22050], resolution=.5)

    # test the getter
    npt.assert_array_equal(GFB.freq_range, [0, 22050])
    assert GFB.resolution == .5
    assert GFB.reference_frequency == 1000
    assert GFB.frequencies.shape == (85, )
    assert GFB.n_bands == 85
    assert GFB.delay == 0.004

    # test the coefficients against AMT toolbox reference
    assert isinstance(GFB.coefficients, np.ndarray)
    assert len(GFB.coefficients) == 85
    npt.assert_array_almost_equal(
        GFB.coefficients[:2],
        [0.996393927986346 + 1j * 0.000253555298760638,
         0.996192240915680 + 1j * 0.00206860301231624])
    npt.assert_array_almost_equal(
        GFB.coefficients[-2:],
        [-0.699557965676674 + 1j * 0.198561122653194,
         -0.709666728508100 + 1j * 0.0826088418784688])

    assert isinstance(GFB.normalizations, np.ndarray)
    assert GFB.normalizations.shape == (85, )
    npt.assert_array_almost_equal(
        GFB.normalizations[:2], [3.38183204919764e-10, 4.19495909178277e-10])
    npt.assert_array_almost_equal(
        GFB.normalizations[-2:], [0.0110779622981509, 0.0132955514268072])

    # test the delay values against AMT toolbox reference
    assert isinstance(GFB.delays, np.ndarray)
    assert GFB.delays.shape == (85, )
    npt.assert_array_equal(GFB.delays[:2], [0, 0])
    npt.assert_array_equal(GFB.delays[-2:], [169, 169])

    # test phase factor values against AMT toolbox reference
    assert isinstance(GFB.phase_factors, np.ndarray)
    assert GFB.phase_factors.shape == (85, )
    npt.assert_array_almost_equal(
        GFB.phase_factors[:2],
        [0.0639595495518330 + 1j * 0.997952491865783,
         0.499802466801202 + 1j * 0.866139419596772])
    npt.assert_array_almost_equal(
        GFB.phase_factors[-2:],
        [0.293917767610258 - 1j * 0.955830709845108,
         -0.696904158943754 - 1j * 0.717164272148926])

    # test gain values against AMT toolbox reference
    assert isinstance(GFB.gains, np.ndarray)
    assert GFB.gains.shape == (85, )
    npt.assert_array_almost_equal(GFB.gains[:2], [0.0786982388842526,
                                                  0.339454449549947])
    npt.assert_array_almost_equal(GFB.gains[-2:], [0.302225125941216,
                                                   0.483240051077418])


@pytest.mark.parametrize('amplitudes,shape_filtered,sampling_rate', (
    [np.array([1]), (85, 2048), 44100],
    [np.array([1]), (85, 2048), 48000],
    [np.array([1]), (85, 2048), 96000],
    [np.array([[1, 2], [3, 4]]), (85, 2, 2, 2048), 44100]
))
def test_gammatone_bands_roundtrip(amplitudes, shape_filtered, sampling_rate):
    """
    Verify the entire filter bank processing with a round trip using single and
    multi channel signals of different sampling rates. Assert shape of
    intermediate results. Assert values of final result.
    """

    # initialize filter bank
    GFB = filter.GammatoneBands(
        [0, 22050], resolution=.5, sampling_rate=sampling_rate)

    # filter and sum an impulse signal
    impulse = pf.signals.impulse(
        2048, amplitude=amplitudes, sampling_rate=sampling_rate)
    real, imag = GFB.process(impulse)
    reconstructed = GFB.reconstruct(real, imag)

    # assert shapes
    assert real.time.shape == shape_filtered
    assert imag.time.shape == shape_filtered
    assert reconstructed.time.shape == impulse.time.shape

    for idx in np.ndindex(impulse.cshape):
        # assert magnitude spectrum: must be constant
        log_mag = pf.dsp.decibel(reconstructed[idx] / amplitudes[idx])
        f_max = reconstructed.find_nearest_frequency(20e3)
        assert np.all(np.abs(log_mag[..., :f_max]) < 0.5)

        # assert group delay in seconds: must be constant above freq. limit
        grp_del = pf.dsp.group_delay(
            reconstructed[idx], [1e3, 2e3, 4e3, 8e3, 16e3], 'scipy')
        assert np.all(np.abs(
            grp_del / reconstructed.sampling_rate - GFB.delay) < .0001)


def test_gammatone_bands_reset_state():

    GFB = filter.GammatoneBands([0, 22050])

    # filter in one block
    real, imag = GFB.process(pf.signals.impulse(2**12))

    # filter in two blocks
    real_a, imag_a = GFB.process(pf.signals.impulse(2**11))
    real_b, imag_b = GFB.process(
        pf.Signal(np.zeros(2**11), 44100), reset=False)

    # check for equality
    npt.assert_array_equal(real_a.time, real.time[:, :2**11])
    npt.assert_array_equal(imag_a.time, imag.time[:, :2**11])

    npt.assert_array_equal(real_b.time, real.time[:, -2**11:])
    npt.assert_array_equal(imag_b.time, imag.time[:, -2**11:])


def test_gammatone_bands_assertions():
    """Test all assertions"""

    # wrong values in freq_range
    with pytest.raises(ValueError, match="Values in freq_range must be"):
        filter.GammatoneBands([-1, 22050])
    with pytest.raises(ValueError, match="Values in freq_range must be"):
        filter.GammatoneBands([0, 24e3])

    # wrong value for delay
    with pytest.raises(ValueError, match="The delay must be larger than zero"):
        filter.GammatoneBands([0, 22050], delay=0)

    # wrong value for resolution
    with pytest.raises(ValueError, match="The resolution must be larger than"):
        filter.GammatoneBands([0, 22050], resolution=0)

    # mismatching type for filter
    GFB = filter.GammatoneBands([0, 22050])
    with pytest.raises(TypeError, match="signal must be"):
        GFB.process([1, 0, 0])

    # mismatching sampling rates
    with pytest.raises(ValueError, match="The sampling rates"):
        GFB.process(pf.Signal([1, 2, 3], 48000))


def test_gammatone_bands_repr():
    """Test string representation"""

    GFB = filter.GammatoneBands([0, 22050])
    assert str(GFB) == ("Reconstructing Gammatone filter bank with 42 bands "
                        "between 0 and 22050 Hz spaced by 1 ERB units "
                        "@ 44100 Hz sampling rate")


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
