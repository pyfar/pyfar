import matplotlib.pyplot as plt
import pytest
import pyfar.plot as plot
import pyfar as pf
import numpy as np
import numpy.testing as npt


def test_prepare_plot():
    # test without arguments
    plot._utils._prepare_plot()

    # test with single axes object
    fig = plt.gcf()
    ax = plt.gca()
    plot._utils._prepare_plot(ax)
    plt.close()

    # test with list of axes
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._utils._prepare_plot(ax)
    plt.close()

    # test with numpy array of axes
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._utils._prepare_plot(ax)
    plt.close()

    # test with list of axes and desired subplot layout
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._utils._prepare_plot(ax, (2, 2))
    plt.close()

    # test with numpy array of axes and desired subplot layout
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._utils._prepare_plot(ax, (2, 2))
    plt.close()

    # test without axes and with desired subplot layout
    plot._utils._prepare_plot(None, (2, 2))
    plt.close()


def test_prepare_plot_2d():
    """
    Test assertion that are not tested directly with plots.
    """
    data = pf.Signal([1, 0, 0], 44100)
    kwargs = {"shading": "flat"}

    # assertion for data type
    with pytest.raises(TypeError, match="Input data has to be of type"):
        plot._utils._prepare_2d_plot(
            data, (pf.FrequencyData, ), 1, [0], 'pcolormesh', plt.gca(), False)

    # assertion for shading
    with pytest.raises(
            ValueError, match="shading is 'flat' but must be 'nearest'"):
        plot._utils._prepare_2d_plot(
            data, (pf.Signal, ), 1, [0], 'pcolormesh', plt.gca(), False,
            **kwargs)

    # assertion for indices
    with pytest.raises(ValueError, match="length of indices must match"):
        plot._utils._prepare_2d_plot(
            data, (pf.Signal, ), 1, [0, 1], 'pcolormesh', plt.gca(), False)

    # assertion for method
    with pytest.raises(ValueError, match="method must be"):
        plot._utils._prepare_2d_plot(
            data, (pf.Signal, ), 1, [0], 'pcontourmesh', plt.gca(), False)


def test_lower_frequency_limit(
        sine, sine_short, frequency_data,
        frequency_data_one_point, time_data):
    """Test the private function plot._utils._lower_frequency_limit."""

    # test Signal with frequencies below 20 Hz
    low = plot._utils._lower_frequency_limit(sine)
    assert low == 20

    # test Signal with frequencies above 20 Hz
    low = plot._utils._lower_frequency_limit(sine_short)
    assert low == 44100/100  # lowest frequency fs=44100 / n_samples=100

    # test with FrequencyData
    # (We only need to test if FrequencyData works. The frequency dependent
    # cases are already tested above)
    low = plot._utils._lower_frequency_limit(frequency_data)
    assert low == 100

    # test only 0 Hz assertions
    with pytest.raises(
            ValueError, match="Signals must have frequencies > 0 Hz"):
        plot._utils._lower_frequency_limit(frequency_data_one_point)

    # test TimeData assertions
    with pytest.raises(TypeError, match="Input data has to be of type"):
        plot._utils._lower_frequency_limit(time_data)


def test_time_auto_unit():
    """Test automatically assigning the unit in group delay plots."""
    assert plot._utils._time_auto_unit(0) == 's'
    assert plot._utils._time_auto_unit(1e-4) == 'mus'
    assert plot._utils._time_auto_unit(2e-2) == 'ms'
    assert plot._utils._time_auto_unit(2) == 's'


def test_default_colors():
    """Test default colors in plotstyles to match
    function used for displaying these.
    """
    color_dict = plot._utils._default_color_dict()
    colors = list(color_dict.values())
    for style in ['light', 'dark']:
        with plot.utils.context(style):
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_style = prop_cycle.by_key()['color']
            assert colors == colors_style


@pytest.mark.parametrize(
    ("fft_norm", "expected"),
    [('none', 20), ('unitary', 20), ('amplitude', 20),
     ('rms', 20), ('power', 10), ('psd', 10)])
def test__log_prefix_norms(sine, fft_norm, expected):
    sine.fft_norm = fft_norm
    assert plot._utils._log_prefix(sine) == expected


def test__log_prefix_frequency_data(frequency_data):
    assert plot._utils._log_prefix(frequency_data) == 20


def test__deal_time_units_mus():
    """Test previous bugfix for unit micro seconds in labels."""
    s = pf.signals.impulse(10, sampling_rate=44100)
    pf.plot.time(s)


def test_assert_and_match_data_to_side_wrong_parameter():
    signal = pf.signals.sine(20, 32)

    with pytest.raises(
            ValueError, match='Invalid `side` parameter, pass either '
            '`left` or `right`.'):
        plot._utils._assert_and_match_data_to_side(
            signal.freq, signal, side='quatsch')


def test_assert_and_match_data_to_side():
    signal = pf.signals.sine(20, 32)

    with pytest.raises(
            ValueError, match='The left side of the spectrum is not '
            'defined.'):
        plot._utils._assert_and_match_data_to_side(
            signal.freq, signal, side='left')

    signal.fft_norm = 'none'
    signal.complex = True

    data, frequencies, _xlabel = plot._utils._assert_and_match_data_to_side(
        signal.freq, signal, side='left')

    assert not np.any(frequencies < 0.0)
    assert data.shape[-1] == frequencies.shape[0]
    assert _xlabel == "Frequency in Hz (left)"

    data, frequencies, _xlabel = plot._utils._assert_and_match_data_to_side(
        signal.freq, signal, side='right')

    assert not np.any(frequencies < 0.0)
    assert data.shape[-1] == frequencies.shape[0]
    assert _xlabel == "Frequency in Hz (right)"


def test_assert_and_match_data_to_side_freq():
    signal = pf.FrequencyData([3, 4, 5, 6, 7],
                              [1, 2, 3, 4, 5])

    with pytest.raises(
            ValueError, match='The left side of the spectrum is not '
            'defined.'):
        plot._utils._assert_and_match_data_to_side(
            signal.freq, signal, side='left')

    data, frequencies, _ = plot._utils._assert_and_match_data_to_side(
        signal.freq, signal, side='right')

    assert not np.any(frequencies < 0.0)
    assert data.shape[-1] == frequencies.shape[0]

    signal = pf.FrequencyData([3, 4, 5, 6, 7],
                              [-5, -4, -3, -2, -1])
    with pytest.raises(ValueError, match='The right side of the spectrum '
                       'is not defined.'):
        plot._utils._assert_and_match_data_to_side(
            signal.freq, signal, side='right')

    data, frequencies, _ = plot._utils._assert_and_match_data_to_side(
        signal.freq, signal, side='left')
    assert not np.any(frequencies < 0.0)
    assert data.shape[-1] == frequencies.shape[0]


@pytest.mark.parametrize(("mode", "ylabel"), [('real', 'Amplitude'),
                                          ('real', 'Amplitude (real)'),
                                          ('imag', 'Amplitude (imaginary)'),
                                          ('abs', 'Amplitude (absolute)')])
def test_assert_and_match_data_to_mode(mode, ylabel):
    signal = pf.signals.sine(20, 32)

    if not ylabel == 'Amplitude':
        signal.fft_norm = 'none'
        signal.complex = True

    data, _ylabel = plot._utils._assert_and_match_data_to_mode(signal.time,
                                                               signal,
                                                               mode)

    if mode == 'real':
        npt.assert_allclose(data,
                            np.real(signal.time), atol=1e-15)
    elif mode == 'imag':
        npt.assert_allclose(data,
                            np.imag(signal.time), atol=1e-15)
    else:
        npt.assert_allclose(data,
                            np.abs(signal.time), atol=1e-15)
    assert _ylabel == ylabel


@pytest.mark.parametrize('side', ['left', 'right'])
def test_assert_and_match_spectrogram_to_side(side):

    signal = pf.Signal(data=[1, 4, 5, 6, 7], sampling_rate=48000,
                       is_complex=True)

    frequencies, _, spec = pf.dsp.spectrogram(signal, window_length=4)

    spec, frequencies, xlabel = \
        plot._utils._assert_and_match_spectrogram_to_side(
            np.squeeze(spec, axis=0), frequencies, signal, side)

    assert not np.any(frequencies < 0.0)
    assert spec.shape[0] == frequencies.shape[0]
    assert xlabel == f'Frequency in Hz ({side})'
