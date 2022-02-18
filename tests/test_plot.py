import os
import pytest
from pytest import raises
import matplotlib.pyplot as plt
import pyfar as pf
import pyfar.plot as plot
from pyfar.testing.plot_utils import create_figure, save_and_compare
import numpy as np
import numpy.testing as npt

# global parameters -----------------------------------------------------------
# flag for creating new baseline plots
# - required if the plot look changed
# - make sure to manually check the new baseline plots located at baseline_path
create_baseline = False

# file type used for saving the plots
file_type = "png"

# if true, the plots will be compared to the baseline and an error is raised
# if there are any differences. In any case, differences are written to
# output_path as images
compare_output = False

# path handling
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

if not os.path.isdir(base_path):
    os.mkdir(base_path)
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# remove old output files
for file in os.listdir(output_path):
    os.remove(os.path.join(output_path, file))


# testing ---------------------------------------------------------------------
@pytest.mark.parametrize('function', [
    (plot.time), (plot.freq), (plot.phase), (plot.group_delay),
    (plot.time_freq), (plot.freq_phase), (plot.freq_group_delay)])
def test_line_plots(function, sine, impulse_group_delay):
    """Test all line plots with default arguments and hold functionality."""
    print(f"Testing: {function.__name__}")

    # initial plot
    filename = 'line_' + function.__name__
    create_figure()
    function(sine)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)

    # test hold functionality
    filename = 'line_' + function.__name__ + '_hold'
    function(impulse_group_delay[0])
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('param', [
    ['line_phase_deg', True, False],
    ['line_phase_unwrap', False, True],
    ['line_phase_deg_unwrap', True, True]])
def test_line_phase_options(param, sine):
    """Test parameters that are unique to the phase plot."""
    print(f"Testing: {param[0]}")

    filename = param[0]
    create_figure()
    plot.phase(sine, deg=param[1], unwrap=param[2])
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


def test_line_phase_unwrap_assertion(sine):
    """Test assertion for unwrap parameter."""
    with raises(ValueError):
        plot.phase(sine, unwrap='infinity')


@pytest.mark.parametrize('function', [
    (plot.time), (plot.freq), (plot.spectrogram)])
def test_line_dB_option(function, sine):
    """Test all line plots that have a dB option."""
    # test if dB option is working
    for dB in [True, False]:
        print(f"Testing: {function.__name__} (dB={dB})")

        filename = 'line_' + function.__name__ + '_dB_' + str(dB)
        create_figure()
        function(sine, dB=dB)
        save_and_compare(create_baseline, baseline_path, output_path,
                         filename, file_type, compare_output)

    # test if log_prefix and log_reference are working
    print(f"Testing: {function.__name__} (log parameters)")

    filename = 'line_' + function.__name__ + '_logParams'
    create_figure()
    function(sine, log_prefix=10, log_reference=.5, dB=True)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.freq), (plot.phase), (plot.group_delay)])
@pytest.mark.parametrize('xscale', [('log'), ('linear')])
def test_line_xscale_option(function, xscale, sine):
    """Test all line plots that have an xscale option."""
    # test if xscale option is working
    print(f"Testing: {function.__name__} (xscale={xscale})")

    filename = 'line_' + function.__name__ + '_xscale_' + xscale
    create_figure()
    function(sine, xscale=xscale)
    save_and_compare(create_baseline, baseline_path, output_path,
                     filename, file_type, compare_output)


def test_line_xscale_assertion(sine):
    """
    Test if all line plots raise an assertion for a wrong scale parameter.
    """

    with raises(ValueError):
        plot.freq(sine, xscale="warped")

    with raises(ValueError):
        plot.phase(sine, xscale="warped")

    with raises(ValueError):
        plot.group_delay(sine, xscale="warped")

    with raises(ValueError):
        plot.spectrogram(sine, yscale="warped")

    plt.close("all")


@pytest.mark.parametrize('function', [
    (plot.time), (plot.group_delay), (plot.spectrogram)])
@pytest.mark.parametrize('unit', [
    (None), ('s'), ('ms'), ('mus'), ('samples')])
def test_time_unit(function, unit, impulse_group_delay):
    """Test plottin with different units."""
    print(f"Testing: {function.__name__} (unit={unit})")

    filename = f'line_{function.__name__}_unit_{str(unit)}'
    create_figure()
    function(impulse_group_delay[0], unit=unit)
    save_and_compare(create_baseline, baseline_path, output_path,
                     filename, file_type, compare_output)


def test_time_unit_assertion(sine):
    """Test if all line plots raise an assertion for a wrong unit parameter."""

    with raises(ValueError):
        plot.time(sine, unit="pascal")

    with raises(ValueError):
        plot.group_delay(sine, unit="pascal")

    with raises(ValueError):
        plot.spectrogram(sine, unit="pascal")

    plt.close("all")


def test_line_custom_subplots(sine, impulse_group_delay):
    """
    Test custom subplots in row, column, and mixed layout including hold
    functionality.
    """

    # plot layouts to be tested
    plots = {
        'row': [plot.time, plot.freq],
        'col': [[plot.time], [plot.freq]],
        'mix': [[plot.time, plot.freq],
                [plot.phase, plot.group_delay]]
    }

    for p in plots:
        print(f"Testing: {p}")

        # test initial plot
        filename = 'line_custom_subplots_' + p
        create_figure()
        plot.custom_subplots(sine, plots[p])
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)

        # test hold functionality
        filename = 'line_custom_subplots_' + p + '_hold'
        plot.custom_subplots(impulse_group_delay[0], plots[p])
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_line_time_data(time_data):
    """Test time data plot with default arguments."""
    function = plot.time
    print(f"Testing: {function.__name__}")

    filename = 'line_time_data_' + function.__name__
    create_figure()
    function(time_data)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.freq), (plot.phase), (plot.freq_phase)])
def test_line_frequency_data(function, frequency_data):
    """Test frequency data plot with default arguments."""
    print(f"Testing: {function.__name__}")

    filename = 'line_frequency_data_' + function.__name__
    create_figure()
    function(frequency_data)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


def test_spectrogram(sine):
    """Test spectrogram with default parameters"""
    function = plot.spectrogram

    print(f"Testing: {function.__name__}")

    filename = '2d_' + function.__name__
    create_figure()
    function(sine)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.time_2d), (plot.freq_2d), (plot.phase_2d), (plot.group_delay_2d),
    (plot.time_freq_2d), (plot.freq_phase_2d), (plot.freq_group_delay_2d)])
def test_2d_plots(function, impulse_45_channels):
    """Test all 2d plots with default arguments."""
    filename = '2d_' + function.__name__
    create_figure()
    function(impulse_45_channels[0])
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.time_2d), (plot.freq_2d), (plot.phase_2d),
    (plot.group_delay_2d), (plot.time_freq_2d), (plot.freq_phase_2d),
    (plot.freq_group_delay_2d)])
@pytest.mark.parametrize('points', [('default'), ('custom')])
@pytest.mark.parametrize('orientation', [('vertical'), ('horizontal')])
def test_2d_points_orientation(
        function, orientation, points, impulse_45_channels):
    """Test 2D plots with varing `points` and `orientation` parameters"""
    print(f"Testing: {function.__name__}")
    points_label = 'points-default' if points == 'default' else 'points-custom'
    signal = impulse_45_channels[0]
    points = impulse_45_channels[1] if points == 'custom' else None
    filename = f'2d_{function.__name__}_{orientation}_{points_label}'
    create_figure()
    function(signal, indices=points, orientation=orientation)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.spectrogram), (plot.time_2d), (plot.freq_2d), (plot.phase_2d),
    (plot.group_delay_2d)])
@pytest.mark.parametrize('colorbar', [('off'), ('axes')])
def test_2d_colorbar_options(function, colorbar, impulse_45_channels):
    """Test 2D color bar options"""
    print(f"Testing: {function.__name__} with colorbar {colorbar}")
    filename = f'2d_{function.__name__}_colorbar-{colorbar}'
    # pad zeros to get minimum signal length for spectrogram
    signal = impulse_45_channels[0]
    if function == plot.spectrogram:
        signal = pf.dsp.pad_zeros(signal, 2048 - signal.n_samples)
    fig = create_figure()
    if colorbar == "off":
        # test not plotting a colorbar
        function(signal, colorbar=False)
    elif colorbar == "axes":
        # test plotting colorbar to specified axis
        fig.clear()
        _, ax = plt.subplots(1, 2, num=fig.number)
        function(signal, ax=ax)
    save_and_compare(create_baseline, baseline_path, output_path,
                     filename, file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.spectrogram), (plot.time_2d), (plot.freq_2d), (plot.phase_2d),
    (plot.group_delay_2d)])
def test_2d_colorbar_assertion(function, impulse_45_channels):
    """
    Test assertion when passing an array of axes but not having a colorbar.
    """
    with raises(ValueError, match="A list of axes"):
        function(impulse_45_channels[0], colorbar=False,
                 ax=[plt.gca(), plt.gca()])


@pytest.mark.parametrize('function', [
    (plot.time_2d), (plot.freq_2d), (plot.phase_2d),
    (plot.group_delay_2d), (plot.time_freq_2d), (plot.freq_phase_2d),
    (plot.freq_group_delay_2d)])
def test_2d_cshape_assertion(function):
    """
    Test assertion when passing a signal with wrong cshape.
    """
    error_str = r"signal.cshape must be \(m, \) with m\>=2 but is \(2, 2\)"
    with raises(ValueError, match=error_str):
        function(pf.signals.impulse(10, [[0, 0], [0, 0]]))


@pytest.mark.parametrize('param', [
    (['2d_phase_deg', True, False]),
    (['2d_phase_unwrap', False, True]),
    (['2d_phase_deg_unwrap', True, True])])
def test_2d_phase_options(param, impulse_45_channels):
    """Test parameters that are unique to the phase plot."""
    print(f"Testing: {param[0]}")

    filename = param[0]
    create_figure()
    plot.phase_2d(impulse_45_channels[0], deg=param[1], unwrap=param[2])
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


def test_phase_2d_unwrap_assertion(impulse_45_channels):
    """Test assertion for unwrap parameter."""
    with raises(ValueError):
        plot.phase_2d(impulse_45_channels[0], unwrap='infinity')


@pytest.mark.parametrize('function', [
    (plot.time_2d), (plot.freq_2d)])
def test_2d_dB_option(function, impulse_45_channels):
    """Test all 2d plots that have a dB option."""
    # test if dB option is working
    for dB in [True, False]:
        print(f"Testing: {function.__name__} (dB={dB})")

        filename = '2d_' + function.__name__ + '_dB_' + str(dB)
        create_figure()
        function(impulse_45_channels[0], dB=dB)
        save_and_compare(create_baseline, baseline_path, output_path,
                         filename, file_type, compare_output)

    # test if log_prefix and log_reference are working
    print(f"Testing: {function.__name__} (log parameters)")

    filename = '2d_' + function.__name__ + '_logParams'
    create_figure()
    function(impulse_45_channels[0], log_prefix=10, log_reference=.5, dB=True)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.freq_2d), (plot.phase_2d), (plot.group_delay_2d)])
@pytest.mark.parametrize('xscale', [('log'), ('linear')])
def test_2d_xscale_option(function, xscale, impulse_45_channels):
    """Test all 2d plots that have an xscale option."""
    # test if xscale option is working
    print(f"Testing: {function.__name__} (xscale={xscale})")

    filename = '2d_' + function.__name__ + '_xscale_' + xscale
    create_figure()
    function(impulse_45_channels[0], xscale=xscale)
    save_and_compare(create_baseline, baseline_path, output_path,
                     filename, file_type, compare_output)


def test_2d_xscale_assertion(impulse_45_channels):
    """
    Test if all 2d plots raise an assertion for a wrong scale parameter.
    """

    with raises(ValueError):
        plot.freq(impulse_45_channels[0], xscale="warped")

    with raises(ValueError):
        plot.phase(impulse_45_channels[0], xscale="warped")

    with raises(ValueError):
        plot.group_delay(impulse_45_channels[0], xscale="warped")

    with raises(ValueError):
        plot.spectrogram(impulse_45_channels[0], yscale="warped")

    plt.close("all")


@pytest.mark.parametrize('function', [
    (plot.time_2d), (plot.group_delay_2d)])
@pytest.mark.parametrize('unit', [
    (None), ('s'), ('ms'), ('mus'), ('samples')])
def test_2d_time_unit(function, unit, impulse_45_channels):
    """Test plottin with different units."""
    print(f"Testing: {function.__name__} (unit={unit})")

    filename = f'2d_{function.__name__}_unit_{str(unit)}'
    create_figure()
    function(impulse_45_channels[0], unit=unit)
    save_and_compare(create_baseline, baseline_path, output_path,
                     filename, file_type, compare_output)


def test_2d_time_unit_assertion(impulse_45_channels):
    """Test if all 2d plots raise an assertion for a wrong unit parameter."""

    with raises(ValueError):
        plot.time_2d(impulse_45_channels[0], unit="pascal")

    with raises(ValueError):
        plot.group_delay_2d(impulse_45_channels[0], unit="pascal")

    plt.close("all")


def test_2d_time_data(impulse_45_channels):
    """Test 2d time data plot with default arguments."""
    function = plot.time
    time_data = pf.TimeData(
        impulse_45_channels[0].time, impulse_45_channels[0].times)
    print(f"Testing: {function.__name__}")

    filename = '2d_time_data_' + function.__name__
    create_figure()
    function(time_data)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


@pytest.mark.parametrize('function', [
    (plot.freq), (plot.phase), (plot.freq_phase)])
def test_2d_frequency_data(function):
    """Test 2d frequency data plot with default arguments."""
    frequency = np.sin(np.linspace(0, 2*np.pi, 45))*500 + 1000
    amplitude = 1 - .5 * np.sin(np.linspace(0, 2*np.pi, 45))
    phase = 2 * np.pi * np.sin(np.linspace(0, 2*np.pi, 45))
    signal = pf.signals.sine(frequency, 4410, amplitude=amplitude, phase=phase)
    frequency_data = pf.FrequencyData(signal.freq, signal.frequencies)
    print(f"Testing: {function.__name__}")

    filename = '2d_frequency_data_' + function.__name__
    create_figure()
    function(frequency_data)
    save_and_compare(create_baseline, baseline_path, output_path, filename,
                     file_type, compare_output)


def test_use():
    """Test if use changes the plot style."""

    for style in ["dark", "default"]:

        filename = 'use_' + style
        plot.utils.use(style)
        create_figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_freq_fft_norm_dB(noise):
    """Test correct log_prefix in plot.freq depending on fft_norm."""
    create_figure()
    noise.fft_norm = 'power'
    ax = plot.freq(noise)
    y_actual = ax.lines[0].get_ydata().flatten()
    y_desired = 10*np.log10(np.abs(noise.freq)).flatten()
    npt.assert_allclose(y_actual, y_desired, atol=1e-6)

    create_figure()
    noise.fft_norm = 'psd'
    ax = plot.freq(noise)
    y_actual = ax.lines[0].get_ydata().flatten()
    y_desired = 10*np.log10(np.abs(noise.freq)).flatten()
    npt.assert_allclose(y_actual, y_desired, atol=1e-6)


def test_time_freq_fft_norm_dB(noise):
    """Test correct log_prefix in plot.time_freq depending on fft_norm."""
    create_figure()
    noise.fft_norm = 'power'
    ax = plot.time_freq(noise)
    y_actual = ax[1].lines[0].get_ydata().flatten()
    y_desired = 10*np.log10(np.abs(noise.freq)).flatten()
    npt.assert_allclose(y_actual, y_desired, atol=1e-6)

    create_figure()
    noise.fft_norm = 'psd'
    ax = plot.time_freq(noise)
    y_actual = ax[1].lines[0].get_ydata().flatten()
    y_desired = 10*np.log10(np.abs(noise.freq)).flatten()
    npt.assert_allclose(y_actual, y_desired, atol=1e-6)
