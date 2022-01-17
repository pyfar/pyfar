import os
from pytest import raises
import matplotlib.pyplot as plt
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
# if there are any differences. In any case, differences are writted to
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
def test_line_plots(sine, impulse_group_delay):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.time,
                     plot.freq,
                     plot.phase,
                     plot.group_delay,
                     plot.time_freq,
                     plot.freq_phase,
                     plot.freq_group_delay]

    for function in function_list:
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


def test_line_phase_options(sine):
    """Test parameters that are unique to the phase plot."""

    parameter_list = [['line_phase_deg', True, False],
                      ['line_phase_unwrap', False, True],
                      ['line_phase_deg_unwrap', True, True]]

    for param in parameter_list:
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


def test_line_dB_option(sine):
    """Test all line plots that have a dB option."""

    function_list = [plot.time,
                     plot.freq,
                     plot.spectrogram]

    # test if dB option is working
    for function in function_list:
        for dB in [True, False]:
            print(f"Testing: {function.__name__} (dB={dB})")

            filename = 'line_' + function.__name__ + '_dB_' + str(dB)
            create_figure()
            function(sine, dB=dB)
            save_and_compare(create_baseline, baseline_path, output_path,
                             filename, file_type, compare_output)

    # test if log_prefix and log_reference are working
    for function in function_list:
        print(f"Testing: {function.__name__} (log parameters)")

        filename = 'line_' + function.__name__ + '_logParams'
        create_figure()
        function(sine, log_prefix=10, log_reference=.5, dB=True)
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_line_xscale_option(sine):
    """Test all line plots that have an xscale option."""

    function_list = [plot.freq,
                     plot.phase,
                     plot.group_delay]

    # test if dB option is working
    for function in function_list:
        for xscale in ['log', 'linear']:
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


def test_time_unit(impulse_group_delay):
    """Test plottin with different units."""
    function_list = [plot.time,
                     plot.group_delay,
                     plot.spectrogram]

    for function in function_list:
        for unit in [None, 's', 'ms', 'mus', 'samples']:
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
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.time]

    for function in function_list:
        print(f"Testing: {function.__name__}")

        filename = 'line_time_data_' + function.__name__
        create_figure()
        function(time_data)
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_line_frequency_data(frequency_data):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.freq,
                     plot.phase,
                     plot.freq_phase]

    for function in function_list:
        print(f"Testing: {function.__name__}")

        filename = 'line_frequency_data_' + function.__name__
        create_figure()
        function(frequency_data)
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_2d_plots(sine):
    """Test all 2D plots with default parameters"""
    function_list = [
        plot.spectrogram]

    for function in function_list:

        print(f"Testing: {function.__name__}")

        filename = '2d_' + function.__name__
        create_figure()
        function(sine)
        save_and_compare(create_baseline, baseline_path, output_path, filename,
                         file_type, compare_output)


def test_2d_colorbar_options(sine):
    """Test all 2D plots with default parameters"""
    function_list = [
        plot.spectrogram]

    for function in function_list:
        for cb_option in ["off", "axes"]:

            print(f"Testing: {function.__name__}")

            filename = '2d_' + cb_option + "_" + function.__name__
            fig = create_figure()
            if cb_option == "off":
                # test not plotting a colobar
                function(sine, colorbar=False)
            elif cb_option == "axes":
                # test plotting colorbar to specified axis
                fig.clear()
                _, ax = plt.subplots(1, 2, num=fig.number)
                function(sine, ax=ax)
            save_and_compare(create_baseline, baseline_path, output_path,
                             filename, file_type, compare_output)


def test_2d_plots_colorbar_assertion(sine):
    function_list = [
        plot.spectrogram]

    # test assertion when passing an array of axes but not having a colobar
    for function in function_list:
        with raises(ValueError, match="A list of axes"):
            function(sine, colorbar=False, ax=[plt.gca(), plt.gca()])


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
