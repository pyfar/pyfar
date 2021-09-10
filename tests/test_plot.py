import matplotlib
import matplotlib.pyplot as plt
import matplotlib.testing as mpt
from matplotlib.testing.compare import compare_images
import os
from pytest import raises

import pyfar.plot as plot

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
compare_output = True

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


# helper functions ------------------------------------------------------------
# Intended to reduce code redundancy and assure reproducibility on different
# operating systems
def create_figure(width=4.8, height=4.8, dpi=100):
    """
    Create figure with defined parameters for reproducible testing.
    Returns: fig
    """

    plt.close('all')
    matplotlib.use('Agg')
    mpt.set_reproducibility_for_testing()
    # force size/dpi for testing
    return plt.figure(1, (width, height), dpi)


def save_and_compare(create_baseline, filename, file_type, compare_output):
    """
    1. Save baseline and test files.
    2. Compare files
    """
    # file names for saving
    baseline = os.path.join(baseline_path, filename + "." + file_type)
    output = os.path.join(output_path, filename + "." + file_type)

    # safe baseline and test image
    if create_baseline:
        plt.savefig(baseline)
    plt.savefig(output)

    # compare images
    comparison = compare_images(baseline, output, tol=10)
    if compare_output:
        assert comparison is None


# testing ---------------------------------------------------------------------
def test_line_plots(sine, impulse_group_delay):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.line.time,
                     plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay,
                     plot.line.time_freq,
                     plot.line.freq_phase,
                     plot.line.freq_group_delay]

    for function in function_list:
        print(f"Testing: {function.__name__}")

        # initial plot
        filename = 'line_' + function.__name__
        create_figure()
        function(sine)
        save_and_compare(create_baseline, filename, file_type, compare_output)

        # test hold functionality
        filename = 'line_' + function.__name__ + '_hold'
        function(impulse_group_delay[0])
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_phase_options(sine):
    """Test parameters that are unique to the phase plot."""

    parameter_list = [['line_phase_deg', True, False],
                      ['line_phase_unwrap', False, True],
                      ['line_phase_deg_unwrap', True, True]]

    for param in parameter_list:
        print(f"Testing: {param[0]}")

        filename = param[0]
        create_figure()
        plot.line.phase(sine, deg=param[1], unwrap=param[2])
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_phase_unwrap_assertion(sine):
    """Test assertion for unwrap parameter."""
    with raises(ValueError):
        plot.line.phase(sine, unwrap='infinity')


def test_line_dB_option(sine):
    """Test all line plots that have a dB option."""

    function_list = [plot.line.time,
                     plot.line.freq,
                     plot.line.spectrogram]

    # test if dB option is working
    for function in function_list:
        for dB in [True, False]:
            print(f"Testing: {function.__name__} (dB={dB})")

            filename = 'line_' + function.__name__ + '_dB_' + str(dB)
            create_figure()
            function(sine, dB=dB)
            save_and_compare(create_baseline, filename, file_type, compare_output)

    # test if log_prefix and log_reference are working
    for function in function_list:
        print(f"Testing: {function.__name__} (log parameters)")

        filename = 'line_' + function.__name__ + '_logParams'
        create_figure()
        function(sine, log_prefix=10, log_reference=.5, dB=True)
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_xscale_option(sine):
    """Test all line plots that have an xscale option."""

    function_list = [plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay]

    # test if dB option is working
    for function in function_list:
        for xscale in ['log', 'linear']:
            print(f"Testing: {function.__name__} (xscale={xscale})")

            filename = 'line_' + function.__name__ + '_xscale_' + xscale
            create_figure()
            function(sine, xscale=xscale)
            save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_xscale_assertion(sine):
    """
    Test if all line plots raise an assertion for a wrong scale parameter.
    """

    with raises(ValueError):
        plot.line.freq(sine, xscale="warped")

    with raises(ValueError):
        plot.line.phase(sine, xscale="warped")

    with raises(ValueError):
        plot.line.group_delay(sine, xscale="warped")

    with raises(ValueError):
        plot.line.spectrogram(sine, yscale="warped")

    plt.close("all")


def test_time_unit(impulse_group_delay):
    """Test plottin with different units."""
    function_list = [plot.line.time,
                     plot.line.group_delay,
                     plot.line.spectrogram]

    for function in function_list:
        for unit in [None, 's', 'ms', 'mus', 'samples']:
            print(f"Testing: {function.__name__} (unit={unit})")

            filename = f'line_{function.__name__}_unit_{str(unit)}'
            create_figure()
            plot.line.group_delay(impulse_group_delay[0], unit=unit)
            save_and_compare(create_baseline, filename, file_type, compare_output)


def test_time_unit_assertion(sine):
    """Test if all line plots raise an assertion for a wrong unit parameter."""

    with raises(ValueError):
        plot.line.time(sine, unit="pascal")

    with raises(ValueError):
        plot.line.group_delay(sine, unit="pascal")

    with raises(ValueError):
        plot.line.spectrogram(sine, unit="pascal")

    plt.close("all")


def test_line_time_auto_unit():
    """Test automatically assigning the unit in group delay plots."""
    assert plot._line._time_auto_unit(0) == 's'
    assert plot._line._time_auto_unit(1e-4) == 'mus'
    assert plot._line._time_auto_unit(2e-2) == 'ms'
    assert plot._line._time_auto_unit(2) == 's'


def test_line_custom_subplots(sine, impulse_group_delay):
    """
    Test custom subplots in row, column, and mixed layout including hold
    functionality.
    """

    # plot layouts to be tested
    plots = {
        'row': [plot.line.time, plot.line.freq],
        'col': [[plot.line.time], [plot.line.freq]],
        'mix': [[plot.line.time, plot.line.freq],
                [plot.line.phase, plot.line.group_delay]]
    }

    for p in plots:
        print(f"Testing: {p}")

        # test initial plot
        filename = 'line_custom_subplots_' + p
        create_figure()
        plot.line.custom_subplots(sine, plots[p])
        save_and_compare(create_baseline, filename, file_type, compare_output)

        # test hold functionality
        filename = 'line_custom_subplots_' + p + '_hold'
        plot.line.custom_subplots(impulse_group_delay[0], plots[p])
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_time_data(time_data):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.line.time]

    for function in function_list:
        print(f"Testing: {function.__name__}")

        filename = 'line_time_data_' + function.__name__
        create_figure()
        function(time_data)
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_line_frequency_data(frequency_data):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.line.freq,
                     plot.line.phase,
                     plot.line.freq_phase]

    for function in function_list:
        print(f"Testing: {function.__name__}")

        filename = 'line_frequency_data_' + function.__name__
        create_figure()
        function(frequency_data)
        save_and_compare(create_baseline, filename, file_type, compare_output)


def test_2d_plots(sine):
    """Test all 2D plots with default parameters"""
    function_list = [
        plot.spectrogram]

    for function in function_list:

        print(f"Testing: {function.__name__}")

        filename = '2d_' + function.__name__
        create_figure()
        function(sine)
        save_and_compare(create_baseline, filename, file_type, compare_output)


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
            save_and_compare(create_baseline, filename, file_type, compare_output)


def test_2d_plots_colorbar_assertion(sine):
    function_list = [
        plot.spectrogram]

    # test assertion when passing an array of axes but not having a colobar
    for function in function_list:
        with raises(ValueError, match="A list of axes"):
            function(sine, colorbar=False, ax=[plt.gca(), plt.gca()])


def test_prepare_plot():
    # test without arguments
    plot._line._prepare_plot()

    # test with single axes object
    fig = plt.gcf()
    ax = plt.gca()
    plot._line._prepare_plot(ax)
    plt.close()

    # test with list of axes
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._line._prepare_plot(ax)
    plt.close()

    # test with numpy array of axes
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._line._prepare_plot(ax)
    plt.close()

    # test with list of axes and desired subplot layout
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._line._prepare_plot(ax, (2, 2))
    plt.close()

    # test with numpy array of axes and desired subplot layout
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._line._prepare_plot(ax, (2, 2))
    plt.close()

    # test without axes and with desired subplot layout
    plot._line._prepare_plot(None, (2, 2))
    plt.close()


def test_lower_frequency_limit(
        sine, sine_short, frequency_data,
        frequency_data_one_point, time_data):
    """Test the private function plot._line._lower_frequency_limit"""

    # test Signal with frequencies below 20 Hz
    low = plot._line._lower_frequency_limit(sine)
    assert low == 20

    # test Signal with frequencies above 20 Hz
    low = plot._line._lower_frequency_limit(sine_short)
    assert low == 44100/100  # lowest frequency fs=44100 / n_samples=100

    # test with FrequencyData
    # (We only need to test if FrequencyData works. The frequency dependent
    # cases are already tested above)
    low = plot._line._lower_frequency_limit(frequency_data)
    assert low == 100

    # test only 0 Hz assertions
    with raises(ValueError, match="Signals must have frequencies > 0 Hz"):
        plot._line._lower_frequency_limit(frequency_data_one_point)

    # test TimeData assertions
    with raises(TypeError, match="Input data has to be of type"):
        plot._line._lower_frequency_limit(time_data)


def test_use():
    """Test if use changes the plot style."""

    for style in ["dark", "default"]:

        filename = 'use_' + style
        plot.utils.use(style)
        create_figure()
        plt.plot([1, 2, 3], [1, 2, 3])
        save_and_compare(create_baseline, filename, file_type, compare_output)
