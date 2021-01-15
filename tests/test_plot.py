import matplotlib
import matplotlib.pyplot as plt
import matplotlib.testing as mpt
from matplotlib.testing.compare import compare_images
import os

import pyfar.plot as plot

# flag for creating new baseline plots (required if the plot look changed)
create_baseline = False

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

# figure parameters
f_width = 4.8
f_height = 4.8
f_dpi = 100


def test_line_plots(sine, impulse_group_delay):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.line.time,
                     plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay,
                     plot.line.spectrogram,
                     plot.line.time_freq,
                     plot.line.freq_phase,
                     plot.line.freq_group_delay]

    for function in function_list:
        print(f"Testing: {function.__name__}")
        # file names
        filename = 'line_' + function.__name__ + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        function(sine)

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
        compare_images(baseline, output, tol=10)

        # test hold functionality
        # file names
        filename = 'line_' + function.__name__ + '_hold.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        function(impulse_group_delay[0])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_phase_options(sine):
    """Test parameters that are unique to the phase plot."""

    parameter_list = [['line_phase_deg.png', True, False],
                      ['line_phase_unwrap.png', False, True],
                      ['line_phase_deg_unwrap.png', True, True]]

    for param in parameter_list:
        print(f"Testing: {param[0]}")
        # file names
        filename = param[0]
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)
        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        plot.line.phase(sine, deg=param[1], unwrap=param[2])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)
        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_dB_option(sine):
    """Test all line plots that have a dB option."""

    function_list = [plot.line.time,
                     plot.line.freq]

    # test if dB option is working
    for function in function_list:
        for dB in [True, False]:
            print(f"Testing: {function.__name__} (dB={dB})")
            # file names
            filename = 'line_' + function.__name__ + '_dB_' + str(dB) + '.png'
            baseline = os.path.join(baseline_path, filename)
            output = os.path.join(output_path, filename)

            # plotting
            matplotlib.use('Agg')
            mpt.set_reproducibility_for_testing()
            plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
            function(sine, dB=dB)

            # save baseline if it does not exist
            # make sure to visually check the baseline uppon creation
            if create_baseline:
                plt.savefig(baseline)
            # safe test image
            plt.savefig(output)

            # close current figure
            plt.close()

            # testing
            compare_images(baseline, output, tol=10)

    # test if log_prefix and log_reference are working
    for function in function_list:
        print(f"Testing: {function.__name__} (log parameters)")
        # file names
        filename = 'line_' + function.__name__ + '_logParams.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
        function(sine, log_prefix=10, log_reference=.5, dB=True)

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_xscale_option(sine):
    """Test all line plots that have an xscale option."""

    function_list = [plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay]

    # test if dB option is working
    for function in function_list:
        for xscale in ['log', 'linear']:
            print(f"Testing: {function.__name__} (xscale={xscale})")
            # file names
            filename = 'line_' + function.__name__ + '_xscale_' + xscale + \
                       '.png'
            baseline = os.path.join(baseline_path, filename)
            output = os.path.join(output_path, filename)

            # plotting
            matplotlib.use('Agg')
            mpt.set_reproducibility_for_testing()
            plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
            function(sine, xscale=xscale)

            # save baseline if it does not exist
            # make sure to visually check the baseline uppon creation
            if create_baseline:
                plt.savefig(baseline)
            # safe test image
            plt.savefig(output)

            # close current figure
            plt.close()

            # testing
            compare_images(baseline, output, tol=10)


def test_line_group_delay_unit(sine, impulse_group_delay):
    """Test plottin the group delay with different units."""

    for unit in [None, 's', 'ms', 'mus', 'samples']:
        print(f"Testing: group_delay (unit={unit})")
        # file names
        filename = 'line_group_delay_unit_' + str(unit) + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
        plot.line.group_delay(impulse_group_delay[0], unit=unit)

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_group_delay_auto_unit():
    """Test automatically assigning the unit in group delay plots."""
    assert plot._line._group_delay_auto_unit(0) == 's'
    assert plot._line._group_delay_auto_unit(1e-4) == 'mus'
    assert plot._line._group_delay_auto_unit(2e-2) == 'ms'
    assert plot._line._group_delay_auto_unit(2) == 's'


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
        # file names
        filename = 'line_custom_subplots_' + p + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        plot.line.custom_subplots(sine, plots[p])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
        compare_images(baseline, output, tol=10)

        # test hold functionality
        # file names
        filename = 'line_custom_subplots_' + p + '_hold.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        plot.line.custom_subplots(impulse_group_delay[0], plots[p])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


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
