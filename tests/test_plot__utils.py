import matplotlib.pyplot as plt
from pytest import raises
import pyfar.plot as plot


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


def test_lower_frequency_limit(
        sine, sine_short, frequency_data,
        frequency_data_one_point, time_data):
    """Test the private function plot._utils._lower_frequency_limit"""

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
    with raises(ValueError, match="Signals must have frequencies > 0 Hz"):
        plot._utils._lower_frequency_limit(frequency_data_one_point)

    # test TimeData assertions
    with raises(TypeError, match="Input data has to be of type"):
        plot._utils._lower_frequency_limit(time_data)


def test_time_auto_unit():
    """Test automatically assigning the unit in group delay plots."""
    assert plot._utils._time_auto_unit(0) == 's'
    assert plot._utils._time_auto_unit(1e-4) == 'mus'
    assert plot._utils._time_auto_unit(2e-2) == 'ms'
    assert plot._utils._time_auto_unit(2) == 's'
