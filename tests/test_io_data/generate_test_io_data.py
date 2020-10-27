import numpy as np
import scipy.io.wavfile as wavfile


def generate_test_wavs(filepath="tests/test_io_data"):
    """ Generate wav file for unit tests."""
    signal, sampling_rate = reference_signal()
    filename = filepath + "/test_wav.wav"
    # Create testfile
    wavfile.write(filename, sampling_rate, signal)


def reference_signal():
    """ Generate sine of 440 Hz as numpy array.
    Returns
    -------
    sine : Signal
        The sine signal
    """
    sampling_rate = 44100
    n_periods = 20
    amplitude = 1
    frequency = 440

    # time signal:
    times = np.arange(0, n_periods*frequency) / sampling_rate
    sine = amplitude * np.sin(2 * np.pi * times * frequency)

    return sine, sampling_rate
