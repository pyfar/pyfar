import numpy as np
import scipy.io.wavfile as wavfile
import sofa


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


def generate_test_sofa(dir='tests/test_io_data/'):
    """ Generate the reference sofa files used for testing the read_sofa function.
    Parameters
    -------
    dir : String
        Path to save the reference plots.
    """
    conventions = ['GeneralFIR', 'GeneralTF']
    n_measurements = 1
    n_receivers = 2
    n_samples = 1000

    for convention in conventions:
        sofafile = sofa.Database.create(
                        (dir + convention + '.sofa'),
                        convention,
                        dimensions={
                            "M": n_measurements,
                            "R": n_receivers,
                            "N": n_samples})
        sofafile.Listener.initialize(fixed=["Position"])
        sofafile.Receiver.initialize(fixed=["Position"])
        sofafile.Source.initialize(variances=["Position"])
        sofafile.Emitter.initialize(fixed=["Position"])

        sofafile.Data.create_attribute(
            "IR",
            np.random.random_sample((n_measurements, n_receivers, n_samples)))
