import numpy as np
import scipy.io.wavfile as wavfile
import sofa
import os.path


baseline_path = 'tests/test_io_data'


def generate_test_wavs(filepath="tests/test_io_data"):
    """ Generate wav file for unit tests."""
    signal, sampling_rate = reference_signal()
    filename = filepath + "/test_wav.wav"
    # Create testfile
    wavfile.write(filename, sampling_rate, signal)


def generate_test_sofas():
    """ Generate the reference sofa files used for testing the read_sofa function.
    Parameters
    -------
    dir : String
        Path to save the reference plots.
    """
    conventions = [
        'GeneralFIR',
        'GeneralTF',
        'GeneralFIR_unit',
        'GeneralFIR_postype']
    n_measurements = 1
    n_receivers = 2
    signal, sampling_rate = reference_signal(
        shape=(n_measurements, n_receivers))
    n_samples = signal.shape[-1]

    for convention in conventions:
        filename = os.path.join(baseline_path, (convention + '.sofa'))
        sofafile = sofa.Database.create(
                        filename,
                        convention.split('_')[0],
                        dimensions={
                            "M": n_measurements,
                            "R": n_receivers,
                            "N": n_samples})
        sofafile.Listener.initialize(fixed=["Position", "View", "Up"])
        sofafile.Source.initialize(fixed=["Position", "View", "Up"])
        sofafile.Source.Position = reference_coordinates()[0]
        sofafile.Receiver.initialize(fixed=["Position", "View", "Up"])
        sofafile.Receiver.Position = reference_coordinates()[1]
        sofafile.Emitter.initialize(fixed=["Position", "View", "Up"], count=1)

        if convention == 'GeneralFIR':
            sofafile.Data.Type = 'FIR'
            sofafile.Data.initialize()
            sofafile.Data.IR = signal
            sofafile.Data.SamplingRate = sampling_rate
        elif convention == 'GeneralTF':
            sofafile.Data.Type = 'TF'
            sofafile.Data.initialize()
            sofafile.Data.Real = signal
            sofafile.Data.Imag = signal
        elif convention == 'GeneralFIR_unit':
            sofafile.Data.Type = 'FIR'
            sofafile.Data.initialize()
            sofafile.Data.IR = signal
            sofafile.Data.SamplingRate = sampling_rate
            sofafile.Data.SamplingRate.Units = 'not_hertz'
        elif convention == 'GeneralFIR_postype':
            sofafile.Data.Type = 'FIR'
            sofafile.Data.initialize()
            sofafile.Data.IR = signal
            sofafile.Data.SamplingRate = sampling_rate
            sofafile.Source.Position.Type = 'not_type'
        sofafile.close()


def reference_signal(shape=(1,)):
    """ Generate sine of 440 Hz as numpy array.
    Returns
    -------
    sine : ndarray
        The sine signal
    sampling_rate : int
        The sampling rate
    """
    sampling_rate = 44100
    n_periods = 20
    amplitude = 1
    frequency = 440

    # time signal
    times = np.arange(0, n_periods*frequency) / sampling_rate
    sine = amplitude * np.sin(2 * np.pi * times * frequency)

    shape + (3,)
    sine = np.ones(shape + (sine.shape[-1],)) * sine

    return sine, sampling_rate


def reference_coordinates():
    """ Generate coordinate array
    Returns
    -------
    coordinates : ndarray
        The coordinates
    """
    source_coordinates = np.ones((1, 3))
    receiver_coordinates = np.ones((2, 3, 1))
    return source_coordinates, receiver_coordinates


generate_test_sofas()
