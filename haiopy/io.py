import numpy as np
import scipy.io.wavfile as wavfile
import os.path
import warnings

from haiopy import Signal


def read_wav(filename):
    """
    Import a WAV file as signal object.

    This method is based on scipy.io.wavfile.read().

    Parameters
    ----------
    filename : string or open file handle
        Input wav file.

    Returns
    -------
    signal : signal instance
        An audio signal object from the haiopy Signal class
        containing the audio data from the WAV file.
    """
    sampling_rate, data = wavfile.read(filename)
    signal = Signal(data.T, sampling_rate, domain='time')
    return signal


def write_wav(signal, filename, overwrite=False):
    """
    Write a signal as a WAV file.

    If the signals maximum absolute amplitude is larger than 1,
    it is normalized before writing the WAV file.
    This method is based on scipy.io.wavfile.write().

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class.

    filename : string or open file handle
        Output wav file.

    overwrite : bool
        Select wether to overwrite the WAV file, if it already exists.
        The default is False.
    """
    sampling_rate = signal.sampling_rate
    data = signal.time.T

    # Normalize, if maximum absolute amplitude is larger than 1
    if np.max(np.abs(data)) > 1:
        warnings.warn("Normalizing the data for wav export.")
        data = 0.99 * data/np.max(np.abs(data))

    # Check for .wav file extension
    if filename.split('.')[-1] != 'wav':
        warnings.warn("Extending filename by .wav.")
        filename += '.wav'

    # Check if file exists and for overwrite
    if overwrite is False and os.path.isfile(filename):
        raise FileExistsError(
                "File already exists,"
                "use overwrite option to disable error.")
    else:
        wavfile.write(filename, sampling_rate, data)
