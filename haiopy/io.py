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

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * This function cannot read wav files with 24-bit data.
    """
    sampling_rate, data = wavfile.read(filename)
    signal = Signal(data.T, sampling_rate, domain='time')
    return signal


def write_wav(signal, filename, overwrite=True):
    """
    Write a signal as a WAV file.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the haiopy Signal class.

    filename : string or open file handle
        Output wav file.

    overwrite : bool
        Select wether to overwrite the WAV file, if it already exists.
        The default is True.

    Notes
    -----
    * This function is based on scipy.io.wavfile.write().
    * Writes a simple uncompressed WAV file.
    * The signal data of dimension larger than 2 is reshaped to 2D.
    * The bits-per-sample and PCM/float will be determined by the data-type.

    Common data types: [1]_

    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============

    Note that 8-bit PCM is unsigned.

    References
    ----------
    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming
       Interface and Data Specifications 1.0", section "Data Format of the
       Samples", August 1991
       http://www.tactilemedia.com/info/MCI_Control_Info.html

    """
    sampling_rate = signal.sampling_rate
    data = signal.time

    # Reshape to 2D
    data = data.reshape(-1, data.shape[-1])

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
        wavfile.write(filename, sampling_rate, data.T)
