
from pyfar import Signal
import multiprocessing
import warnings
import scipy.signal

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    scipy.fftpack = pyfftw.interfaces.scipy_fftpack
except ImportError:
    warnings.warn(
        "Using numpy FFT implementation.\
        Install pyfftw for improved performance.")


def convolve_overlap_add(signal1, signal2, mode='full'):
    """Convolve two signals using the overlap-add algorithm.

    Parameters
    ----------
    signal1 : pyfar.Signal
        The first signal
    signal2 : pyfar.Signal
        The second signal
    mode : str {‘full’, ‘valid’, ‘same’}, optional
        A string indicating the size of the output:
            - `full`: The output is the full discrete linear convolution of
                the inputs. (Default)
            - `valid` : The output consists only of those elements that do not
                rely on the zero-padding. In ‘valid’ mode, either  in1 or in2
                must be at least as large as the other in every dimension.
            - `same` :  The output is the same size as in1, centered with
                respect to the ‘full’ output.

    Returns
    -------
    pyfar.Signal
        The result as a signal object.
    """
    if not signal1.sampling_rate == signal2.sampling_rate:
        raise ValueError("The sampling rates do not match")
    if not signal1.fft_norm == signal2.fft_norm:
        raise ValueError("FFT norms do not match.")

    res = scipy.signal.oaconvolve(
        signal1.time, signal2.time, mode=mode, axes=-1)

    return Signal(
        res, signal1.sampling_rate, domain='time', fft_norm=signal1.fft_norm)
