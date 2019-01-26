import numpy as np

class Audio(object):

    """Docstring for Audio. """

    def __init__(self):
        """TODO: to be defined1. """


class Signal(Audio):
    """

    """

    def __init__(self,
                 data,
                 samplingrate,
                 domain='time',
                 dtype=None,
                 position=None,
                 orientation=None):
        """TODO: to be defined1.

        Parameters
        ----------
        data : TODO
        samplingrate : TODO
        domain : TODO
        dtype : TODO, optional


        """
        Audio.__init__(self)

        if not dtype:
            self._dtype = data.dtype
        else:
            self._dtype = dtype
        self._data = np.asarray(data, dtype=dtype)
        self._samplingrate = samplingrate
        self._domain = domain

    @property
    def n_samples(self):
        """Number of samples."""
        return self._data.shape[-1]

    @property
    def frequencies(self):
        """Frequencies of the discrete signal spectrum."""
        return np.fft.rfftfreq(self.n_samples, d=1/self.samplingrate)

    @property
    def times(self):
        """Time instances the signal is sampled at."""
        return np.arange(0, self.n_samples) / self.samplingrate

    @property
    def time(self):
        """The signal data in the time domain."""
        return self._data

    @time.setter
    def time(self, value):
        self._data = value

    @property
    def freq(self):
        """The signal data in the frequency domain."""
        if self.iscomplex:
            freq = np.fft.rfft(self._data)
        else:
            freq = np.fft.fft(self._data)
        return freq

    @freq.setter
    def freq(self, value):
        if self.iscomplex:
            self.data = np.fft.ifft(value)
        else:
            self.data = np.fft.irfft(value)

    @property
    def samplingrate(self):
        """The sampling rate of the signal."""
        return self._samplingrate

    @samplingrate.setter
    def samplingrate(self, value):
        self._samplingrate = value

    @property
    def dtype(self):
        """The data type of the signal. This can be any data type and precision
        supported by numpy."""
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def iscomplex(self):
        if 'complex' in str(self.dtype):
            iscomplex = True
        else:
            iscomplex = False
        return iscomplex
