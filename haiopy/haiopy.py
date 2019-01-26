import numpy as np

class Audio(object):

    """Docstring for Audio. """

    def __init__(self):
        """TODO: to be defined1. """


class Signal(Audio):

    """Docstring for Signal. """

    def __init__(self,
                 data,
                 samplingrate,
                 domain='time',
                 dtype=np.double,
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

        self._data = data
        self._samplingrate = samplingrate
        self._domain = domain
        self._dtype = dtype

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
        return np.fft.rfft(self._data)

    @freq.setter
    def freq(self, value):
        self._data = np.fft.irfft(value)

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
